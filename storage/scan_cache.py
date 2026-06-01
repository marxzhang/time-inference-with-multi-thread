"""
storage/scan_cache.py — ScanStage 断点续传缓存

解决的问题
----------
sha1 计算（读全文件）+ phash（解码图片）在大批量文件时极慢。
相同文件第二次扫描时，这些结果完全不需要重算。

Cache Key 设计
--------------
    key = "{relpath}|{filesize}|{mtime_int}"

    不用 sha1 作 key（sha1 本身就是最慢的操作）。
    用 (relpath, filesize, mtime) 组合：
    - stat() 微秒级，无需读文件内容
    - 三者任一变化 → key 变 → cache miss → 重新扫描
    - 文件移动但内容不变：relpath 变 → miss（重新扫描，但 sha1 cache 仍有效）

Value 设计
----------
    存储 item.to_dict() 的完整序列化，恢复时还原所有字段。
    不存 evidence / time_result（这些由后续 stage 填充，每次重跑）。

存储格式
--------
    cache_dir/scan_cache.jsonl
    每行：{"key": "...", "value": {item_dict}}
    append-only，compact() 去重。

与 Cache 类的区别
-----------------
    Cache   : key = sha1（内容寻址），跨任务复用
    ScanCache: key = (relpath, filesize, mtime)（路径+stat 寻址），单任务加速
    两者存储格式相同，但语义和失效策略不同，独立实现避免混淆。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional


_CACHE_FILENAME = "scan_cache.jsonl"

# item.to_dict() 中哪些字段要存入 scan cache
# 不存 evidence / time_result / logs（这些每次重新生成）
_CACHED_FIELDS = {
    "id", "path", "relpath", "filename", "ext",
    "filesize", "ctime", "mtime",
    "sha1", "phash", "dhash",
    "width", "height", "format_real",
    "is_screenshot", "is_scan",
    "livp_source_path",
    "duplicate_group_id", "duplicate_rank", "duplicate_kind",
    "flags",     # UNSUPPORTED 等 scan 阶段打的 flag
    "warnings",  # image open failed 等
    "exif",      # ExifStage 之前为空，但结构已建立
}


class ScanCache:
    """
    ScanStage 断点续传缓存。

    使用示例（scan.py 内部）
    -------------------------
    sc = ScanCache(ctx.config.cache_dir)
    sc.preload()                         # 启动时一次性加载进内存

    # 扫描时
    cached = sc.get(relpath, filesize, mtime)
    if cached:
        item = Item.from_dict(cached)    # 直接恢复，跳过 sha1/phash
    else:
        item = self._build_item(...)     # 正常扫描
        sc.set(relpath, filesize, mtime, item.to_dict())  # 写入缓存

    sc.flush()                           # 扫描完成后写盘（或每 N 条自动写）
    """

    def __init__(self, cache_dir: str, flush_interval: int = 500) -> None:
        self._path = Path(cache_dir) / _CACHE_FILENAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict[str, Any]] = {}  # key → item_dict
        self._dirty: list[tuple[str, dict]] = []      # 待写盘的条目
        self._loaded = False
        self._lock = threading.Lock()  # 保护 _store / _dirty 和文件写入
        # 积累多少条 dirty 自动写盘。500条崩溃最多丢500条，单次写盘<5ms。
        self._flush_interval = flush_interval

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def preload(self) -> int:
        """
        从磁盘加载缓存到内存。
        在 scan 开始前调用一次，之后 get() 全是内存操作。
        返回加载的条目数。
        """
        if self._loaded:
            return len(self._store)
        self._load()
        self._loaded = True
        return len(self._store)

    def get(
        self,
        relpath: str,
        filesize: int,
        mtime: float,
    ) -> Optional[dict[str, Any]]:
        """
        查询缓存。命中返回 item_dict，未命中返回 None。
        """
        if not self._loaded:
            self.preload()
        return self._store.get(_make_key(relpath, filesize, mtime))

    def set(
        self,
        relpath: str,
        filesize: int,
        mtime: float,
        item_dict: dict[str, Any],
    ) -> None:
        """
        写入缓存（内存 + 加入待写盘队列）。
        item_dict 会被过滤，只保留 _CACHED_FIELDS。
        """
        key = _make_key(relpath, filesize, mtime)
        filtered = {k: v for k, v in item_dict.items() if k in _CACHED_FIELDS}
        with self._lock:
            self._store[key] = filtered
            self._dirty.append((key, filtered))
            should_flush = len(self._dirty) >= self._flush_interval
        # flush 在锁外执行，不阻塞其他线程继续 set
        if should_flush:
            self.flush()

    def flush(self, compact_threshold: int = 5000) -> int:
        """
        把待写盘条目追加到文件。
        dirty 条目超过 compact_threshold 时自动压缩去重。
        返回写入的条目数。
        """
        with self._lock:
            if not self._dirty:
                return 0
            to_write = self._dirty[:]
            self._dirty.clear()
            store_size = len(self._store)

        with open(self._path, "a", encoding="utf-8") as f:
            for key, value in to_write:
                f.write(
                    json.dumps({"key": key, "value": value}, ensure_ascii=False)
                    + "\n"
                )

        if store_size > compact_threshold:
            self.compact()

        return len(to_write)

    def compact(self) -> int:
        """
        重写文件，去除重复 key（保留最新值）。
        返回压缩后的条目数。
        """
        if not self._loaded:
            self.preload()

        with open(self._path, "w", encoding="utf-8") as f:
            for key, value in self._store.items():
                f.write(
                    json.dumps({"key": key, "value": value}, ensure_ascii=False)
                    + "\n"
                )
        return len(self._store)

    def stats(self) -> dict[str, Any]:
        return {
            "file": str(self._path),
            "loaded": self._loaded,
            "entries": len(self._store),
            "pending_flush": len(self._dirty),
            "file_exists": self._path.exists(),
            "file_size_kb": (
                round(self._path.stat().st_size / 1024, 1)
                if self._path.exists() else 0
            ),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._store[record["key"]] = record["value"]
                except (json.JSONDecodeError, KeyError):
                    pass  # 损坏的行跳过

    def __repr__(self) -> str:
        return (
            f"ScanCache(entries={len(self._store)}, "
            f"pending={len(self._dirty)}, "
            f"loaded={self._loaded})"
        )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _make_key(relpath: str, filesize: int, mtime: float) -> str:
    """
    生成 cache key。
    mtime 取整到秒（跨平台精度统一，避免浮点抖动）。
    """
    mtime_int = int(mtime)
    return f"{relpath}|{filesize}|{mtime_int}"