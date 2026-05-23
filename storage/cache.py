"""
storage/cache.py — 计算结果缓存（item 级，跨运行持久化）

解决的问题
----------
CLIP embedding、phash、OCR 等计算成本高，同一张图（相同 sha1）
不应该在每次运行时重复计算。

设计
----
- key   = "{namespace}:{sha1}"
          namespace 区分不同类型的缓存（clip / phash / ocr / face …）
          sha1 唯一标识图片内容（内容变了自动失效）
- value = 任意 JSON-serializable 对象
- 存储  = 每个 namespace 一个 .jsonl 文件，懒加载到内存 dict
- 失效  = 不设 TTL，靠 sha1 内容寻址自动失效；
          手动调用 clear(namespace) 可清除整个 namespace

目录结构（cache_dir 下）
------------------------
    cache/
    ├── clip.jsonl       # CLIP embedding
    ├── phash.jsonl      # 感知 hash（如果改用独立计算）
    ├── ocr.jsonl        # OCR 文字
    └── face.jsonl       # 人脸 embedding

.jsonl 格式（每行一条记录）
----------------------------
    {"key": "clip:abc123...", "value": [[0.1, 0.2, ...]]}

为什么用 .jsonl 而不是 sqlite / shelve？
    - 零依赖，纯标准库
    - 人类可读，方便调试
    - append-only 写入，不需要加锁（单进程场景）
    - 规模上限：单个文件 ~10 万条以内都没问题
    后期数据量大了可以无缝换成 sqlite，接口不变。

线程安全
--------
    当前版本：单进程单线程，不加锁。
    后期多线程：每个 namespace 加一把 threading.Lock 即可。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


class Cache:
    """
    计算结果缓存，跨运行持久化。

    使用示例
    --------
    cache = Cache("/path/to/cache_dir")

    # 写入
    cache.set("clip", item.sha1, embedding_list)

    # 读取
    embedding = cache.get("clip", item.sha1)   # 未命中返回 None
    if embedding is None:
        embedding = model.encode(image)
        cache.set("clip", item.sha1, embedding)

    # 批量预加载（处理大量 item 前调用，避免逐条 I/O）
    cache.preload("clip")
    """

    def __init__(self, cache_dir: str) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # namespace → {sha1: value} 的内存 dict
        # 懒加载：第一次访问某 namespace 时才读文件
        self._store: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def get(self, namespace: str, sha1: str) -> Optional[Any]:
        """
        读取缓存值。未命中返回 None。

        参数
        ----
        namespace : 缓存类型，如 "clip" / "ocr" / "phash"
        sha1      : 图片内容的 SHA-1 哈希
        """
        self._ensure_loaded(namespace)
        return self._store[namespace].get(sha1)

    def set(self, namespace: str, sha1: str, value: Any) -> None:
        """
        写入缓存值，同时追加到磁盘文件。

        参数
        ----
        value : 任意 JSON-serializable 对象
                （list、dict、str、float、int、None）
        """
        self._ensure_loaded(namespace)
        self._store[namespace][sha1] = value

        # append-only 写磁盘
        record = json.dumps({"key": sha1, "value": value}, ensure_ascii=False)
        with open(self._namespace_path(namespace), "a", encoding="utf-8") as f:
            f.write(record + "\n")

    def has(self, namespace: str, sha1: str) -> bool:
        """key 是否已缓存。"""
        self._ensure_loaded(namespace)
        return sha1 in self._store[namespace]

    # ------------------------------------------------------------------
    # 批量操作
    # ------------------------------------------------------------------

    def preload(self, namespace: str) -> int:
        """
        强制加载整个 namespace 到内存，返回加载的条目数。
        处理大批量 item 前调用，避免 get() 时逐条触发 I/O。
        """
        self._load(namespace)
        return len(self._store.get(namespace, {}))

    def preload_all(self) -> dict[str, int]:
        """
        加载 cache_dir 下所有已有的 namespace，返回 {namespace: 条目数}。
        """
        result = {}
        for path in self._dir.glob("*.jsonl"):
            ns = path.stem
            result[ns] = self.preload(ns)
        return result

    # ------------------------------------------------------------------
    # 维护操作
    # ------------------------------------------------------------------

    def clear(self, namespace: str) -> None:
        """
        清除整个 namespace（内存 + 磁盘）。
        适用于：模型换版本后 embedding 全部失效。
        """
        self._store.pop(namespace, None)
        path = self._namespace_path(namespace)
        if path.exists():
            path.unlink()

    def compact(self, namespace: str) -> int:
        """
        压缩 .jsonl 文件：重写去重（append-only 可能有重复 key）。
        返回压缩后的条目数。

        append-only 写入时，同一个 sha1 被 set 两次会产生两行记录，
        _load() 读取时后者覆盖前者（正确），但文件会有冗余。
        定期调用 compact() 清理。
        """
        self._ensure_loaded(namespace)
        store = self._store.get(namespace, {})
        path = self._namespace_path(namespace)

        with open(path, "w", encoding="utf-8") as f:
            for sha1, value in store.items():
                record = json.dumps({"key": sha1, "value": value}, ensure_ascii=False)
                f.write(record + "\n")

        return len(store)

    def stats(self) -> dict[str, int]:
        """
        返回各 namespace 的内存中条目数（只统计已加载的）。
        """
        return {ns: len(store) for ns, store in self._store.items()}

    # ------------------------------------------------------------------
    # 预留槽位（后期按需实现）
    # ------------------------------------------------------------------

    # def get_batch(self, namespace: str, sha1s: list[str]) -> dict[str, Any]:
    #     """批量读取，返回命中的 {sha1: value} dict。"""
    #     self._ensure_loaded(namespace)
    #     store = self._store[namespace]
    #     return {s: store[s] for s in sha1s if s in store}

    # def set_batch(self, namespace: str, items: dict[str, Any]) -> None:
    #     """批量写入，减少文件 I/O 次数。"""
    #     self._ensure_loaded(namespace)
    #     self._store[namespace].update(items)
    #     path = self._namespace_path(namespace)
    #     with open(path, "a", encoding="utf-8") as f:
    #         for sha1, value in items.items():
    #             f.write(json.dumps({"key": sha1, "value": value}) + "\n")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _namespace_path(self, namespace: str) -> Path:
        return self._dir / f"{namespace}.jsonl"

    def _ensure_loaded(self, namespace: str) -> None:
        """懒加载：首次访问时才读磁盘。"""
        if namespace not in self._store:
            self._load(namespace)

    def _load(self, namespace: str) -> None:
        """
        从 .jsonl 文件加载到内存。
        重复 key 取最后一条（compact 前的正确行为）。
        文件不存在则初始化为空 dict。
        """
        store: dict[str, Any] = {}
        path = self._namespace_path(namespace)

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        store[record["key"]] = record["value"]
                    except (json.JSONDecodeError, KeyError):
                        # 损坏的行跳过，不影响其他记录
                        pass

        self._store[namespace] = store

    def __repr__(self) -> str:
        loaded = list(self._store.keys())
        return f"Cache(dir={self._dir}, loaded={loaded})"