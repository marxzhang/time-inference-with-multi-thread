"""
storage/checkpoint.py — 运行状态缓存（pipeline 级，断点续传）

解决的问题
----------
处理几万张照片时，SimilarStage / CLIP 等耗时 stage 中途崩溃，
重启后不应该从头开始——已完成的 item 直接跳过。

与 cache.py 的区别
------------------
    Cache      : 存计算"结果"（embedding、OCR文字）
                 key = sha1（内容寻址），跨任务复用
    Checkpoint : 存运行"状态"（哪些 item 的哪些 stage 已完成）
                 key = item_id（路径/任务相关），任务完成后可清除

设计
----
- 存储  = cache_dir/checkpoint.jsonl，每行一条 item 快照
- 触发  = 每处理 N 个 item（save_interval），或显式调用 save_item()
- 恢复  = 启动时调用 load()，拿到上次未完成的 item 列表继续跑
- 格式  = item.to_dict() 的完整序列化，包含 completed_stages

文件格式
--------
    每行一个 item 的完整 JSON 快照（to_dict() 结果）：
    {"id": "uuid", "path": "...", "completed_stages": [...], ...}

    同一 item 可能出现多行（多次保存），加载时取最后一条。

恢复策略
--------
    恢复后，item 的 completed_stages 已有记录，
    Stage.run() 的幂等检查（idempotent=True）会自动跳过已完成的 stage，
    无需 Checkpoint 和 Stage 之间有任何耦合。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.item import Item


_CHECKPOINT_FILENAME = "checkpoint.jsonl"


class Checkpoint:
    """
    运行状态持久化，支持断点续传。

    使用示例（scheduler 中）
    -------------------------
    ckpt = Checkpoint(cache_dir, save_interval=100)

    # 启动时：尝试恢复
    items = ckpt.restore(items)  # 已完成的 item 保留状态，未完成的重新跑

    # 每处理一个 item 后：
    ckpt.on_item_done(item)      # 达到 interval 时自动批量写盘

    # 正常结束时：
    ckpt.flush()                 # 把剩余未写的 item 写盘
    ckpt.clear()                 # 任务成功完成，删除 checkpoint 文件
    """

    def __init__(
        self,
        cache_dir: str,
        save_interval: int = 50,
    ) -> None:
        """
        参数
        ----
        cache_dir     : checkpoint 文件的存放目录
        save_interval : 每处理多少个 item 写一次盘（权衡安全性和 I/O 开销）
        """
        self._path = Path(cache_dir) / _CHECKPOINT_FILENAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval

        # 待写盘的 item 队列
        self._pending: list["Item"] = []
        self._done_count: int = 0

    # ------------------------------------------------------------------
    # 断点恢复
    # ------------------------------------------------------------------

    def restore(self, items: list["Item"]) -> list["Item"]:
        """
        用 checkpoint 数据恢复 item 状态。

        流程：
            1. 读取 checkpoint，建立 {item_id: item_dict} 映射
            2. 对 items 列表中每个 item，若 checkpoint 中有记录，
               则用记录中的 completed_stages / evidence / time_result 等字段覆盖
            3. 返回更新后的 items 列表（顺序不变，只恢复状态）

        注意：
            不过滤掉已完成的 item，只恢复状态。
            Stage.run() 的幂等检查负责跳过已完成的 stage。
            这样 pipeline 结构无需感知 checkpoint。
        """
        saved = self._load_saved()
        if not saved:
            return items

        restored_count = 0
        for item in items:
            if item.id in saved:
                self._restore_item(item, saved[item.id])
                restored_count += 1

        return items

    # ------------------------------------------------------------------
    # 增量保存
    # ------------------------------------------------------------------

    def on_item_done(self, item: "Item") -> None:
        """
        每个 item 处理完后调用。
        达到 save_interval 时自动批量写盘。
        """
        self._pending.append(item)
        self._done_count += 1

        if len(self._pending) >= self.save_interval:
            self.flush()

    def flush(self) -> int:
        """
        把所有待写盘的 item 写入 checkpoint 文件。
        返回成功写入的条目数。

        逐条写入并捕获异常：单条序列化失败不影响其他 item，
        失败的 item 会被跳过（下次运行时重新处理该 item，代价可接受）。
        """
        if not self._pending:
            return 0

        count = 0
        with open(self._path, "a", encoding="utf-8") as f:
            for item in self._pending:
                try:
                    record = json.dumps(item.to_dict(), ensure_ascii=False)
                    f.write(record + "\n")
                    count += 1
                except (TypeError, ValueError) as e:
                    # 序列化失败：跳过该 item，不中断整体流程
                    # 该 item 下次运行时会被重新处理
                    import warnings
                    warnings.warn(
                        f"[Checkpoint] failed to serialize item "
                        f"{getattr(item, 'filename', '?')!r}: {e}",
                        stacklevel=2,
                    )

        self._pending.clear()
        return count

    # ------------------------------------------------------------------
    # 维护操作
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        删除 checkpoint 文件（任务成功完成后调用）。
        不影响 Cache（计算结果缓存，应当保留）。
        """
        if self._path.exists():
            self._path.unlink()
        self._pending.clear()
        self._done_count = 0

    def exists(self) -> bool:
        """是否存在未完成的 checkpoint（上次运行中断过）。"""
        return self._path.exists()

    def stats(self) -> dict:
        """返回当前 checkpoint 状态摘要。"""
        saved = self._load_saved()
        return {
            "file": str(self._path),
            "exists": self._path.exists(),
            "saved_items": len(saved),
            "pending_flush": len(self._pending),
            "done_this_run": self._done_count,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _load_saved(self) -> dict[str, dict]:
        """
        从 checkpoint 文件加载所有记录。
        同一 item_id 取最后一条（最新状态）。
        返回 {item_id: item_dict}。
        """
        if not self._path.exists():
            return {}

        saved: dict[str, dict] = {}
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "id" in d:
                        saved[d["id"]] = d
                except json.JSONDecodeError:
                    pass  # 损坏的行跳过

        return saved

    @staticmethod
    def _restore_item(item: "Item", saved: dict) -> None:
        """
        用 checkpoint 数据恢复 item 的运行状态字段。

        只恢复"运行状态"相关字段，不覆盖路径/文件系统字段
        （路径可能在两次运行之间发生变化）。
        """
        from core.item import TimeResult
        from core.evidence import Evidence

        item.completed_stages = saved.get("completed_stages", [])
        item.failed_stages    = saved.get("failed_stages", [])
        item.flags            = saved.get("flags", [])
        item.logs             = saved.get("logs", [])
        item.warnings         = saved.get("warnings", [])

        # 恢复 evidence
        raw_evidence = saved.get("evidence", [])
        item.evidence = [Evidence.from_dict(e) for e in raw_evidence]

        # 恢复 time_result
        raw_tr = saved.get("time_result")
        if raw_tr:
            item.time_result = TimeResult.from_dict(raw_tr)

        # 恢复图片基础字段（hash 等，避免重算）
        for field in ("sha1", "phash", "dhash", "width", "height",
                      "format_real", "is_screenshot", "is_scan", "ocr_text"):
            if field in saved and saved[field]:
                setattr(item, field, saved[field])

    def __repr__(self) -> str:
        return (
            f"Checkpoint(path={self._path}, "
            f"interval={self.save_interval}, "
            f"pending={len(self._pending)})"
        )