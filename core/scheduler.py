"""
core/scheduler.py — 运行调度

职责：
    拿到 items 和 pipeline，负责"怎么跑"。

并发策略
--------
    Stage 层（逐 item）：ThreadPoolExecutor 并发。
        - max_workers=1 时退化为串行，行为与旧版完全一致
        - 每个 item 被一个线程独占处理（item 内串行，item 间并发）
        - item 之间无共享状态，天然线程安全
        - 需要保护的共享资源：Cache / ScanCache / Checkpoint（已加锁）

    BatchStage 层（全量）：始终串行。
        BatchStage 需要访问全量 item，内部可能修改多个 item 的字段
        （如 SimilarStage 的 _re_resolve），强行并发收益极低且危险。

进度条
------
    多线程下用 tqdm 的线程安全模式（position 参数），
    未安装 tqdm 时降级为每完成 10% 打一条日志。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item
    from core.pipeline import Pipeline


class Scheduler:

    def __init__(self, pipeline: "Pipeline") -> None:
        self.pipeline = pipeline

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, items: list["Item"], ctx: "Context") -> None:
        """
        对所有 items 执行 pipeline。

        流程：
            1. checkpoint 恢复
            2. Stage 层（并发，workers 由 config.max_workers 控制）
            3. BatchStage 层（串行）
            4. checkpoint 清除
            5. 统计摘要
        """
        if not items:
            ctx.logger.warning("[Scheduler] no items to process")
            return

        if ctx.checkpoint is not None:
            items = ctx.checkpoint.restore(items)

        workers = getattr(ctx.config, "max_workers", 1)
        ctx.logger.info(
            f"[Scheduler] start — {len(items)} items, "
            f"workers={workers}, "
            f"stages: {self.pipeline.stage_names()}"
        )
        t0 = time.monotonic()

        self._run_stages(items, ctx, workers)

        if self.pipeline.batch_stages:
            self._run_batch_stages(items, ctx)

        if ctx.checkpoint is not None:
            ctx.checkpoint.flush()
            ctx.checkpoint.clear()
            ctx.logger.info("[Scheduler] checkpoint cleared")

        elapsed = time.monotonic() - t0
        self._log_summary(items, ctx, elapsed)

    # ------------------------------------------------------------------
    # Stage 层
    # ------------------------------------------------------------------

    def _run_stages(
        self,
        items: list["Item"],
        ctx: "Context",
        workers: int,
    ) -> None:
        """
        逐 item 执行所有 Stage。
        workers=1 串行，workers>1 并发。
        每个 item 在单个线程内串行执行所有 stage，item 之间并发。
        """
        if workers <= 1:
            self._run_stages_serial(items, ctx)
        else:
            self._run_stages_parallel(items, ctx, workers)

    # ？？
    @staticmethod
    def _make_progress(items: list, desc: str = ""):
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc, unit="file", dynamic_ncols=True)
        except ImportError:
            return None

    def _run_stages_serial(
        self,
        items: list["Item"],
        ctx: "Context",
    ) -> None:
        """串行版本（workers=1 或 tqdm 不可用时的降级）。"""
        total = len(items)
        progress = self._make_progress(items, desc="processing")

        for i, item in enumerate(progress if progress else items):
            if progress is None and total >= 10 and i % max(1, total // 10) == 0:
                ctx.logger.info(f"[Scheduler] {i}/{total} ({i * 100 // total}%)")

            self._process_one(item, ctx)

    def _run_stages_parallel(
        self,
        items: list["Item"],
        ctx: "Context",
        workers: int,
    ) -> None:
        """
        并发版本。

        进度计数用 threading.Lock 保护，tqdm 用线程安全的 update() 模式。
        """
        total = len(items)
        completed = 0
        lock = threading.Lock()

        # tqdm 的并发安全用法：在主线程创建，子线程调用 update()
        try:
            from tqdm import tqdm
            progress = tqdm(total=total, desc="processing", unit="item",
                            dynamic_ncols=True)
        except ImportError:
            progress = None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_one, item, ctx): item
                for item in items
            }
            for future in as_completed(futures):
                # stage 内部已 try/except，这里只需取结果触发异常传播
                try:
                    future.result()
                except Exception as e:
                    item = futures[future]
                    ctx.logger.error(
                        f"[Scheduler] unexpected error on {item.filename}: {e}"
                    )

                with lock:
                    completed += 1
                    if progress is not None:
                        progress.update(1)
                    elif total >= 10 and completed % max(1, total // 10) == 0:
                        ctx.logger.info(
                            f"[Scheduler] {completed}/{total} "
                            f"({completed * 100 // total}%)"
                        )

        if progress is not None:
            progress.close()

    def _process_one(self, item: "Item", ctx: "Context") -> None:
        """
        对单个 item 串行执行所有 Stage，并在完成后通知 checkpoint。
        此方法在子线程中调用，item 是该线程独占的，无需加锁。
        """
        for stage in self.pipeline.stages:
            stage.run(item, ctx)

        # checkpoint 的 on_item_done 内部已加锁，并发调用安全
        if ctx.checkpoint is not None:
            ctx.checkpoint.on_item_done(item)

    # ------------------------------------------------------------------
    # BatchStage 层（始终串行）
    # ------------------------------------------------------------------

    def _run_batch_stages(self, items: list["Item"], ctx: "Context") -> None:
        for batch_stage in self.pipeline.batch_stages:
            ctx.logger.info(f"[Scheduler] batch stage: {batch_stage.name}")
            t0 = time.monotonic()
            try:
                batch_stage.run_batch(items, ctx)
                ctx.logger.info(
                    f"[Scheduler] {batch_stage.name} done "
                    f"({time.monotonic() - t0:.2f}s)"
                )
            except Exception:
                import traceback
                ctx.logger.error(
                    f"[Scheduler] {batch_stage.name} FAILED:\n"
                    f"{traceback.format_exc()}"
                )

    # ------------------------------------------------------------------
    # 统计摘要
    # ------------------------------------------------------------------

    def _log_summary(
        self,
        items: list["Item"],
        ctx: "Context",
        elapsed: float,
    ) -> None:
        total      = len(items)
        resolved   = sum(1 for it in items if it.time_result.is_resolved)
        has_exif   = sum(1 for it in items if it.has_exif_time)
        failed_any = sum(1 for it in items if it.failed_stages)
        no_exif    = sum(1 for it in items if it.has_flag("NO_EXIF"))
        threshold  = ctx.config.confidence_threshold
        low_conf   = sum(
            1 for it in items
            if not it.time_result.is_resolved
            or it.time_result.confidence < threshold
        )
        ctx.logger.info(
            f"[Scheduler] ── summary ──────────────────────────────\n"
            f"  total      : {total}\n"
            f"  resolved   : {resolved} ({resolved * 100 // total if total else 0}%)\n"
            f"  has_exif   : {has_exif}\n"
            f"  no_exif    : {no_exif}\n"
            f"  low_conf   : {low_conf}\n"
            f"  failed_any : {failed_any}\n"
            f"  elapsed    : {elapsed:.2f}s  ({elapsed / total:.3f}s/item)\n"
            f"  ─────────────────────────────────────────────────"
        )

