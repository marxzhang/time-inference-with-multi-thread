"""
core/scheduler.py — 运行调度

职责：
    拿到 items 和 pipeline，负责"怎么跑"。
    Stage 内部逻辑不归这里管。

当前版本：极简串行
    - 单线程，item 逐个处理
    - tqdm 进度条（可选，没装则降级为日志）
    - 每个 item 的异常已在 Stage.run() 内部捕获，scheduler 不需要额外 try/catch
    - 运行完输出简单统计

预留槽位（后期增强，不影响当前接口）：
    [ ] 多线程：ThreadPoolExecutor，max_workers 来自 config
    [ ] Checkpoint：每 N 个 item 序列化一次，支持断点恢复
    [ ] Retry：failed_stages 非空的 item 重新入队
    [ ] Batch 资源协调：BatchStage 运行前等待所有 Stage 完成
    [ ] 进度持久化：写 progress.json，供前端轮询
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item
    from core.pipeline import Pipeline


class Scheduler:
    """
    串行调度器。

    使用示例
    --------
    scheduler = Scheduler(pipeline)
    scheduler.run(items, ctx)
    """

    def __init__(self, pipeline: "Pipeline") -> None:
        self.pipeline = pipeline

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, items: list["Item"], ctx: "Context") -> None:
        """
        对所有 items 依次执行 pipeline 中的每个 stage。

        流程：
            1. checkpoint 恢复（ctx.checkpoint 存在时）
            2. 逐 item × 逐 stage 串行执行（Stage 层）
            3. 全部完成后执行 BatchStage 层
            4. checkpoint 清除（正常完成时）
            5. 输出统计摘要
        """
        if not items:
            ctx.logger.warning("[Scheduler] no items to process")
            return

        # ── Checkpoint 恢复 ───────────────────────────────────────────
        if ctx.checkpoint is not None:
            items = ctx.checkpoint.restore(items)

        ctx.logger.info(
            f"[Scheduler] start — {len(items)} items, "
            f"stages: {self.pipeline.stage_names()}"
        )
        t0 = time.monotonic()

        # ── Stage 层（逐 item）──────────────────────────────────────
        self._run_stages(items, ctx)

        # ── BatchStage 层（全量）────────────────────────────────────
        if self.pipeline.batch_stages:
            self._run_batch_stages(items, ctx)

        # ── Checkpoint 清除（正常完成）───────────────────────────────
        if ctx.checkpoint is not None:
            ctx.checkpoint.flush()   # 写入最后一批
            ctx.checkpoint.clear()   # 任务成功，删除 checkpoint 文件
            ctx.logger.info("[Scheduler] checkpoint cleared")

        # ── 统计摘要 ─────────────────────────────────────────────────
        elapsed = time.monotonic() - t0
        self._log_summary(items, ctx, elapsed)

    # ------------------------------------------------------------------
    # Stage 层：逐 item 串行
    # ------------------------------------------------------------------

    def _run_stages(self, items: list["Item"], ctx: "Context") -> None:
        """对每个 item 依次执行所有 Stage。"""
        total = len(items)
        stages = self.pipeline.stages

        # 尝试使用 tqdm 进度条，没有则降级
        progress = self._make_progress(items, desc="processing")

        for i, item in enumerate(progress if progress else items):
            if progress is None:
                # 降级：每 10% 打一条日志
                if total >= 10 and i % max(1, total // 10) == 0:
                    ctx.logger.info(
                        f"[Scheduler] {i}/{total} "
                        f"({i * 100 // total}%)"
                    )

            for stage in stages:
                stage.run(item, ctx)

            # checkpoint 增量保存
            if ctx.checkpoint is not None:
                ctx.checkpoint.on_item_done(item)

    # ------------------------------------------------------------------
    # BatchStage 层：全量
    # ------------------------------------------------------------------

    def _run_batch_stages(self, items: list["Item"], ctx: "Context") -> None:
        """在所有 Stage 完成后，依次执行 BatchStage。"""
        for batch_stage in self.pipeline.batch_stages:
            ctx.logger.info(f"[Scheduler] batch stage: {batch_stage.name}")
            t0 = time.monotonic()
            try:
                batch_stage.run_batch(items, ctx)
                elapsed = time.monotonic() - t0
                ctx.logger.info(
                    f"[Scheduler] {batch_stage.name} done ({elapsed:.2f}s)"
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
        total = len(items)
        resolved   = sum(1 for it in items if it.time_result.is_resolved)
        has_exif   = sum(1 for it in items if it.has_exif_time)
        failed_any = sum(1 for it in items if it.failed_stages)
        no_exif    = sum(1 for it in items if it.has_flag("NO_EXIF"))

        # 直接从 time_result 计算，不依赖 LOW_CONFIDENCE flag。
        # 原因：flag 由 main.py 在 scheduler 结束后才打上，
        # 若此处读 flag 会永远为 0。统计和业务标记应相互独立。
        threshold = ctx.config.confidence_threshold
        low_conf = sum(
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

    # ------------------------------------------------------------------
    # 工具：进度条
    # ------------------------------------------------------------------

    @staticmethod
    def _make_progress(items: list, desc: str = ""):
        """
        尝试创建 tqdm 进度条。
        没有安装 tqdm 时返回 None，调用方降级为日志输出。
        """
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc, unit="item", dynamic_ncols=True)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # 预留：后期扩展接口（签名已定，实现留空）
    # ------------------------------------------------------------------

    # def _run_stages_parallel(self, items, ctx) -> None:
    #     """多线程版本，替换 _run_stages()。"""
    #     from concurrent.futures import ThreadPoolExecutor, as_completed
    #     with ThreadPoolExecutor(max_workers=ctx.config.max_workers) as executor:
    #         futures = {executor.submit(self._process_one, item, ctx): item
    #                    for item in items}
    #         for future in as_completed(futures):
    #             future.result()  # stage 内部已捕获异常，这里不会抛

    # def _process_one(self, item, ctx) -> None:
    #     for stage in self.pipeline.stages:
    #         stage.run(item, ctx)

    # def _save_checkpoint(self, items, ctx) -> None:
    #     """序列化当前进度到 cache_dir/checkpoint.json。"""
    #     pass

    # def _load_checkpoint(self, ctx) -> list | None:
    #     """从 checkpoint.json 恢复，返回已完成的 item id 集合。"""
    #     pass