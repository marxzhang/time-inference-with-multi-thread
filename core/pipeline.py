"""
core/pipeline.py — Stage 执行顺序定义

职责：
    只管"stages 怎么串"——定义顺序、分组、依赖关系。
    不管调度细节（那是 scheduler 的事）。

两类 Stage 的区别：
    Stage      → 逐 item 处理，由 scheduler 并发调度
    BatchStage → 需要全部 item 才能运行（相似图、时间轴），
                 在所有 Stage 完成后统一执行

典型用法（main.py 中）：
    pipeline = Pipeline.default(ctx)
    scheduler = Scheduler(pipeline)
    scheduler.run(items, ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.stage_base import Stage, BatchStage
from stages.clip import ClipStage
from stages.exif import ExifStage
from stages.filename import FilenameStage
from stages.resolver import ResolverStage
from stages.similar import SimilarStage
from stages.dedup import DedupStage

if TYPE_CHECKING:
    from core.context import Context
    from utils.patterns import DatePattern


@dataclass
class Pipeline:
    """
    Stage 执行顺序的容器。

    属性
    ----
    stages       : 逐 item 执行的 Stage 列表，按顺序执行。
    batch_stages : 需要全量 item 的 BatchStage 列表，在 stages 全部完成后执行。
    """

    stages: list[Stage] = field(default_factory=list)
    batch_stages: list[BatchStage] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def default(
        cls,
        ctx: "Context",
        extra_patterns: list["DatePattern"] | None = None,
    ) -> "Pipeline":
        """
        构建默认 pipeline。

        参数
        ----
        ctx             : 全局上下文（用于按 config 决定哪些 stage 启用）。
        extra_patterns  : PatternMiner 挖掘出的动态模式，注入 FilenameStage。

        Stage 顺序设计原则：
            1. 直接证据先于间接证据（EXIF > 文件名 > 相似图 > 时间轴）
            2. 快速 stage 先于慢速 stage（EXIF/文件名 < OCR < CLIP）
            3. 依赖其他 stage 结果的 stage 排在后面
        """
        stages: list[Stage] = [

            # ── 第一优先级：直接元数据 ──────────────────────────────────
            ExifStage(),
            # 命中率最高，速度最快，confidence=1.0

            FilenameStage(extra_patterns=extra_patterns),
            # 第二高价值，覆盖 IMG_20200101_ 等大量格式

            # ── CLIP Embedding（可选）────────────────────────────────────
            *(  [ClipStage()]
                if ctx.config.enable_clip
                else []
            ),
            # 必须在 SimilarStage（BatchStage）之前，在 ResolverStage 之前

            # ── 预留槽位（后期按序解除注释）────────────────────────────
            # OcrStage(),           # 需要 ctx.config.enable_ocr
            # GpsTimezoneStage(),   # GPS 时区校正，依赖 ExifStage 填充的 gps_*

            # ── 决策层：必须在所有证据收集 stage 之后 ───────────────────
            ResolverStage(),
            # 从所有 evidence 中选出最优结果，写入 item.time_result
            # 是唯一有权修改 time_result 的 stage
        ]

        batch_stages: list[BatchStage] = [
            *(  [SimilarStage(min_score=ctx.config.clip_min_score)]
                if ctx.config.enable_clip and ctx.config.enable_similar
                else []
            ),
            # TimelineStage 预留槽位
            DedupStage(
                dedup_mode=ctx.config.dedup_mode,
                phash_threshold=ctx.config.dedup_phash_threshold,
                clip_min_score=ctx.config.clip_min_score,
            ),
            # 必须在 SimilarStage 之后（相似度结果会影响 time_result，
            # 而 time_result.confidence 参与去重质量评分）
        ]

        # TimelineStage 预留槽位
        # if ctx.config.enable_timeline:
        #     batch_stages.append(TimelineStage())

        return cls(stages=stages, batch_stages=batch_stages)

    # ------------------------------------------------------------------
    # 信息查询
    # ------------------------------------------------------------------

    def stage_names(self) -> list[str]:
        """返回所有 stage 名称（用于日志和 checkpoint）。"""
        names = [s.name for s in self.stages]
        names += [s.name for s in self.batch_stages]
        return names

    def __repr__(self) -> str:
        stage_str = " → ".join(s.name for s in self.stages)
        batch_str = " → ".join(s.name for s in self.batch_stages)
        return (
            f"Pipeline(\n"
            f"  stages      : {stage_str or '(empty)'}\n"
            f"  batch_stages: {batch_str or '(empty)'}\n"
            f")"
        )