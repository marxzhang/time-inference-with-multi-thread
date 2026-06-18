"""
core/pipeline.py — Stage 执行顺序定义（更新版）

新增：LearnedDateNERStage，插入 FilenameStage 之后、ClipStage 之前。
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
from stages.timeline.learned.date_ner_stage import LearnedDateNERStage

if TYPE_CHECKING:
    from core.context import Context
    from utils.patterns import DatePattern


@dataclass
class Pipeline:
    """
    Stage 执行顺序的容器。
    """

    stages: list[Stage] = field(default_factory=list)
    batch_stages: list[BatchStage] = field(default_factory=list)

    @classmethod
    def default(
        cls,
        ctx: "Context",
        extra_patterns: list["DatePattern"] | None = None,
    ) -> "Pipeline":
        """
        构建默认 pipeline。

        Stage 顺序：
            ExifStage               直接元数据，最高优先级
            FilenameStage           规则正则，覆盖已知命名格式
            LearnedDateNERStage     模型补充，覆盖规则未知格式
            ClipStage（可选）       CLIP embedding
            ResolverStage           从所有 evidence 决策最终时间
        """
        stages: list[Stage] = [

            # ── 第一优先级：直接元数据 ──────────────────────────────────
            ExifStage(),

            # ── 文件名规则正则 ───────────────────────────────────────────
            FilenameStage(extra_patterns=extra_patterns),

            # ── 模型补充（规则未覆盖的片段）─────────────────────────────
            # 模型文件不存在时自动跳过（LearnedDateNERStage.skip_reason 处理）
            LearnedDateNERStage(),

            # ── CLIP Embedding（可选）────────────────────────────────────
            *(  [ClipStage()]
                if ctx.config.enable_clip
                else []
            ),

            # ── 决策层 ───────────────────────────────────────────────────
            ResolverStage(),
        ]

        batch_stages: list[BatchStage] = [
            *(  [SimilarStage(min_score=ctx.config.clip_min_score)]
                if ctx.config.enable_clip and ctx.config.enable_similar
                else []
            ),
            DedupStage(
                dedup_mode=ctx.config.dedup_mode,
                phash_threshold=ctx.config.dedup_phash_threshold,
                clip_min_score=ctx.config.clip_min_score,
            ),
        ]

        return cls(stages=stages, batch_stages=batch_stages)

    def stage_names(self) -> list[str]:
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