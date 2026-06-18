"""
stages/timeline/learned/date_ner_stage.py

将训练好的 MTL DateNER 模型集成到 pipeline，
作为 FilenameStage 的补充，覆盖其未能解析的路径片段。

定位
----
    - 不替换 FilenameStage / PatternMiner，只做补充
    - 只处理 FilenameStage 未覆盖的片段（避免重复产出 evidence）
    - 置信度上限 0.82，低于 FilenameStage 规则正则的最高置信度（0.9）
    - 模型文件不存在时优雅降级，不报错

集成位置（pipeline 顺序）
--------------------------
    ExifStage
    FilenameStage          ← 规则正则，覆盖已知格式
    LearnedDateNERStage    ← 模型补充，覆盖规则未知格式
    ClipStage（可选）
    ResolverStage

输出的 Evidence
---------------
    source      : "filename" 或 "foldername"（与 FilenameStage 一致）
    stage       : "LearnedDateNERStage"
    confidence  : clf_prob × span_conf_mean × DISCOUNT，上限 0.82
    precision   : 由识别出的最细时间字段决定
    metadata    : {"matched_text": ..., "pattern": "DateNER", "spans": [...]}

模型加载
--------
    从 ctx.models.date_ner 取（由 setup_models() 初始化）。
    ctx.models.date_ner 是一个 (model, vocab) tuple。
    若为 None，跳过整个 stage（无模型时优雅降级）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.stage_base import Stage, StageSkip

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item


# 模型产出 evidence 的置信度折扣（相对于规则正则）
_CONFIDENCE_DISCOUNT = 0.95
# 最高置信度上限（低于 FilenameStage 的 0.9）
_MAX_CONFIDENCE = 0.82
# 最低置信度门槛（低于此值不产出 evidence）
_MIN_CONFIDENCE = 0.30

# 时间字段精度映射（由识别出的最细字段决定）
_FIELD_PRECISION = {
    "SEC":   "second",
    "MIN":   "minute",
    "HOUR":  "hour",
    "DAY":   "day",
    "MONTH": "month",
    "YEAR":  "year",
}

# 文件夹相对于文件名的置信度折扣（与 FilenameStage 保持一致）
_FOLDER_BASE_DISCOUNT = 0.85
_FOLDER_DEPTH_DECAY   = [1.0, 0.85, 0.7]


class LearnedDateNERStage(Stage):
    """
    用 MTL DateNER 模型补充识别 FilenameStage 未覆盖的路径片段。
    """

    name = "LearnedDateNERStage"

    def __init__(self, max_seq_len: int = 64) -> None:
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    # 前置检查
    # ------------------------------------------------------------------

    def skip_reason(self, item: "Item", ctx: "Context") -> Optional[str]:
        # 模型未加载 → 跳过（优雅降级）
        if not self._get_model(ctx):
            return "DateNER model not loaded (weights/date_ner.pt not found)"
        return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def process(self, item: "Item", ctx: "Context") -> None:
        model_tuple = self._get_model(ctx)
        if model_tuple is None:
            raise StageSkip("no model")

        model, vocab = model_tuple

        # 找出 FilenameStage 已覆盖的文件夹名（不重复处理）
        covered_folders = self._covered_folder_names(item)
        covered_stem    = self._is_stem_covered(item)

        # 收集待推理的片段：(text, source, depth)
        segments: list[tuple[str, str, int]] = []

        if not covered_stem:
            stem = Path(item.relpath).stem
            if stem and len(stem) <= self.max_seq_len:
                segments.append((stem, "stem", -1))

        folder_parts = list(Path(item.relpath).parts[:-1])
        for depth, folder in enumerate(reversed(folder_parts[:3])):
            if folder and folder not in covered_folders and len(folder) <= self.max_seq_len:
                segments.append((folder, "folder", depth))

        if not segments:
            item.log(self.name, "all segments already covered by FilenameStage")
            return

        # 批量推理
        texts = [s[0] for s in segments]
        device = next(model.parameters()).device
        results = model.predict(texts, vocab, device)

        produced = 0
        for (text, source, depth), result in zip(segments, results):
            if not result["datetime_candidates"]:
                continue

            cand = result["datetime_candidates"][0]
            if cand["datetime"] is None and not cand["fields"]:
                continue

            dt = None
            if cand["datetime"]:
                try:
                    dt = datetime.fromisoformat(cand["datetime"])
                except ValueError:
                    pass

            # 计算置信度
            raw_conf = cand["confidence"]
            if source == "folder" and depth >= 0:
                decay = _FOLDER_DEPTH_DECAY[min(depth, 2)]
                raw_conf = raw_conf * _FOLDER_BASE_DISCOUNT * decay
            conf = min(raw_conf * _CONFIDENCE_DISCOUNT, _MAX_CONFIDENCE)

            if conf < _MIN_CONFIDENCE:
                continue

            # 推断精度（由最细的识别字段决定）
            fields = cand["fields"]
            precision = "year"
            for field_name in ["SEC", "MIN", "HOUR", "DAY", "MONTH", "YEAR"]:
                if field_name in fields:
                    precision = _FIELD_PRECISION[field_name]
                    break

            # 文件夹名精度上限为 day（与 FilenameStage 策略一致）
            if source == "folder":
                _prec_rank = {"second": 6, "minute": 5, "hour": 4,
                              "day": 3, "month": 2, "year": 1}
                if _prec_rank.get(precision, 0) > _prec_rank["day"]:
                    precision = "day"

            ev_source = "filename" if source == "stem" else "foldername"
            reason = (
                f'DateNER matched {text!r} → '
                f'{cand["datetime"] or str(fields)}'
            )

            item.add_evidence(self.make_evidence(
                source=ev_source,
                dt=dt,
                precision=precision,
                confidence=round(conf, 3),
                is_direct=(source == "stem"),
                reason=reason,
                metadata={
                    "matched_text": text,
                    "pattern":      "DateNER",
                    "fields":       fields,
                    "spans":        cand.get("char_spans", []),
                    "clf_prob":     result["clf_prob"],
                    "folder_depth": depth if source == "folder" else None,
                },
            ))
            produced += 1

        item.log(self.name, f"produced {produced} evidence from {len(segments)} segments")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _get_model(self, ctx: "Context"):
        """从 ctx.models 取 (DateNERModel, VocabHelper) tuple，不存在返回 None。"""
        models = getattr(ctx, "models", None)
        if models is None:
            return None
        return getattr(models, "date_ner", None)

    @staticmethod
    def _covered_folder_names(item: "Item") -> set[str]:
        """FilenameStage 已产出 foldername evidence 的文件夹名集合。"""
        covered: set[str] = set()
        for ev in item.evidence:
            if ev.source == "foldername":
                name = ev.metadata.get("folder_name", "")
                if name:
                    covered.add(name)
        return covered

    @staticmethod
    def _is_stem_covered(item: "Item") -> bool:
        """FilenameStage 是否已对 stem 产出过 filename evidence。"""
        return any(ev.source == "filename" for ev in item.evidence)