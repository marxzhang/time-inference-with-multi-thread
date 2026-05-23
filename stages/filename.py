"""
FilenameStage — 从文件名和文件夹名推断时间

职责：
    对每个 item，用预置正则 + 动态挖掘模式匹配其文件名和文件夹路径各级名称，
    将命中结果转为 Evidence 追加到 item.evidence。

两个信号来源，分别产出 Evidence：
    1. 文件名（stem）         source="filename"
    2. 文件夹名（各级路径）   source="foldername"
       - 文件夹名产出的置信度整体低于文件名（因为文件夹是人整理的，精度通常较低）
       - 越靠近文件的文件夹级别置信度越高（直接父级 > 祖父级）

与 PatternMiner 的协作：
    FilenameStage 接受 extra_patterns 参数，把挖掘出的动态模式追加到匹配池。
    pipeline 中的典型用法：

        miner = PatternMiner()
        dynamic = miner.mine(items, ctx)
        pipeline = [
            ...
            FilenameStage(extra_patterns=dynamic),
            ...
        ]

设计决策：同一来源可能产出多条 Evidence：
    文件名 "IMG_20200101_143022.jpg" 同时被 DATETIME8_6 和 DATE8 命中。
    两条都会追加，让决策层自行选择最优，而不是在这里静默丢弃低精度的结果。
    （参考 ExifStage 的设计原则：Stage 只收集证据，不做最终决策。）
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.stage_base import Stage, StageSkip
from utils.patterns import DatePattern, MatchResult, match_all, BUILTIN_PATTERNS

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item


# 文件夹级别置信度衰减系数
# 直接父级 × 1.0，祖父级 × 0.85，更高层 × 0.7
_FOLDER_DEPTH_DECAY = [1.0, 0.85, 0.7]

# 文件夹相对于文件名的整体置信度折扣
# （文件夹是人工整理，精度低，但意图更明确）
_FOLDER_BASE_DISCOUNT = 0.85


class FilenameStage(Stage):
    """
    从文件名 / 文件夹名推断时间，产出 Evidence。
    """

    name = "FilenameStage"

    def __init__(self, extra_patterns: Optional[list[DatePattern]] = None) -> None:
        """
        参数
        ----
        extra_patterns : PatternMiner 挖掘出的动态模式，追加到内置模式之后。
                         None 表示只使用内置模式。
        """
        self._patterns: list[DatePattern] = list(BUILTIN_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    # ------------------------------------------------------------------
    # 前置检查
    # ------------------------------------------------------------------

    def skip_reason(self, item: "Item", ctx: "Context") -> Optional[str]:
        # 已有高可信度直接证据（EXIF DateTimeOriginal）时，仍然运行。
        # 文件名证据可以用于交叉验证，不应跳过。
        return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def process(self, item: "Item", ctx: "Context") -> None:
        path = Path(item.path)
        found_any = False

        # 1. 匹配文件名（不含扩展名）
        stem = path.stem
        filename_results = match_all(stem, self._patterns)
        for result in filename_results:
            item.add_evidence(self._make_filename_evidence(item, result, stem))
            found_any = True

        # 2. 匹配文件夹各级名称（从最近的父级往上，最多取 3 级）
        folder_parts = self._get_folder_parts(path, item)
        for depth, folder_name in enumerate(folder_parts):
            folder_results = match_all(folder_name, self._patterns)
            for result in folder_results:
                item.add_evidence(
                    self._make_foldername_evidence(item, result, folder_name, depth)
                )
                found_any = True

        if not found_any:
            item.log(self.name, "no date pattern matched in filename or foldername")

    # ------------------------------------------------------------------
    # 构建 Evidence
    # ------------------------------------------------------------------

    def _make_filename_evidence(
        self,
        item: "Item",
        result: MatchResult,
        stem: str,
    ) -> object:
        return self.make_evidence(
            source="filename",
            dt=result.dt,
            precision=result.precision,
            confidence=result.confidence,
            is_direct=True,
            reason=f'matched {result.matched_text!r} in filename {stem!r}',
            metadata={
                "pattern_name": result.pattern_name,
                "matched_text": result.matched_text,
                "span": list(result.span),
                "stem": stem,
            },
        )

    def _make_foldername_evidence(
        self,
        item: "Item",
        result: MatchResult,
        folder_name: str,
        depth: int,
    ) -> object:
        # 深度衰减：越远的祖先文件夹置信度越低
        decay = _FOLDER_DEPTH_DECAY[min(depth, len(_FOLDER_DEPTH_DECAY) - 1)]
        adjusted_confidence = result.confidence * _FOLDER_BASE_DISCOUNT * decay

        # 文件夹时间通常精度较低（人工整理，很少精确到秒）
        # 如果模式精度高于 day，降为 day（文件夹名里的时分秒通常是巧合）
        effective_precision = result.precision
        _PRECISION_ORDER = {"second": 6, "minute": 5, "hour": 4,
                            "day": 3, "month": 2, "year": 1, "unknown": 0}
        if _PRECISION_ORDER.get(result.precision, 0) > _PRECISION_ORDER["day"]:
            effective_precision = "day"

        depth_label = ["parent", "grandparent", "ancestor"][min(depth, 2)]

        return self.make_evidence(
            source="foldername",
            dt=result.dt,
            precision=effective_precision,
            confidence=round(adjusted_confidence, 3),
            is_direct=False,  # 文件夹名是间接证据
            reason=f'matched {result.matched_text!r} in {depth_label} folder {folder_name!r}',
            metadata={
                "pattern_name": result.pattern_name,
                "matched_text": result.matched_text,
                "span": list(result.span),
                "folder_name": folder_name,
                "folder_depth": depth,
            },
        )

    # ------------------------------------------------------------------
    # 工具：提取文件夹各级名称
    # ------------------------------------------------------------------

    @staticmethod
    def _get_folder_parts(path: Path, item: "Item") -> list[str]:
        """
        从文件路径提取最多 3 级父文件夹名称（不含根目录和驱动器名）。
        最近的父级排在最前面。

        示例：
            /photos/2020/Japan/IMG_001.jpg
            → ["Japan", "2020", "photos"]  （最多取前 3 个）
        """
        parts = []
        current = path.parent
        for _ in range(3):
            name = current.name
            if not name or name in ("", "/", "\\"):
                break
            parts.append(name)
            current = current.parent
        return parts