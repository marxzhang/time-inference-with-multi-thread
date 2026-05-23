"""
stages/clip.py — 计算 CLIP embedding

职责：
    对每个 item 计算 CLIP embedding，写入 item.clip_embedding。
    优先从 cache 读取，命中则跳过计算。

    注意：本 stage 只负责"计算并存储 embedding"，
    不产出任何 Evidence——时间推断由 SimilarStage（BatchStage）完成。

运行位置：
    pipeline 中排在 ExifStage / FilenameStage 之后，
    SimilarStage（BatchStage）之前。
    SimilarStage 依赖本 stage 填充的 clip_embedding。

Cache 使用：
    namespace = "clip"
    key       = item.sha1（内容寻址，文件移动不影响缓存命中）
    value     = list[float]（归一化 embedding 向量）
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.stage_base import Stage, StageSkip

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item

# 支持计算 embedding 的扩展名（视频不支持）
_SUPPORTED_EXT = frozenset({
    "jpg", "jpeg", "png", "webp", "heic", "heif", "avif",
    "tiff", "tif", "bmp", "gif",
})


class ClipStage(Stage):
    """
    计算 CLIP embedding，写入 item.clip_embedding。
    需要 ctx.models.clip 已初始化（由 main.py 调用 ctx.setup_models() 完成）。
    """

    name = "ClipStage"

    # ------------------------------------------------------------------
    # 前置检查
    # ------------------------------------------------------------------

    def skip_reason(self, item: "Item", ctx: "Context") -> Optional[str]:
        if item.ext not in _SUPPORTED_EXT:
            return f"unsupported extension: {item.ext}"
        if not item.sha1:
            return "sha1 not computed (ScanStage may have failed)"
        return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def process(self, item: "Item", ctx: "Context") -> None:

        # 1. cache 命中 → 直接用，跳过计算
        if ctx.cache is not None and ctx.cache.has("clip", item.sha1):
            item.clip_embedding = ctx.cache.get("clip", item.sha1)
            item.log(self.name, "cache hit")
            return

        # 2. 确认模型已加载
        model = self._get_model(ctx)
        if model is None:
            raise StageSkip("CLIP model not initialized (call ctx.setup_models())")

        # 3. 计算 embedding
        embedding = model.encode(item.path)
        if embedding is None:
            item.warn(self.name, f"encode failed for {item.filename}")
            return

        item.clip_embedding = embedding

        # 4. 写入 cache
        if ctx.cache is not None:
            ctx.cache.set("clip", item.sha1, embedding)

        item.log(self.name, f"embedding computed (dim={len(embedding)})")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model(ctx: "Context"):
        """从 ctx.models 取 CLIP 模型，不存在返回 None。"""
        models = getattr(ctx, "models", None)
        if models is None:
            return None
        return getattr(models, "clip", None)