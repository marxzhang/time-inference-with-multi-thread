"""
models/ — ML 模型容器

ModelContainer 是所有模型的统一持有者，挂在 ctx.models 上。
Stage 通过 ctx.models.clip / ctx.models.face 等访问。

当前支持：
    clip  : ClipModel（CLIP embedding）
    face  : 预留（人脸识别，未实现）
    ocr   : 预留（OCR，未实现）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import Config


class ModelContainer:
    """
    所有 ML 模型的统一容器。
    挂在 ctx.models，各 Stage 通过 ctx.models.xxx 取用。

    懒加载：各模型只在第一次 encode() 时真正加载权重，
    __init__ 只创建对象，不占 GPU 显存。
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self.clip: Optional[object] = None
        self.face: Optional[object] = None   # 预留
        self.ocr:  Optional[object] = None   # 预留

        if config.enable_clip:
            self._init_clip()

    def _init_clip(self) -> None:
        from models.clip import ClipModel
        self.clip = ClipModel()

    def __repr__(self) -> str:
        loaded = [k for k in ("clip", "face", "ocr")
                  if getattr(self, k) is not None]
        return f"ModelContainer(loaded={loaded})"