"""
models/ — ML 模型容器

ModelContainer 是所有模型的统一持有者，挂在 ctx.models 上。
Stage 通过 ctx.models.clip / ctx.models.date_ner 等访问。

当前支持：
    clip      : ClipModel（CLIP embedding）
    date_ner  : (DateNERModel, VocabHelper) tuple，懒加载，文件不存在时为 None
    face      : 预留（人脸识别，未实现）
    ocr       : 预留（OCR，未实现）
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import Config

# DateNER 权重的默认路径（相对于项目根目录）
_DATE_NER_WEIGHTS = Path("weights/date_ner.pt")
_DATE_NER_VOCAB   = Path("weights/vocab.json")


class ModelContainer:
    """
    所有 ML 模型的统一容器。
    挂在 ctx.models，各 Stage 通过 ctx.models.xxx 取用。

    懒加载：各模型只在第一次 encode() 时真正加载权重，
    __init__ 只创建对象，不占 GPU 显存（DateNER 例外，因为模型很小，
    __init__ 阶段直接加载，省去运行时判断逻辑）。
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self.clip:     Optional[object] = None
        self.face:     Optional[object] = None   # 预留
        self.ocr:      Optional[object] = None   # 预留
        self.date_ner: Optional[tuple]  = None   # (DateNERModel, VocabHelper)

        if config.enable_clip:
            self._init_clip()

        self._init_date_ner()

    def _init_clip(self) -> None:
        from models.clip import ClipModel
        self.clip = ClipModel()

    def _init_date_ner(self) -> None:
        """
        加载 DateNER MTL 模型（models/date_ner.py 定义）。

        权重路径：weights/date_ner.pt（由 training/date_ner/train.py 生成）
        词表路径：weights/vocab.json

        文件不存在或加载异常时静默跳过（优雅降级）。
        LearnedDateNERStage.skip_reason() 会检测 ctx.models.date_ner is None，
        自动跳过该 stage，不影响 pipeline 其余部分运行。
        """
        if not _DATE_NER_WEIGHTS.exists() or not _DATE_NER_VOCAB.exists():
            return

        try:
            import torch
            from models.date_ner import load_model

            device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps"  if (hasattr(torch.backends, "mps")
                           and torch.backends.mps.is_available()) else "cpu"
            )
            model, vocab = load_model(_DATE_NER_WEIGHTS, _DATE_NER_VOCAB, device)
            model.eval()
            self.date_ner = (model, vocab)
        except Exception:
            self.date_ner = None

    def __repr__(self) -> str:
        loaded = [k for k in ("clip", "face", "ocr")
                  if getattr(self, k) is not None]
        if self.date_ner is not None:
            loaded.append("date_ner")
        return f"ModelContainer(loaded={loaded})"
