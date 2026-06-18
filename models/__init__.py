"""
models/__init__.py — ML 模型容器（更新版）

新增：date_ner — DateNER MTL 模型（(DateNERModel, VocabHelper) tuple）
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
    __init__ 只创建对象，不占 GPU 显存。
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
        加载 DateNER MTL 模型。

        权重路径：weights/date_ner.pt（由 train.py 生成）
        词表路径：weights/vocab.json

        文件不存在时静默跳过（优雅降级），不报错。
        LearnedDateNERStage.skip_reason() 会检测 ctx.models.date_ner is None，
        自动跳过该 stage。
        """
        if not _DATE_NER_WEIGHTS.exists() or not _DATE_NER_VOCAB.exists():
            return  # 模型文件不存在，降级运行

        try:
            import torch
            # 导入路径：tools/train_date_ner/model.py
            # 在 pipeline 里用 sys.path 或包结构来 import
            import sys
            _tools_path = str(Path(__file__).parent.parent / "tools" / "train_data_ner")
            if _tools_path not in sys.path:
                sys.path.insert(0, _tools_path)
            # pycharm 静态代码检查会标红
            from model import load_model  # tools/train_data_ner/model.py

            device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps"  if (hasattr(torch.backends, "mps")
                           and torch.backends.mps.is_available()) else "cpu"
            )
            model, vocab = load_model(_DATE_NER_WEIGHTS, _DATE_NER_VOCAB, device)
            model.eval()
            self.date_ner = (model, vocab)
        except Exception:
            # 任何加载异常（torch 未安装、权重损坏等）都静默跳过
            self.date_ner = None

    def __repr__(self) -> str:
        loaded = [k for k in ("clip", "face", "ocr")
                  if getattr(self, k) is not None]
        if self.date_ner is not None:
            loaded.append("date_ner")
        return f"ModelContainer(loaded={loaded})"