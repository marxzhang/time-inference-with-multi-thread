"""
models/clip.py — CLIP 模型封装

职责：
    加载 CLIP 模型，对图片编码为归一化 embedding 向量。
    对上层（ClipStage）屏蔽模型加载、设备选择、批处理等细节。

依赖：
    pip install torch torchvision open-clip-torch Pillow

设计决策：
    - 懒加载：第一次 encode() 时才加载模型，不在 __init__ 占用资源
    - 归一化：输出向量 L2 归一化，余弦相似度 = 点积，faiss 用 IndexFlatIP
    - 批处理：encode_batch() 比逐张 encode() 快 3-10 倍（GPU 场景尤其明显）
    - 设备优先级：CUDA > MPS（Apple Silicon）> CPU

模型选择（open_clip_model / open_clip_pretrained）：
    "ViT-B-32" / "laion2b_s34b_b79k"  — 默认，速度快，512 维
    "ViT-L-14" / "laion2b_s32b_b82k"  — 更准，768 维，慢 3x
    "ViT-H-14" / "laion2b_s32b_b79k"  — 最准，1024 维，慢 6x
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np


# 默认模型配置
_DEFAULT_MODEL      = "ViT-B-32"
# _DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
_DEFAULT_PRETRAINED = "/home/marx/code/time_inference/weights/open_clip_model.safetensors"
_EMBEDDING_DIM      = 512   # ViT-B-32 的输出维度


class ClipModel:
    """
    CLIP 模型封装，懒加载，线程不安全（每个线程应有自己的实例）。

    使用示例
    --------
    model = ClipModel()
    embedding = model.encode("/path/to/photo.jpg")  # list[float], 长度 512
    embeddings = model.encode_batch([path1, path2, path3])
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        pretrained: str = _DEFAULT_PRETRAINED,
        device: Optional[str] = None,
    ) -> None:
        """
        参数
        ----
        model_name  : open_clip 模型名称
        pretrained  : 预训练权重名称
        device      : "cuda" / "mps" / "cpu"，None 表示自动选择
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self._device_hint = device

        # 懒加载，首次 encode 时初始化
        self._model = None
        self._preprocess = None
        self._device = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """embedding 向量维度。"""
        # ViT-B-32=512, ViT-L-14=768, ViT-H-14=1024
        _DIM_MAP = {
            "ViT-B-32": 512,
            "ViT-L-14": 768,
            "ViT-H-14": 1024,
        }
        return _DIM_MAP.get(self.model_name, 512)

    def encode(self, image_path: str) -> Optional[list[float]]:
        """
        对单张图片编码，返回归一化 embedding（list[float]）。
        图片无法读取时返回 None。
        """
        result = self.encode_batch([image_path])
        return result[0] if result else None

    def encode_batch(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> list[Optional[list[float]]]:
        """
        批量编码，返回与 image_paths 等长的列表。
        无法读取的图片对应位置为 None。

        参数
        ----
        batch_size : 每次送入模型的图片数，受 GPU 显存限制
        """
        self._ensure_loaded()
        import torch

        results: list[Optional[list[float]]] = [None] * len(image_paths)

        # 分批处理
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start: batch_start + batch_size]
            tensors = []
            valid_indices = []

            for i, path in enumerate(batch_paths):
                tensor = self._load_image(path)
                if tensor is not None:
                    tensors.append(tensor)
                    valid_indices.append(i)

            if not tensors:
                continue

            # 批量推理
            batch_tensor = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                features = self._model.encode_image(batch_tensor)
                # L2 归一化：余弦相似度 = 点积，方便 faiss IndexFlatIP
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy().astype(np.float32)

            for local_i, global_i in enumerate(valid_indices):
                results[batch_start + global_i] = features_np[local_i].tolist()

        return results

    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # 内部：懒加载
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import open_clip
            import torch
        except ImportError as e:
            raise RuntimeError(
                f"CLIP dependencies not installed: {e}\n"
                f"Run: pip install open-clip-torch torch torchvision"
            ) from e

        self._device = self._select_device()
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self._device,
        )
        self._model.eval()

    def _select_device(self) -> str:
        if self._device_hint:
            return self._device_hint
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # 内部：图片加载
    # ------------------------------------------------------------------

    def _load_image(self, path: str):
        """加载单张图片为预处理后的 tensor，失败返回 None。
        HEIC/HEIF 需要 pillow-heif：pip install pillow-heif
        """
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass  # 未安装则 HEIC 会在 Image.open 时抛异常，由下面的 except 捕获

        try:
            from PIL import Image
            # img = Image.open(path).convert("RGB")
            with Image.open(path) as img:  # ← 关键修改
                img = img.convert("RGB")
            return self._preprocess(img)
        except Exception:
            return None

    def __repr__(self) -> str:
        status = f"loaded on {self._device}" if self._model else "not loaded"
        return f"ClipModel({self.model_name}/{self.pretrained}, {status})"