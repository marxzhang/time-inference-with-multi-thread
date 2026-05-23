"""
全局运行上下文（Context）

整个 pipeline 运行期间唯一的共享资源容器。
在 main.py 里构建一次，以参数形式传入每个 Stage.run()，
Stage 通过 ctx.xxx 取用所需资源，不自己持有全局状态。

当前包含（极简版）：
    ctx.config   —— 全局配置
    ctx.logger   —— 统一日志

预留槽位（后期按需启用）：
    ctx.cache    —— 计算结果缓存（storage/cache.py）
    ctx.index    —— 向量 / hash 索引（storage/index.py）
    ctx.models   —— ML 模型容器（services/）
    ctx.db       —— 数据库连接（storage/database.py）

设计原则：
- Context 只是容器，不包含业务逻辑
- 所有字段初始为 None，使用前由 main.py 按需初始化
- Stage 只读取 ctx，不向 ctx 写入新字段
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Optional

from config import Config

# 后期解除注释，按需引入
# from storage.cache import Cache
# from storage.index import Index
# from storage.database import Database


class Context:
    """
    全局运行上下文，main.py 构建，传入所有 Stage。

    使用示例（main.py）
    -------------------
    cfg = Config(input_dir="/photos", max_workers=8)
    ctx = Context(cfg)
    ctx.setup_logger()          # 初始化日志
    # ctx.setup_cache()         # 后期启用
    # ctx.setup_models()        # 后期启用

    使用示例（Stage 内部）
    ----------------------
    def process(self, item, ctx):
        ctx.logger.debug(f"processing {item.filename}")
        threshold = ctx.config.confidence_threshold
    """

    def __init__(self, config: Config) -> None:
        self.config: Config = config

        # 日志：先设为 None，调用 setup_logger() 后可用
        self.logger: logging.Logger = self._build_logger(config)

        # ------------------------------------------------------------------
        # 预留槽位（后期按需初始化，目前保持 None）
        # ------------------------------------------------------------------

        self.cache: Optional[Any] = None
        """storage.cache.Cache 实例，setup_cache() 后可用。"""

        self.checkpoint: Optional[Any] = None
        """storage.checkpoint.Checkpoint 实例，setup_checkpoint() 后可用。"""

        self.index: Optional[Any] = None
        """storage.index.Index 实例，setup_index() 后可用。"""

        self.clip_model: Optional[Any] = None
        """models.clip.ClipModel 实例，setup_clip_model() 后可用。"""

        self.models: Optional[Any] = None
        """ModelContainer 实例，setup_models() 后可用。"""

        self.db: Optional[Any] = None
        """数据库连接，setup_db() 后可用。"""

    # ------------------------------------------------------------------
    # 初始化方法（main.py 按需调用）
    # ------------------------------------------------------------------

    def setup_logger(self) -> None:
        """重新按 config 配置日志（config 变更后可重调）。"""
        self.logger = self._build_logger(self.config)

    def setup_cache(self) -> None:
        """初始化计算结果缓存（CLIP embedding、OCR 等跨运行复用）。"""
        from storage.cache import Cache
        self.cache = Cache(self.cache_dir)
        loaded = self.cache.preload_all()
        if loaded:
            self.logger.info(
                f"[Context] cache loaded: "
                + ", ".join(f"{ns}={n}" for ns, n in loaded.items())
            )

    def setup_checkpoint(self, save_interval: int = 50) -> None:
        """初始化断点续传（scheduler 启动前调用）。"""
        from storage.checkpoint import Checkpoint
        self.checkpoint = Checkpoint(self.cache_dir, save_interval=save_interval)
        if self.checkpoint.exists():
            self.logger.warning(
                "[Context] checkpoint found — previous run was interrupted, "
                "will resume from last saved state"
            )

    def setup_clip_model(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ) -> None:
        """初始化 CLIP 模型（懒加载，首次 encode 时才真正加载权重）。"""
        from models.clip import ClipModel
        self.clip_model = ClipModel(model_name, pretrained, device)
        if self.clip_model.is_available:
            self.logger.info(
                f"[Context] CLIP model ready: {model_name!r} / {pretrained!r}"
            )
        else:
            self.logger.warning(
                "[Context] CLIP backend not found. "
                "Install with: pip install open-clip-torch"
            )

    def setup_models(self) -> None:
        """初始化 ML 模型容器（CLIP 等，按 config 中的开关决定加载哪些）。"""
        from models import ModelContainer
        self.models = ModelContainer(self.config)
        self.logger.info(f"[Context] models: {self.models}")

    # 后期按需解除注释并实现：
    #
    # def setup_index(self) -> None:
    #     from storage.index import VectorIndex, HashIndex
    #     self.index = VectorIndex(dim=512)
    #
    # def setup_db(self) -> None:
    #     from storage.database import Database
    #     self.db = Database(self.config.cache_dir)

    @property
    def cache_dir(self) -> str:
        return self.config.cache_dir

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _build_logger(config: Config) -> logging.Logger:
        """
        构建统一 logger。

        格式：  2024-01-01 12:00:00  INFO  [ExifStage] done (0.012s)
        输出：  终端（始终），+ 文件（config.log_file 非空时）
        """
        logger = logging.getLogger("time_inference")
        logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

        # 避免重复添加 handler（热重载 / 多次调用时）
        if logger.handlers:
            logger.handlers.clear()

        fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 终端输出
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        # 文件输出（可选）
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

        return logger

    def __repr__(self) -> str:
        return (
            f"Context("
            f"input_dir={self.config.input_dir!r}, "
            f"log_level={self.config.log_level!r}, "
            f"cache={'yes' if self.cache else 'no'}, "
            f"index={'yes' if self.index else 'no'}, "
            f"clip={'yes' if self.clip_model else 'no'}, "
            f"models={'yes' if self.models else 'no'})"
        )