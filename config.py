"""
全局配置

使用方式：
    from config import Config
    cfg = Config()                        # 全默认值
    cfg = Config(input_dir="/photos")     # 覆盖部分字段
    cfg = Config.from_dict({...})         # 从 dict 加载（适合 JSON / YAML 配置文件）

设计原则：
- 用 dataclass，字段有类型和默认值，IDE 补全友好
- 不做复杂逻辑，只存值
- from_dict / to_dict 支持后期接入配置文件
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------

    input_dir: str = ""
    """扫描根目录（必填，运行前赋值）。"""

    output_dir: str = ""
    """输出目录，空字符串表示原地写回。"""

    cache_dir: str = ".cache"
    """cache / checkpoint 存放目录。"""

    # ------------------------------------------------------------------
    # Pipeline 行为
    # ------------------------------------------------------------------

    dry_run: bool = False
    """True = 只推断，不写回任何文件。"""

    overwrite_exif: bool = False
    """True = 允许覆盖原始 EXIF（谨慎）。"""

    write_mtime: bool = True
    """True = 将推断时间写入文件 mtime。"""

    # ------------------------------------------------------------------
    # 置信度阈值
    # ------------------------------------------------------------------

    confidence_threshold: float = 0.6
    """低于此值标记 LOW_CONFIDENCE flag，不写回。"""

    # ------------------------------------------------------------------
    # 性能
    # ------------------------------------------------------------------

    max_workers: int = 32
    """并发线程数。"""

    batch_size: int = 64
    """每批处理的 item 数（用于 BatchStage）。"""

    # ------------------------------------------------------------------
    # 功能开关
    # ------------------------------------------------------------------

    enable_ocr: bool = False
    """OCR 推断（慢，默认关闭）。"""

    enable_clip: bool = False
    """CLIP embedding（需要 GPU，默认关闭）。"""

    clip_model: str = "ViT-B-32"
    """CLIP 模型名称，见 models/clip.py。"""

    clip_pretrained: str = "laion2b_s34b_b79k"
    """CLIP 预训练权重名称。"""

    clip_min_score: float = 0.85
    """相似图检索的最低余弦相似度阈值。"""

    clip_batch_size: int = 64
    """CLIP 批量编码每批大小。"""

    enable_face: bool = False
    """人脸识别（默认关闭）。"""

    enable_similar: bool = True
    """相似图推断（需要 enable_clip=True）。"""

    dedup_mode: str = "sha1"
    """
    去重强度：no / sha1 / phash / clip。
    每一档包含上一档效果：clip ⊃ phash ⊃ sha1。
    clip 模式需要 enable_clip=True。
    """

    dedup_phash_threshold: int = 4
    """phash 汉明距离阈值（dedup_mode >= phash 时生效，默认 4）。"""

    skip_extensions: frozenset = None  # type: ignore[assignment]
    """完全跳过的文件扩展名集合（不建 item，不统计）。
    None 表示使用 scan.py 中的 DEFAULT_SKIP_EXTENSIONS。
    用户可通过 --skip-ext 指定，格式：逗号分隔，不含点，如 "pdf,txt,xml"。"""

    enable_timeline: bool = True
    """时间轴连续性推断。"""

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    log_level: str = "INFO"
    """日志级别：DEBUG / INFO / WARNING / ERROR。"""

    log_file: str = ""
    """日志文件路径，空字符串表示只输出到终端。"""

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "dry_run": self.dry_run,
            "overwrite_exif": self.overwrite_exif,
            "write_mtime": self.write_mtime,
            "confidence_threshold": self.confidence_threshold,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "enable_ocr": self.enable_ocr,
            "enable_clip": self.enable_clip,
            "clip_model": self.clip_model,
            "clip_pretrained": self.clip_pretrained,
            "clip_min_score": self.clip_min_score,
            "clip_batch_size": self.clip_batch_size,
            "enable_face": self.enable_face,
            "enable_similar": self.enable_similar,
            "dedup_mode": self.dedup_mode,
            "dedup_phash_threshold": self.dedup_phash_threshold,
            "skip_extensions": list(self.skip_extensions) if self.skip_extensions else None,
            "enable_timeline": self.enable_timeline,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg