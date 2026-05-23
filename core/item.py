"""
照片 Item

设计原则：
- 单张照片的全部属性的唯一数据载体，贯穿整个 pipeline
- JSON-friendly：to_dict / from_dict 双向转换
- evidence 只追加，不覆盖
- time_result 是最终综合结论，由 writer stage 统一写入
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from core.evidence import Evidence, TimePrecision


# ---------------------------------------------------------------------------
# JSON 序列化工具
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: Any) -> Any:
    """
    递归清洗 piexif / Pillow 原始 EXIF 中 JSON 不支持的类型：
        bytes       → hex 字符串（"0x..."），保留信息可调试
        tuple       → list（JSON 只有 array）
        int / float → 原样保留
        其他无法序列化的对象 → str(obj)

    只在 ExifData.to_dict() 处理 raw 字段时调用，
    结构化字段（datetime_original 等）已单独处理，不经过这里。
    """
    if isinstance(obj, bytes):
        # 尝试 UTF-8 解码（Make/Model 等文本字段）
        # 额外要求解码结果全为可打印字符，否则视为二进制数据转 hex
        try:
            text = obj.decode("utf-8", errors="strict").rstrip("\x00").strip()
            if not text or text.isprintable():
                return text
            return "0x" + obj.hex()
        except UnicodeDecodeError:
            return "0x" + obj.hex()
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    # fallback：未知类型转字符串
    return str(obj)


# ---------------------------------------------------------------------------
# 常量：item flags
# ---------------------------------------------------------------------------

ItemFlag = Literal[
    "NO_EXIF",        # 完全没有 EXIF
    "LOW_CONFIDENCE", # 最终置信度低于阈值
    "CONFLICT_TIME",  # 多条 evidence 时间矛盾
    "SCREENSHOT",     # 截图（非相机拍摄）
    "SCAN",           # 扫描件
    "DUPLICATE",      # 与其他 item 重复
    "NEEDS_REVIEW",   # 需要人工审核
    "UNSUPPORTED",    # 不支持的文件格式，不参与后续 pipeline
]


# ---------------------------------------------------------------------------
# 子结构：ExifData
# ---------------------------------------------------------------------------

@dataclass
class ExifData:
    """从 EXIF 提取的结构化字段。"""

    datetime_original: Optional[datetime] = None
    """DateTimeOriginal：快门按下的时刻，最权威。"""

    datetime_digitized: Optional[datetime] = None
    """DateTimeDigitized：数字化时刻（扫描件常用）。"""

    datetime_modify: Optional[datetime] = None
    """DateTime（文件修改时间，可靠性较低）。"""

    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
    gps_timestamp: Optional[datetime] = None
    """GPS 记录的 UTC 时间（可用于时区校正）。"""

    device_make: Optional[str] = None
    device_model: Optional[str] = None

    raw: dict[str, Any] = field(default_factory=dict)
    """完整原始 EXIF，备用。"""

    def to_dict(self) -> dict[str, Any]:
        def _dt(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt is not None else None

        return {
            "datetime_original": _dt(self.datetime_original),
            "datetime_digitized": _dt(self.datetime_digitized),
            "datetime_modify": _dt(self.datetime_modify),
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "gps_alt": self.gps_alt,
            "gps_timestamp": _dt(self.gps_timestamp),
            "device_make": self.device_make,
            "device_model": self.device_model,
            "raw": _sanitize_for_json(self.raw),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExifData":
        def _dt(s: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(s) if s else None

        return cls(
            datetime_original=_dt(d.get("datetime_original")),
            datetime_digitized=_dt(d.get("datetime_digitized")),
            datetime_modify=_dt(d.get("datetime_modify")),
            gps_lat=d.get("gps_lat"),
            gps_lon=d.get("gps_lon"),
            gps_alt=d.get("gps_alt"),
            gps_timestamp=_dt(d.get("gps_timestamp")),
            device_make=d.get("device_make"),
            device_model=d.get("device_model"),
            raw=d.get("raw", {}),
        )


# ---------------------------------------------------------------------------
# 子结构：TimeResult
# ---------------------------------------------------------------------------

@dataclass
class TimeResult:
    """
    时间推理的最终综合结论。

    由 pipeline 末尾的决策逻辑（writer stage 之前）统一写入，
    不要在各 stage 内部修改此字段。
    """

    final_datetime: Optional[datetime] = None
    """最终采用的时间点。"""

    confidence: float = 0.0
    """综合置信度 [0.0, 1.0]。"""

    precision: TimePrecision = "unknown"
    """最终时间精度。"""

    range_start: Optional[datetime] = None
    range_end: Optional[datetime] = None
    """综合置信区间。"""

    primary_source: Optional[str] = None
    """决定 final_datetime 的主要 evidence source（可读性用）。"""

    source_summary: list[str] = field(default_factory=list)
    """所有参与决策的 evidence source 列表，按权重降序。"""

    def to_dict(self) -> dict[str, Any]:
        def _dt(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt is not None else None

        return {
            "final_datetime": _dt(self.final_datetime),
            "confidence": self.confidence,
            "precision": self.precision,
            "range_start": _dt(self.range_start),
            "range_end": _dt(self.range_end),
            "primary_source": self.primary_source,
            "source_summary": self.source_summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimeResult":
        def _dt(s: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(s) if s else None

        return cls(
            final_datetime=_dt(d.get("final_datetime")),
            confidence=d.get("confidence", 0.0),
            precision=d.get("precision", "unknown"),
            range_start=_dt(d.get("range_start")),
            range_end=_dt(d.get("range_end")),
            primary_source=d.get("primary_source"),
            source_summary=d.get("source_summary", []),
        )

    @property
    def is_resolved(self) -> bool:
        """是否已完成时间推理。"""
        return self.final_datetime is not None

    @property
    def is_confident(self, threshold: float = 0.6) -> bool:
        """是否达到可信阈值（默认 0.6）。"""
        return self.confidence >= threshold


# ---------------------------------------------------------------------------
# Item dataclass
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """
    一张照片（或视频）在 pipeline 中的完整数据载体。

    使用约定
    --------
    - 所有 stage 只追加 evidence，不修改已有 evidence
    - 所有 stage 通过 add_evidence() 方法添加证据
    - time_result 由 pipeline 末尾统一写入，stage 不直接修改
    - 用 add_flag() / has_flag() 管理 flags，不要直接操作列表
    """

    # ------------------------------------------------------------------
    # 必填：路径信息
    # ------------------------------------------------------------------

    path: str
    """绝对路径。"""

    relpath: str
    """相对于扫描根目录的路径（用于展示和日志）。"""

    # ------------------------------------------------------------------
    # 基础身份
    # ------------------------------------------------------------------

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """全局唯一 ID，自动生成。"""

    filename: str = ""
    ext: str = ""
    """文件扩展名，小写，不含点，如 'jpg'。"""

    # ------------------------------------------------------------------
    # 文件系统信息
    # ------------------------------------------------------------------

    filesize: int = 0
    """字节数。"""

    ctime: float = 0.0
    """文件创建时间（Unix timestamp）。不可信，仅作参考。"""

    mtime: float = 0.0
    """文件修改时间（Unix timestamp）。不可信，仅作参考。"""

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    sha1: str = ""
    """SHA-1，用于去重。"""

    phash: str = ""
    """感知哈希，用于相似图检索。"""

    dhash: str = ""
    """差值哈希，备用相似图检索。"""

    # ------------------------------------------------------------------
    # 图片基础信息
    # ------------------------------------------------------------------

    width: int = 0
    height: int = 0

    format_real: str = ""
    """真实格式（通过文件头检测，可能与扩展名不同），如 'JPEG'、'PNG'。"""

    is_screenshot: bool = False
    """是否为截图（通过分辨率比例 / EXIF 缺失等启发式判断）。"""

    is_scan: bool = False
    """是否为扫描件。"""

    # ------------------------------------------------------------------
    # Live Photo 来源（由 LivePhotoStage 填充）
    # ------------------------------------------------------------------

    livp_source_path: str = ""
    """
    若本 item 是从 .livp 文件解压出的图片，此字段记录原始 .livp 的绝对路径。
    空字符串表示不是来自 livp。
    Writer 据此字段决定把输出写回到原始 .livp 对应的位置。
    """

    # ------------------------------------------------------------------
    # 去重（由 DedupStage 填充）
    # ------------------------------------------------------------------

    duplicate_group_id: str = ""
    """
    同一重复组的共享 ID（sha1 相同或 phash 极近）。
    空字符串表示尚未去重分析，或该 item 无重复。
    同一组内所有 item 共享同一个 group_id。
    """

    duplicate_rank: int = 0
    """
    在重复组内的质量排名，0 = 最优（被保留），越大越差。
    仅在 duplicate_group_id 非空时有意义。
    """

    duplicate_kind: str = ""
    """
    重复类型：
      "exact"    sha1 完全相同
      "near"     phash 汉明距离 <= 4（极可能是同一张图的不同格式/质量）
      ""         无重复
    """

    # ------------------------------------------------------------------
    # EXIF
    # ------------------------------------------------------------------

    exif: ExifData = field(default_factory=ExifData)

    # ------------------------------------------------------------------
    # AI / Embedding（计算后填入，初始为 None 表示未计算）
    # ------------------------------------------------------------------

    clip_embedding: Optional[list[float]] = None
    """CLIP 向量，维度取决于模型（通常 512 或 768）。"""

    face_embedding: Optional[list[float]] = None
    """人脸特征向量，None 表示未检测到人脸或未运行。"""

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    ocr_text: Optional[str] = None
    """OCR 提取的全部文字，None 表示未运行 OCR。"""

    # ------------------------------------------------------------------
    # 时间推理（核心）
    # ------------------------------------------------------------------

    evidence: list[Evidence] = field(default_factory=list)
    """所有时间证据，只追加，不删除不覆盖。"""

    time_result: TimeResult = field(default_factory=TimeResult)
    """最终时间推理结论，由 pipeline 末尾写入。"""

    # ------------------------------------------------------------------
    # 日志与调试
    # ------------------------------------------------------------------

    logs: list[str] = field(default_factory=list)
    """运行日志，格式：'[StageName] message'。"""

    warnings: list[str] = field(default_factory=list)
    """警告信息（不影响运行，但值得注意）。"""

    flags: list[ItemFlag] = field(default_factory=list)
    """状态标志，见 ItemFlag。"""

    # ------------------------------------------------------------------
    # Pipeline 状态
    # ------------------------------------------------------------------

    completed_stages: list[str] = field(default_factory=list)
    """已成功完成的 stage 名称列表。"""

    failed_stages: list[str] = field(default_factory=list)
    """执行失败的 stage 名称列表。"""

    # ------------------------------------------------------------------
    # 临时缓存（不持久化）
    # ------------------------------------------------------------------

    cache: dict[str, Any] = field(default_factory=dict)
    """
    stage 内部临时数据，不写入最终输出。
    例如：{'phash_computed': True, 'ocr_raw_result': [...]}
    """

    # ------------------------------------------------------------------
    # Evidence 操作
    # ------------------------------------------------------------------

    def add_evidence(self, ev: Evidence) -> None:
        """追加一条时间证据（不覆盖已有证据）。"""
        self.evidence.append(ev)

    def get_evidence_by_source(self, source: str) -> list[Evidence]:
        """按 source 过滤证据。"""
        return [e for e in self.evidence if e.source == source]

    def get_direct_evidence(self) -> list[Evidence]:
        """获取所有直接证据。"""
        return [e for e in self.evidence if e.is_direct]

    def best_evidence(self) -> Optional[Evidence]:
        """返回置信度最高的证据，无证据时返回 None。"""
        if not self.evidence:
            return None
        return max(self.evidence, key=lambda e: e.confidence)

    # ------------------------------------------------------------------
    # Flag 操作
    # ------------------------------------------------------------------

    def add_flag(self, flag: ItemFlag) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def has_flag(self, flag: ItemFlag) -> bool:
        return flag in self.flags

    def remove_flag(self, flag: ItemFlag) -> None:
        if flag in self.flags:
            self.flags.remove(flag)

    # ------------------------------------------------------------------
    # Stage 状态操作
    # ------------------------------------------------------------------

    def mark_stage_done(self, stage_name: str) -> None:
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)

    def mark_stage_failed(self, stage_name: str) -> None:
        if stage_name not in self.failed_stages:
            self.failed_stages.append(stage_name)

    def stage_done(self, stage_name: str) -> bool:
        return stage_name in self.completed_stages

    # ------------------------------------------------------------------
    # 日志操作
    # ------------------------------------------------------------------

    def log(self, stage_name: str, message: str) -> None:
        self.logs.append(f"[{stage_name}] {message}")

    def warn(self, stage_name: str, message: str) -> None:
        self.warnings.append(f"[{stage_name}] {message}")

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def has_exif_time(self) -> bool:
        return self.exif.datetime_original is not None

    @property
    def has_gps(self) -> bool:
        return self.exif.gps_lat is not None and self.exif.gps_lon is not None

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    @property
    def is_time_resolved(self) -> bool:
        return self.time_result.is_resolved

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON-friendly dict。"""
        return {
            # 身份
            "id": self.id,
            "path": self.path,
            "relpath": self.relpath,
            "filename": self.filename,
            "ext": self.ext,
            # 文件系统
            "filesize": self.filesize,
            "ctime": self.ctime,
            "mtime": self.mtime,
            # hash
            "sha1": self.sha1,
            "phash": self.phash,
            "dhash": self.dhash,
            # 去重
            "duplicate_group_id": self.duplicate_group_id,
            "duplicate_rank": self.duplicate_rank,
            "duplicate_kind": self.duplicate_kind,
            # 图片信息
            "width": self.width,
            "height": self.height,
            "format_real": self.format_real,
            "is_screenshot": self.is_screenshot,
            "is_scan": self.is_scan,
            "livp_source_path": self.livp_source_path,
            # EXIF
            "exif": self.exif.to_dict(),
            # AI
            "clip_embedding": self.clip_embedding,
            "face_embedding": self.face_embedding,
            # OCR
            "ocr_text": self.ocr_text,
            # 时间推理
            "evidence": [e.to_dict() for e in self.evidence],
            "time_result": self.time_result.to_dict(),
            # 日志
            "logs": self.logs,
            "warnings": self.warnings,
            "flags": self.flags,
            # pipeline 状态
            "completed_stages": self.completed_stages,
            "failed_stages": self.failed_stages,
            # cache 不序列化（临时数据）
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        """从 dict 反序列化（用于 checkpoint 恢复）。"""
        item = cls(
            path=d["path"],
            relpath=d.get("relpath", ""),
        )
        item.id = d.get("id", item.id)
        item.filename = d.get("filename", "")
        item.ext = d.get("ext", "")
        item.filesize = d.get("filesize", 0)
        item.ctime = d.get("ctime", 0.0)
        item.mtime = d.get("mtime", 0.0)
        item.sha1 = d.get("sha1", "")
        item.phash = d.get("phash", "")
        item.dhash = d.get("dhash", "")
        item.width = d.get("width", 0)
        item.height = d.get("height", 0)
        item.format_real = d.get("format_real", "")
        item.is_screenshot = d.get("is_screenshot", False)
        item.is_scan = d.get("is_scan", False)
        item.livp_source_path = d.get("livp_source_path", "")
        item.duplicate_group_id = d.get("duplicate_group_id", "")
        item.duplicate_rank = d.get("duplicate_rank", 0)
        item.duplicate_kind = d.get("duplicate_kind", "")
        item.exif = ExifData.from_dict(d.get("exif", {}))
        item.clip_embedding = d.get("clip_embedding")
        item.face_embedding = d.get("face_embedding")
        item.ocr_text = d.get("ocr_text")
        item.evidence = [Evidence.from_dict(e) for e in d.get("evidence", [])]
        item.time_result = TimeResult.from_dict(d.get("time_result", {}))
        item.logs = d.get("logs", [])
        item.warnings = d.get("warnings", [])
        item.flags = d.get("flags", [])
        item.completed_stages = d.get("completed_stages", [])
        item.failed_stages = d.get("failed_stages", [])
        return item

    def __repr__(self) -> str:
        resolved = self.time_result.final_datetime
        dt_str = resolved.isoformat() if resolved else "unresolved"
        return (
            f"Item(id={self.id[:8]}…, "
            f"filename={self.filename!r}, "
            f"evidence={self.evidence_count}, "
            f"time={dt_str})"
        )