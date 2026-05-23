"""
时间证据（Evidence）

设计原则：
- 每次推断只追加，不覆盖——一张图可以有多条 evidence
- JSON-friendly：to_dict / from_dict 双向转换
- source / precision / confidence 使用字面量常量，避免魔法字符串
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# 常量：证据来源
# ---------------------------------------------------------------------------

EvidenceSource = Literal[
    "exif",          # EXIF 元数据
    "filename",      # 文件名解析
    "foldername",    # 文件夹名解析
    "filesystem",    # 文件系统 ctime / mtime
    "similar",       # 相似图推断
    "timeline",      # 前后时间连续性推断
    "ocr",           # OCR 识别到的时间文字
    "gps",           # GPS 反查时区 / 时间校正
    "video_pair",    # 视频与截帧关联
    "livephoto",     # Live Photo 关联
    "user_manual",   # 用户手动标注
]

# ---------------------------------------------------------------------------
# 常量：时间精度
# ---------------------------------------------------------------------------

TimePrecision = Literal[
    "second",   # 精确到秒
    "minute",   # 精确到分
    "hour",     # 精确到小时
    "day",      # 精确到天
    "month",    # 精确到月
    "year",     # 精确到年
    "unknown",  # 精度未知
]

# confidence 取值约定（供调用方参考，不做强制）
#   1.0   完全可信（EXIF DateTimeOriginal 来自正常相机）
#   0.9   高度可信（文件名含完整 YYYYMMDDHHmmss）
#   0.7   较可信（文件名含 YYYYMMDD）
#   0.5   一般（相邻图推断 / OCR）
#   0.3   弱（文件系统时间 / 仅有年份）
#   0.0   不可信 / 占位


# ---------------------------------------------------------------------------
# Evidence dataclass
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """
    一条时间证据。

    使用示例
    --------
    ev = Evidence(
        source="filename",
        dt=datetime(2020, 1, 1),
        precision="day",
        confidence=0.7,
        reason='matched "IMG_20200101_143022.jpg"',
        stage="FilenameStage",
        metadata={"matched_text": "20200101", "pattern": "DATE8"},
    )
    """

    # ------------------------------------------------------------------
    # 必填字段
    # ------------------------------------------------------------------

    source: EvidenceSource
    """证据来源，见 EvidenceSource 枚举。"""

    stage: str
    """产生该证据的 Stage 类名，便于 debug 追溯。"""

    # ------------------------------------------------------------------
    # 时间信息
    # ------------------------------------------------------------------

    dt: Optional[datetime] = None
    """推断出的时间点（可为 None，表示只有范围没有精确时间）。"""

    precision: TimePrecision = "unknown"
    """dt 字段的精度。"""

    # ------------------------------------------------------------------
    # 可信度
    # ------------------------------------------------------------------

    confidence: float = 0.5
    """
    [0.0, 1.0]。
    - 1.0 完全可信（EXIF DateTimeOriginal）
    - 0.0 不可信 / 占位
    """

    # ------------------------------------------------------------------
    # 时间范围
    # ------------------------------------------------------------------

    range_start: Optional[datetime] = None
    """置信时间范围下界（闭区间）。"""

    range_end: Optional[datetime] = None
    """置信时间范围上界（闭区间）。"""

    # ------------------------------------------------------------------
    # 推理说明
    # ------------------------------------------------------------------

    reason: str = ""
    """人类可读的推理说明，例如 'matched IMG_20200101.jpg'。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """
    附加结构化信息，按 source 类型约定内容：

    filename / foldername::

        {
            "matched_text": "20200101",
            "pattern": "DATE8",
            "position": [4, 12],
        }

    similar::

        {
            "matched_item_id": "uuid-of-similar-photo",
            "distance": 0.03,
            "method": "phash",
        }

    timeline::

        {
            "prev_item_id": "...",
            "next_item_id": "...",
            "gap_seconds": 180,
        }

    ocr::

        {
            "raw_text": "2020年1月1日",
            "bounding_box": [x, y, w, h],
        }
    """

    # ------------------------------------------------------------------
    # 关联 item
    # ------------------------------------------------------------------

    related_item_ids: list[str] = field(default_factory=list)
    """
    产生该证据所参考的其他 item id。
    - similar stage：被参考的相似图
    - timeline stage：前后相邻图
    """

    # ------------------------------------------------------------------
    # 直接 vs 间接证据
    # ------------------------------------------------------------------

    is_direct: bool = True
    """
    True  = 直接证据（EXIF、filename 含完整时间戳）。
    False = 间接 / 推断证据（相邻推断、时间轴插值）。
    """

    # ------------------------------------------------------------------
    # 内部字段（自动生成）
    # ------------------------------------------------------------------

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """全局唯一 ID，自动生成。"""

    debug: dict[str, Any] = field(default_factory=dict)
    """调试用任意信息，生产环境可忽略。"""

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON-friendly dict（datetime → ISO 字符串）。"""

        def _fmt(d: Optional[datetime]) -> Optional[str]:
            return d.isoformat() if d is not None else None

        return {
            "id": self.id,
            "source": self.source,
            "stage": self.stage,
            "datetime": _fmt(self.dt),          # JSON key 保持 "datetime"，对外接口不变
            "precision": self.precision,
            "confidence": self.confidence,
            "range_start": _fmt(self.range_start),
            "range_end": _fmt(self.range_end),
            "reason": self.reason,
            "metadata": self.metadata,
            "related_item_ids": self.related_item_ids,
            "is_direct": self.is_direct,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Evidence":
        """从 dict 反序列化（ISO 字符串 → datetime）。"""

        def _parse(s: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(s) if s else None

        return cls(
            id=d.get("id", str(uuid.uuid4())),
            source=d["source"],
            stage=d.get("stage", ""),
            dt=_parse(d.get("datetime")),       # JSON key 仍读 "datetime"
            precision=d.get("precision", "unknown"),
            confidence=d.get("confidence", 0.5),
            range_start=_parse(d.get("range_start")),
            range_end=_parse(d.get("range_end")),
            reason=d.get("reason", ""),
            metadata=d.get("metadata", {}),
            related_item_ids=d.get("related_item_ids", []),
            is_direct=d.get("is_direct", True),
            debug=d.get("debug", {}),
        )

    def __repr__(self) -> str:
        dt_str = self.dt.isoformat() if self.dt else "None"
        return (
            f"Evidence(source={self.source!r}, "
            f"dt={dt_str}, "
            f"precision={self.precision!r}, "
            f"confidence={self.confidence:.2f}, "
            f"is_direct={self.is_direct})"
        )