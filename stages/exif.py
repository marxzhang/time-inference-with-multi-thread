"""
ExifStage — EXIF 提取与时间推断

职责：
    读取图片 EXIF，填充 item.exif，并根据找到的时间字段
    生成对应的 Evidence 追加到 item.evidence。

产出的 Evidence（按优先级）：
    1. exif / DateTimeOriginal   confidence=1.0  precision=second  is_direct=True
    2. exif / DateTimeDigitized  confidence=0.9  precision=second  is_direct=True
    3. exif / DateTime           confidence=0.6  precision=second  is_direct=True
    4. filesystem / mtime        confidence=0.2  precision=second  is_direct=False
       （仅在完全没有任何 EXIF 时间时作为兜底）
    5. gps                       单独追加，用于后期时区校正
       （不产生独立时间 evidence，只填充 item.exif.gps_* 字段）

填充的 item 字段：
    item.exif.datetime_original
    item.exif.datetime_digitized
    item.exif.datetime_modify
    item.exif.gps_lat / gps_lon / gps_alt / gps_timestamp
    item.exif.device_make / device_model
    item.exif.raw
    item.flags  ← NO_EXIF（完全无 EXIF 时）

依赖：
    piexif   pip install piexif    （主力，精确控制每个 IFD tag）
    Pillow   pip install Pillow    （fallback，兼容更多格式）

两者都没有时：只生成 filesystem evidence，不抛异常。

EXIF 时间格式说明：
    标准格式："%Y:%m:%d %H:%M:%S"
    少数相机会写 "%Y-%m-%d %H:%M:%S" 或带毫秒 "%Y:%m:%d %H:%M:%S.%f"
    本模块全部处理。
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.stage_base import Stage, StageSkip

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# EXIF tag ID（Exif IFD）
_TAG_DATETIME_ORIGINAL  = 36867   # DateTimeOriginal
_TAG_DATETIME_DIGITIZED = 36868   # DateTimeDigitized
_TAG_DATETIME           = 306     # DateTime（IFD 0）
_TAG_MAKE               = 271     # Make
_TAG_MODEL              = 272     # Model

# GPS IFD tag ID
_TAG_GPS_LAT_REF  = 1
_TAG_GPS_LAT      = 2
_TAG_GPS_LON_REF  = 3
_TAG_GPS_LON      = 4
_TAG_GPS_ALT_REF  = 5
_TAG_GPS_ALT      = 6
_TAG_GPS_TIME_UTC = 7
_TAG_GPS_DATE     = 29

# 支持解析的文件扩展名（视频没有 EXIF，跳过）
_SUPPORTED_EXT = frozenset({
    "jpg", "jpeg", "tiff", "tif", "heic", "heif", "avif",
    "raw", "cr2", "cr3", "nef", "arw", "orf", "rw2", "dng",
    "png",   # PNG 可能有 Exif chunk
    "webp",  # WebP 可能有 Exif chunk
})

# EXIF 时间字符串的所有合法格式
_DATETIME_FORMATS = [
    "%Y:%m:%d %H:%M:%S",     # 标准 EXIF
    "%Y-%m-%d %H:%M:%S",     # 部分相机
    "%Y:%m:%d %H:%M:%S.%f",  # 带毫秒
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y:%m:%d",              # 只有日期（极少数）
]


# ---------------------------------------------------------------------------
# ExifStage
# ---------------------------------------------------------------------------

class ExifStage(Stage):
    """
    读取 EXIF，填充 item.exif，产出时间 Evidence。

    优先使用 piexif（精确），fallback 到 Pillow._getexif()（兼容）。
    """

    name = "ExifStage"

    # ------------------------------------------------------------------
    # 前置检查
    # ------------------------------------------------------------------

    def skip_reason(self, item: "Item", ctx: "Context") -> Optional[str]:
        if item.ext not in _SUPPORTED_EXT:
            return f"unsupported extension: {item.ext}"
        return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def process(self, item: "Item", ctx: "Context") -> None:
        path = Path(item.path)

        # 尝试用 piexif 读取
        raw_exif = self._read_with_piexif(path, item, ctx)

        # piexif 失败时 fallback 到 Pillow
        if raw_exif is None:
            raw_exif = self._read_with_pillow(path, item, ctx)

        # 两种方式都失败 → 无 EXIF
        if raw_exif is None:
            item.add_flag("NO_EXIF")
            item.log(self.name, "no EXIF found")
            self._add_filesystem_evidence(item)
            return

        # 解析结构化字段
        self._parse_datetime_fields(raw_exif, item)
        self._parse_gps_fields(raw_exif, item)
        self._parse_device_fields(raw_exif, item)
        item.exif.raw = raw_exif

        # 产出 Evidence
        self._emit_evidence(item)

        # 没有任何 EXIF 时间 → 兜底文件系统时间
        if item.exif.datetime_original is None \
                and item.exif.datetime_digitized is None \
                and item.exif.datetime_modify is None:
            item.add_flag("NO_EXIF")
            item.log(self.name, "EXIF present but no datetime fields")
            self._add_filesystem_evidence(item)

    # ------------------------------------------------------------------
    # 读取：piexif（主力）
    # ------------------------------------------------------------------

    def _read_with_piexif(
        self,
        path: Path,
        item: "Item",
        ctx: "Context",
    ) -> Optional[dict]:
        """
        用 piexif 读取，返回扁平化的 tag dict；失败返回 None。
        扁平化结构：{tag_id: value, ...}，合并 IFD0 + ExifIFD + GPS IFD。
        """
        try:
            import piexif
        except ImportError:
            ctx.logger.debug(f"[{self.name}] piexif not installed")
            return None

        try:
            exif_dict = piexif.load(str(path))
        except Exception as e:
            ctx.logger.debug(f"[{self.name}] piexif failed for {path.name}: {e}")
            return None

        # 把各 IFD 合并到一个扁平 dict，GPS 单独保留
        flat: dict = {}
        for ifd_name in ("0th", "Exif"):
            ifd = exif_dict.get(ifd_name, {})
            if isinstance(ifd, dict):
                flat.update(ifd)

        # GPS 保持独立 key
        gps = exif_dict.get("GPS", {})
        if isinstance(gps, dict) and gps:
            flat["_gps"] = gps

        return flat if flat else None

    # ------------------------------------------------------------------
    # 读取：Pillow fallback
    # ------------------------------------------------------------------

    def _read_with_pillow(
        self,
        path: Path,
        item: "Item",
        ctx: "Context",
    ) -> Optional[dict]:
        """
        用 Pillow._getexif() 读取，返回 {tag_id: value} dict；失败返回 None。
        Pillow 的 _getexif 返回已解码的值（字节串已解码为字符串）。
        """
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
        except ImportError:
            ctx.logger.debug(f"[{self.name}] Pillow not installed")
            return None

        try:
            with Image.open(path) as img:
                exif_data = img._getexif()  # type: ignore[attr-defined]
                if not exif_data:
                    return None
                return exif_data
        except Exception as e:
            ctx.logger.debug(f"[{self.name}] Pillow fallback failed for {path.name}: {e}")
            return None

    # ------------------------------------------------------------------
    # 解析：时间字段
    # ------------------------------------------------------------------

    def _parse_datetime_fields(self, raw: dict, item: "Item") -> None:
        """从扁平 raw dict 解析三个时间字段，写入 item.exif。"""
        mapping = [
            (_TAG_DATETIME_ORIGINAL,  "datetime_original"),
            (_TAG_DATETIME_DIGITIZED, "datetime_digitized"),
            (_TAG_DATETIME,           "datetime_modify"),
        ]
        for tag_id, attr in mapping:
            raw_val = raw.get(tag_id)
            if raw_val is None:
                continue
            dt = self._parse_exif_datetime(raw_val)
            if dt is not None:
                setattr(item.exif, attr, dt)

    # ------------------------------------------------------------------
    # 解析：GPS 字段
    # ------------------------------------------------------------------

    def _parse_gps_fields(self, raw: dict, item: "Item") -> None:
        """解析 GPS IFD，写入 item.exif.gps_*。"""
        gps = raw.get("_gps")
        if not gps:
            return

        lat = self._parse_gps_coord(
            gps.get(_TAG_GPS_LAT),
            gps.get(_TAG_GPS_LAT_REF),
        )
        lon = self._parse_gps_coord(
            gps.get(_TAG_GPS_LON),
            gps.get(_TAG_GPS_LON_REF),
        )
        if lat is not None:
            item.exif.gps_lat = lat
        if lon is not None:
            item.exif.gps_lon = lon

        # 海拔
        alt_raw = gps.get(_TAG_GPS_ALT)
        alt_ref = gps.get(_TAG_GPS_ALT_REF, 0)
        if alt_raw is not None:
            alt = self._rational_to_float(alt_raw)
            if alt is not None:
                item.exif.gps_alt = alt * (-1 if alt_ref == 1 else 1)

        # GPS UTC 时间
        time_raw = gps.get(_TAG_GPS_TIME_UTC)
        date_raw = gps.get(_TAG_GPS_DATE)
        if time_raw and date_raw:
            gps_dt = self._parse_gps_datetime(time_raw, date_raw)
            if gps_dt is not None:
                item.exif.gps_timestamp = gps_dt

    # ------------------------------------------------------------------
    # 解析：设备信息
    # ------------------------------------------------------------------

    def _parse_device_fields(self, raw: dict, item: "Item") -> None:
        make = raw.get(_TAG_MAKE)
        model = raw.get(_TAG_MODEL)
        if make:
            item.exif.device_make = self._decode_bytes(make).strip()
        if model:
            item.exif.device_model = self._decode_bytes(model).strip()

    # ------------------------------------------------------------------
    # 产出 Evidence
    # ------------------------------------------------------------------

    def _emit_evidence(self, item: "Item") -> None:
        """
        按优先级产出 EXIF 时间 Evidence。
        三个时间字段各自独立产出，让后续决策层自行选择最优。
        """
        if item.exif.datetime_original is not None:
            item.add_evidence(self.make_evidence(
                source="exif",
                dt=item.exif.datetime_original,
                precision="second",
                confidence=1.0,
                is_direct=True,
                reason="EXIF DateTimeOriginal",
                metadata={"exif_tag": "DateTimeOriginal"},
            ))

        if item.exif.datetime_digitized is not None:
            item.add_evidence(self.make_evidence(
                source="exif",
                dt=item.exif.datetime_digitized,
                precision="second",
                # 略低于 original：digitized 常见于扫描件，不一定是拍摄时间
                confidence=0.9,
                is_direct=True,
                reason="EXIF DateTimeDigitized",
                metadata={"exif_tag": "DateTimeDigitized"},
            ))

        if item.exif.datetime_modify is not None:
            item.add_evidence(self.make_evidence(
                source="exif",
                dt=item.exif.datetime_modify,
                precision="second",
                # DateTime 容易被编辑软件改写，可信度明显低于 original
                confidence=0.6,
                is_direct=True,
                reason="EXIF DateTime (last modified)",
                metadata={"exif_tag": "DateTime"},
            ))

    def _add_filesystem_evidence(self, item: "Item") -> None:
        """
        兜底：用文件系统 mtime 产出低可信度 evidence。
        mtime 容易被复制操作改写，仅在完全没有 EXIF 时使用。
        """
        from datetime import datetime as dt_cls
        mtime_dt = dt_cls.fromtimestamp(item.mtime)
        item.add_evidence(self.make_evidence(
            source="filesystem",
            dt=mtime_dt,
            precision="second",
            confidence=0.2,
            is_direct=False,
            reason="filesystem mtime (no EXIF available)",
            metadata={"mtime_raw": item.mtime},
        ))

    # ------------------------------------------------------------------
    # 工具：时间字符串解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_exif_datetime(value: object) -> Optional[datetime]:
        """
        解析 EXIF 时间字符串，兼容多种格式和编码。
        返回 None 表示解析失败（无效值 / 全零占位符）。
        """
        if value is None:
            return None

        # bytes → str
        if isinstance(value, (bytes, bytearray)):
            try:
                value = value.decode("utf-8", errors="ignore")
            except Exception:
                return None

        if not isinstance(value, str):
            return None

        value = value.strip().rstrip("\x00")

        # 全零占位符（相机未设时间时写入）
        if not value or value.startswith("0000"):
            return None

        for fmt in _DATETIME_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        return None

    # ------------------------------------------------------------------
    # 工具：GPS 坐标解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gps_coord(
        dms: object,
        ref: object,
    ) -> Optional[float]:
        """
        将 DMS（度分秒）格式转为十进制浮点数。
        dms 是三元组：((度分子,度分母), (分分子,分分母), (秒分子,秒分母))
        ref 是 b'N'/'S'/'E'/'W' 或字符串。
        """
        if dms is None:
            return None
        try:
            d = ExifStage._rational_to_float(dms[0])
            m = ExifStage._rational_to_float(dms[1])
            s = ExifStage._rational_to_float(dms[2])
            if d is None or m is None or s is None:
                return None
            decimal = d + m / 60.0 + s / 3600.0
            # 南纬 / 西经为负
            if isinstance(ref, bytes):
                ref = ref.decode("utf-8", errors="ignore")
            if isinstance(ref, str) and ref.upper() in ("S", "W"):
                decimal = -decimal
            return round(decimal, 7)
        except Exception:
            return None

    @staticmethod
    def _rational_to_float(value: object) -> Optional[float]:
        """
        piexif 返回的 rational 是 (分子, 分母) 元组。
        Pillow 有时直接返回 float。
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            if den == 0:
                return None
            return num / den
        return None

    # ------------------------------------------------------------------
    # 工具：GPS UTC 时间解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gps_datetime(
        time_raw: object,
        date_raw: object,
    ) -> Optional[datetime]:
        """
        GPS 时间是三个 rational（时、分、秒），日期是 "YYYY:MM:DD" 字符串。
        返回 UTC datetime（aware）。
        """
        try:
            h = ExifStage._rational_to_float(time_raw[0])
            m = ExifStage._rational_to_float(time_raw[1])
            s = ExifStage._rational_to_float(time_raw[2])
            if h is None or m is None or s is None:
                return None

            if isinstance(date_raw, bytes):
                date_raw = date_raw.decode("utf-8", errors="ignore")
            date_str = str(date_raw).strip().rstrip("\x00")
            date_obj = datetime.strptime(date_str, "%Y:%m:%d")

            return datetime(
                date_obj.year, date_obj.month, date_obj.day,
                int(h), int(m), int(s),
                tzinfo=timezone.utc,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 工具：bytes 解码
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_bytes(value: object) -> str:
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        return str(value) if value is not None else ""