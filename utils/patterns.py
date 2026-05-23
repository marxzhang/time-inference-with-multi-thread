"""
utils/patterns.py — 预置时间模式库（静态正则）

职责：
    定义所有已知的文件名 / 文件夹名时间模式，
    提供统一的 match() 接口供 FilenameStage / FoldernameStage 调用。

设计：
    每个 DatePattern 描述一种命名规范：
      - regex     正则（带命名捕获组 year/month/day/hour/minute/second）
      - precision 能解析到的最细精度
      - confidence 置信度基准（挖掘出的动态模式会在此基础上浮动）
      - examples  示例，方便维护和测试

    match_all() 返回所有命中的结果，让 FilenameStage 自行决策，
    不在这里做"最优选择"——选择逻辑属于业务层。

命名捕获组约定：
    (?P<year>...)   四位年份
    (?P<month>...)  两位月份
    (?P<day>...)    两位日期
    (?P<hour>...)   两位小时（可选）
    (?P<minute>...) 两位分钟（可选）
    (?P<second>...) 两位秒（可选）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class DatePattern:
    """一条预置时间模式。"""

    name: str
    """模式唯一标识，用于 Evidence.metadata 和日志。"""

    regex: re.Pattern
    """预编译正则，带命名捕获组。"""

    precision: str
    """能解析到的最细精度：second / minute / hour / day / month / year。"""

    confidence: float
    """置信度基准。"""

    examples: list[str] = field(default_factory=list)
    """示例文件名（文档 + 单元测试用）。"""


@dataclass
class MatchResult:
    """一次匹配的结果。"""

    pattern_name: str
    dt: datetime
    precision: str
    confidence: float
    matched_text: str       # 正则匹配到的原始子串
    span: tuple[int, int]  # 在文件名中的位置


# ---------------------------------------------------------------------------
# 辅助：从命名捕获组构建 datetime
# ---------------------------------------------------------------------------

def _build_datetime(groups: dict[str, str]) -> Optional[datetime]:
    """
    从正则命名捕获组构建 datetime。
    缺失字段用最小合法值填充（month/day=1，时分秒=0）。
    返回 None 表示数值非法（如 month=13）。
    """
    try:
        year   = int(groups["year"])
        month  = int(groups.get("month") or 1)
        day    = int(groups.get("day")   or 1)
        hour   = int(groups.get("hour")   or 0)
        minute = int(groups.get("minute") or 0)
        second = int(groups.get("second") or 0)

        # 基本合法性检查（datetime() 本身也会抛 ValueError）
        if not (1900 <= year <= 2100):
            return None

        return datetime(year, month, day, hour, minute, second)
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# 预置模式库
# ---------------------------------------------------------------------------
#
# 排列原则：
#   1. 精度高的排前面（秒 > 分 > 天 > 月 > 年）
#   2. 同精度下，置信度高的排前面
#   3. 每个模式的 regex 尽量精确，避免误匹配
#

_RAW_PATTERNS: list[tuple[str, str, str, float, list[str]]] = [
    # (name, pattern_str, precision, confidence, examples)

    # ------------------------------------------------------------------
    # 精确到秒：YYYYMMDDHHmmss 连续
    # ------------------------------------------------------------------
    (
        "DATETIME14",
        r"(?<![0-9])(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"
        r"(?P<hour>[01]\d|2[0-3])(?P<minute>[0-5]\d)(?P<second>[0-5]\d)(?![0-9])",
        "second", 0.9,
        ["IMG_20200101143022.jpg", "微信图片_20211103153244.jpg"],
    ),
    # ------------------------------------------------------------------
    # 精确到秒：YYYY-MM-DD HH-mm-ss / YYYY-MM-DD HH.mm.ss 等分隔
    # ------------------------------------------------------------------
    (
        "DATETIME_SEP",
        r"(?P<year>(?:19|20)\d{2})[-_. ](?P<month>0[1-9]|1[0-2])[-_. ](?P<day>0[1-9]|[12]\d|3[01])"
        r"[-_. T](?P<hour>[01]\d|2[0-3])[-_.:h](?P<minute>[0-5]\d)[-_.:m](?P<second>[0-5]\d)",
        "second", 0.9,
        ["2023-12-05 08-30-11.png", "2020_01_01_14_30_22.jpg", "2020-01-01T14:30:22.jpg"],
    ),
    # ------------------------------------------------------------------
    # 精确到分：YYYYMMDD_HHmm
    # ------------------------------------------------------------------
    (
        "DATETIME12",
        r"(?<![0-9])(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"
        r"_(?P<hour>[01]\d|2[0-3])(?P<minute>[0-5]\d)(?![0-9])",
        "minute", 0.85,
        ["IMG_20200101_1430.jpg"],
    ),
    # ------------------------------------------------------------------
    # 精确到秒：YYYYMMDD_HHmmss（最常见：相机 / Android）
    # ------------------------------------------------------------------
    (
        "DATETIME8_6",
        r"(?<![0-9])(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"
        r"[_\-](?P<hour>[01]\d|2[0-3])(?P<minute>[0-5]\d)(?P<second>[0-5]\d)(?![0-9])",
        "second", 0.9,
        ["IMG_20200101_143022.jpg", "Screenshot_20231205_083011.png",
         "DCIM_20190821-092211.jpg"],
    ),
    # ------------------------------------------------------------------
    # 精确到天：YYYYMMDD 连续（最常见的纯日期格式）
    # ------------------------------------------------------------------
    (
        "DATE8",
        r"(?<![0-9])(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])(?![0-9])",
        "day", 0.75,
        ["WeiXin_20190821.jpg", "photo_20200101.jpg"],
    ),
    # ------------------------------------------------------------------
    # 精确到天：YYYY-MM-DD / YYYY.MM.DD / YYYY_MM_DD
    # ------------------------------------------------------------------
    (
        "DATE_SEP",
        r"(?P<year>(?:19|20)\d{2})[-_.](?P<month>0[1-9]|1[0-2])[-_.](?P<day>0[1-9]|[12]\d|3[01])",
        "day", 0.8,
        ["2020-01-01.jpg", "backup_2019.08.21.jpg", "photo_2023_12_05.jpg"],
    ),
    # ------------------------------------------------------------------
    # 精确到月：YYYYMM 连续（需要排除被 DATE8 捕获的前缀）
    # ------------------------------------------------------------------
    (
        "YEARMONTH6",
        r"(?<![0-9])(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?![0-9])",
        "month", 0.5,
        ["album_202001/", "backup_201908.zip"],
    ),
    # ------------------------------------------------------------------
    # 精确到月：YYYY-MM
    # ------------------------------------------------------------------
    (
        "YEARMONTH_SEP",
        r"(?P<year>(?:19|20)\d{2})[-_.](?P<month>0[1-9]|1[0-2])(?![-_.0-9])",
        "month", 0.55,
        ["2020-01/", "photos_2019-08/"],
    ),
    # ------------------------------------------------------------------
    # 精确到年：独立四位年份
    # ------------------------------------------------------------------
    (
        "YEAR4",
        r"(?<![0-9])(?P<year>(?:19|20)\d{2})(?![0-9])",
        "year", 0.3,
        ["2020/", "旅行2019/", "backup_2018.zip"],
    ),
    # ------------------------------------------------------------------
    # 中文日期：YYYY年MM月DD日
    # ------------------------------------------------------------------
    (
        "DATE_CN",
        r"(?P<year>(?:19|20)\d{2})年(?P<month>0?[1-9]|1[0-2])月(?P<day>0?[1-9]|[12]\d|3[01])日",
        "day", 0.85,
        ["2020年01月01日.jpg", "照片_2019年8月21日.png"],
    ),
    # ------------------------------------------------------------------
    # 中文年月：YYYY年MM月
    # ------------------------------------------------------------------
    (
        "YEARMONTH_CN",
        r"(?P<year>(?:19|20)\d{2})年(?P<month>0?[1-9]|1[0-2])月",
        "month", 0.6,
        ["2020年1月/", "相册_2019年8月/"],
    ),
    # ------------------------------------------------------------------
    # 中文年份：YYYY年
    # ------------------------------------------------------------------
    (
        "YEAR_CN",
        r"(?P<year>(?:19|20)\d{2})年",
        "year", 0.35,
        ["2020年/", "旅行_2019年.zip"],
    ),
]

# 编译为 DatePattern 对象（模块加载时执行一次）
BUILTIN_PATTERNS: list[DatePattern] = [
    DatePattern(
        name=name,
        regex=re.compile(pattern_str),
        precision=precision,
        confidence=confidence,
        examples=examples,
    )
    for name, pattern_str, precision, confidence, examples in _RAW_PATTERNS
]


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def match_all(text: str, patterns: Optional[list[DatePattern]] = None) -> list[MatchResult]:
    """
    对 text（文件名或文件夹名）运行所有模式，返回全部命中结果。

    参数
    ----
    text     : 去掉扩展名后的文件名，或文件夹名。
    patterns : 要使用的模式列表，默认使用 BUILTIN_PATTERNS。
               FilenameStage 会把挖掘出的动态模式追加到这里。

    返回
    ----
    list[MatchResult]，按 confidence 降序排列。
    空列表表示完全没有命中。

    注意
    ----
    同一段文本可能被多个模式命中（比如 DATE8 和 YEAR4 都会匹配到年份部分）。
    调用方负责决策使用哪条——通常取第一条（精度最高 + 置信度最高）。
    """
    if patterns is None:
        patterns = BUILTIN_PATTERNS

    results: list[MatchResult] = []

    for pat in patterns:
        for m in pat.regex.finditer(text):
            dt = _build_datetime(m.groupdict())
            if dt is None:
                continue
            results.append(MatchResult(
                pattern_name=pat.name,
                dt=dt,
                precision=pat.precision,
                confidence=pat.confidence,
                matched_text=m.group(0),
                span=m.span(),
            ))

    # 按置信度降序，置信度相同时精度更高的排前面
    _PRECISION_ORDER = {"second": 6, "minute": 5, "hour": 4,
                        "day": 3, "month": 2, "year": 1, "unknown": 0}
    results.sort(
        key=lambda r: (r.confidence, _PRECISION_ORDER.get(r.precision, 0)),
        reverse=True,
    )
    return results


def match_best(text: str, patterns: Optional[list[DatePattern]] = None) -> Optional[MatchResult]:
    """match_all 的快捷版，只返回最优的一条，无命中返回 None。"""
    results = match_all(text, patterns)
    return results[0] if results else None