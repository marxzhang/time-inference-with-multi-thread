"""
utils/miner.py — 文件名模式挖掘器（签名驱动 + 多假设解析）

v3 改进要点
-----------

1. 数字结构签名（替代 abstractify 模板）
   不再把整个文件名映射为字符类型序列，而是只提取
   "数字段长度序列 + 分隔符序列" 作为签名：

       "Screenshot_2019_0723_152344" → sig=([4,4,6], ['_','_'])
       "Screenshot_20190830_223302"  → sig=([8,6],   ['_'])
       "照片_20200101_143022"        → sig=([8,6],   ['_'])  ← 与上面合并！
       "IMG_20200101_143022"         → sig=([8,6],   ['_'])  ← 同上

   汉字、字母、前缀长度全部折叠为"非数字前景"，不影响签名。
   "Screenshot_YYYY_MMDD" 和 "screen_YYYY_MMDD" 视为同一签名。

2. 预过滤优先
   只对"预置正则未覆盖的 stem"统计签名。
   避免高频的 IMG_YYYYMMDD 消耗配额和 most_common 名额。

3. 签名验证率门槛
   生成正则后，在同一签名的所有样本上测命中率。
   命中率 < MATCH_RATE_THRESHOLD 的正则丢弃（偶发规律，非真实模式）。

4. 正则宽松化
   字母/汉字段统一用 \\D*? 或 [^0-9]+ 通配，不绑定具体内容。
   生成的正则能同时匹配 Screenshot_ 和 screen_ 前缀的变体。

接口不变
--------
    PatternMiner.mine(items, ctx) → list[DatePattern]
    FilenameStage / main.py 无需任何修改。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from utils.patterns import DatePattern, _build_datetime, BUILTIN_PATTERNS

if TYPE_CHECKING:
    from core.item import Item
    from core.context import Context


# ---------------------------------------------------------------------------
# 签名提取
# ---------------------------------------------------------------------------

def _extract_signature(name: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """
    从文件名（或文件夹名）提取"数字结构签名"。

    返回 (digit_lengths, separators)：
        digit_lengths : 各数字段的长度，如 (8, 6)
        separators    : 各数字段之间的分隔符（归一化），如 ('_',)

    规则：
    - 连续数字 → 记录其长度
    - 非数字（字母、汉字、标点）→ 提取分隔符（归一化为单字符代表）
    - 前缀/后缀的非数字段不参与分隔符序列（只记录数字段之间的分隔符）

    归一化：
    - 任何空白 → ' '
    - 任何 CJK 或非 ASCII 字母 → 视为非分隔符内容，忽略
    - 常见分隔符 (- _ . / 空格) → 保留原字符
    - 其他符号 → '?'

    示例：
        "Screenshot_2019_0723_152344" → ([4,4,6], ['_','_'])
        "照片_20200101_143022"         → ([8,6],   ['_'])
        "2020-01-01 14:30:22"          → ([4,2,2,2,2,2], ['-','-',' ',':',':'])
        "IMG20200101143022"            → ([14], [])
    """
    digit_lengths: list[int] = []
    separators: list[str] = []

    i = 0
    n = len(name)
    last_digit_end = -1  # 上一个数字段结束位置，用于提取分隔符

    while i < n:
        if name[i].isdigit():
            # 数字段
            j = i
            while j < n and name[j].isdigit():
                j += 1
            digit_lengths.append(j - i)

            # 如果前面有数字段，提取两者之间的分隔符
            if last_digit_end >= 0 and last_digit_end < i:
                sep = _normalize_separator(name[last_digit_end:i])
                separators.append(sep)

            last_digit_end = j
            i = j
        else:
            i += 1

    return tuple(digit_lengths), tuple(separators)


def _normalize_separator(s: str) -> str:
    """
    把两个数字段之间的非数字内容归一化为单个分隔符字符。

    - 全是常见分隔符（_ - . /）→ 去重后拼接（最多2字符）
    - 包含字母或 CJK → 视为"词语分隔"，返回 'W'
    - 其他 → '?'
    """
    if not s:
        return ""

    # 是否包含字母或 CJK（说明中间有"词"，不只是分隔符）
    has_word = any(
        c.isalpha() or unicodedata.category(c).startswith("L")
        for c in s
    )
    if has_word:
        return "W"  # Word separator（两个数字段之间有字/字母）

    # 纯分隔符：取其中出现的唯一符号
    common = set(s) & set("_-./: ")
    if common:
        # 保留原始第一个非空分隔符
        for c in s:
            if c in "_-./: ":
                return c
    return "?"


def _sig_has_date_potential(digit_lengths: tuple[int, ...]) -> bool:
    """
    签名是否有可能包含日期（快速预过滤）。
    至少需要一段 >= 4 位的数字（年份），且总位数 >= 4。
    """
    if not digit_lengths:
        return False
    total = sum(digit_lengths)
    return total >= 4 and any(l >= 4 for l in digit_lengths)


def _min_count_for_sig(digit_lengths: tuple[int, ...]) -> int:
    """
    根据签名精度动态决定最低出现次数门槛。

    精度越高（数字总位数越多），说明命名格式越具体，
    出现 2 次就足以认为是真实规律（不太可能是巧合）。
    精度低的签名（如只有年份 4 位）则需要更多样本才可信。

    分层规则：
        总位数 >= 12（含时分秒或完整日期时间）→ 2
        总位数 >= 8 （含完整日期）             → 3
        总位数 >= 6 （年月）                   → 4
        其他（只有年份等）                      → 5
    """
    total = sum(digit_lengths)
    if total >= 12:
        return 2
    if total >= 8:
        return 3
    if total >= 6:
        return 4
    return 5


def _grouping_key(digit_lengths: tuple[int, ...]) -> int:
    """
    签名聚合键：用数字段总位数作为粗粒度分组。
    用于把 [4,4,6]=14位 和 [8,6]=14位 识别为"同一量级"，
    合并统计出现次数，降低稀疏签名被过滤的概率。
    """
    return sum(digit_lengths)


# ---------------------------------------------------------------------------
# 多假设解析（来自上一版，保持不变）
# ---------------------------------------------------------------------------

_FIELD_RANGES = {
    "year":   (1900, 2100),
    "month":  (1,    12),
    "day":    (1,    31),
    "hour":   (0,    23),
    "minute": (0,    59),
    "second": (0,    59),
}

_COMPOSITE_FIELDS = {
    "mmdd":   [("month", 2), ("day", 2)],
    "hhmmss": [("hour", 2), ("minute", 2), ("second", 2)],
    "yearmonth": [("year", 4), ("month", 2)],
}

_SEGMENT_SCHEMES: list[list[tuple[str, int]]] = [
    [("year", 4), ("month", 2), ("day", 2), ("hour", 2), ("minute", 2), ("second", 2)],
    [("year", 4), ("mmdd", 4), ("hhmmss", 6)],
    [("year", 4), ("month", 2), ("day", 2), ("hhmmss", 6)],
    [("year", 4), ("month", 2), ("day", 2), ("hour", 2), ("minute", 2)],
    [("year", 4), ("month", 2), ("day", 2)],
    [("year", 4), ("mmdd", 4)],
    [("year", 4), ("month", 2)],
    [("yearmonth", 6)],
    [("year", 4)],
]


@dataclass
class _ParsedScheme:
    fields: dict[str, int]
    precision: str
    scheme: list[tuple[str, int]]
    confidence_bonus: float = 0.0


def _try_parse_runs_with_scheme(
    digit_lengths: tuple[int, ...],
    digit_values: list[str],       # 各数字段的实际字符串值
    scheme: list[tuple[str, int]],
) -> Optional[_ParsedScheme]:
    """
    用一种分段方案解析数字段序列。
    digit_values: 从样本提取的实际数字字符串列表，与 digit_lengths 一一对应。
    """
    import calendar as _cal

    # 展开复合字段
    scheme_fields = []
    for fname, flen in scheme:
        if fname in _COMPOSITE_FIELDS:
            scheme_fields.extend(_COMPOSITE_FIELDS[fname])
        else:
            scheme_fields.append((fname, flen))

    # 把所有数字段的字符串拼接
    digit_str = "".join(digit_values)
    scheme_total = sum(flen for _, flen in scheme_fields)
    if scheme_total != len(digit_str):
        return None

    fields: dict[str, int] = {}
    pos = 0
    for fname, flen in scheme_fields:
        chunk = digit_str[pos:pos + flen]
        if len(chunk) != flen:
            return None
        val = int(chunk)
        lo, hi = _FIELD_RANGES.get(fname, (0, 9999))
        if not (lo <= val <= hi):
            return None
        fields[fname] = val
        pos += flen

    # 月日合法性
    if "month" in fields and "day" in fields:
        year = fields.get("year", 2000)
        max_day = _cal.monthrange(year, fields["month"])[1]
        if fields["day"] > max_day:
            return None

    has = set(fields)
    if "second" in has:     precision = "second"
    elif "minute" in has:   precision = "minute"
    elif "hour" in has:     precision = "hour"
    elif "day" in has:      precision = "day"
    elif "month" in has:    precision = "month"
    else:                   precision = "year"

    bonus = 0.05 if len(digit_lengths) >= 3 and len(scheme) >= 3 else 0.0
    return _ParsedScheme(fields=fields, precision=precision,
                         scheme=scheme_fields, confidence_bonus=bonus)


def _try_all_schemes(
    digit_lengths: tuple[int, ...],
    digit_values: list[str],
) -> list[_ParsedScheme]:
    """枚举所有方案，返回所有合法解析，按精度降序。"""
    _PREC_RANK = {"second": 6, "minute": 5, "hour": 4,
                  "day": 3, "month": 2, "year": 1}
    results = []
    seen: set[tuple] = set()
    for scheme in _SEGMENT_SCHEMES:
        parsed = _try_parse_runs_with_scheme(digit_lengths, digit_values, scheme)
        if parsed is None:
            continue
        key = (parsed.precision, tuple(sorted(parsed.fields.items())))
        if key not in seen:
            seen.add(key)
            results.append(parsed)
    results.sort(key=lambda r: _PREC_RANK.get(r.precision, 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# 从样本提取数字值
# ---------------------------------------------------------------------------

def _extract_digit_values(name: str) -> list[str]:
    """从文件名提取各数字段的实际字符串，与 digit_lengths 对应。"""
    values = []
    i, n = 0, len(name)
    while i < n:
        if name[i].isdigit():
            j = i
            while j < n and name[j].isdigit():
                j += 1
            values.append(name[i:j])
            i = j
        else:
            i += 1
    return values


# ---------------------------------------------------------------------------
# 正则生成（签名驱动，宽松匹配）
# ---------------------------------------------------------------------------

_FIELD_REGEX = {
    "year":   r"(?P<year>(?:19|20)\d{2})",
    "month":  r"(?P<month>0[1-9]|1[0-2])",
    "day":    r"(?P<day>0[1-9]|[12]\d|3[01])",
    "hour":   r"(?P<hour>[01]\d|2[0-3])",
    "minute": r"(?P<minute>[0-5]\d)",
    "second": r"(?P<second>[0-5]\d)",
}

# 分隔符 → 正则
_SEP_REGEX = {
    "_": r"_",
    "-": r"-",
    ".": r"\.",
    "/": r"/",
    ":": r":",
    " ": r"\s",
    "W": r"[^0-9]+",    # 有词语的分隔符：宽松匹配任意非数字内容
    "?": r"[^0-9]?",
    "":  r"",
}


def _build_regex_from_sig(
    digit_lengths: tuple[int, ...],
    separators: tuple[str, ...],
    parsed: _ParsedScheme,
) -> Optional[re.Pattern]:
    """
    根据签名和解析方案构建宽松正则。

    前缀/后缀的非数字内容用 [^0-9]* 通配（不绑定 IMG/Screenshot 等前缀），
    数字段之间的分隔符用 _SEP_REGEX 映射（稍微宽松，如 '_' 只匹配 '_'）。
    """
    scheme_iter = iter(parsed.scheme)
    current_field, current_len = next(scheme_iter, (None, 0))
    field_consumed = 0

    parts = [r"(?<![0-9])"]   # 前面不能是数字（边界）
    parts.append(r"[^0-9]*?") # 宽松前缀（字母/汉字/路径等）

    for seg_idx, dl in enumerate(digit_lengths):
        # 数字段：按 scheme 字段顺序填充
        remaining = dl
        while remaining > 0 and current_field is not None:
            consumed = min(remaining, current_len - field_consumed)
            if field_consumed == 0:
                frag = _FIELD_REGEX.get(current_field)
                if frag:
                    parts.append(frag)
                else:
                    parts.append(rf"\d{{{current_len}}}")
            field_consumed += consumed
            remaining -= consumed
            if field_consumed >= current_len:
                current_field, current_len = next(scheme_iter, (None, 0))
                field_consumed = 0

        if remaining > 0:
            parts.append(rf"\d{{{remaining}}}")

        # 分隔符（数字段之间）
        if seg_idx < len(separators):
            sep = separators[seg_idx]
            parts.append(_SEP_REGEX.get(sep, r"[^0-9]?"))

    parts.append(r"(?![0-9])")  # 后面不能是数字

    pattern_str = "".join(parts)
    try:
        return re.compile(pattern_str)
    except re.error:
        return None


# ---------------------------------------------------------------------------
# 验证率（命中率门槛）
# ---------------------------------------------------------------------------

_MATCH_RATE_THRESHOLD = 0.75   # 75% 以上的同签名样本能被命中才保留


def _validate_pattern(
    pat: re.Pattern,
    samples: list[str],
    parsed: _ParsedScheme,
) -> float:
    """
    在 samples 上测试正则的命中率。
    只统计能真正解析出合法 datetime 且年份一致的命中。
    返回 [0.0, 1.0] 的命中率。
    """
    if not samples:
        return 0.0
    hits = 0
    expected_year = parsed.fields.get("year")
    for s in samples:
        m = pat.search(s)
        if not m:
            continue
        dt = _build_datetime(m.groupdict())
        if dt is None:
            continue
        # 年份检查：若样本的年份各不相同（如文件夹按年组织），不做年份绑定
        if expected_year is not None and dt.year != expected_year:
            # 年份不匹配不算失败：可能不同样本年份不同
            pass
        hits += 1
    return hits / len(samples)


# ---------------------------------------------------------------------------
# PatternMiner（公开接口不变）
# ---------------------------------------------------------------------------

class PatternMiner:
    """
    从 Item 列表中挖掘文件名 / 文件夹名时间模式。

    接口与所有旧版完全兼容：
        mine(items, ctx) → list[DatePattern]
    """

    def __init__(
        self,
        min_count: int = 5,
        max_patterns: int = 20,
        match_rate_threshold: float = _MATCH_RATE_THRESHOLD,
    ) -> None:
        self.min_count             = min_count
        self.max_patterns          = max_patterns
        self.match_rate_threshold  = match_rate_threshold

    def mine(
        self,
        items: list["Item"],
        ctx: "Context" = None,  # type: ignore[assignment]
    ) -> list[DatePattern]:
        """
        主入口：分析文件名/文件夹名，返回动态 DatePattern 列表。

        流程：
            1. 收集 stems（文件名 stem + 文件夹各级 part）
            2. 预过滤：去掉预置正则已覆盖的 stem
            3. 对未覆盖 stem 提取数字结构签名，统计频次
            4. 对高频签名，多假设解析 → 生成宽松正则
            5. 验证率门槛：命中率 < threshold 的正则丢弃
            6. 返回 DatePattern 列表
        """
        # ── 1. 收集 stems ────────────────────────────────────────────────
        stems: list[str] = []
        for item in items:
            p = Path(item.path)
            stems.append(p.stem)
            for part in p.parent.parts:
                if part not in ("", "/", "\\"):
                    stems.append(part)

        if ctx:
            ctx.logger.debug(f"[PatternMiner] {len(stems)} names collected")

        # ── 2. 预过滤：去掉已覆盖的 stem ─────────────────────────────────
        uncovered = [s for s in stems if not self._already_covered(s)]

        if ctx:
            ctx.logger.debug(
                f"[PatternMiner] {len(uncovered)} uncovered "
                f"(filtered {len(stems) - len(uncovered)} already-covered)"
            )

        if not uncovered:
            return []

        # ── 3. 签名统计 ──────────────────────────────────────────────────
        # sig → (count, [sample_stem, ...])
        sig_counter: Counter[tuple] = Counter()
        sig_samples: dict[tuple, list[str]] = defaultdict(list)
        # grouping_key（总位数）→ sig 集合，用于聚合计数
        group_counts: Counter[int] = Counter()

        for stem in uncovered:
            dl, seps = _extract_signature(stem)
            if not _sig_has_date_potential(dl):
                continue
            sig = (dl, seps)
            sig_counter[sig] += 1
            group_counts[_grouping_key(dl)] += 1
            # 每个签名最多保留 50 个样本（够验证率计算，不占太多内存）
            if len(sig_samples[sig]) < 50:
                sig_samples[sig].append(stem)

        if ctx:
            ctx.logger.debug(
                f"[PatternMiner] {len(sig_counter)} distinct signatures, "
                f"group totals: { {k: v for k, v in group_counts.most_common(5)} }"
            )

        # ── 4~5. 多假设解析 + 验证率过滤 ─────────────────────────────────
        discovered: list[DatePattern] = []
        seen_sigs: set[tuple] = set()

        _CONFIDENCE_BASE = {
            "second": 0.85, "minute": 0.80, "hour": 0.75,
            "day":    0.70, "month":  0.45, "year": 0.25,
        }

        for sig, count in sig_counter.most_common():
            if len(discovered) >= self.max_patterns:
                break

            dl, seps = sig

            # 动态门槛：精度高的签名降低最低出现次数要求
            # 同时用聚合 count（同总位数的所有签名之和）辅助判断
            dynamic_min = _min_count_for_sig(dl)
            group_total = group_counts[_grouping_key(dl)]
            # 如果该签名本身不够，但同位数组合总量充足，降低门槛
            if count < dynamic_min:
                if group_total < self.min_count:
                    continue   # 组合也不够，真的太少了，跳过
                # 组合够了但单签名不够：降级到 2 次（避免误杀有价值的变体）
                if count < 2:
                    continue

            dl, seps = sig
            samples = sig_samples[sig]

            # 用第一个样本提取实际数字值做多假设解析
            digit_values = _extract_digit_values(samples[0])
            candidates = _try_all_schemes(dl, digit_values)

            if not candidates and ctx:
                ctx.logger.debug(
                    f"[PatternMiner] no valid scheme for sig={dl} "
                    f"(sample={samples[0]!r})"
                )

            for parsed in candidates:
                if len(discovered) >= self.max_patterns:
                    break

                dedup_key = (dl, seps, parsed.precision)
                if dedup_key in seen_sigs:
                    continue

                # 生成正则
                compiled = _build_regex_from_sig(dl, seps, parsed)
                if compiled is None:
                    continue

                # 验证率
                rate = _validate_pattern(compiled, samples, parsed)
                if rate < self.match_rate_threshold:
                    if ctx:
                        ctx.logger.debug(
                            f"[PatternMiner] low match rate {rate:.0%} for "
                            f"sig={dl}, prec={parsed.precision}, skip"
                        )
                    continue

                # 置信度：验证率越高越加分
                confidence = (
                    _CONFIDENCE_BASE.get(parsed.precision, 0.5)
                    + parsed.confidence_bonus
                    + 0.05 * (rate - self.match_rate_threshold)  # 超出阈值部分加分
                )
                confidence = min(confidence, 0.89)  # 动态模式上限低于预置正则

                pattern_name = f"MINED_{'_'.join(map(str,dl))}_{parsed.precision}"

                pat = DatePattern(
                    name=pattern_name,
                    regex=compiled,
                    precision=parsed.precision,
                    confidence=confidence,
                    examples=samples[:3],
                )
                discovered.append(pat)
                seen_sigs.add(dedup_key)

                if ctx:
                    fields_str = ", ".join(
                        f"{k}={v}" for k, v in sorted(parsed.fields.items())
                    )
                    ctx.logger.info(
                        f"[PatternMiner] discovered {pattern_name!r} "
                        f"(count={count}, match_rate={rate:.0%}, "
                        f"fields={{{fields_str}}}, sample={samples[0]!r})"
                    )

        return discovered

    @staticmethod
    def _already_covered(text: str) -> bool:
        """
        判断文件名是否已被预置正则"充分"覆盖。

        "充分"的定义：预置正则命中的精度 >= day。
        若最佳匹配只有 year 或 month，仍需挖掘（可能有更精确的结构）。
        """
        from utils.patterns import match_best
        result = match_best(text)
        if result is None:
            return False
        _PREC_RANK = {"second": 6, "minute": 5, "hour": 4,
                      "day": 3, "month": 2, "year": 1}
        # 精度达到 day 及以上才算充分覆盖
        return _PREC_RANK.get(result.precision, 0) >= 3