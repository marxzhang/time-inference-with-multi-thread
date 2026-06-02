"""
utils/miner.py — 文件名模式挖掘器（结构驱动 + 多假设解析）

与旧版的区别
------------
旧版（长度驱动）：
    看到 8 位数字 → 硬认为是 YYYYMMDD。
    看到 4+4+6 → 不知道怎么处理，直接跳过。

新版（结构驱动）：
    对每个数字段序列，枚举所有合理的"分段方案"（假设），
    用实际数值范围验证，选出自洽的那个。

    示例：Screenshot_2019_0723_152344
        数字段序列：[4, 4, 6]
        假设 A：[year=4] + [mmdd=4] + [hhmmss=6]  → 2019-07-23 15:23:44 ✓
        假设 B：[year=4] + [year=4(非法)] + ...    → ✗
    → 选 A，生成精确到秒的正则

多假设解析（fallback）
-----------------------
    对一个数字段序列，可能存在多种合法假设（如 [4,4] 可以是 year+mmdd 或两个4位数）。
    全部生成，都加入候选池，置信度略有差异。
    FilenameStage 的 match_all() 会把所有结果都返回，由决策层选最优。

接口不变
--------
    PatternMiner.mine(items, ctx) → list[DatePattern]
    与旧版完全兼容，FilenameStage 无需任何修改。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from utils.patterns import DatePattern, _build_datetime, BUILTIN_PATTERNS

if TYPE_CHECKING:
    from core.item import Item
    from core.context import Context


# ---------------------------------------------------------------------------
# 字符类型抽象
# ---------------------------------------------------------------------------

def _abstractify(name: str) -> str:
    """
    文件名 → 字符类型序列。
        大写 → A，小写 → a，数字 → N，其他 → 原字符
    示例：
        "Screenshot_2019_0723_152344" → "AaaaaaaaaaaA_NNNN_NNNN_NNNNNN"
    """
    result = []
    for ch in name:
        if ch.isupper():
            result.append("A")
        elif ch.islower():
            result.append("a")
        elif ch.isdigit():
            result.append("N")
        else:
            result.append(ch)
    return "".join(result)


def _find_digit_runs(template: str) -> list[tuple[int, int]]:
    """
    模板 → 连续 N 段列表 [(start_pos, length), ...]
    """
    runs, i = [], 0
    while i < len(template):
        if template[i] == "N":
            j = i
            while j < len(template) and template[j] == "N":
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


# ---------------------------------------------------------------------------
# 语义分段：枚举所有合理的假设
# ---------------------------------------------------------------------------

# 每个字段的合法值范围
_FIELD_RANGES = {
    "year":   (1900, 2100),
    "month":  (1,    12),
    "day":    (1,    31),
    "hour":   (0,    23),
    "minute": (0,    59),
    "second": (0,    59),
}

# 完整日期时间结构：各字段的固定位数
# 格式：[(field_name, bit_length), ...]
# 所有合法的"分段方案"——每种方案是一种对数字段的语义解读
_SEGMENT_SCHEMES: list[list[tuple[str, int]]] = [
    # 精确到秒
    [("year", 4), ("month", 2), ("day", 2), ("hour", 2), ("minute", 2), ("second", 2)],  # YYYYMMDDHHMMSS (14)
    [("year", 4), ("mmdd", 4), ("hhmmss", 6)],   # YYYY + MMDD + HHMMSS (特殊: Screenshot_)
    [("year", 4), ("month", 2), ("day", 2), ("hhmmss", 6)],  # YYYY + MM + DD + HHMMSS (分隔)
    [("year", 4), ("month", 2), ("day", 2), ("hour", 2), ("minute", 2)],  # YYYYMMDDHHMI (12)
    # 精确到天
    [("year", 4), ("month", 2), ("day", 2)],       # YYYYMMDD (8) 或分隔
    [("year", 4), ("mmdd", 4)],                    # YYYY + MMDD (8, 分两段)
    # 精确到月
    [("year", 4), ("month", 2)],                   # YYYY + MM
    [("yearmonth", 6)],                            # YYYYMM (6)
    # 精确到年
    [("year", 4)],                                 # YYYY
]

# 复合字段的解包规则
_COMPOSITE_FIELDS = {
    "mmdd":   [("month", 2), ("day", 2)],
    "hhmmss": [("hour", 2), ("minute", 2), ("second", 2)],
    "yearmonth": [("year", 4), ("month", 2)],
}

# 方案对应的精度
_SCHEME_PRECISION = {
    14: "second", 12: "second",
    8:  "day",
    6:  "month",
    4:  "year",
}


@dataclass
class _ParsedScheme:
    """一次成功的假设解析结果。"""
    fields: dict[str, int]       # {"year": 2019, "month": 7, ...}
    precision: str
    scheme: list[tuple[str, int]]
    confidence_bonus: float = 0.0  # 某些结构更可靠，给额外加分


def _try_parse_runs_with_scheme(
    runs: list[tuple[int, int]],   # [(pos, length), ...]
    sample: str,
    scheme: list[tuple[str, int]],
) -> Optional[_ParsedScheme]:
    """
    尝试用一种分段方案解析数字段序列。

    scheme 里每个 (field, bit_len) 对应一段数字，
    bit_len 是该字段应该消耗的位数，多个 field 可以映射到同一个 run
    （如 mmdd=4 由 month=2 + day=2 组成）。

    返回 None 表示：
        - 数字段总长度与方案不符
        - 某个字段的实际数值超出合法范围
    """
    # 1. 把 runs 展平为连续数字字符串（忽略分隔符位置，按位数分配）
    # 注意：runs 的 pos 是相对于 template 的，sample 的字符 1:1 对应 template
    digit_str = "".join(sample[pos:pos + length] for pos, length in runs)

    # 2. 计算方案需要的总位数（展开复合字段）
    scheme_fields = []
    for fname, flen in scheme:
        if fname in _COMPOSITE_FIELDS:
            scheme_fields.extend(_COMPOSITE_FIELDS[fname])
        else:
            scheme_fields.append((fname, flen))

    scheme_total = sum(flen for _, flen in scheme_fields)
    if scheme_total != len(digit_str):
        return None

    # 3. 逐字段切割并验证
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

    # 4. 基本语义检查（月+日的组合合法性）
    if "month" in fields and "day" in fields:
        m, d = fields["month"], fields["day"]
        import calendar
        year = fields.get("year", 2000)
        max_day = calendar.monthrange(year, m)[1]
        if d > max_day:
            return None

    # 5. 推断精度
    has = set(fields)
    if "second" in has:
        precision = "second"
    elif "minute" in has:
        precision = "minute"
    elif "hour" in has:
        precision = "hour"
    elif "day" in has:
        precision = "day"
    elif "month" in has:
        precision = "month"
    else:
        precision = "year"

    # 6. 置信度加成：分隔符明确的结构比连续数字更可靠
    bonus = 0.05 if len(runs) >= 3 and len(scheme) >= 3 else 0.0

    return _ParsedScheme(fields=fields, precision=precision, scheme=scheme_fields, confidence_bonus=bonus)


def _try_all_schemes(
    runs: list[tuple[int, int]],
    sample: str,
) -> list[_ParsedScheme]:
    """
    对同一组数字段，枚举所有方案，返回所有合法解析结果（多假设）。
    按精度降序排列（精度高的在前）。
    """
    _PRECISION_RANK = {"second": 6, "minute": 5, "hour": 4,
                       "day": 3, "month": 2, "year": 1}
    results = []
    for scheme in _SEGMENT_SCHEMES:
        parsed = _try_parse_runs_with_scheme(runs, sample, scheme)
        if parsed is not None:
            results.append(parsed)

    # 去重：相同 precision + fields 只保留一个
    seen = set()
    unique = []
    for r in results:
        key = (r.precision, tuple(sorted(r.fields.items())))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda r: _PRECISION_RANK.get(r.precision, 0), reverse=True)
    return unique


# ---------------------------------------------------------------------------
# 正则生成（结构驱动）
# ---------------------------------------------------------------------------

# 各字段的正则片段
_FIELD_REGEX = {
    "year":   r"(?P<year>(?:19|20)\d{2})",
    "month":  r"(?P<month>0[1-9]|1[0-2])",
    "day":    r"(?P<day>0[1-9]|[12]\d|3[01])",
    "hour":   r"(?P<hour>[01]\d|2[0-3])",
    "minute": r"(?P<minute>[0-5]\d)",
    "second": r"(?P<second>[0-5]\d)",
}


def _build_regex_from_scheme(
    template: str,
    sample: str,
    parsed: _ParsedScheme,
    runs: list[tuple[int, int]],
) -> Optional[re.Pattern]:
    """
    根据成功解析的方案，构建正则表达式。

    策略：
    - 非数字字符（字母、分隔符）：从 sample 取字面量并转义
    - 数字字符：按 parsed.scheme 的字段顺序填充对应正则片段
    - 字母段不做精确匹配（只要求相同类型），用 [A-Za-z]+ 匹配
      避免"样本是 Screenshot，其他文件可能是 screen_shot"这类问题
    """
    pattern_parts = []
    scheme_iter = iter(parsed.scheme)
    current_field, current_len = next(scheme_iter, (None, 0))
    field_consumed = 0  # 当前字段已消耗的位数

    i = 0
    while i < len(template):
        ch = template[i]

        if ch == "N":
            # 数字字符：从 scheme 中取对应字段的正则
            if current_field is None:
                pattern_parts.append(r"\d")
                i += 1
                continue

            # 第一个数字字符：插入整个字段的正则（之后的同字段字符跳过）
            if field_consumed == 0:
                frag = _FIELD_REGEX.get(current_field)
                if frag:
                    pattern_parts.append(frag)
                else:
                    # 未知字段（理论上不会发生）
                    pattern_parts.append(rf"\d{{{current_len}}}")

            field_consumed += 1
            if field_consumed >= current_len:
                # 当前字段消耗完毕，移到下一个
                current_field, current_len = next(scheme_iter, (None, 0))
                field_consumed = 0
            i += 1

        elif ch in "Aa":
            # 字母段：用 [A-Za-z]+ 匹配（不绑定具体字母内容）
            j = i
            while j < len(template) and template[j] in "Aa":
                j += 1
            pattern_parts.append(r"[A-Za-z]+")
            i = j

        else:
            # 分隔符：字面量转义
            pattern_parts.append(re.escape(ch))
            i += 1

    full_pattern = r"(?<![0-9A-Za-z])" + "".join(pattern_parts) + r"(?![0-9A-Za-z])"
    try:
        return re.compile(full_pattern)
    except re.error:
        return None


# ---------------------------------------------------------------------------
# 模板筛选
# ---------------------------------------------------------------------------

_MIN_TEMPLATE_LEN = 6
_MIN_TOTAL_DIGITS = 4  # 至少包含年份


def _template_has_date_potential(template: str) -> bool:
    """模板是否值得尝试解析（快速预过滤，避免对无效模板做昂贵的多假设枚举）。"""
    runs = _find_digit_runs(template)
    if not runs:
        return False
    total_digits = sum(length for _, length in runs)
    # 至少有 4 位数字，且至少有一段 >= 4 位（年份）
    return total_digits >= _MIN_TOTAL_DIGITS and any(l >= 4 for _, l in runs)


# ---------------------------------------------------------------------------
# PatternMiner（接口不变）
# ---------------------------------------------------------------------------

class PatternMiner:
    """
    从 Item 列表中挖掘文件名 / 文件夹名时间模式。

    接口与旧版完全兼容：
        mine(items, ctx) → list[DatePattern]
    """

    def __init__(
        self,
        min_count: int = 5,
        max_patterns: int = 20,
    ) -> None:
        self.min_count   = min_count
        self.max_patterns = max_patterns

    def mine(
        self,
        items: list["Item"],
        ctx: "Context" = None,  # type: ignore[assignment]
    ) -> list[DatePattern]:
        """
        主入口。分析 items 的文件名和文件夹名，返回动态 DatePattern 列表。

        改进：
        - 对每个高频模板枚举所有合理的分段假设（多假设解析）
        - 每个合法假设各自生成一条 DatePattern，由 FilenameStage 统一竞争
        - 已被预置正则覆盖的样本跳过
        """
        # 收集所有待分析的 stem
        stems: list[str] = []
        for item in items:
            p = Path(item.path)
            stems.append(p.stem)
            for part in p.parent.parts:
                if part not in ("", "/", "\\"):
                    stems.append(part)

        if ctx:
            ctx.logger.debug(f"[PatternMiner] analyzing {len(stems)} names")

        # 模板统计
        template_counter: Counter[str] = Counter()
        template_samples: dict[str, str] = {}

        for stem in stems:
            tmpl = _abstractify(stem)
            template_counter[tmpl] += 1
            if tmpl not in template_samples:
                template_samples[tmpl] = stem

        # 挖掘
        discovered: list[DatePattern] = []
        seen_sigs: set[tuple] = set()  # (pattern_name, precision) 去重

        _CONFIDENCE_BASE = {
            "second": 0.85, "minute": 0.80, "hour": 0.75,
            "day":    0.70, "month":  0.45, "year": 0.25,
        }

        for tmpl, count in template_counter.most_common():
            if count < self.min_count:
                break
            if len(tmpl) < _MIN_TEMPLATE_LEN:
                continue
            if not _template_has_date_potential(tmpl):
                continue
            if len(discovered) >= self.max_patterns:
                break

            sample = template_samples[tmpl]

            # 已被预置正则覆盖 → 跳过
            if self._already_covered(sample):
                if ctx:
                    ctx.logger.debug(
                        f"[PatternMiner] skip {tmpl[:20]!r}: already covered by builtin"
                    )
                continue

            runs = _find_digit_runs(tmpl)
            if not runs:
                continue

            # 多假设解析
            candidates = _try_all_schemes(runs, sample)
            if not candidates:
                if ctx:
                    ctx.logger.debug(
                        f"[PatternMiner] no valid scheme for {tmpl[:20]!r} "
                        f"(sample={sample!r})"
                    )
                continue

            for parsed in candidates:
                if len(discovered) >= self.max_patterns:
                    break

                # 去重签名：同一模板 + 同一精度只保留一条
                sig = (tmpl[:20], parsed.precision)
                if sig in seen_sigs:
                    continue

                # 生成正则
                compiled = _build_regex_from_scheme(tmpl, sample, parsed, runs)
                if compiled is None:
                    continue

                # 验证：用样本确认能解析出合法 datetime
                m = compiled.search(sample)
                if m is None:
                    continue
                dt = _build_datetime(m.groupdict())
                if dt is None:
                    continue

                # 确认 dt 和 parsed.fields 一致（双重验证）
                if dt.year != parsed.fields.get("year", dt.year):
                    continue

                pattern_name = f"MINED_{tmpl[:20]}_{parsed.precision}"
                confidence = (
                    _CONFIDENCE_BASE.get(parsed.precision, 0.5)
                    + parsed.confidence_bonus
                )

                pat = DatePattern(
                    name=pattern_name,
                    regex=compiled,
                    precision=parsed.precision,
                    confidence=confidence,
                    examples=[sample],
                )
                discovered.append(pat)
                seen_sigs.add(sig)

                if ctx:
                    fields_str = ", ".join(
                        f"{k}={v}" for k, v in sorted(parsed.fields.items())
                    )
                    ctx.logger.info(
                        f"[PatternMiner] discovered {pattern_name!r} "
                        f"(count={count}, fields={{{fields_str}}}, "
                        f"sample={sample!r})"
                    )

        return discovered

    @staticmethod
    def _already_covered(text: str) -> bool:
        from utils.patterns import match_best
        return match_best(text) is not None