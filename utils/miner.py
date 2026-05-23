"""
utils/miner.py — 文件名模式挖掘器

职责：
    在 ScanStage 之后、FilenameStage 之前运行一次，
    从大量文件名 / 文件夹名中自动归纳出未被预置正则覆盖的命名规范，
    以动态 DatePattern 的形式返回给 FilenameStage 使用。

算法：字符类型序列对齐（Character-Type Abstraction）

    原理：
        把每个文件名映射成字符类型序列（模板），然后统计模板频次。
        高频模板 + 含有足够长数字段 = 有可能是日期格式。

        "IMG_20200101_143022.jpg"   →  "AAA_NNNNNNNN_NNNNNN.EEE"
        "IMG_20211103_092211.jpg"   →  "AAA_NNNNNNNN_NNNNNN.EEE"
        "WeiXin_20190821.jpg"       →  "AAAAAA_NNNNNNNN.EEE"

    步骤：
        1. 对所有文件名做模板化（abstractify）
        2. 统计模板频次，过滤低频（< min_count）
        3. 对高频模板，找出数字段位置和长度
        4. 判断数字段是否符合日期格式（8/14位，或含分隔符的日期结构）
        5. 构造正则，验证样本，生成 DatePattern

局限：
    - 挖掘出的正则置信度会比预置正则略低（-0.05），因为可靠性未经人工验证
    - 对于极度混乱的文件名集合效果有限
    - 挖掘结果是运行时对象，不持久化（每次启动重新挖掘，速度很快）

运行时机（pipeline 中）：
    ScanStage → PatternMiner.mine() → FilenameStage（注入动态模式）
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from utils.patterns import DatePattern, MatchResult, _build_datetime, BUILTIN_PATTERNS

if TYPE_CHECKING:
    from core.item import Item


# ---------------------------------------------------------------------------
# 字符类型映射
# ---------------------------------------------------------------------------

def _abstractify(name: str) -> str:
    """
    将文件名（不含扩展名）映射为字符类型序列。

    映射规则：
        大写字母    → A
        小写字母    → a  （和大写分开，保留大小写信息用于模板匹配）
        数字        → N
        其他可见字符 → 原字符（保留分隔符，是模板的关键结构）

    合并连续同类：
        "IMG"  → "AAA"（不合并，保留长度信息）

    示例：
        "IMG_20200101_143022" → "AAA_NNNNNNNN_NNNNNN"
        "WeiXin_20190821"     → "AaaaaA_NNNNNNNN"
        "photo-2020-01-01"    → "aaaaa-NNNN-NN-NN"
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
            result.append(ch)  # 保留分隔符原样
    return "".join(result)


# ---------------------------------------------------------------------------
# 模板分析
# ---------------------------------------------------------------------------

# 可能包含日期的数字段长度（连续 N 的个数）
_DATE_DIGIT_LENGTHS = {4, 6, 8, 12, 14}

# 最短有意义的模板（过短的模板太泛化）
_MIN_TEMPLATE_LEN = 6


def _find_digit_runs(template: str) -> list[tuple[int, int]]:
    """
    找出模板中所有连续 N 的段落。
    返回 [(start, length), ...] 列表。

    示例：
        "AAA_NNNNNNNN_NNNNNN" → [(4, 8), (13, 6)]
    """
    runs = []
    i = 0
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


def _template_has_date_potential(template: str) -> bool:
    """
    判断模板是否有可能包含日期。
    条件：存在长度为 4、8、14 的数字段，或存在多个合适长度的数字段（分隔日期）。
    """
    runs = _find_digit_runs(template)
    lengths = [length for _, length in runs]

    # 单字段：4位年、8位日期、14位完整时间戳
    if any(l in {4, 8, 14} for l in lengths):
        return True

    # 多字段分隔格式：如 NNNN-NN-NN（年-月-日）
    if len(lengths) >= 2 and lengths[0] == 4 and lengths[1] == 2:
        return True

    return False


# ---------------------------------------------------------------------------
# 正则生成
# ---------------------------------------------------------------------------

_DIGIT_REGEX_MAP: dict[int, str] = {
    4:  r"(?P<year>(?:19|20)\d{2})",
    2:  r"(?P<_slot_>\d{2})",   # _slot_ 会被替换为 month/day/hour/minute/second
    8:  r"(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])",
    6:  r"(?P<hour>[01]\d|2[0-3])(?P<minute>[0-5]\d)(?P<second>[0-5]\d)",
    14: r"(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"
        r"(?P<hour>[01]\d|2[0-3])(?P<minute>[0-5]\d)(?P<second>[0-5]\d)",
}

_SLOT_NAMES = ["month", "day", "hour", "minute", "second"]


def _build_regex_from_template(
    template: str,
    sample_name: str,
) -> Optional[tuple[re.Pattern, str]]:
    """
    根据模板和一个样本文件名，尝试构建正则。
    返回 (compiled_pattern, precision) 或 None（无法构建时）。

    策略：
    - 把模板中非数字字符转义后直接用于正则
    - 数字段按长度匹配 _DIGIT_REGEX_MAP
    - 逐个数字段分配命名捕获组（month/day/hour/...）
    """
    runs = _find_digit_runs(template)
    if not runs:
        return None

    # 分析数字段结构，决定解析策略
    lengths = [length for _, length in runs]
    total_digits = sum(lengths)

    # 只支持几种已知结构，其余跳过（避免生成意义不明的正则）
    if total_digits not in {4, 6, 8, 10, 12, 14}:
        # 尝试看是否是 NNNN + NN + NN 结构（年月日分隔）
        if not (len(lengths) >= 3 and lengths[0] == 4 and lengths[1] == 2 and lengths[2] == 2):
            return None

    # 逐字符构建正则
    pattern_parts = []
    slot_idx = 0
    i = 0

    while i < len(template):
        ch = template[i]

        if ch == "N":
            # 找连续 N 的长度
            j = i
            while j < len(template) and template[j] == "N":
                j += 1
            run_len = j - i

            mapped = _DIGIT_REGEX_MAP.get(run_len)
            if mapped is None:
                # 未知长度，用通用数字段
                mapped = rf"\d{{{run_len}}}"
            elif "_slot_" in mapped:
                # 需要分配具体命名
                if slot_idx < len(_SLOT_NAMES):
                    mapped = mapped.replace("_slot_", _SLOT_NAMES[slot_idx])
                    slot_idx += 1
                else:
                    mapped = rf"\d{{{run_len}}}"

            pattern_parts.append(mapped)
            i = j

        elif ch in "AaB":
            # 字母段：匹配原始对应子串（用字面量，不用 \w+ 避免过于宽泛）
            j = i
            while j < len(template) and template[j] in "Aa":
                j += 1
            run_len = j - i
            # 从样本文件名取出对应的字母内容
            literal = sample_name[i:i + run_len]
            pattern_parts.append(re.escape(literal))
            i = j

        else:
            # 分隔符：直接转义
            pattern_parts.append(re.escape(ch))
            i += 1

    # 拼成完整正则，加前后边界
    full_pattern = r"(?<![0-9A-Za-z])" + "".join(pattern_parts) + r"(?![0-9A-Za-z])"

    try:
        compiled = re.compile(full_pattern)
    except re.error:
        return None

    # 推断精度
    all_groups = compiled.groupindex
    if "second" in all_groups:
        precision = "second"
    elif "minute" in all_groups:
        precision = "minute"
    elif "day" in all_groups:
        precision = "day"
    elif "month" in all_groups:
        precision = "month"
    elif "year" in all_groups:
        precision = "year"
    else:
        return None

    return compiled, precision


# ---------------------------------------------------------------------------
# PatternMiner
# ---------------------------------------------------------------------------

class PatternMiner:
    """
    从 Item 列表中挖掘文件名 / 文件夹名时间模式。

    使用示例（pipeline 中）
    ----------------------
    miner = PatternMiner()
    dynamic_patterns = miner.mine(items, ctx)
    # 把 dynamic_patterns 注入 FilenameStage
    filename_stage = FilenameStage(extra_patterns=dynamic_patterns)
    """

    def __init__(
        self,
        min_count: int = 5,
        max_patterns: int = 20,
    ) -> None:
        """
        参数
        ----
        min_count    : 模板至少出现多少次才尝试挖掘，过低会产生噪声。
        max_patterns : 最多返回多少条动态模式，防止结果爆炸。
        """
        self.min_count = min_count
        self.max_patterns = max_patterns

    def mine(
        self,
        items: list["Item"],
        ctx: "Context" = None,  # type: ignore[assignment]
    ) -> list[DatePattern]:
        """
        主入口。分析 items 的文件名和文件夹名，返回动态 DatePattern 列表。

        已被预置正则覆盖的模式会被过滤掉（避免重复）。
        """
        # 收集所有待模板化的字符串
        # 文件名去掉扩展名（扩展名不含日期信息，会干扰模板统计）
        # 文件夹名本身没有扩展名，直接使用
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
        template_samples: dict[str, str] = {}  # template → 代表性样本（用于正则生成）

        for stem in stems:
            tmpl = _abstractify(stem)
            template_counter[tmpl] += 1
            if tmpl not in template_samples:
                template_samples[tmpl] = stem

        # 过滤 + 挖掘
        discovered: list[DatePattern] = []
        seen_names: set[str] = set()

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
            result = _build_regex_from_template(tmpl, sample)
            if result is None:
                continue

            compiled, precision = result
            pattern_name = f"MINED_{tmpl[:20]}"

            if pattern_name in seen_names:
                continue

            # 验证：用样本跑一遍，确认能真正解析出 datetime
            m = compiled.search(sample)
            if m is None:
                continue
            dt = _build_datetime(m.groupdict())
            if dt is None:
                continue

            # 过滤掉已被预置正则覆盖的情况
            if self._already_covered(sample):
                if ctx:
                    ctx.logger.debug(
                        f"[PatternMiner] skip {pattern_name!r}: already covered by builtin"
                    )
                continue

            # 动态模式置信度比预置正则略低
            base_confidence = {"second": 0.85, "minute": 0.8, "day": 0.7,
                               "month": 0.45, "year": 0.25}.get(precision, 0.5)

            pat = DatePattern(
                name=pattern_name,
                regex=compiled,
                precision=precision,
                confidence=base_confidence,
                examples=[sample],
            )
            discovered.append(pat)
            seen_names.add(pattern_name)

            if ctx:
                ctx.logger.info(
                    f"[PatternMiner] discovered pattern {pattern_name!r} "
                    f"(count={count}, precision={precision}, sample={sample!r})"
                )

        return discovered

    @staticmethod
    def _already_covered(text: str) -> bool:
        """检查文本是否已被预置正则覆盖。"""
        from utils.patterns import match_best
        return match_best(text) is not None