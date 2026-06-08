"""
stages/timeline/struct_resolver.py — 结构时间 Token 消歧

职责
----
利用 Session 的时间上下文（year_candidates, time_start/end）
对 StructToken 的候选集做逐步约束，得出最可信的 datetime 解释，
并产出 Evidence 追加到 session 内符合条件的 item。

消歧四步
--------
    Step 1  年份过滤：非占位年的候选，年份不在 year_candidates 则淘汰
    Step 2  月份约束：time_start/end 提供月份范围，淘汰超出范围的候选
    Step 3  占位年补全：year==2000 的候选（来自 MMDD 解释）补全真实年份
    Step 4  共识兜底：仍有多个候选时取 year/month 共识，降精度输出

置信度构成
----------
    base                   0.50（精确消歧）/ 0.30（共识兜底）
    +year_filter_bonus     0.10（年份过滤有效）
    +month_filter_bonus    0.10（月份约束有效）
    +anchor_bonus          min(0.05 × anchor_count, 0.20)
    -depth_penalty         0.10 × source_folder（文件夹层级越远越不可信）

Evidence 追加条件
-----------------
    只追加给 session 内 time_result 未解析或 confidence < 阈值 的 item。
    已有高置信度时间的 item 不追加（避免干扰已确定的结论）。

与 StructToken 的关系
---------------------
    消歧成功后，StructToken.resolved_* 字段会被填入（in-place）。
    这样调用方（TimelineStage / dev 脚本）可以读取 token 的消歧结果，
    便于调试和日志。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.item import Item
    from session import Session
    from tokens import StructToken

# 占位年（Tokenizer 中 MMDD 解释使用）
_PLACEHOLDER_YEAR = 2000

# 月份约束的容差（±N 个月）
_MONTH_TOLERANCE = 1

# 消歧成功 vs 共识兜底的基础置信度
_BASE_CONF_RESOLVED = 0.50
_BASE_CONF_CONSENSUS = 0.30

# 置信度各项加成的上下限
_YEAR_FILTER_BONUS  = 0.10
_MONTH_FILTER_BONUS = 0.10
_ANCHOR_BONUS_PER   = 0.05
_ANCHOR_BONUS_MAX   = 0.20
_DEPTH_PENALTY_PER  = 0.10   # 每深一层文件夹扣分

# 只向 confidence 低于此值的 item 追加 evidence
_TARGET_MAX_CONFIDENCE = 0.6


# ---------------------------------------------------------------------------
# 消歧结果
# ---------------------------------------------------------------------------

class _DisambigResult:
    """单个 StructToken 的消歧中间结果，供内部传递。"""

    __slots__ = (
        "dt", "precision", "confidence", "reason",
        "used_year_filter", "used_month_filter", "is_consensus",
    )

    def __init__(
        self,
        dt: datetime,
        precision: str,
        confidence: float,
        reason: str,
        used_year_filter: bool = False,
        used_month_filter: bool = False,
        is_consensus: bool = False,
    ):
        self.dt = dt
        self.precision = precision
        self.confidence = confidence
        self.reason = reason
        self.used_year_filter = used_year_filter
        self.used_month_filter = used_month_filter
        self.is_consensus = is_consensus


# ---------------------------------------------------------------------------
# StructResolver
# ---------------------------------------------------------------------------

class StructResolver:
    """
    利用 Session 时间上下文对 StructToken 进行消歧，产出 Evidence。

    使用示例
    --------
    resolver = StructResolver()
    resolver.resolve(session, min_confidence=0.6, verbose=True)
    """

    def __init__(self, min_evidence_confidence: float = 0.25) -> None:
        """
        参数
        ----
        min_evidence_confidence  产出 Evidence 的最低置信度门槛
                                 低于此值的消歧结果不追加（太不可信）
        """
        self.min_evidence_confidence = min_evidence_confidence

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def resolve(
        self,
        session: "Session",
        min_confidence: float = _TARGET_MAX_CONFIDENCE,
        verbose: bool = False,
    ) -> int:
        """
        对 session.struct_tokens 里的每个 StructToken 进行消歧，
        并将结果以 Evidence 形式追加到 session 内符合条件的 item。

        参数
        ----
        session         含 struct_tokens、year_candidates、time_start/end
        min_confidence  只向低于此置信度的 item 追加 evidence
        verbose         True 时打印每个 token 的消歧过程（调试用）

        返回
        ----
        成功追加 evidence 的 item 数量
        """
        if not session.struct_tokens:
            return 0

        # 找出需要帮助的 item（时间未解析，或置信度低于阈值）
        target_items = [
            it for it in session.items
            if not it.time_result.is_resolved
            or it.time_result.confidence < min_confidence
        ]
        if not target_items:
            return 0

        anchor_count = len(session.anchor_items)
        improved_ids: set[str] = set()

        for token in session.struct_tokens:
            result = self._disambiguate(token, session, verbose)
            if result is None:
                continue

            # 填入 StructToken.resolved_*（in-place，供调试读取）
            token.resolved_dt = result.dt
            token.resolved_precision = result.precision
            token.resolved_confidence = result.confidence
            token.resolved_reason = result.reason

            if result.confidence < self.min_evidence_confidence:
                if verbose:
                    print(
                        f"  [StructResolver] {token.folder_text!r} → "
                        f"confidence={result.confidence:.2f} 低于门槛，跳过"
                    )
                continue

            # 追加 Evidence 到目标 item
            ev = self._make_evidence(result, token, session)
            for item in target_items:
                # 只追加给文件夹路径与 token 来源文件夹有关联的 item
                if self._item_in_scope(item, token, session):
                    item.add_evidence(ev)
                    improved_ids.add(item.id)

        return len(improved_ids)

    # ------------------------------------------------------------------
    # 消歧核心（四步）
    # ------------------------------------------------------------------

    def _disambiguate(
        self,
        token: "StructToken",
        session: "Session",
        verbose: bool,
    ) -> Optional[_DisambigResult]:
        """
        对单个 StructToken 执行消歧四步，返回结果或 None（无法消歧）。
        """
        candidates = list(token.candidates)   # 工作副本
        used_year_filter = False
        used_month_filter = False

        if verbose:
            cands_str = [d.strftime("%Y-%m-%d") for d in candidates]
            print(
                f"\n  [StructResolver] {token.folder_text!r} "
                f"depth={token.source_folder} "
                f"candidates={cands_str}"
            )
            print(
                f"    session: year_candidates={session.year_candidates}  "
                f"range={session.time_start} ~ {session.time_end}"
            )

        # ── Step 1: 年份过滤 ───────────────────────────────────────────
        if session.year_candidates:
            filtered = self._filter_by_year(candidates, session.year_candidates)
            if filtered and filtered != candidates:
                used_year_filter = True
                candidates = filtered
                if verbose:
                    print(
                        f"    Step1 年份过滤 → "
                        f"{[d.strftime('%Y-%m-%d') for d in candidates]}"
                    )

        # ── Step 2: 月份约束（Step 1 后仍有多个候选）──────────────────
        if len(candidates) > 1 and (session.time_start or session.time_end):
            filtered = self._filter_by_month(
                candidates, session.time_start, session.time_end
            )
            if filtered and filtered != candidates:
                used_month_filter = True
                candidates = filtered
                if verbose:
                    print(
                        f"    Step2 月份约束 → "
                        f"{[d.strftime('%Y-%m-%d') for d in candidates]}"
                    )

        # ── Step 3: 占位年补全 ─────────────────────────────────────────
        candidates = self._fill_placeholder_years(
            candidates, session.year_candidates
        )
        if verbose and any(d.year == _PLACEHOLDER_YEAR for d in token.candidates):
            print(
                f"    Step3 占位年补全 → "
                f"{[d.strftime('%Y-%m-%d') for d in candidates]}"
            )

        # ── Step 4: 结果判定 ───────────────────────────────────────────
        anchor_count = len(session.anchor_items)
        depth = token.source_folder

        if len(candidates) == 1:
            # 精确消歧
            dt = candidates[0]
            precision = self._infer_precision(dt, token)
            conf = self._calc_confidence(
                base=_BASE_CONF_RESOLVED,
                used_year_filter=used_year_filter,
                used_month_filter=used_month_filter,
                anchor_count=anchor_count,
                depth=depth,
            )
            reason = self._build_reason(
                token, dt, used_year_filter, used_month_filter, session
            )
            if verbose:
                print(f"    → 精确消歧: {dt.strftime('%Y-%m-%d')} "
                      f"precision={precision} conf={conf:.2f}")
            return _DisambigResult(
                dt=dt, precision=precision, confidence=conf, reason=reason,
                used_year_filter=used_year_filter,
                used_month_filter=used_month_filter,
                is_consensus=False,
            )

        elif len(candidates) > 1:
            # 共识兜底
            consensus = self._find_consensus(candidates)
            if consensus is None:
                if verbose:
                    print("    → 无共识，跳过")
                return None
            dt, precision = consensus
            conf = self._calc_confidence(
                base=_BASE_CONF_CONSENSUS,
                used_year_filter=used_year_filter,
                used_month_filter=used_month_filter,
                anchor_count=anchor_count,
                depth=depth,
            )
            reason = (
                f"consensus from {len(candidates)} candidates "
                f"(precision={precision}): "
                f"{[d.strftime('%Y-%m-%d') for d in candidates]}"
            )
            if verbose:
                print(f"    → 共识: {dt.strftime('%Y-%m')} "
                      f"precision={precision} conf={conf:.2f}")
            return _DisambigResult(
                dt=dt, precision=precision, confidence=conf, reason=reason,
                used_year_filter=used_year_filter,
                used_month_filter=used_month_filter,
                is_consensus=True,
            )

        else:
            # 过滤后无候选（所有候选都被淘汰）
            if verbose:
                print("    → 所有候选被淘汰，跳过")
            return None

    # ------------------------------------------------------------------
    # 消歧四步的具体实现
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_year(
        candidates: list[datetime],
        year_candidates: list[int],
    ) -> list[datetime]:
        """
        Step 1：用 session 年份过滤候选。

        规则：
            - year == _PLACEHOLDER_YEAR（2000）→ 保留（等待 Step 3 补全）
            - year 在 year_candidates 中 → 保留
            - 其他 → 淘汰

        若过滤后列表为空，返回原列表（防止过度过滤）。
        """
        year_set = set(year_candidates)
        filtered = [
            d for d in candidates
            if d.year == _PLACEHOLDER_YEAR or d.year in year_set
        ]
        return filtered if filtered else candidates

    @staticmethod
    def _filter_by_month(
        candidates: list[datetime],
        time_start: Optional[datetime],
        time_end: Optional[datetime],
    ) -> list[datetime]:
        """
        Step 2：用 session 时间范围的月份过滤候选。

        取 time_start.month 和 time_end.month（若都有）的并集，
        再扩展 ±_MONTH_TOLERANCE 个月作为容差范围。

        注意：month 范围可能跨年（如 12 月 ±1 → 包含 1 月）。
        用集合存合法月份，统一处理跨年问题。
        """
        ref_months: set[int] = set()
        for ref in (time_start, time_end):
            if ref:
                for delta in range(-_MONTH_TOLERANCE, _MONTH_TOLERANCE + 1):
                    m = (ref.month - 1 + delta) % 12 + 1
                    ref_months.add(m)

        if not ref_months:
            return candidates

        # 占位年的候选不按月过滤（年份不确定时月份过滤意义不大）
        filtered = [
            d for d in candidates
            if d.year == _PLACEHOLDER_YEAR or d.month in ref_months
        ]
        return filtered if filtered else candidates

    @staticmethod
    def _fill_placeholder_years(
        candidates: list[datetime],
        year_candidates: list[int],
    ) -> list[datetime]:
        """
        Step 3：把 year==2000 的占位年替换为真实年份。

        若 year_candidates 为空，保持占位年不变（无法补全）。
        若补全后与已有候选重复，合并去重。
        """
        if not year_candidates:
            return candidates

        real_year = year_candidates[0]   # 取最早/最可能的年份
        result: list[datetime] = []
        seen_iso: set[str] = set()

        for d in candidates:
            if d.year == _PLACEHOLDER_YEAR:
                try:
                    filled = d.replace(year=real_year)
                    key = filled.isoformat()
                    if key not in seen_iso:
                        seen_iso.add(key)
                        result.append(filled)
                except ValueError:
                    # 极端情况：闰年 2 月 29 日补全到非闰年
                    result.append(d)   # 保留占位年，不补全
            else:
                key = d.isoformat()
                if key not in seen_iso:
                    seen_iso.add(key)
                    result.append(d)

        return result

    @staticmethod
    def _find_consensus(
        candidates: list[datetime],
    ) -> Optional[tuple[datetime, str]]:
        """
        Step 4：在多个候选中找共识部分。

        共识规则（从精到粗）：
            所有候选 year + month + day 相同 → precision="day"（极少见）
            所有候选 year + month 相同       → precision="month"，dt 取月初
            所有候选 year 相同              → precision="year"，dt 取年初
            无共识                          → 返回 None

        返回 (consensus_dt, precision) 或 None。
        """
        if not candidates:
            return None

        years  = {d.year  for d in candidates}
        months = {d.month for d in candidates}
        days   = {d.day   for d in candidates}

        if len(years) == 1 and len(months) == 1 and len(days) == 1:
            return candidates[0], "day"

        if len(years) == 1 and len(months) == 1:
            return candidates[0].replace(day=1, hour=0, minute=0, second=0), "month"

        if len(years) == 1:
            year = next(iter(years))
            return datetime(year, 1, 1), "year"

        return None

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_precision(dt: datetime, token: "StructToken") -> str:
        """
        从消歧结果和 token 结构推断精度。

        占位年补全后：月日已知 → "day"
        候选集含占位年（MMDD 解释）：消歧结果月日也确定 → "day"
        普通情况按数字总位数判断（与 Tokenizer 一致）。
        """
        # 候选集中有占位年 → 说明包含了 MMDD 解释，月日已知
        if any(c.year == _PLACEHOLDER_YEAR for c in token.candidates):
            return "day"
        # 按总位数判断
        total = sum(token.digit_lengths)
        if total >= 6:  return "day"    # YYMMDD 或更长
        if total == 4:
            # 4位：YYMM 精确到月，MMDD 精确到日
            # 若消歧后 dt.day > 1，说明原始候选有 day 信息
            if dt.day > 1:
                return "day"
            return "month"
        return "year"

    @staticmethod
    def _calc_confidence(
        base: float,
        used_year_filter: bool,
        used_month_filter: bool,
        anchor_count: int,
        depth: int,
    ) -> float:
        """计算最终置信度，结果 clamp 到 [0.1, 0.85]。"""
        conf = base
        if used_year_filter:
            conf += _YEAR_FILTER_BONUS
        if used_month_filter:
            conf += _MONTH_FILTER_BONUS
        conf += min(anchor_count * _ANCHOR_BONUS_PER, _ANCHOR_BONUS_MAX)
        conf -= depth * _DEPTH_PENALTY_PER
        return round(max(0.10, min(0.85, conf)), 3)

    @staticmethod
    def _build_reason(
        token: "StructToken",
        dt: datetime,
        used_year_filter: bool,
        used_month_filter: bool,
        session: "Session",
    ) -> str:
        steps = []
        if used_year_filter:
            steps.append(f"year∈{session.year_candidates}")
        if used_month_filter:
            ref = session.time_start or session.time_end
            steps.append(f"month≈{ref.month}" if ref else "month-filter")
        constraint = " + ".join(steps) if steps else "single-candidate"
        return (
            f"StructToken {token.folder_text!r} → {dt.strftime('%Y-%m-%d')} "
            f"({constraint})"
        )

    @staticmethod
    def _item_in_scope(
        item: "Item",
        token: "StructToken",
        session: "Session",
    ) -> bool:
        """
        判断 item 是否在 token 的适用范围内。

        StructToken 来自某一层文件夹，只对位于该文件夹（或其子目录）
        下的 item 有效，不向同 session 内不相关文件夹的 item 追加。

        例：token.folder_text="0213" 来自 "旅行/0213/" 这一层
            item.relpath="旅行/0213/IMG_001.jpg" → 在范围内
            item.relpath="旅行/重庆/IMG_002.jpg" → 不在范围内
        """
        from pathlib import Path
        item_parts = Path(item.relpath).parts
        # token.folder_text 是文件夹名（不含路径），检查 item 路径中是否包含该层
        return token.folder_text in item_parts

    @staticmethod
    def _make_evidence(
        result: _DisambigResult,
        token: "StructToken",
        session: "Session",
    ):
        """构造 Evidence，追加到 item.evidence（与 Stage.make_evidence 一致）。"""
        from core.evidence import Evidence
        return Evidence(
            source="timeline",
            stage="TimelineStage",
            dt=result.dt,
            precision=result.precision,
            confidence=result.confidence,
            is_direct=False,
            reason=result.reason,
            metadata={
                "token_folder_text":  token.folder_text,
                "token_digit_runs":   token.digit_runs,
                "token_candidates":   [d.isoformat() for d in token.candidates],
                "session_id":         session.id,
                "year_candidates":    session.year_candidates,
                "used_year_filter":   result.used_year_filter,
                "used_month_filter":  result.used_month_filter,
                "is_consensus":       result.is_consensus,
            },
        )


# ---------------------------------------------------------------------------
# 独立测试（__main__）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datetime import datetime, timedelta

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from core.item import Item, TimeResult
    from core.evidence import Evidence
    from tokens import StructToken
    from session import Session
    from struct_resolver import StructResolver

    def _anchor_item(relpath: str, dt: datetime, conf: float = 1.0) -> Item:
        item = Item(path=f"/fake/{relpath}", relpath=relpath)
        item.filename = Path(relpath).name
        item.mtime = dt.timestamp()
        item.time_result = TimeResult(
            final_datetime=dt, confidence=conf,
            precision="second", primary_source="exif"
        )
        return item

    def _float_item(relpath: str) -> Item:
        item = Item(path=f"/fake/{relpath}", relpath=relpath)
        item.filename = Path(relpath).name
        return item

    resolver = StructResolver()

    print("=" * 60)
    print("StructResolver 消歧测试（verbose=True）")
    print("=" * 60)

    # ── 测试 1：年份过滤精确消歧 ─────────────────────────────────────
    print("\n[测试1] 年份过滤：candidates=[2013-02-01, 2019-02-13], year=[2019]")
    base = datetime(2019, 2, 13)
    session1 = Session(id="s0")
    session1.anchor_items = [_anchor_item("旅行/0213/A.jpg", base)]
    session1.items = [
        _anchor_item("旅行/0213/A.jpg", base),
        _float_item("旅行/0213/B.jpg"),
    ]
    session1.year_candidates = [2019]
    session1.time_start = base
    session1.time_end   = base + timedelta(hours=2)
    session1.struct_tokens = [StructToken(
        folder_text="0213",
        digit_runs=["0213"],
        digit_lengths=(4,),
        candidates=[datetime(2013, 2, 1), datetime(2019, 2, 13)],
        source_folder=0,
    )]
    n = resolver.resolve(session1, verbose=True)
    tok = session1.struct_tokens[0]
    assert tok.is_resolved, "期望消歧成功"
    assert tok.resolved_dt == datetime(2019, 2, 13), f"期望 2019-02-13，实际 {tok.resolved_dt}"
    float_item = session1.items[1]
    tl_evs = [e for e in float_item.evidence if e.source == "timeline"]
    assert len(tl_evs) == 1, f"期望 1 条 timeline evidence，实际 {len(tl_evs)}"
    print(f"  ✓ resolved={tok.resolved_dt.strftime('%Y-%m-%d')}, "
          f"conf={tok.resolved_confidence:.2f}, "
          f"float_item evidence={len(tl_evs)}")

    # ── 测试 2：占位年补全（MMDD → 年份未知，补全） ───────────────────
    print("\n[测试2] 占位年补全：candidates=[2000-09-01, 2009-01-01], year=[2019]")
    base2 = datetime(2019, 9, 5)
    session2 = Session(id="s1")
    session2.anchor_items = [_anchor_item("日记/0901/A.jpg", base2)]
    session2.items = [
        _anchor_item("日记/0901/A.jpg", base2),
        _float_item("日记/0901/B.jpg"),
    ]
    session2.year_candidates = [2019]
    session2.time_start = base2
    session2.time_end   = base2 + timedelta(hours=3)
    session2.struct_tokens = [StructToken(
        folder_text="0901",
        digit_runs=["0901"],
        digit_lengths=(4,),
        candidates=[datetime(2000, 9, 1), datetime(2009, 1, 1)],
        source_folder=0,
    )]
    n2 = resolver.resolve(session2, verbose=True)
    tok2 = session2.struct_tokens[0]
    assert tok2.is_resolved
    assert tok2.resolved_dt == datetime(2019, 9, 1), f"期望 2019-09-01，实际 {tok2.resolved_dt}"
    print(f"  ✓ resolved={tok2.resolved_dt.strftime('%Y-%m-%d')}, "
          f"conf={tok2.resolved_confidence:.2f}, precision={tok2.resolved_precision}")

    # ── 测试 3：月份约束 ──────────────────────────────────────────────
    print("\n[测试3] 月份约束：candidates=[2019-02-01, 2019-09-01], session 在 2月")
    base3 = datetime(2019, 2, 10)
    session3 = Session(id="s2")
    session3.anchor_items = [_anchor_item("相册/年初/A.jpg", base3)]
    session3.items = [
        _anchor_item("相册/年初/A.jpg", base3),
        _float_item("相册/年初/B.jpg"),
    ]
    session3.year_candidates = [2019]
    session3.time_start = base3
    session3.time_end   = base3 + timedelta(days=5)
    session3.struct_tokens = [StructToken(
        folder_text="年初",   # 文件夹名是中文，数字 token 来自子文件夹
        digit_runs=[],
        digit_lengths=(),
        candidates=[datetime(2019, 2, 1), datetime(2019, 9, 1)],
        source_folder=0,
    )]
    # 手动触发消歧（无数字段的 token 仍可消歧）
    tok3 = session3.struct_tokens[0]
    result3 = resolver._disambiguate(tok3, session3, verbose=True)
    assert result3 is not None
    assert result3.dt.month == 2, f"期望 2 月，实际 {result3.dt.month}"
    print(f"  ✓ month={result3.dt.month}, conf={result3.confidence:.2f}")

    # ── 测试 4：共识兜底 ──────────────────────────────────────────────
    print("\n[测试4] 共识兜底：candidates=[2019-02-13, 2019-09-01]，无月份约束")
    session4 = Session(id="s3")
    session4.year_candidates = [2019]
    # 无 time_start/end → 月份约束无效
    session4.struct_tokens = [StructToken(
        folder_text="test",
        digit_runs=[],
        digit_lengths=(),
        candidates=[datetime(2019, 2, 13), datetime(2019, 9, 1)],
        source_folder=0,
    )]
    tok4 = session4.struct_tokens[0]
    result4 = resolver._disambiguate(tok4, session4, verbose=True)
    assert result4 is not None
    assert result4.is_consensus
    assert result4.precision == "year", f"期望 year，实际 {result4.precision}"
    assert result4.dt.year == 2019
    print(f"  ✓ consensus precision={result4.precision}, "
          f"dt={result4.dt.strftime('%Y')}, conf={result4.confidence:.2f}")

    # ── 测试 5：无锚点 session（year_candidates 为空）────────────────
    print("\n[测试5] 无锚点 session，year_candidates=[]，仅靠候选自身判断")
    session5 = Session(id="s4")
    session5.year_candidates = []
    session5.struct_tokens = [StructToken(
        folder_text="0213",
        digit_runs=["0213"],
        digit_lengths=(4,),
        candidates=[datetime(2000, 2, 13), datetime(2002, 1, 3)],
        source_folder=0,
    )]
    tok5 = session5.struct_tokens[0]
    result5 = resolver._disambiguate(tok5, session5, verbose=True)
    # 无锚点时：占位年无法补全，但 2002-01-03 是合法年份
    print(f"  结果: {'消歧成功' if result5 and not result5.is_consensus else '共识/失败'}")
    if result5:
        print(f"  dt={result5.dt}, conf={result5.confidence:.2f}")

    # ── 测试 6：depth 越大，置信度越低 ───────────────────────────────
    print("\n[测试6] depth 惩罚：depth=0 vs depth=2 的置信度差异")
    base6 = datetime(2019, 5, 1)
    for depth in [0, 1, 2]:
        tok_d = StructToken(
            folder_text="0501",
            digit_runs=["0501"],
            digit_lengths=(4,),
            candidates=[datetime(2000, 5, 1), datetime(2005, 1, 1)],
            source_folder=depth,
        )
        s_d = Session(id=f"sd{depth}")
        s_d.year_candidates = [2019]
        s_d.time_start = base6
        s_d.struct_tokens = [tok_d]
        r = resolver._disambiguate(tok_d, s_d, verbose=False)
        if r:
            print(f"  depth={depth}: conf={r.confidence:.2f}")

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)

    # ── 模式 B：读真实 snapshot ─────────────────────────────────────
    if len(sys.argv) > 1:
        snap_path = Path(sys.argv[1])
        print(f"\n模式 B：真实 snapshot — {snap_path}")
        print("=" * 60)

        from storage.snapshot import load
        from session import SessionBuilder
        from tokenizer import Tokenizer

        items = load(snap_path)
        print(f"加载 {len(items)} 个 item")

        # 建 session
        sessions = SessionBuilder().build(items)
        print(f"聚类为 {len(sessions)} 个 session")

        # 提取 struct token，填入各 session
        tokenizer = Tokenizer()
        for item in items:
            _, structs, _ = tokenizer.extract(item)
            for tok in structs:
                for s in sessions:
                    if item in s.items:
                        s.struct_tokens.append(tok)
                        break

        # 消歧
        total_tokens = sum(len(s.struct_tokens) for s in sessions)
        print(f"StructToken 总数: {total_tokens}")

        total_improved = 0
        for s in sessions:
            if not s.struct_tokens:
                continue
            n = resolver.resolve(s, verbose=True)
            total_improved += n

        print(f"\n成功追加 evidence 的 item 数: {total_improved}")