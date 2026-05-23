"""
ResolverStage — 从多条 Evidence 中决策出最终时间

在所有证据收集 stage（Exif、Filename…）之后、WriterStage 之前运行。
唯一有权写入 item.time_result 的 stage。

决策流程
--------
1. 过滤掉 dt=None 的 evidence（只有范围没有时间点的，暂不参与决策）
2. 直接证据（is_direct=True）优先于间接证据
3. 在候选集内按 confidence 降序排列，取最高者为主证据
4. 冲突检测：用主证据与其他高可信度证据比较时间差
   - 超过 conflict_threshold_days → 打 CONFLICT_TIME flag，记录冲突详情
5. 计算置信区间：由所有候选证据的时间范围取并集
6. 写入 item.time_result

冲突后的行为
------------
发现冲突 **不等于** 放弃决策。
仍然写入最高可信度的结果，同时打 CONFLICT_TIME + NEEDS_REVIEW flag，
让用户/写回阶段自行决定是否采纳。
理由：静默丢弃比带 flag 的错误结果更难排查。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from core.stage_base import Stage, StageSkip
from core.item import TimeResult

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item
    from core.evidence import Evidence


# 精度排序（数字越大越精确）
_PRECISION_RANK: dict[str, int] = {
    "unknown": 0,
    "year":    1,
    "month":   2,
    "day":     3,
    "hour":    4,
    "minute":  5,
    "second":  6,
}

# 冲突检测：两条 evidence 时间差超过此值才算冲突（天）
_DEFAULT_CONFLICT_DAYS = 30

# 参与冲突检测的最低 confidence 门槛
# 低可信度的 evidence（如 filesystem mtime）不参与冲突检测，避免误报
_CONFLICT_MIN_CONFIDENCE = 0.5


class ResolverStage(Stage):
    """
    从 item.evidence 中决策出 time_result。
    """

    name = "ResolverStage"

    def __init__(self, conflict_threshold_days: int = _DEFAULT_CONFLICT_DAYS) -> None:
        self.conflict_threshold = timedelta(days=conflict_threshold_days)

    # ------------------------------------------------------------------
    # 前置检查
    # ------------------------------------------------------------------

    def skip_reason(self, item: "Item", ctx: "Context") -> Optional[str]:
        if not item.evidence:
            return "no evidence to resolve"
        return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def process(self, item: "Item", ctx: "Context") -> None:

        # 1. 过滤出有时间点的 evidence
        candidates = [ev for ev in item.evidence if ev.dt is not None]
        if not candidates:
            item.log(self.name, "all evidence have dt=None, cannot resolve")
            return

        # 2. 直接证据优先；同类内按 confidence 降序
        direct   = sorted(
            [ev for ev in candidates if ev.is_direct],
            key=lambda e: (e.confidence, _PRECISION_RANK.get(e.precision, 0)),
            reverse=True,
        )
        indirect = sorted(
            [ev for ev in candidates if not ev.is_direct],
            key=lambda e: (e.confidence, _PRECISION_RANK.get(e.precision, 0)),
            reverse=True,
        )
        ranked = direct + indirect  # 直接证据整体优先于间接证据

        # 3. 主证据 = ranked[0]
        primary = ranked[0]

        # 4. 冲突检测
        conflicts = self._find_conflicts(primary, ranked[1:], ctx, item)

        if conflicts:
            item.add_flag("CONFLICT_TIME")
            item.add_flag("NEEDS_REVIEW")
            for c_ev, delta in conflicts:
                item.warn(
                    self.name,
                    f"CONFLICT: primary={primary.source}({primary.dt.date()}, "
                    f"conf={primary.confidence:.2f}) vs "
                    f"{c_ev.source}({c_ev.dt.date()}, "   # type: ignore[union-attr]
                    f"conf={c_ev.confidence:.2f}), "
                    f"gap={delta.days}d",
                )

        # 5. 置信区间
        range_start, range_end = self._compute_range(primary, candidates)

        # 6. source_summary（按 confidence 降序的来源列表）
        source_summary = list(dict.fromkeys(
            ev.source for ev in ranked  # dict.fromkeys 去重并保序
        ))

        # 7. 写入 time_result
        item.time_result = TimeResult(
            final_datetime=primary.dt,
            confidence=primary.confidence,
            precision=primary.precision,
            range_start=range_start,
            range_end=range_end,
            primary_source=primary.source,
            source_summary=source_summary,
        )

        item.log(
            self.name,
            f"resolved: {primary.dt.isoformat()} "
            f"[{primary.source}, {primary.precision}, "
            f"conf={primary.confidence:.2f}]"
            + (f" ⚠ {len(conflicts)} conflict(s)" if conflicts else ""),
        )

    # ------------------------------------------------------------------
    # 冲突检测
    # ------------------------------------------------------------------

    def _find_conflicts(
        self,
        primary: "Evidence",
        others: list["Evidence"],
        ctx: "Context",
        item: "Item",
    ) -> list[tuple["Evidence", timedelta]]:
        """
        找出与主证据时间差超过阈值的高可信度 evidence。

        只检测 confidence >= _CONFLICT_MIN_CONFIDENCE 的证据，
        避免 filesystem mtime（confidence=0.2）之类的噪声触发冲突。

        返回 [(冲突证据, 时间差), ...]
        """
        conflicts = []
        for ev in others:
            if ev.dt is None:
                continue
            if ev.confidence < _CONFLICT_MIN_CONFIDENCE:
                continue
            # 同一来源的不同字段（如 EXIF DateTimeOriginal vs DateTime）
            # 如果来源相同，允许更大的时间差（编辑软件可能改写 DateTime）
            threshold = (
                self.conflict_threshold * 2
                if ev.source == primary.source
                else self.conflict_threshold
            )
            delta = abs(primary.dt - ev.dt)  # type: ignore[operator]
            if delta > threshold:
                conflicts.append((ev, delta))

        return conflicts

    # ------------------------------------------------------------------
    # 置信区间
    # ------------------------------------------------------------------

    def _compute_range(
        self,
        primary: "Evidence",
        all_candidates: list["Evidence"],
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        计算综合置信区间。

        策略（按优先级）：
        1. 主证据自带 range_start/range_end → 直接用
        2. 主证据有精度信息 → 由精度推算区间
           second → ±0s（点区间）
           minute → ±30s
           hour   → ±30min
           day    → 当天 00:00 ~ 23:59:59
           month  → 当月第一天 ~ 最后一天
           year   → 当年第一天 ~ 最后一天
        3. fallback → None（不输出区间）
        """
        # 优先用证据自带区间
        if primary.range_start and primary.range_end:
            return primary.range_start, primary.range_end

        dt = primary.dt
        if dt is None:
            return None, None

        precision = primary.precision

        if precision == "second":
            return dt, dt

        if precision == "minute":
            return (
                dt - timedelta(seconds=30),
                dt + timedelta(seconds=30),
            )

        if precision == "hour":
            return (
                dt - timedelta(minutes=30),
                dt + timedelta(minutes=30),
            )

        if precision == "day":
            start = dt.replace(hour=0,  minute=0,  second=0,  microsecond=0)
            end   = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            return start, end

        if precision == "month":
            import calendar
            start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            end = dt.replace(day=last_day, hour=23, minute=59,
                             second=59, microsecond=999999)
            return start, end

        if precision == "year":
            start = dt.replace(month=1,  day=1,  hour=0,  minute=0,
                               second=0,  microsecond=0)
            end   = dt.replace(month=12, day=31, hour=23, minute=59,
                               second=59, microsecond=999999)
            return start, end

        return None, None
