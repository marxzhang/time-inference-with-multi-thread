"""
stages/timeline/interpolator.py — Session 内时间插值

职责
----
在 StructResolver 运行之后，对 session 内仍无可信时间的 item
利用已知锚点（anchor_items）做时间插值或外推，追加 Evidence。

插值前提
--------
必须先对 session 内 items 建立可靠的"拍摄顺序"，
才能在顺序上做线性插值。顺序依据（优先级依次降低）：
    1. 已有可信时间 → 直接用 final_datetime 排序
    2. 文件名末尾数字序号（IMG_1234 → 1234）→ 相机连拍顺序
    3. mtime → 最后兜底，最不可靠

同文件夹的 item 序号可信度更高（同一次拍摄的连拍）；
跨文件夹的 item 置信度打折（可能是不同场景）。

插值模式
--------
    内插（两侧都有锚点）
        ratio = (pos - left_pos) / (right_pos - left_pos)
        dt    = left_dt + (right_dt - left_dt) × ratio
        conf  = base_conf × (1 - distance_factor)

    外推（只有单侧锚点）
        dt    = nearest_anchor_dt（同一天，精度降为 "day"）
        conf  = anchor_conf - dist × decay_per_step
        外推置信度显著低于内插，随距离快速衰减

产出
----
Evidence(source="timeline", stage="TimelineStage", is_direct=False)
不直接修改 time_result，只追加 evidence。
ResolverStage 重跑后才更新 time_result（与现有 pipeline 设计一致）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.item import Item
    from session import Session


# 每步（每张图）的外推置信度衰减
_DECAY_PER_STEP = 0.08

# 同文件夹置信度加成
_SAME_FOLDER_BONUS = 0.10

# 内插置信度的距离因子权重（越大，距离越远时置信度下降越快）
_DISTANCE_WEIGHT = 0.6

# 产出 Evidence 的最低置信度门槛
_MIN_CONFIDENCE = 0.10

# 只向 confidence 低于此值的 item 追加 evidence
_TARGET_MAX_CONFIDENCE = 0.6

# 提取文件名末尾数字的正则（IMG_1234.jpg → 1234）
_TAIL_NUMBER_RE = re.compile(r'(\d+)[^\d]*$')


# ---------------------------------------------------------------------------
# 排序后的序列项（内部数据结构）
# ---------------------------------------------------------------------------

@dataclass
class _OrderedItem:
    """排序序列中的一个位置。"""
    item: "Item"
    pos: float          # 在序列中的位置（0-based，可以是小数）
    is_anchor: bool     # 是否是可信锚点
    folder: str         # 所在文件夹路径


# ---------------------------------------------------------------------------
# Interpolator
# ---------------------------------------------------------------------------

class Interpolator:
    """
    Session 内时间插值。

    使用示例
    --------
    interp = Interpolator()
    n = interp.interpolate(session)
    print(f"追加了 {n} 条 timeline evidence")
    """

    def __init__(
        self,
        decay_per_step: float = _DECAY_PER_STEP,
        same_folder_bonus: float = _SAME_FOLDER_BONUS,
        min_confidence: float = _MIN_CONFIDENCE,
    ) -> None:
        self.decay_per_step   = decay_per_step
        self.same_folder_bonus = same_folder_bonus
        self.min_confidence   = min_confidence

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def interpolate(
        self,
        session: "Session",
        target_max_confidence: float = _TARGET_MAX_CONFIDENCE,
        verbose: bool = False,
    ) -> int:
        """
        对 session 内缺少可信时间的 item 追加插值 Evidence。

        参数
        ----
        session                 包含 items 和 anchor_items 的 Session
        target_max_confidence   只向低于此置信度的 item 追加 evidence
        verbose                 True 时打印每个 item 的插值过程

        返回
        ----
        追加了 evidence 的 item 数量
        """
        if not session.has_anchor:
            return 0

        # 找目标 item（需要帮助的）
        targets = {
            it.id
            for it in session.items
            if not it.time_result.is_resolved
            or it.time_result.confidence < target_max_confidence
        }
        if not targets:
            return 0

        # 建立排序序列
        ordered = self._build_ordered_sequence(session)
        if len(ordered) < 2:
            return 0

        # 找出锚点位置
        anchor_positions = [
            oi for oi in ordered if oi.is_anchor
        ]
        if not anchor_positions:
            return 0

        improved = 0

        for oi in ordered:
            if oi.item.id not in targets:
                continue
            if oi.is_anchor:
                continue

            ev = self._interpolate_one(oi, ordered, session, verbose)
            if ev is not None:
                oi.item.add_evidence(ev)
                improved += 1

        return improved

    # ------------------------------------------------------------------
    # 排序序列构建
    # ------------------------------------------------------------------

    def _build_ordered_sequence(self, session: "Session") -> list[_OrderedItem]:
        """
        把 session 内所有 item 排列成有序序列，位置用浮点数表示。

        策略：
          1. 按文件夹分组
          2. 各文件夹内用文件名序号统一排序（有时间和无时间交错排列）
          3. 有锚点的文件夹优先排列
          4. 无锚点的文件夹追加在末尾
          5. 用锚点真实时间重新校准位置轴
        """
        from pathlib import Path

        # 按文件夹分组
        folder_groups: dict[str, list["Item"]] = {}
        for item in session.items:
            folder = str(Path(item.relpath).parent)
            folder_groups.setdefault(folder, []).append(item)

        # 对每个文件夹内的 item 按序号统一排序（有时间和无时间放在一起）
        for folder, group in folder_groups.items():
            group.sort(key=lambda it: self._sort_key(it))

        # 合并到有序序列
        ordered: list[_OrderedItem] = []
        pos = 0.0

        anchor_folders = {
            str(Path(it.relpath).parent)
            for it in session.anchor_items
        }

        # 先排有锚点的文件夹
        for folder in sorted(folder_groups.keys()):
            if folder not in anchor_folders:
                continue
            for item in folder_groups[folder]:
                is_anchor = (
                    item.time_result.is_resolved
                    and item.time_result.confidence >= 0.6
                )
                ordered.append(_OrderedItem(
                    item=item, pos=pos,
                    is_anchor=is_anchor, folder=folder
                ))
                pos += 1.0

        # 再排无锚点的文件夹
        for folder in sorted(folder_groups.keys()):
            if folder in anchor_folders:
                continue
            for item in folder_groups[folder]:
                ordered.append(_OrderedItem(
                    item=item, pos=pos,
                    is_anchor=False, folder=folder
                ))
                pos += 1.0

        # 用锚点的真实时间重新校准位置轴
        self._realign_positions(ordered)

        return ordered

    def _sort_key(self, item: "Item") -> tuple:
        """
        文件夹内排序 key。

        核心原则：有时间和无时间的 item 用同一把尺（文件名序号）排在一起，
        使 IMG_001、IMG_005（无时间）、IMG_010 能正确交错排列。

        有时间的 item：序号作为主 key，时间作为 tiebreaker（确认序号正确）
        无时间的 item：序号作为主 key，mtime 作为 tiebreaker
        """
        tail = self._tail_number(item)
        tr = item.time_result
        if tr.is_resolved and tr.confidence >= 0.6:
            return (tail, 0, tr.final_datetime.timestamp())
        else:
            return (tail, 1, item.mtime or 0)

    @staticmethod
    def _tail_number(item: "Item") -> int:
        """提取文件名末尾数字（IMG_1234.jpg → 1234）。"""
        stem = Path(item.filename).stem
        m = _TAIL_NUMBER_RE.search(stem)
        return int(m.group(1)) if m else 0

    def _realign_positions(self, ordered: list[_OrderedItem]) -> None:
        """
        根据锚点的真实时间，把整数位置轴重新映射到连续时间轴。

        目的：使 pos 值能直接反映时间间隔，而不只是文件序号。
        映射方式：
            找出所有锚点，按 (pos, dt) 建立分段线性映射
            非锚点 item 的 pos 在相邻锚点之间线性插值

        如果只有一个锚点，pos 保持整数（只做外推，不需要精确 pos）。
        """
        anchors = [(oi.pos, oi.item.time_result.final_datetime)
                   for oi in ordered if oi.is_anchor]
        if len(anchors) < 2:
            return   # 外推模式不需要 realign

        # 把 pos 转换为"从第一个锚点起的秒数"
        t0_pos, t0_dt = anchors[0]
        t1_pos, t1_dt = anchors[-1]

        total_secs = (t1_dt - t0_dt).total_seconds()
        total_pos  = t1_pos - t0_pos
        if total_pos == 0 or total_secs == 0:
            return

        scale = total_secs / total_pos   # 秒 / 位置单位

        for oi in ordered:
            oi.pos = (oi.pos - t0_pos) * scale   # 转换为秒数（相对 t0_dt）

    # ------------------------------------------------------------------
    # 单 item 插值
    # ------------------------------------------------------------------

    def _interpolate_one(
        self,
        target: _OrderedItem,
        ordered: list[_OrderedItem],
        session: "Session",
        verbose: bool,
    ) -> Optional[object]:
        """对单个 item 找最近锚点，决定用内插还是外推。"""
        left  = self._nearest_anchor(target, ordered, direction="left")
        right = self._nearest_anchor(target, ordered, direction="right")

        if left and right:
            result = self._interpolate(target, left, right, session)
            mode = "内插"
        elif left or right:
            anchor = left or right
            result = self._extrapolate(target, anchor, session)
            mode = "外推"
        else:
            return None

        if result is None:
            return None

        dt, conf, precision, reason = result

        if conf < self.min_confidence:
            return None

        if verbose:
            print(
                f"  [{mode}] {target.item.filename!r:30s} "
                f"→ {dt.strftime('%Y-%m-%d %H:%M')} "
                f"precision={precision} conf={conf:.2f}"
            )

        return self._make_evidence(dt, conf, precision, reason,
                                   left, right, target)

    def _nearest_anchor(
        self,
        target: _OrderedItem,
        ordered: list[_OrderedItem],
        direction: str,
    ) -> Optional[_OrderedItem]:
        """找 target 左侧或右侧最近的锚点。"""
        if direction == "left":
            candidates = [oi for oi in ordered
                          if oi.is_anchor and oi.pos < target.pos]
            return candidates[-1] if candidates else None
        else:
            candidates = [oi for oi in ordered
                          if oi.is_anchor and oi.pos > target.pos]
            return candidates[0] if candidates else None

    def _interpolate(
        self,
        target: _OrderedItem,
        left: _OrderedItem,
        right: _OrderedItem,
        session: "Session",
    ) -> Optional[tuple]:
        """两侧锚点线性内插。"""
        left_dt  = left.item.time_result.final_datetime
        right_dt = right.item.time_result.final_datetime

        span = right.pos - left.pos
        if span == 0:
            dt = left_dt
        else:
            ratio = (target.pos - left.pos) / span
            total_secs = (right_dt - left_dt).total_seconds()
            dt = left_dt + timedelta(seconds=total_secs * ratio)

        # 置信度：基于两侧锚点的最低置信度，距离越近越高
        base_conf = min(
            left.item.time_result.confidence,
            right.item.time_result.confidence,
        )
        # distance_factor：target 距较远侧锚点的比例
        if span > 0:
            left_dist  = target.pos - left.pos
            right_dist = right.pos - target.pos
            distance_factor = max(left_dist, right_dist) / span
        else:
            distance_factor = 0.0

        conf = base_conf * (1.0 - _DISTANCE_WEIGHT * distance_factor)

        # 同文件夹加成
        if target.folder == left.folder or target.folder == right.folder:
            conf += self.same_folder_bonus

        conf = round(max(self.min_confidence, min(0.85, conf)), 3)

        # 精度：两侧锚点时间差决定精度
        total_secs = abs((right_dt - left_dt).total_seconds())
        precision = self._duration_to_precision(total_secs, span)

        reason = (
            f"interpolated between "
            f"{left.item.filename!r}(@{left_dt.strftime('%H:%M')}) and "
            f"{right.item.filename!r}(@{right_dt.strftime('%H:%M')})"
        )
        return dt, conf, precision, reason

    def _extrapolate(
        self,
        target: _OrderedItem,
        anchor: _OrderedItem,
        session: "Session",
    ) -> Optional[tuple]:
        """单侧锚点外推。"""
        anchor_dt   = anchor.item.time_result.final_datetime
        anchor_conf = anchor.item.time_result.confidence

        # 外推：时间与锚点相同（只知道"同一天"），精度降为 day
        dt = anchor_dt

        # 距离（以 realign 后的秒数为单位；若未 realign 则为位置步数）
        dist_steps = abs(target.pos - anchor.pos)
        # 将距离归一化到"步数"（每步 ~1 张图）
        # realign 后 pos 是秒数，用平均间隔估算步数
        avg_interval = self._estimate_avg_interval(session)
        if avg_interval > 0:
            dist_steps = dist_steps / avg_interval

        conf = anchor_conf - dist_steps * self.decay_per_step

        # 同文件夹加成
        if target.folder == anchor.folder:
            conf += self.same_folder_bonus

        conf = round(max(self.min_confidence, min(0.60, conf)), 3)
        # 外推置信度上限 0.60（低于内插，反映不确定性）

        precision = "day"   # 外推只知道"这一天"

        side = "after" if anchor.pos < target.pos else "before"
        reason = (
            f"extrapolated {side} anchor "
            f"{anchor.item.filename!r}(@{anchor_dt.strftime('%Y-%m-%d %H:%M')})"
        )
        return dt, conf, precision, reason

    @staticmethod
    def _estimate_avg_interval(session: "Session") -> float:
        """估算锚点之间的平均时间间隔（秒/张）。"""
        anchors = sorted(
            session.anchor_items,
            key=lambda it: it.time_result.final_datetime,
        )
        if len(anchors) < 2:
            return 60.0  # 默认 1 分钟/张

        total_secs = (
            anchors[-1].time_result.final_datetime
            - anchors[0].time_result.final_datetime
        ).total_seconds()
        return total_secs / max(len(anchors) - 1, 1)

    @staticmethod
    def _duration_to_precision(total_secs: float, span_steps: float) -> str:
        """
        根据两侧锚点的时间跨度决定插值结果的精度。

        跨度越小，序列越密集，插值越精确。
        """
        if span_steps == 0:
            return "second"
        # 平均每步的时间（秒/步）
        secs_per_step = total_secs / span_steps
        if secs_per_step < 120:      # <2分钟/张 → 连拍，精确到分
            return "minute"
        if secs_per_step < 3600:     # <1小时/张
            return "minute"
        if secs_per_step < 86400:    # <1天/张
            return "hour"
        return "day"

    @staticmethod
    def _make_evidence(
        dt: datetime,
        conf: float,
        precision: str,
        reason: str,
        left: Optional[_OrderedItem],
        right: Optional[_OrderedItem],
        target: _OrderedItem,
    ):
        from core.evidence import Evidence
        related = []
        if left:
            related.append(left.item.id)
        if right:
            related.append(right.item.id)

        mode = "interpolated" if (left and right) else "extrapolated"
        return Evidence(
            source="timeline",
            stage="TimelineStage",
            dt=dt,
            precision=precision,
            confidence=conf,
            is_direct=False,
            reason=reason,
            metadata={
                "interpolation_mode": mode,
                "left_anchor_id":  left.item.id  if left  else None,
                "right_anchor_id": right.item.id if right else None,
                "target_folder":   target.folder,
            },
            related_item_ids=related,
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
    from session import Session
    from interpolator import Interpolator

    def _anchor(relpath: str, dt: datetime, conf: float = 1.0) -> Item:
        item = Item(path=f"/fake/{relpath}", relpath=relpath)
        item.filename = Path(relpath).name
        item.mtime = dt.timestamp()
        item.time_result = TimeResult(
            final_datetime=dt, confidence=conf,
            precision="second", primary_source="exif"
        )
        return item

    def _float(relpath: str, mtime_dt: datetime = None) -> Item:
        item = Item(path=f"/fake/{relpath}", relpath=relpath)
        item.filename = Path(relpath).name
        if mtime_dt:
            item.mtime = mtime_dt.timestamp()
        return item

    interp = Interpolator()
    base = datetime(2019, 2, 13, 10, 0, 0)

    print("=" * 60)
    print("Interpolator 测试（verbose=True）")
    print("=" * 60)

    # ── 测试 1：基本内插（两侧锚点，序号连续）────────────────────────
    print("\n[测试1] 两侧锚点内插：IMG_001@10:00, IMG_005@?, IMG_010@12:00")
    s1 = Session(id="s0")
    a_left  = _anchor("成都/IMG_001.jpg", base)
    a_right = _anchor("成都/IMG_010.jpg", base + timedelta(hours=2))
    f_mid   = _float("成都/IMG_005.jpg")
    s1.items        = [a_left, a_right, f_mid]
    s1.anchor_items = [a_left, a_right]
    s1.time_start, s1.time_end = base, base + timedelta(hours=2)

    n1 = interp.interpolate(s1, verbose=True)
    ev1 = [e for e in f_mid.evidence if e.source == "timeline"]
    assert n1 == 1, f"期望 1，实际 {n1}"
    assert len(ev1) == 1
    dt1 = ev1[0].dt
    # realign 后位置轴是时间轴（秒数），IMG_005 在两锚点时间中点
    # 001@pos=0s, 005@pos=3600s(中点), 010@pos=7200s → ratio=0.5 → 11:00
    expected_approx = base + timedelta(hours=1)   # 10:00 + 1h = 11:00
    diff = abs((dt1 - expected_approx).total_seconds())
    assert diff < 60, f"时间误差过大: {diff:.0f}s（期望 < 60s）"
    print(f"  ✓ n={n1}, dt={dt1.strftime('%H:%M')}, "
          f"conf={ev1[0].confidence:.2f}, precision={ev1[0].precision}")

    # ── 测试 2：单侧外推（只有右侧锚点）──────────────────────────────
    print("\n[测试2] 单侧外推：IMG_003@? → 最近锚点 IMG_005@11:00")
    s2 = Session(id="s1")
    a_right2 = _anchor("成都/IMG_005.jpg", base + timedelta(hours=1))
    f_early  = _float("成都/IMG_003.jpg")
    s2.items        = [f_early, a_right2]
    s2.anchor_items = [a_right2]
    s2.time_start = s2.time_end = base + timedelta(hours=1)

    n2 = interp.interpolate(s2, verbose=True)
    ev2 = [e for e in f_early.evidence if e.source == "timeline"]
    assert n2 == 1
    assert ev2[0].precision == "day"          # 外推精度降为 day
    assert ev2[0].confidence < 1.0
    print(f"  ✓ n={n2}, precision={ev2[0].precision}, conf={ev2[0].confidence:.2f}")

    # ── 测试 3：无锚点 session → 跳过 ────────────────────────────────
    print("\n[测试3] 无锚点 session → 跳过，返回 0")
    s3 = Session(id="s2")
    s3.items = [_float("照片/A.jpg"), _float("照片/B.jpg")]
    s3.anchor_items = []

    n3 = interp.interpolate(s3, verbose=True)
    assert n3 == 0, f"期望 0，实际 {n3}"
    print(f"  ✓ n={n3}（跳过）")

    # ── 测试 4：已有高置信度时间的 item 不追加 ───────────────────────
    print("\n[测试4] target_max_confidence 过滤：高置信 item 不追加")
    s4 = Session(id="s3")
    a1 = _anchor("成都/IMG_001.jpg", base)
    a2 = _anchor("成都/IMG_010.jpg", base + timedelta(hours=2))
    already_good = _anchor("成都/IMG_005.jpg",
                           base + timedelta(hours=1), conf=0.9)
    s4.items = [a1, a2, already_good]
    s4.anchor_items = [a1, a2]

    n4 = interp.interpolate(s4, target_max_confidence=0.6, verbose=True)
    assert n4 == 0, f"已有高置信度时间，期望 0，实际 {n4}"
    print(f"  ✓ n={n4}（已有高置信度时间，跳过）")

    # ── 测试 5：多个浮动 item，距离递增，置信度递减 ───────────────────
    print("\n[测试5] 外推置信度随距离衰减")
    s5 = Session(id="s4")
    anchor5 = _anchor("相册/IMG_010.jpg", base)
    floats5 = [_float(f"相册/IMG_{10+i:03d}.jpg") for i in range(1, 5)]
    s5.items = [anchor5] + floats5
    s5.anchor_items = [anchor5]
    s5.time_start = s5.time_end = base

    n5 = interp.interpolate(s5, verbose=True)
    confs = [
        e.confidence
        for f in floats5
        for e in f.evidence if e.source == "timeline"
    ]
    print(f"  n={n5}, 置信度序列: {[f'{c:.2f}' for c in confs]}")
    # 置信度应单调递减（离锚点越远越低）
    for i in range(len(confs) - 1):
        assert confs[i] >= confs[i+1], f"置信度未单调递减: {confs}"
    print("  ✓ 置信度单调递减验证通过")

    # ── 测试 6：同文件夹加成 ─────────────────────────────────────────
    print("\n[测试6] 同文件夹 bonus：同文件夹 item 置信度更高")
    s6 = Session(id="s5")
    anc6 = _anchor("成都/IMG_001.jpg", base)
    same_folder = _float("成都/IMG_002.jpg")
    diff_folder = _float("重庆/IMG_100.jpg")
    s6.items = [anc6, same_folder, diff_folder]
    s6.anchor_items = [anc6]
    s6.time_start = s6.time_end = base

    interp.interpolate(s6, verbose=True)
    ev_same = [e for e in same_folder.evidence if e.source == "timeline"]
    ev_diff = [e for e in diff_folder.evidence if e.source == "timeline"]
    if ev_same and ev_diff:
        assert ev_same[0].confidence >= ev_diff[0].confidence, \
            f"同文件夹应更高: {ev_same[0].confidence:.2f} vs {ev_diff[0].confidence:.2f}"
        print(f"  ✓ 同文件夹={ev_same[0].confidence:.2f} >= "
              f"跨文件夹={ev_diff[0].confidence:.2f}")

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)

    # ── 模式 B：真实 snapshot ─────────────────────────────────────────
    if len(sys.argv) > 1:
        snap_path = Path(sys.argv[1])
        print(f"\n模式 B：真实 snapshot — {snap_path}")
        print("=" * 60)

        from storage.snapshot import load
        from session import SessionBuilder
        from tokenizer import Tokenizer
        from struct_resolver import StructResolver

        items = load(snap_path)
        print(f"加载 {len(items)} 个 item")

        sessions = SessionBuilder().build(items)

        # StructResolver 先跑
        tokenizer = Tokenizer()
        resolver  = StructResolver()
        for item in items:
            _, structs, _ = tokenizer.extract(item)
            for s in sessions:
                if item in s.items:
                    s.struct_tokens.extend(structs)
                    break
        for s in sessions:
            resolver.resolve(s)

        # Interpolator
        total_before = sum(
            1 for it in items if not it.time_result.is_resolved
        )
        total_ev = 0
        for s in sessions:
            n = interp.interpolate(s, verbose=False)
            total_ev += n

        print(f"\n插值追加 evidence 的 item 数: {total_ev}")
        print(f"插值前无时间 item 数: {total_before}")
        print("（需重跑 ResolverStage 才能更新 time_result）")

        # 展示部分插值结果
        print("\n前 10 个插值结果：")
        shown = 0
        for it in items:
            evs = [e for e in it.evidence if e.source == "timeline"
                   and "interpolat" in e.metadata.get("interpolation_mode", "")]
            if evs and shown < 10:
                print(
                    f"  {it.filename!r:35s} → "
                    f"{evs[0].dt.strftime('%Y-%m-%d %H:%M')} "
                    f"precision={evs[0].precision} "
                    f"conf={evs[0].confidence:.2f} "
                    f"[{evs[0].metadata['interpolation_mode']}]"
                )
                shown += 1