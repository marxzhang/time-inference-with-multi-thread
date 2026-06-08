"""
stages/timeline/session.py — Session 数据结构 + 聚类

Session 是 StructResolver 和 Interpolator 工作的基本单元，
代表一次连续拍摄活动（如一天的出行、一个事件）的图片集合。

Session 与文件夹的区别
----------------------
    文件夹 = 用户主观整理的目录结构，可能横跨多次拍摄
    Session = 由时间连续性自动推断的拍摄活动单元

    同一文件夹的图片可能属于不同 session（文件夹按主题整理，不按时间）。
    不同文件夹的图片可能属于同一 session（旅行图片分散在多个子目录）。

SessionBuilder 两阶段流程
--------------------------
    阶段 1：锚点聚类
        有可信时间的 item（锚点）按时间排序，
        相邻间隔 > gap_hours → 切分新 session。

    阶段 2：浮动归属
        无时间的 item（浮动）按文件夹前缀归入最匹配的 session。
        若无 session 可归属（全部浮动），建一个 fallback session。

struct_tokens 字段
-------------------
    Session.struct_tokens 初始为空列表。
    TimelineStage 在建完 session 后，把 Tokenizer 产出的 StructToken
    按归属关系填入各 session。SessionBuilder 不负责此字段。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.item import Item
    from tokens import StructToken, SemanticToken


# ---------------------------------------------------------------------------
# Session 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    一次拍摄活动的图片集合及其时间上下文。

    字段说明
    --------
    id              全局唯一标识，格式 "s{idx}"
    items           该 session 的全部 item（含锚点和浮动）
    anchor_items    有可信时间的子集（confidence >= threshold），时序已排序
    time_start      最早锚点的时间（可 None，表示 session 无锚点）
    time_end        最晚锚点的时间（可 None）
    folder_path     该 session 内所有 item 的最深公共文件夹路径
    year_candidates 由锚点时间推算的可能年份，供 StructResolver 消歧
    struct_tokens   待消歧的 StructToken，由 TimelineStage 填入（初始空列表）
    semantic_tokens 语义 Token，由 TimelineStage 填入（D 阶段）
    """

    id: str

    items:        list["Item"] = field(default_factory=list)
    anchor_items: list["Item"] = field(default_factory=list)

    time_start: Optional[datetime] = None
    time_end:   Optional[datetime] = None

    folder_path: str = ""

    year_candidates: list[int] = field(default_factory=list)

    # 由 TimelineStage 填入，SessionBuilder 不负责
    struct_tokens:   list["StructToken"]   = field(default_factory=list)
    semantic_tokens: list["SemanticToken"] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def has_anchor(self) -> bool:
        """是否有至少一个时间锚点。"""
        return len(self.anchor_items) > 0

    @property
    def duration(self) -> Optional[timedelta]:
        """session 的时间跨度（无锚点时为 None）。"""
        if self.time_start and self.time_end:
            return self.time_end - self.time_start
        return None

    @property
    def n_unresolved(self) -> int:
        """未能确定时间的 item 数量。"""
        return sum(1 for it in self.items if not it.time_result.is_resolved)

    def __repr__(self) -> str:
        time_str = (
            f"{self.time_start.strftime('%Y-%m-%d')}~{self.time_end.strftime('%Y-%m-%d')}"
            if self.time_start and self.time_end
            else "no-anchor"
        )
        return (
            f"Session({self.id}, "
            f"items={len(self.items)}, "
            f"anchors={len(self.anchor_items)}, "
            f"time={time_str}, "
            f"folder={self.folder_path!r})"
        )


# ---------------------------------------------------------------------------
# SessionBuilder
# ---------------------------------------------------------------------------

class SessionBuilder:
    """
    把 items 列表聚类为 Session 列表。

    使用示例
    --------
    builder = SessionBuilder(gap_hours=6.0, anchor_min_confidence=0.6)
    sessions = builder.build(items)
    for s in sessions:
        print(s)
    """

    def __init__(
        self,
        gap_hours: float = 6.0,
        anchor_min_confidence: float = 0.6,
    ) -> None:
        """
        参数
        ----
        gap_hours               相邻锚点超过此间隔（小时）则切分新 session
        anchor_min_confidence   item 成为锚点的最低 confidence
        """
        self.gap = timedelta(hours=gap_hours)
        self.anchor_min_confidence = anchor_min_confidence

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def build(self, items: list["Item"]) -> list[Session]:
        """
        聚类入口，返回 session 列表（按时间升序，无锚点 session 在最后）。

        流程：
            1. 分拣：锚点 vs 浮动
            2. 锚点聚类（按时间间隔切分）
            3. 浮动归属（按文件夹前缀匹配）
            4. 计算各 session 的 time_range / folder_path / year_candidates
        """
        if not items:
            return []

        anchored, floating = self._split(items)
        sessions = self._cluster_anchored(anchored)
        self._assign_floating(floating, sessions)
        self._finalize(sessions)

        return sessions

    # ------------------------------------------------------------------
    # 阶段 1：分拣
    # ------------------------------------------------------------------

    def _split(
        self,
        items: list["Item"],
    ) -> tuple[list["Item"], list["Item"]]:
        """按是否有可信时间分为锚点和浮动两组。"""
        anchored, floating = [], []
        for item in items:
            tr = item.time_result
            if tr.is_resolved and tr.confidence >= self.anchor_min_confidence:
                anchored.append(item)
            else:
                floating.append(item)
        return anchored, floating

    # ------------------------------------------------------------------
    # 阶段 2：锚点聚类
    # ------------------------------------------------------------------

    def _cluster_anchored(self, anchored: list["Item"]) -> list[Session]:
        """
        按时间排序锚点，相邻间隔 > gap → 切分新 session。

        全部无锚点时返回单个空 session（fallback）。
        """
        if not anchored:
            return [Session(id="s0")]

        sorted_items = sorted(
            anchored,
            key=lambda it: it.time_result.final_datetime,
        )

        sessions: list[Session] = []
        current_batch: list["Item"] = []

        for item in sorted_items:
            if not current_batch:
                current_batch.append(item)
                continue

            last_dt = current_batch[-1].time_result.final_datetime
            this_dt = item.time_result.final_datetime

            if this_dt - last_dt > self.gap:
                # 超过间隔 → 结束当前 session，开始新的
                sessions.append(self._make_session(current_batch, len(sessions)))
                current_batch = [item]
            else:
                current_batch.append(item)

        if current_batch:
            sessions.append(self._make_session(current_batch, len(sessions)))

        return sessions

    def _make_session(self, anchor_batch: list["Item"], idx: int) -> Session:
        """从一批锚点 item 构造 Session（不含 floating，待后续归属）。"""
        s = Session(id=f"s{idx}")
        s.anchor_items = list(anchor_batch)
        s.items = list(anchor_batch)
        return s

    # ------------------------------------------------------------------
    # 阶段 3：浮动归属
    # ------------------------------------------------------------------

    def _assign_floating(
        self,
        floating: list["Item"],
        sessions: list[Session],
    ) -> None:
        """
        把浮动 item 归入最匹配的 session。

        匹配优先级：
            1. 文件夹前缀重叠最长的 session（文件夹结构相近 → 同一拍摄活动）
            2. 若多个 session 前缀重叠相同长度，取时间最近的 session
            3. 若完全无重叠（孤立文件夹），归入时间最近的 session
            4. 若无 session 有时间（全部 fallback），归入 s0

        修改 session.items（in-place），不返回值。
        """
        if not floating:
            return

        if not sessions:
            return

        # 预计算每个 session 的文件夹路径（用第一个 anchor item 代表）
        session_folders = [
            _item_folder(s.anchor_items[0]) if s.anchor_items else ""
            for s in sessions
        ]

        for item in floating:
            item_folder = _item_folder(item)
            best_session = self._find_best_session(
                item_folder, item, sessions, session_folders
            )
            best_session.items.append(item)

    def _find_best_session(
        self,
        item_folder: str,
        item: "Item",
        sessions: list[Session],
        session_folders: list[str],
    ) -> Session:
        """找出最适合归入的 session。"""
        # 计算每个 session 与 item 文件夹的前缀重叠长度
        overlaps = [
            _folder_overlap(item_folder, sf)
            for sf in session_folders
        ]
        max_overlap = max(overlaps)

        # 找出重叠最长的 session 集合
        candidates = [
            sessions[i]
            for i, ov in enumerate(overlaps)
            if ov == max_overlap
        ]

        if len(candidates) == 1:
            return candidates[0]

        # 平局：取时间最近的 session（用 time_start 或 time_end 比较）
        item_mtime = datetime.fromtimestamp(item.mtime) if item.mtime else None
        if item_mtime:
            def _time_dist(s: Session) -> float:
                ref = s.time_start or s.time_end
                if ref is None:
                    return float("inf")
                return abs((item_mtime - ref).total_seconds())
            candidates.sort(key=_time_dist)

        return candidates[0]

    # ------------------------------------------------------------------
    # 阶段 4：收尾计算
    # ------------------------------------------------------------------

    def _finalize(self, sessions: list[Session]) -> None:
        """
        为每个 session 计算：
            time_start / time_end    由 anchor_items 时间取 min/max
            folder_path              session 内所有 item 的最深公共父目录
            year_candidates          由锚点时间收集可能年份
        """
        for s in sessions:
            self._fill_time_range(s)
            self._fill_folder_path(s)
            self._fill_year_candidates(s)

    @staticmethod
    def _fill_time_range(s: Session) -> None:
        times = [
            it.time_result.final_datetime
            for it in s.anchor_items
            if it.time_result.final_datetime
        ]
        if times:
            s.time_start = min(times)
            s.time_end   = max(times)

    @staticmethod
    def _fill_folder_path(s: Session) -> None:
        """所有 item（含 floating）的公共父目录前缀。"""
        if not s.items:
            return
        folders = [str(Path(it.relpath).parent) for it in s.items]
        s.folder_path = _common_folder_prefix(folders)

    @staticmethod
    def _fill_year_candidates(s: Session) -> None:
        """从锚点时间收集去重年份列表（升序）。"""
        years: list[int] = []
        seen: set[int] = set()
        for it in s.anchor_items:
            dt = it.time_result.final_datetime
            if dt and dt.year not in seen:
                seen.add(dt.year)
                years.append(dt.year)
        s.year_candidates = sorted(years)


# ---------------------------------------------------------------------------
# 工具函数（模块级，供测试和其他模块复用）
# ---------------------------------------------------------------------------

def _item_folder(item: "Item") -> str:
    """取 item.relpath 的父目录路径字符串（统一用正斜杠）。"""
    return str(Path(item.relpath).parent).replace(os.sep, "/")


def _folder_overlap(a: str, b: str) -> int:
    """
    计算两个文件夹路径的前缀重叠字符数。

    以路径分隔符为边界，避免 "abc" 和 "abcdef" 被误判为高度重叠。

    示例：
        "旅行/成都"  vs "旅行/成都"  → 9  (完全相同)
        "旅行/成都"  vs "旅行/重庆"  → 3  ("旅行" 部分)
        "旅行"       vs "旅行/成都"  → 3  ("旅行" 部分)
        "工作"       vs "旅行/成都"  → 0
    """
    if not a or not b:
        return 0

    # 按路径分隔符切分，逐段比较
    parts_a = a.replace("\\", "/").split("/")
    parts_b = b.replace("\\", "/").split("/")

    overlap_chars = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            overlap_chars += len(pa) + 1   # +1 for separator
        else:
            break

    return overlap_chars


def _common_folder_prefix(folders: list[str]) -> str:
    """
    多个文件夹路径的最深公共前缀（以路径组件为粒度）。

    示例：
        ["旅行/成都", "旅行/成都", "旅行/成都/锦里"] → "旅行/成都"
        ["旅行/成都", "旅行/重庆"]                   → "旅行"
        ["a/b/c", "a/b/d", "a/e"]                    → "a"
        ["x", "y"]                                   → ""
    """
    if not folders:
        return ""
    if len(folders) == 1:
        return folders[0]

    # 统一分隔符
    split_folders = [
        f.replace("\\", "/").split("/")
        for f in folders
    ]

    common: list[str] = []
    for parts in zip(*split_folders):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break

    return "/".join(common)


# ---------------------------------------------------------------------------
# 独立测试（__main__）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import datetime, timedelta
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from core.item import Item, TimeResult
    from core.evidence import Evidence
    from session import SessionBuilder, _folder_overlap, _common_folder_prefix

    # ── 工具函数测试 ───────────────────────────────────────────────────
    print("=" * 60)
    print("工具函数测试")
    print("=" * 60)

    cases_overlap = [
        ("旅行/成都",  "旅行/成都",      True,  "完全相同"),
        ("旅行/成都",  "旅行/重庆",      False, "同父不同子"),
        ("旅行",       "旅行/成都",      False, "父子关系"),
        ("工作",       "旅行/成都",      False, "无重叠"),
        ("a/b/c",      "a/b/d",          False, "英文路径"),
    ]
    print("\n_folder_overlap：")
    for a, b, should_be_greater, desc in cases_overlap:
        ov = _folder_overlap(a, b)
        ov_rev = _folder_overlap(b, a)
        assert ov == ov_rev, f"不对称: {a!r} vs {b!r}"
        if should_be_greater:
            assert ov > 0, f"期望 >0: {desc}"
        print(f"  {a!r:15s} vs {b!r:15s} → {ov:3d}  ({desc})")

    print("\n_common_folder_prefix：")
    pfx_cases = [
        (["旅行/成都", "旅行/成都/锦里"],   "旅行/成都"),
        (["旅行/成都", "旅行/重庆"],         "旅行"),
        (["a/b/c", "a/b/d", "a/e"],          "a"),
        (["x", "y"],                          ""),
        (["single"],                          "single"),
    ]
    for folders, expected in pfx_cases:
        got = _common_folder_prefix(folders)
        status = "✓" if got == expected else f"✗ (期望 {expected!r})"
        print(f"  {folders} → {got!r} {status}")
        assert got == expected

    # ── SessionBuilder 测试 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SessionBuilder 测试")
    print("=" * 60)

    base_time = datetime(2019, 2, 13, 10, 0, 0)

    def _anchor(relpath: str, dt: datetime, conf: float = 1.0) -> Item:
        """构造有可信时间的锚点 item。"""
        item = Item(path=f"/fake/{relpath}", relpath=relpath)
        item.filename = Path(relpath).name
        item.mtime = dt.timestamp()
        item.time_result = TimeResult(
            final_datetime=dt, confidence=conf,
            precision="second", primary_source="exif"
        )
        return item

    def _floating(relpath: str, mtime_dt: datetime = None) -> Item:
        """构造无时间的浮动 item。"""
        item = Item(path=f"/fake/{relpath}", relpath=relpath)
        item.filename = Path(relpath).name
        if mtime_dt:
            item.mtime = mtime_dt.timestamp()
        return item

    builder = SessionBuilder(gap_hours=6.0, anchor_min_confidence=0.6)

    # 测试 1：纯锚点，两个 session
    print("\n[测试1] 纯锚点，gap=6h，应切分为 2 个 session")
    items1 = [
        _anchor("旅行/成都/IMG_001.jpg", base_time + timedelta(hours=0)),
        _anchor("旅行/成都/IMG_002.jpg", base_time + timedelta(hours=1)),
        _anchor("旅行/成都/IMG_003.jpg", base_time + timedelta(hours=2)),
        # gap > 6h
        _anchor("旅行/成都/IMG_004.jpg", base_time + timedelta(hours=10)),
        _anchor("旅行/成都/IMG_005.jpg", base_time + timedelta(hours=11)),
    ]
    s1 = builder.build(items1)
    print(f"  session 数量: {len(s1)}（期望 2）")
    for s in s1:
        print(f"    {s}")
    assert len(s1) == 2
    assert len(s1[0].items) == 3
    assert len(s1[1].items) == 2
    assert s1[0].year_candidates == [2019]
    print("  ✓")

    # 测试 2：混合，浮动按文件夹归属
    print("\n[测试2] 混合 item，浮动按文件夹前缀归属")
    items2 = [
        _anchor("旅行/成都/IMG_001.jpg", base_time),
        _floating("旅行/成都/IMG_002.jpg"),       # 同文件夹，应归入 s0
        _floating("旅行/成都/子目录/IMG_003.jpg"), # 子目录，应归入 s0
        _floating("工作/doc.jpg"),                 # 无重叠，归入唯一 session
    ]
    s2 = builder.build(items2)
    print(f"  session 数量: {len(s2)}（期望 1）")
    print(f"  session[0] items: {len(s2[0].items)}（期望 4）")
    for s in s2:
        print(f"    {s}")
    assert len(s2) == 1
    assert len(s2[0].items) == 4
    print("  ✓")

    # 测试 3：两个 session，浮动分别归属
    print("\n[测试3] 两个 session，浮动分别归属对应的 session")
    items3 = [
        _anchor("旅行/成都/IMG_001.jpg", base_time),
        _anchor("旅行/重庆/IMG_010.jpg", base_time + timedelta(hours=12)),
        _floating("旅行/成都/float_001.jpg"),  # 应归入成都 session
        _floating("旅行/重庆/float_010.jpg"),  # 应归入重庆 session
    ]
    s3 = builder.build(items3)
    print(f"  session 数量: {len(s3)}（期望 2）")
    for s in s3:
        anchors = [it.filename for it in s.anchor_items]
        all_items = [it.filename for it in s.items]
        print(f"    {s}")
        print(f"      anchors={anchors}, all={all_items}")
    assert len(s3) == 2
    # 成都 session 应有 2 个 item
    chengdu_session = next(s for s in s3 if "成都" in s.folder_path)
    chongqing_session = next(s for s in s3 if "重庆" in s.folder_path)
    assert len(chengdu_session.items) == 2, f"成都 session 应有 2 items，实际 {len(chengdu_session.items)}"
    assert len(chongqing_session.items) == 2
    print("  ✓")

    # 测试 4：全部浮动，建 fallback session
    print("\n[测试4] 全部无时间，建 fallback session s0")
    items4 = [
        _floating("照片/IMG_001.jpg"),
        _floating("照片/IMG_002.jpg"),
        _floating("文档/scan.jpg"),
    ]
    s4 = builder.build(items4)
    print(f"  session 数量: {len(s4)}（期望 1，fallback s0）")
    print(f"  {s4[0]}")
    assert len(s4) == 1
    assert s4[0].id == "s0"
    assert not s4[0].has_anchor
    assert len(s4[0].items) == 3
    print("  ✓")

    # 测试 5：year_candidates 跨年 session
    # 三张图间隔均 < 6h：23:00 → 00:30 → 01:00（跨年夜）
    print("\n[测试5] 跨年 session 的 year_candidates")
    items5 = [
        _anchor("年末/IMG_001.jpg", datetime(2019, 12, 31, 23, 0)),
        _anchor("年末/IMG_002.jpg", datetime(2020, 1,  1,  0, 30)),  # gap=1.5h
        _anchor("年末/IMG_003.jpg", datetime(2020, 1,  1,  1, 0)),   # gap=0.5h
    ]
    s5 = builder.build(items5)
    print(f"  session 数量: {len(s5)}（期望 1，间隔均 < 6h）")
    print(f"  year_candidates: {s5[0].year_candidates}（期望 [2019, 2020]）")
    assert len(s5) == 1, f"期望 1 个 session，实际 {len(s5)}"
    assert s5[0].year_candidates == [2019, 2020], f"实际 {s5[0].year_candidates}"
    print("  ✓")

    # 测试 6：低置信锚点被视为浮动
    print("\n[测试6] confidence=0.4 的 item 被视为浮动（低于阈值 0.6）")
    items6 = [
        _anchor("相册/IMG_001.jpg", base_time, conf=1.0),
        _anchor("相册/IMG_002.jpg", base_time + timedelta(hours=1), conf=0.4),  # 浮动
    ]
    s6 = builder.build(items6)
    print(f"  anchors: {len(s6[0].anchor_items)}（期望 1）")
    print(f"  all items: {len(s6[0].items)}（期望 2）")
    assert len(s6[0].anchor_items) == 1
    assert len(s6[0].items) == 2
    print("  ✓")

    print("\n" + "=" * 60)

    # ── 模式 B：读真实 snapshot ────────────────────────────────────────
    if len(sys.argv) > 1:
        snap_path = Path(sys.argv[1])
        print(f"\n模式 B：真实 snapshot — {snap_path}")
        print("=" * 60)

        from storage.snapshot import load
        items = load(snap_path)
        print(f"加载 {len(items)} 个 item")

        sessions = builder.build(items)
        print(f"聚类结果：{len(sessions)} 个 session\n")

        anchored_total = sum(len(s.anchor_items) for s in sessions)
        floating_total = sum(len(s.items) - len(s.anchor_items) for s in sessions)
        print(f"  锚点 item: {anchored_total}")
        print(f"  浮动 item: {floating_total}")

        print("\n各 session 详情：")
        for s in sessions:
            dur = s.duration
            dur_str = f"{dur.total_seconds()/3600:.1f}h" if dur else "—"
            print(
                f"  {s.id:4s}  items={len(s.items):4d}  "
                f"anchors={len(s.anchor_items):4d}  "
                f"dur={dur_str:8s}  "
                f"years={s.year_candidates}  "
                f"folder={s.folder_path!r}"
            )