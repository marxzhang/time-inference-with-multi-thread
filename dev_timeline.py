#!/usr/bin/env python3
"""
dev_timeline.py — TimelineStage 独立开发调试入口

从 snapshot 加载第一阶段产物，运行 Tokenizer → SessionBuilder →
StructResolver → Interpolator，输出详细报告。

不依赖第一阶段代码，秒级启动，反复修改 stages/timeline/ 后直接重跑。

典型用法
--------
    # 快速验证整体流程
    python dev_timeline.py --snapshot .cache/phase1.jsonl

    # 缩小范围
    python dev_timeline.py --snapshot .cache/phase1.jsonl --limit 200

    # 聚焦单个 session，显示详细消歧过程
    python dev_timeline.py --snapshot .cache/phase1.jsonl --session 2 --verbose

    # 只看 session 聚类结果，不做消歧
    python dev_timeline.py --snapshot .cache/phase1.jsonl --sessions-only

    # 过滤某个文件夹
    python dev_timeline.py --snapshot .cache/phase1.jsonl --folder "旅行/成都"

    # 接受真实 snapshot 路径作为位置参数（便捷写法）
    python dev_timeline.py .cache/phase1.jsonl --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

# 把项目根目录加入 sys.path（无论从哪里运行都能找到模块）
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dev_timeline",
        description="TimelineStage 独立调试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "snapshot_pos", nargs="?", default="",
        metavar="SNAPSHOT",
        help="snapshot 文件路径（位置参数，与 --snapshot 等价）",
    )
    p.add_argument(
        "--snapshot", "-s", default="",
        metavar="FILE",
        help="snapshot 文件路径（.cache/phase1.jsonl）",
    )
    p.add_argument(
        "--limit", "-n", type=int, default=0,
        metavar="N",
        help="只处理前 N 个 item（0=全部）",
    )
    p.add_argument(
        "--session", type=int, default=-1,
        metavar="IDX",
        help="只处理第 IDX 个 session（0-based，-1=全部）",
    )
    p.add_argument(
        "--folder", "-f", default="",
        metavar="PATH",
        help="只显示路径包含此字符串的 item",
    )
    p.add_argument(
        "--sessions-only", action="store_true",
        help="只显示 session 聚类结果，不运行消歧和插值",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示每个 token 的消歧细节和每个 item 的插值过程",
    )
    p.add_argument(
        "--gap-hours", type=float, default=6.0,
        metavar="H",
        help="session 切分的时间间隔（小时，默认 6.0）",
    )
    p.add_argument(
        "--anchor-conf", type=float, default=0.6,
        metavar="C",
        help="锚点最低 confidence（默认 0.6）",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # 合并两种传入方式
    snap_path_str = args.snapshot or args.snapshot_pos
    if not snap_path_str:
        print("错误：请提供 snapshot 文件路径", file=sys.stderr)
        print("  用法: python dev_timeline.py .cache/phase1.jsonl", file=sys.stderr)
        return 1

    snap_path = Path(snap_path_str)
    if not snap_path.exists():
        print(f"错误：snapshot 文件不存在: {snap_path}", file=sys.stderr)
        return 1

    t_start = time.monotonic()

    # ── 步骤 1：加载 snapshot ──────────────────────────────────────────
    _header("1. 加载 Snapshot")
    from storage.snapshot import load, stats as snap_stats

    s = snap_stats(snap_path)
    print(f"  文件: {snap_path}  ({s['file_size_mb']} MB)")

    items = load(snap_path)
    total_loaded = len(items)

    if args.limit > 0:
        items = items[:args.limit]
        print(f"  加载 {total_loaded} 个 item，限制处理前 {len(items)} 个")
    else:
        print(f"  加载 {total_loaded} 个 item")

    if args.folder:
        items = [it for it in items if args.folder in it.relpath]
        print(f"  文件夹过滤 {args.folder!r} → {len(items)} 个 item")

    if not items:
        print("  无 item 可处理，退出")
        return 0

    _print_snapshot_stats(items)

    # ── 步骤 2：Tokenizer ─────────────────────────────────────────────
    _header("2. Tokenizer")
    from stages.timeline.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    item_tokens: dict[str, tuple] = {}   # item.id → (direct, struct, semantic)
    struct_count = direct_extra_count = 0

    for item in items:
        d, s_toks, sem = tokenizer.extract(item)
        item_tokens[item.id] = (d, s_toks, sem)
        # direct 里来自 foldername_unambiguous 的是 tokenizer 新增的
        extra = [t for t in d if t.source_name == "foldername_unambiguous"]
        direct_extra_count += len(extra)
        struct_count += len(s_toks)

    print(f"  foldername DirectToken（新增低置信）: {direct_extra_count}")
    print(f"  StructToken（待消歧）              : {struct_count}")

    if struct_count > 0:
        _print_struct_token_samples(items, item_tokens, max_show=10)

    # ── 步骤 3：SessionBuilder ────────────────────────────────────────
    _header("3. SessionBuilder")
    from stages.timeline.session import SessionBuilder

    builder = SessionBuilder(
        gap_hours=args.gap_hours,
        anchor_min_confidence=args.anchor_conf,
    )
    sessions = builder.build(items)

    # 过滤单个 session
    if args.session >= 0:
        if args.session >= len(sessions):
            print(f"  错误：session {args.session} 不存在（共 {len(sessions)} 个）")
            return 1
        sessions = [sessions[args.session]]
        print(f"  聚焦 session[{args.session}]")

    _print_session_summary(sessions)

    if args.sessions_only:
        _footer(t_start)
        return 0

    # 把 StructToken 分配到各 session
    _assign_struct_tokens(items, sessions, item_tokens)

    # ── 步骤 4：StructResolver ────────────────────────────────────────
    _header("4. StructResolver")
    from stages.timeline.struct_resolver import StructResolver

    resolver = StructResolver()
    total_struct_resolved = 0
    total_struct_failed = 0

    for s in sessions:
        if not s.struct_tokens:
            continue
        n = resolver.resolve(
            s,
            min_confidence=args.anchor_conf,
            verbose=args.verbose,
        )
        for tok in s.struct_tokens:
            if tok.is_resolved:
                total_struct_resolved += 1
            else:
                total_struct_failed += 1

    print(f"  消歧成功: {total_struct_resolved}  失败/跳过: {total_struct_failed}")

    if args.verbose or total_struct_resolved > 0:
        _print_resolver_results(sessions)

    # ── 步骤 5：Interpolator ──────────────────────────────────────────
    _header("5. Interpolator")
    from stages.timeline.interpolator import Interpolator

    interp = Interpolator()

    # 记录插值前的状态（用于 before/after 对比）
    before_state = {
        it.id: (it.time_result.is_resolved, it.time_result.confidence)
        for it in items
    }

    total_interp = 0
    for s in sessions:
        n = interp.interpolate(s, verbose=args.verbose)
        total_interp += n

    print(f"  追加 timeline evidence 的 item 数: {total_interp}")
    print()
    print("  注意：只追加了 evidence，需要重跑 ResolverStage 才能更新 time_result")
    print("        在完整 pipeline 里，TimelineStage 之后 ResolverStage 会自动重跑")

    # ── 步骤 6：before/after 对比 ─────────────────────────────────────
    _header("6. 效果预估（需重跑 Resolver 才精确）")
    _print_before_after(items, before_state, item_tokens, verbose=args.verbose)

    _footer(t_start)
    return 0


# ---------------------------------------------------------------------------
# 报告函数
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _footer(t_start: float) -> None:
    elapsed = time.monotonic() - t_start
    print(f"\n{'=' * 60}")
    print(f"  完成  耗时 {elapsed:.2f}s")
    print(f"{'=' * 60}")


def _print_snapshot_stats(items: list) -> None:
    resolved   = sum(1 for it in items if it.time_result.is_resolved)
    has_exif   = sum(1 for it in items if it.exif.datetime_original)
    has_clip   = sum(1 for it in items if it.clip_embedding)
    unresolved = len(items) - resolved

    conf_bins: Counter = Counter()
    for it in items:
        if it.time_result.is_resolved:
            c = it.time_result.confidence
            if c >= 0.9:   conf_bins["≥0.9"] += 1
            elif c >= 0.7: conf_bins["0.7~0.9"] += 1
            elif c >= 0.5: conf_bins["0.5~0.7"] += 1
            else:          conf_bins["<0.5"] += 1

    print(f"  总计: {len(items)}  "
          f"resolved: {resolved}({resolved*100//len(items)}%)  "
          f"unresolved: {unresolved}")
    print(f"  has_exif: {has_exif}  has_clip: {has_clip}")
    print(f"  置信度分布: "
          + "  ".join(f"{k}:{v}" for k, v in sorted(conf_bins.items())))


def _print_struct_token_samples(items, item_tokens, max_show=10) -> None:
    print(f"\n  StructToken 样本（前 {max_show} 个）：")
    shown = 0
    for item in items:
        _, structs, _ = item_tokens.get(item.id, ([], [], []))
        for tok in structs:
            if shown >= max_show:
                return
            cands = [d.strftime("%Y-%m-%d") for d in tok.candidates]
            print(
                f"    [{tok.source_folder}层] "
                f"{item.relpath!r:45s} "
                f"folder={tok.folder_text!r:10s} "
                f"→ {tok.n_candidates} 候选: {cands}"
            )
            shown += 1


def _print_session_summary(sessions: list) -> None:
    anchored_total  = sum(len(s.anchor_items) for s in sessions)
    floating_total  = sum(len(s.items) - len(s.anchor_items) for s in sessions)
    no_anchor_count = sum(1 for s in sessions if not s.has_anchor)

    print(f"  session 数: {len(sessions)}  "
          f"（有锚点: {len(sessions)-no_anchor_count}  "
          f"无锚点: {no_anchor_count}）")
    print(f"  锚点 item: {anchored_total}  浮动 item: {floating_total}")
    print()

    for s in sessions:
        dur = s.duration
        dur_str = (
            f"{dur.total_seconds()/3600:.1f}h"
            if dur and dur.total_seconds() < 86400 * 7
            else (f"{dur.days}d" if dur else "—")
        )
        unresolved = s.n_unresolved
        print(
            f"  {s.id:4s}  "
            f"items={len(s.items):4d}  "
            f"anchors={len(s.anchor_items):4d}  "
            f"unresolved={unresolved:3d}  "
            f"dur={dur_str:8s}  "
            f"years={s.year_candidates}  "
            f"folder={s.folder_path!r}"
        )


def _assign_struct_tokens(items, sessions, item_tokens) -> None:
    """把 Tokenizer 产出的 StructToken 分配到对应的 session。"""
    # 建立 item_id → session 的映射
    item_to_session: dict[str, object] = {}
    for s in sessions:
        for it in s.items:
            item_to_session[it.id] = s

    for item in items:
        _, structs, _ = item_tokens.get(item.id, ([], [], []))
        if not structs:
            continue
        s = item_to_session.get(item.id)
        if s:
            s.struct_tokens.extend(structs)


def _print_resolver_results(sessions: list) -> None:
    resolved_toks = []
    failed_toks   = []

    for s in sessions:
        for tok in s.struct_tokens:
            if tok.is_resolved:
                resolved_toks.append((s, tok))
            else:
                failed_toks.append((s, tok))

    if resolved_toks:
        print(f"\n  消歧成功 ({len(resolved_toks)} 个)：")
        for s, tok in resolved_toks[:15]:
            cands = [d.strftime("%Y-%m-%d") for d in tok.candidates]
            print(
                f"    {s.id:4s}  {tok.folder_text!r:12s}  "
                f"候选={cands}  "
                f"→ {tok.resolved_dt.strftime('%Y-%m-%d')}  "
                f"precision={tok.resolved_precision}  "
                f"conf={tok.resolved_confidence:.2f}"
            )
        if len(resolved_toks) > 15:
            print(f"    ... 共 {len(resolved_toks)} 个")

    if failed_toks:
        print(f"\n  消歧失败/跳过 ({len(failed_toks)} 个，前 10 个）：")
        for s, tok in failed_toks[:10]:
            cands = [d.strftime("%Y-%m-%d") for d in tok.candidates]
            print(
                f"    {s.id:4s}  {tok.folder_text!r:12s}  "
                f"候选={cands}  "
                f"year_candidates={s.year_candidates}"
            )


def _print_before_after(items, before_state, item_tokens, verbose=False) -> None:
    """统计有多少 item 通过 timeline evidence 可能得到改善。"""
    newly_have_ev = []    # 之前无时间，现在有了 timeline evidence
    conf_boost    = []    # 之前有时间但置信度低，现在有更多证据

    for item in items:
        was_resolved, was_conf = before_state[item.id]
        tl_evs = [e for e in item.evidence if e.source == "timeline"]
        if not tl_evs:
            continue

        if not was_resolved:
            newly_have_ev.append(item)
        elif was_conf < 0.6:
            conf_boost.append(item)

    print(f"  预计新增 timeline evidence 后可改善：")
    print(f"    之前 unresolved，现有 evidence 可 resolve: {len(newly_have_ev)}")
    print(f"    之前 resolved 但低置信，现有补充证据:     {len(conf_boost)}")

    if verbose and newly_have_ev:
        print(f"\n  新增 evidence 的 item（前 20 个）：")
        for item in newly_have_ev[:20]:
            tl_evs = [e for e in item.evidence if e.source == "timeline"]
            best   = max(tl_evs, key=lambda e: e.confidence)
            mode   = best.metadata.get("interpolation_mode", "struct")
            print(
                f"    {item.relpath!r:50s}  "
                f"→ {best.dt.strftime('%Y-%m-%d')}  "
                f"precision={best.precision}  "
                f"conf={best.confidence:.2f}  [{mode}]"
            )
        if len(newly_have_ev) > 20:
            print(f"    ... 共 {len(newly_have_ev)} 个")

    # 统计 timeline evidence 的 precision 分布
    all_tl_evs = [
        e for item in items
        for e in item.evidence
        if e.source == "timeline"
    ]
    if all_tl_evs:
        prec_dist: Counter = Counter(e.precision for e in all_tl_evs)
        conf_dist: Counter = Counter()
        for e in all_tl_evs:
            if e.confidence >= 0.6:   conf_dist["≥0.6"] += 1
            elif e.confidence >= 0.4: conf_dist["0.4~0.6"] += 1
            else:                     conf_dist["<0.4"] += 1

        print(f"\n  timeline evidence 统计（共 {len(all_tl_evs)} 条）：")
        print(f"    precision: " +
              "  ".join(f"{k}:{v}" for k, v in sorted(prec_dist.items())))
        print(f"    confidence: " +
              "  ".join(f"{k}:{v}" for k, v in sorted(conf_dist.items())))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())