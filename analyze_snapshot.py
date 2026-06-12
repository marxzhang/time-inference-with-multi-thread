"""
analyze_snapshot.py — 分析 snapshot.jsonl 的数据分布

用法：
    python analyze_snapshot.py                          # 默认读 .cache/phase1.jsonl
    python analyze_snapshot.py --input /path/to/x.jsonl
    python analyze_snapshot.py --sample 5000            # 只读前 N 条（快速预览）

输出：
    1. 基础统计（总量、relpath 层级、字符长度）
    2. EXIF 质量分布（弱监督标签来源）
    3. 文本字符分布（语言、字符类型）
    4. 规则正则覆盖率（现有 patterns.py 能覆盖多少）
    5. 正负样本比例估算
    6. 高频 stem / 文件夹名 Top20（感受数据面貌）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="分析 snapshot.jsonl 数据分布")
    p.add_argument("--input", default=".cache/phase1.jsonl",
                   help="snapshot jsonl 路径（默认 .cache/phase1.jsonl）")
    p.add_argument("--sample", type=int, default=0,
                   help="只读前 N 条（0=全量）")
    p.add_argument("--top", type=int, default=20,
                   help="高频统计显示 Top N")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 内联简化版 match_all（不依赖项目模块，脚本独立运行）
# ---------------------------------------------------------------------------

# 用于判断某个文本片段是否含日期信息的轻量正则集合
_DATE_PATTERNS = [
    re.compile(r"(?<!\d)(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:[_\-](?:[01]\d|2[0-3])(?:[0-5]\d)(?:[0-5]\d))?(?!\d)"),  # YYYYMMDD[_HHmmss]
    re.compile(r"(?:19|20)\d{2}[-_.](0[1-9]|1[0-2])[-_.](?:0[1-9]|[12]\d|3[01])"),  # YYYY-MM-DD
    re.compile(r"(?:19|20)\d{2}年(?:0?[1-9]|1[0-2])月(?:0?[1-9]|[12]\d|3[01])日"),   # 中文全日期
    re.compile(r"(?<!\d)(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?!\d)"),                      # YYYYMM
    re.compile(r"(?:19|20)\d{2}[-_.](?:0[1-9]|1[0-2])(?![-_.\d])"),                   # YYYY-MM
    re.compile(r"(?:19|20)\d{2}年(?:0?[1-9]|1[0-2])月"),                              # 中文年月
    re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)"),                                        # 纯年份
    re.compile(r"(?:19|20)\d{2}年"),                                                   # 中文年份
]

def _has_date(text: str) -> bool:
    return any(p.search(text) for p in _DATE_PATTERNS)


# ---------------------------------------------------------------------------
# 字符类型判断
# ---------------------------------------------------------------------------

def _char_category(c: str) -> str:
    cat = unicodedata.category(c)
    name = unicodedata.name(c, "")
    if c.isdigit():
        return "digit"
    if c.isascii() and c.isalpha():
        return "latin"
    if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name or "HANGUL" in name:
        if "CJK" in name:
            return "cjk"
        if "HIRAGANA" in name or "KATAKANA" in name:
            return "japanese"
        return "korean"
    if cat in ("Po", "Pd", "Pc", "Pe", "Ps", "Pi", "Pf"):
        return "punct"
    if c in "_-. /\\()[]{}":
        return "separator"
    return "other"


def _classify_text(text: str) -> dict[str, int]:
    counts: dict[str, int] = Counter(_char_category(c) for c in text)
    return counts


# ---------------------------------------------------------------------------
# 主分析函数
# ---------------------------------------------------------------------------

def analyze(path: str, sample: int, top_n: int) -> None:
    fpath = Path(path)
    if not fpath.exists():
        print(f"[ERROR] 文件不存在: {fpath}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  分析目标: {fpath}")
    print(f"{'='*60}\n")

    # ── 读取数据 ──────────────────────────────────────────────────────
    items = []
    errors = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if sample and i >= sample:
                break
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1

    total = len(items)
    if total == 0:
        print("[ERROR] 没有读取到任何有效数据")
        sys.exit(1)

    print(f"读取: {total} 条  (解析错误: {errors} 条)\n")

    # ================================================================
    # 1. relpath 层级分布
    # ================================================================
    print("─" * 50)
    print("【1】relpath 层级数分布")
    print("─" * 50)

    depth_counter: Counter = Counter()
    all_folder_names: list[str] = []
    all_stems: list[str] = []
    all_segment_lengths: list[int] = []

    for item in items:
        relpath = item.get("relpath", "")
        parts = Path(relpath).parts   # ('folder1', 'folder2', 'file.jpg')
        depth = len(parts)
        depth_counter[depth] += 1

        # 收集文件夹名
        for folder in parts[:-1]:
            all_folder_names.append(folder)
            all_segment_lengths.append(len(folder))

        # 收集 stem
        if parts:
            stem = Path(parts[-1]).stem
            all_stems.append(stem)
            all_segment_lengths.append(len(stem))

    _print_counter_hist(depth_counter, label="层级数", top=10)
    print(f"\n  文件夹名 数量: {len(all_folder_names)}")
    print(f"  文件名 stem 数量: {len(all_stems)}")
    print(f"  所有片段 总数: {len(all_segment_lengths)}")
    if all_segment_lengths:
        print(f"  片段字符长度  avg={_avg(all_segment_lengths):.1f}  "
              f"max={max(all_segment_lengths)}  "
              f"p95={_percentile(all_segment_lengths, 95)}")

    # ================================================================
    # 2. EXIF 质量分布（弱监督标签来源）
    # ================================================================
    print("\n" + "─" * 50)
    print("【2】EXIF 时间质量分布（弱监督标签来源）")
    print("─" * 50)

    n_has_original    = 0
    n_has_digitized   = 0
    n_has_modify      = 0
    n_no_exif         = 0
    n_filesystem_only = 0
    confidence_vals: list[float] = []
    high_conf_threshold = 0.85

    for item in items:
        exif = item.get("exif", {})
        tr   = item.get("time_result", {})
        conf = tr.get("confidence", 0.0)
        confidence_vals.append(conf)

        has_orig = bool(exif.get("datetime_original"))
        has_dig  = bool(exif.get("datetime_digitized"))
        has_mod  = bool(exif.get("datetime_modify"))

        if has_orig:
            n_has_original += 1
        if has_dig:
            n_has_digitized += 1
        if has_mod:
            n_has_modify += 1

        flags = item.get("flags", [])
        if "NO_EXIF" in flags:
            n_no_exif += 1

        primary = tr.get("primary_source", "")
        if primary == "filesystem":
            n_filesystem_only += 1

    def pct(n): return f"{n/total*100:.1f}%"

    print(f"  有 DateTimeOriginal : {n_has_original:>6}  ({pct(n_has_original)})")
    print(f"  有 DateTimeDigitized: {n_has_digitized:>6}  ({pct(n_has_digitized)})")
    print(f"  有 DateTime(modify) : {n_has_modify:>6}  ({pct(n_has_modify)})")
    print(f"  NO_EXIF flag        : {n_no_exif:>6}  ({pct(n_no_exif)})")
    print(f"  filesystem only     : {n_filesystem_only:>6}  ({pct(n_filesystem_only)})")

    n_high_conf = sum(1 for c in confidence_vals if c >= high_conf_threshold)
    print(f"\n  confidence >= {high_conf_threshold}: {n_high_conf:>6}  ({pct(n_high_conf)})  ← 可用弱监督样本")
    print(f"  confidence 分布:")
    conf_buckets = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in conf_buckets:
        n = sum(1 for c in confidence_vals if lo <= c < hi)
        bar = "█" * int(n / total * 40)
        print(f"    [{lo:.1f}, {hi:.1f}): {n:>6}  {bar}")

    # ================================================================
    # 3. 字符类型分布
    # ================================================================
    print("\n" + "─" * 50)
    print("【3】文本字符类型分布")
    print("─" * 50)

    total_char_counts: Counter = Counter()
    for seg in all_folder_names + all_stems:
        total_char_counts.update(_classify_text(seg))

    grand = sum(total_char_counts.values()) or 1
    for cat, cnt in sorted(total_char_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / grand * 40)
        print(f"  {cat:<12}: {cnt:>8}  ({cnt/grand*100:.1f}%)  {bar}")

    # ================================================================
    # 4. 规则正则覆盖率
    # ================================================================
    print("\n" + "─" * 50)
    print("【4】现有正则规则覆盖率（片段级）")
    print("─" * 50)

    all_segs = all_folder_names + all_stems
    n_covered   = sum(1 for s in all_segs if _has_date(s))
    n_uncovered = len(all_segs) - n_covered

    print(f"  总片段数  : {len(all_segs)}")
    print(f"  有日期    : {n_covered:>6}  ({n_covered/len(all_segs)*100:.1f}%)")
    print(f"  无日期    : {n_uncovered:>6}  ({n_uncovered/len(all_segs)*100:.1f}%)")

    # item 级：至少一个片段含日期
    n_item_covered = 0
    n_item_exif_only = 0
    n_item_nothing = 0
    for item in items:
        relpath = item.get("relpath", "")
        parts = Path(relpath).parts
        segs = [Path(parts[-1]).stem] + list(parts[:-1]) if parts else []
        has_any = any(_has_date(s) for s in segs)
        has_exif = bool(item.get("exif", {}).get("datetime_original"))
        if has_any:
            n_item_covered += 1
        elif has_exif:
            n_item_exif_only += 1
        else:
            n_item_nothing += 1

    print(f"\n  item 级（至少一个片段含日期）:")
    print(f"    路径有日期              : {n_item_covered:>6}  ({pct(n_item_covered)})")
    print(f"    路径无日期 但有 EXIF    : {n_item_exif_only:>6}  ({pct(n_item_exif_only)})")
    print(f"    路径无日期 且无 EXIF    : {n_item_nothing:>6}  ({pct(n_item_nothing)})")

    # ================================================================
    # 5. 正负样本比估算（训练集视角）
    # ================================================================
    print("\n" + "─" * 50)
    print("【5】训练样本正负比估算")
    print("─" * 50)

    # 正样本：片段含日期 AND 对应 item 有高可信 EXIF（可生成标签）
    # 负样本：片段不含日期

    relpath_to_item = {}
    for item in items:
        relpath_to_item[item.get("relpath", "")] = item

    n_pos_segs = 0
    n_neg_segs = 0
    n_ambiguous = 0  # 片段含日期但 item 无高可信 EXIF，无法确认标签

    for item in items:
        relpath = item.get("relpath", "")
        parts = Path(relpath).parts
        segs = ([Path(parts[-1]).stem] + list(parts[:-1])) if parts else []

        has_high_conf_exif = (
            bool(item.get("exif", {}).get("datetime_original")) and
            item.get("time_result", {}).get("confidence", 0) >= high_conf_threshold
        )

        for seg in segs:
            if _has_date(seg):
                if has_high_conf_exif:
                    n_pos_segs += 1
                else:
                    n_ambiguous += 1
            else:
                n_neg_segs += 1

    total_labelable = n_pos_segs + n_neg_segs
    print(f"  正样本（有日期 + 高置信EXIF）: {n_pos_segs:>6}")
    print(f"  负样本（无日期）              : {n_neg_segs:>6}")
    print(f"  模糊样本（有日期但无高置信）  : {n_ambiguous:>6}  ← 暂跳过")
    if n_pos_segs > 0:
        ratio = n_neg_segs / n_pos_segs
        print(f"\n  正:负 ≈ 1:{ratio:.1f}", end="")
        if ratio > 10:
            print("  ⚠  严重不平衡，训练时需要加权或过采样")
        elif ratio > 5:
            print("  ⚠  不平衡，建议加权")
        else:
            print("  ✓  尚可接受")

    # ================================================================
    # 6. 高频片段 Top N
    # ================================================================
    print("\n" + "─" * 50)
    print(f"【6】高频文件夹名 Top{top_n}（感受数据面貌）")
    print("─" * 50)
    folder_counter: Counter = Counter(all_folder_names)
    for name, cnt in folder_counter.most_common(top_n):
        has_d = "📅" if _has_date(name) else "  "
        print(f"  {has_d} {cnt:>5}x  {name!r}")

    print(f"\n【6b】高频 stem Top{top_n}")
    print("─" * 50)
    stem_counter: Counter = Counter(all_stems)
    for name, cnt in stem_counter.most_common(top_n):
        has_d = "📅" if _has_date(name) else "  "
        print(f"  {has_d} {cnt:>5}x  {name!r}")

    # ================================================================
    # 7. stem 长度分布
    # ================================================================
    print("\n" + "─" * 50)
    print("【7】stem 字符长度分布（决定 max_seq_len）")
    print("─" * 50)
    stem_lens = [len(s) for s in all_stems]
    if stem_lens:
        buckets = [0, 10, 20, 30, 40, 50, 75, 100, 200, 9999]
        for lo, hi in zip(buckets, buckets[1:]):
            n = sum(1 for l in stem_lens if lo <= l < hi)
            bar = "█" * int(n / len(stem_lens) * 35)
            print(f"  [{lo:>3}, {hi if hi<9999 else '∞':>4}): {n:>6}  {bar}")
        print(f"\n  avg={_avg(stem_lens):.1f}  "
              f"p50={_percentile(stem_lens,50)}  "
              f"p95={_percentile(stem_lens,95)}  "
              f"p99={_percentile(stem_lens,99)}  "
              f"max={max(stem_lens)}")
        print(f"\n  → 建议 max_seq_len = {_percentile(stem_lens,99)}  "
              f"（覆盖 99% 的 stem，截断剩余 1%）")

    # ================================================================
    # 8. 扩展名分布（了解数据构成）
    # ================================================================
    print("\n" + "─" * 50)
    print("【8】文件扩展名分布")
    print("─" * 50)
    ext_counter: Counter = Counter()
    for item in items:
        ext = item.get("ext", "unknown")
        ext_counter[ext] += 1
    for ext, cnt in ext_counter.most_common(15):
        bar = "█" * int(cnt / total * 40)
        print(f"  .{ext:<8}: {cnt:>6}  ({cnt/total*100:.1f}%)  {bar}")

    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "=" * 60)
    print("  总结 & 下一步建议")
    print("=" * 60)
    print(f"  总 item 数          : {total}")
    print(f"  可用弱监督样本      : {n_high_conf}  ({pct(n_high_conf)})")
    print(f"  可生成正样本片段    : {n_pos_segs}")
    print(f"  负样本片段          : {n_neg_segs}")
    if all_segment_lengths:
        print(f"  建议 max_seq_len    : {_percentile(all_segment_lengths, 99)}")
    print()

    if n_high_conf < 1000:
        print("  ⚠  高置信 EXIF 样本不足 1000，弱监督质量可能受限")
        print("     → 考虑降低 confidence 阈值至 0.7，或引入文件名规则直接标注")
    elif n_high_conf < 5000:
        print("  ℹ  高置信样本在 1000-5000 区间，可训练但建议做数据增强")
    else:
        print("  ✓  高置信样本充足，可以直接训练")

    if n_pos_segs > 0 and n_neg_segs / n_pos_segs > 10:
        print("  ⚠  正负比超过 1:10，建议在 train.py 里对正样本 loss 加权 x3")

    print()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _percentile(vals: list, p: int) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s)-1)]

def _print_counter_hist(counter: Counter, label: str = "", top: int = 10) -> None:
    total = sum(counter.values())
    for val, cnt in sorted(counter.items())[:top]:
        bar = "█" * int(cnt / total * 30)
        print(f"  {label}={val}: {cnt:>6}  ({cnt/total*100:.1f}%)  {bar}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    analyze(args.input, args.sample, args.top)