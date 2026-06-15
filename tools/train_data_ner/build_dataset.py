"""
tools/train_date_ner/build_dataset.py

从 snapshot.jsonl 生成 MTL DateNER 训练数据。

输出：
    dataset.jsonl   每行一个样本（片段级）
    vocab.json      字符词表（训练 + 推理共用）
    stats.json      数据集统计摘要

样本格式（dataset.jsonl 每行）：
    {
        "text":       "IMG_20190823_143022",   # 原始文本片段
        "chars":      ["I","M","G","_",...],   # 字符列表
        "labels":     ["O","O","O","O",...],   # BIO 标签列表
        "clf_label":  1,                       # 0=无时间 1=有时间
        "source":     "stem",                  # "stem" | "folder"
        "item_id":    "uuid...",               # 来源 item（调试用）
        "relpath":    "...",                   # 来源路径（调试用）
        "exif_dt":    "2019-08-23T14:30:22",  # EXIF ground truth（调试用）
    }

标签集（13 个）：
    O  B-YEAR I-YEAR  B-MONTH I-MONTH  B-DAY I-DAY
       B-HOUR I-HOUR  B-MIN   I-MIN    B-SEC I-SEC

方案 B 的核心过滤逻辑：
    正样本：现有正则能精确匹配 AND 匹配日期与 item EXIF 一致（年月日相符）
    负样本：现有正则完全匹配不到任何日期（全 O，直接使用）
    丢弃：  正则匹配到但与 EXIF 日期不一致的片段（噪声，跳过）
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="生成 DateNER 训练数据集")
    p.add_argument("--input",   default="/home/marx/code/AlbumTimeFixSystem/.cache/phase1.jsonl",
                   help="snapshot jsonl 路径")
    p.add_argument("--out-dir", default="/home/marx/code/AlbumTimeFixSystem/tools/train_data_ner/data",
                   help="输出目录")
    p.add_argument("--min-conf", type=float, default=0.85,
                   help="item 最低 confidence，低于此值的 item 不生成正样本（默认 0.85）")
    p.add_argument("--val-ratio",  type=float, default=0.1,
                   help="验证集比例（默认 0.1）")
    p.add_argument("--test-ratio", type=float, default=0.1,
                   help="测试集比例（默认 0.1）")
    p.add_argument("--neg-ratio",  type=float, default=5.0,
                   help="负样本保留比例（相对于正样本，默认保留全部负样本，0=全保留）")
    p.add_argument("--max-seq-len", type=int, default=64,
                   help="片段最大字符数，超过则跳过（默认 64，覆盖所有实际数据）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 标签体系
# ---------------------------------------------------------------------------

LABELS = [
    "O",
    "B-YEAR",  "I-YEAR",
    "B-MONTH", "I-MONTH",
    "B-DAY",   "I-DAY",
    "B-HOUR",  "I-HOUR",
    "B-MIN",   "I-MIN",
    "B-SEC",   "I-SEC",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# 时间字段名 → BIO 前缀
_FIELD_TO_LABEL = {
    "year":   "YEAR",
    "month":  "MONTH",
    "day":    "DAY",
    "hour":   "HOUR",
    "minute": "MIN",
    "second": "SEC",
}


# ---------------------------------------------------------------------------
# 正则模式（复用 utils/patterns.py 的思路，但这里独立实现以脱离项目依赖）
# ---------------------------------------------------------------------------
# 每个模式描述：(name, compiled_regex, field_group_order)
# field_group_order: 按正则命名捕获组出现顺序列出字段名
# 用于在匹配后知道每个捕获组对应哪个时间字段

_RAW_PATTERNS = [
    # YYYYMMDD_HHmmss 或 YYYYMMDD-HHmmss（最常见）
    (
        "DATETIME8_6",
        re.compile(
            r"(?<!\d)"
            r"(?P<year>(?:19|20)\d{2})"
            r"(?P<month>0[1-9]|1[0-2])"
            r"(?P<day>0[1-9]|[12]\d|3[01])"
            r"[_\-]"
            r"(?P<hour>[01]\d|2[0-3])"
            r"(?P<minute>[0-5]\d)"
            r"(?P<second>[0-5]\d)"
            r"(?!\d)"
        ),
        ["year", "month", "day", "hour", "minute", "second"],
    ),
    # YYYYMMDD HHmmss（空格分隔）
    (
        "DATETIME8_6_SPC",
        re.compile(
            r"(?<!\d)"
            r"(?P<year>(?:19|20)\d{2})"
            r"(?P<month>0[1-9]|1[0-2])"
            r"(?P<day>0[1-9]|[12]\d|3[01])"
            r" "
            r"(?P<hour>[01]\d|2[0-3])"
            r"(?P<minute>[0-5]\d)"
            r"(?P<second>[0-5]\d)"
            r"(?!\d)"
        ),
        ["year", "month", "day", "hour", "minute", "second"],
    ),
    # YYYY-MM-DD HH-mm-ss / YYYY_MM_DD_HH_mm_ss 等分隔
    (
        "DATETIME_SEP",
        re.compile(
            r"(?P<year>(?:19|20)\d{2})"
            r"(?P<sep1>[-_.T])"
            r"(?P<month>0[1-9]|1[0-2])"
            r"(?P=sep1)"
            r"(?P<day>0[1-9]|[12]\d|3[01])"
            r"[-_. T]"
            r"(?P<hour>[01]\d|2[0-3])"
            r"[-_.:h]"
            r"(?P<minute>[0-5]\d)"
            r"[-_.:m]"
            r"(?P<second>[0-5]\d)"
        ),
        ["year", "month", "day", "hour", "minute", "second"],
    ),
    # YYYYMMDD（纯日期，无时间）
    (
        "DATE8",
        re.compile(
            r"(?<!\d)"
            r"(?P<year>(?:19|20)\d{2})"
            r"(?P<month>0[1-9]|1[0-2])"
            r"(?P<day>0[1-9]|[12]\d|3[01])"
            r"(?!\d)"
        ),
        ["year", "month", "day"],
    ),
    # YYYY-MM-DD / YYYY.MM.DD / YYYY_MM_DD
    (
        "DATE_SEP",
        re.compile(
            r"(?P<year>(?:19|20)\d{2})"
            r"(?P<sep1>[-_.])"
            r"(?P<month>0[1-9]|1[0-2])"
            r"(?P=sep1)"
            r"(?P<day>0[1-9]|[12]\d|3[01])"
            r"(?!\d)"
        ),
        ["year", "month", "day"],
    ),
    # YYYY年MM月DD日（中文）
    (
        "DATE_CN",
        re.compile(
            r"(?P<year>(?:19|20)\d{2})年"
            r"(?P<month>0?[1-9]|1[0-2])月"
            r"(?P<day>0?[1-9]|[12]\d|3[01])日"
        ),
        ["year", "month", "day"],
    ),
    # YYYY年MM月（中文年月）
    (
        "YEARMONTH_CN",
        re.compile(
            r"(?P<year>(?:19|20)\d{2})年"
            r"(?P<month>0?[1-9]|1[0-2])月"
        ),
        ["year", "month"],
    ),
    # YYYY-MM / YYYY.MM
    (
        "YEARMONTH_SEP",
        re.compile(
            r"(?P<year>(?:19|20)\d{2})"
            r"[-_.]"
            r"(?P<month>0[1-9]|1[0-2])"
            r"(?![-_.\d])"
        ),
        ["year", "month"],
    ),
    # YYYYMM（连续年月，需排除被 DATE8 捕获的情况）
    # 注意：这个模式会被 DATE8 的 finditer 优先匹配，此处作兜底
    (
        "YEARMONTH6",
        re.compile(
            r"(?<!\d)"
            r"(?P<year>(?:19|20)\d{2})"
            r"(?P<month>0[1-9]|1[0-2])"
            r"(?!\d)"
        ),
        ["year", "month"],
    ),
    # YYYY年（中文纯年份）
    (
        "YEAR_CN",
        re.compile(r"(?P<year>(?:19|20)\d{2})年"),
        ["year"],
    ),
    # 纯四位年份（最后兜底，精度最低）
    (
        "YEAR4",
        re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})(?!\d)"),
        ["year"],
    ),
]

# 精度排序（越大越精确）
_PREC_RANK = {
    "year": 1, "month": 2, "day": 3,
    "hour": 4, "minute": 5, "second": 6,
}

def _pattern_precision(fields: list[str]) -> int:
    return max(_PREC_RANK.get(f, 0) for f in fields)


# ---------------------------------------------------------------------------
# 核心：对一个文本片段生成 BIO 标签
# ---------------------------------------------------------------------------

class SegmentLabeler:
    """
    对单个文本片段（文件名 stem 或文件夹名）生成 BIO 标签序列。

    流程：
        1. 遍历所有正则，收集所有匹配（可能有重叠）
        2. 用 EXIF ground truth 过滤：只保留与 EXIF 日期一致的匹配
        3. 去重叠：多个匹配重叠时，保留精度最高的那个
        4. 将匹配 span 内的字符按捕获组映射到 BIO 标签
        5. 未被任何匹配覆盖的字符标为 O
    """

    def label(
        self,
        text: str,
        exif_dt: Optional[datetime],
        min_conf_for_positive: float,
        item_conf: float,
        source: str,  # "stem" | "folder"
    ) -> Optional[dict]:
        """
        返回标注结果 dict，或 None（表示应跳过此片段）。

        返回 None 的情况：
            - 有正则匹配，但与 EXIF 不一致（噪声，丢弃）
            - 文本为空

        返回全 O 的情况（合法负样本）：
            - 没有任何正则命中
        """
        if not text:
            return None

        chars = list(text)
        n = len(chars)

        # ── 步骤1：收集所有正则匹配 ─────────────────────────────────
        candidates = []  # (start, end, fields_dict, pattern_name, precision)
        for pat_name, pat, field_order in _RAW_PATTERNS:
            for m in pat.finditer(text):
                groups = m.groupdict()
                # 提取有效的时间字段（排除辅助组如 sep1）
                fields = {
                    f: groups[f]
                    for f in field_order
                    if f in groups and groups[f] is not None
                    and f in _FIELD_TO_LABEL
                }
                if not fields:
                    continue
                prec = _pattern_precision(list(fields.keys()))
                candidates.append((m.start(), m.end(), fields, pat_name, prec, m))

        # ── 步骤2：无匹配 → 全 O 负样本 ─────────────────────────────
        if not candidates:
            return self._make_result(text, chars, ["O"] * n, 0, source)

        # ── 步骤3：与 EXIF 做一致性验证 ────────────────────────────
        # 只有 item_conf >= min_conf_for_positive 时才做正样本
        # 否则有匹配但无法验证 → 丢弃（返回 None）
        if item_conf < min_conf_for_positive or exif_dt is None:
            # 不能验证 → 跳过（不生成负样本，因为可能含日期信息）
            return None

        valid_candidates = []
        for cand in candidates:
            start, end, fields, pat_name, prec, m = cand
            if self._is_consistent(fields, exif_dt):
                valid_candidates.append(cand)

        # 有匹配但全部不一致 → 噪声片段，丢弃
        if not valid_candidates and candidates:
            return None

        # 全部一致 → 生成正样本
        candidates = valid_candidates

        # ── 步骤4：去重叠，保留精度最高的匹配 ──────────────────────
        # 按精度降序，再按匹配长度降序（更长的优先）
        candidates.sort(key=lambda c: (c[4], c[1] - c[0]), reverse=True)
        selected = []
        occupied = [False] * n
        for cand in candidates:
            start, end = cand[0], cand[1]
            if any(occupied[i] for i in range(start, end)):
                continue  # 与已选匹配重叠，跳过
            selected.append(cand)
            for i in range(start, end):
                occupied[i] = True

        if not selected:
            # 理论上不会到这里，保险起见返回全 O
            return self._make_result(text, chars, ["O"] * n, 0, source)

        # ── 步骤5：将 span 映射为 BIO 标签 ─────────────────────────
        labels = ["O"] * n
        for start, end, fields, pat_name, prec, m in selected:
            self._fill_bio(text, labels, m, fields)

        clf_label = 1 if any(l != "O" for l in labels) else 0
        return self._make_result(text, chars, labels, clf_label, source)

    def _is_consistent(
        self,
        fields: dict[str, str],
        exif_dt: datetime,
    ) -> bool:
        """
        判断匹配到的时间字段是否与 EXIF 日期一致。

        一致的定义：
            - year 存在 → 必须与 exif_dt.year 一致
            - month 存在 → 必须与 exif_dt.month 一致
            - day 存在 → 必须与 exif_dt.day 一致
            - hour/minute/second 不做强制（文件名时间精度可能只到天）

        容忍：
            - 文件夹名含年份与 EXIF 年份一致即可（月日可以不一致）
            - 因为文件夹往往只有粗粒度年份
        """
        try:
            if "year" in fields:
                if int(fields["year"]) != exif_dt.year:
                    return False
            if "month" in fields:
                if int(fields["month"]) != exif_dt.month:
                    return False
            if "day" in fields:
                if int(fields["day"]) != exif_dt.day:
                    return False
            return True
        except (ValueError, TypeError):
            return False

    def _fill_bio(
        self,
        text: str,
        labels: list[str],
        match: re.Match,
        fields: dict[str, str],
    ) -> None:
        """
        将正则匹配的各捕获组位置填入 BIO 标签。

        对每个有效时间字段（year/month/day/...）：
            - 找到该组在原始字符串中的 span
            - 第一个字符标 B-FIELD，后续字符标 I-FIELD
        """
        for field_name, field_val in fields.items():
            label_base = _FIELD_TO_LABEL.get(field_name)
            if label_base is None:
                continue
            try:
                span = match.span(field_name)
            except IndexError:
                continue
            if span[0] < 0:
                continue
            start, end = span
            if start >= end:
                continue
            labels[start] = f"B-{label_base}"
            for i in range(start + 1, end):
                labels[i] = f"I-{label_base}"

    @staticmethod
    def _make_result(
        text: str,
        chars: list[str],
        labels: list[str],
        clf_label: int,
        source: str,
    ) -> dict:
        return {
            "text":      text,
            "chars":     chars,
            "labels":    labels,
            "clf_label": clf_label,
            "source":    source,
        }


# ---------------------------------------------------------------------------
# 词表构建
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<PAD>", "<UNK>"]

def build_vocab(samples: list[dict], min_freq: int = 1) -> dict[str, int]:
    """
    从所有样本的 chars 里统计字符频次，构建词表。
    低频字符（< min_freq）统一映射到 <UNK>。
    """
    freq: Counter = Counter()
    for s in samples:
        freq.update(s["chars"])

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for char, cnt in freq.most_common():
        if cnt < min_freq:
            break
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


# ---------------------------------------------------------------------------
# 数据集划分（按 item_id hash，保证同一 item 的片段不跨集合）
# ---------------------------------------------------------------------------

def split_by_item(
    samples: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    按 item_id 划分，保证同一 item 的所有片段都在同一个集合里。
    item_id 不存在时（负样本可能没有 item_id）按样本本身 hash。
    """
    # 收集所有唯一 item_id
    item_ids = list({s.get("item_id", s["text"]) for s in samples})
    rng = random.Random(seed)
    rng.shuffle(item_ids)

    n = len(item_ids)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_ids  = set(item_ids[:n_test])
    val_ids   = set(item_ids[n_test: n_test + n_val])

    train, val, test = [], [], []
    for s in samples:
        iid = s.get("item_id", s["text"])
        if iid in test_ids:
            test.append(s)
        elif iid in val_ids:
            val.append(s)
        else:
            train.append(s)

    return train, val, test


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] 找不到输入文件: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"读取: {input_path}")

    # ── 读取 snapshot ────────────────────────────────────────────────
    raw_items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"共 {len(raw_items)} 条 item")

    labeler = SegmentLabeler()

    # 统计计数器
    stats = {
        "total_items":       len(raw_items),
        "total_segments":    0,
        "pos_samples":       0,
        "neg_samples":       0,
        "discarded_noise":   0,
        "discarded_no_conf": 0,
        "discarded_toolong": 0,
        "label_freq":        Counter(),
        "source_freq":       Counter(),
        "pattern_hit":       Counter(),
    }

    all_samples: list[dict] = []

    # ── 遍历 item，提取片段并标注 ─────────────────────────────────
    for item in raw_items:
        relpath  = item.get("relpath", "")
        item_id  = item.get("id", "")
        item_conf = item.get("time_result", {}).get("confidence", 0.0)

        # 解析 EXIF ground truth
        exif_dt_str = item.get("exif", {}).get("datetime_original")
        exif_dt: Optional[datetime] = None
        if exif_dt_str:
            try:
                exif_dt = datetime.fromisoformat(exif_dt_str)
            except ValueError:
                pass

        # 拆分路径为片段
        parts = Path(relpath).parts  # ('folder1', ..., 'file.ext')
        if not parts:
            continue

        # stem（文件名去掉扩展名）
        stem = Path(parts[-1]).stem
        # 文件夹名（从最近到最远，最多 3 层）
        folder_names = list(parts[:-1])[-3:]  # 最近的 3 级文件夹

        segments: list[tuple[str, str]] = []  # (text, source)
        if stem:
            segments.append((stem, "stem"))
        for fname in reversed(folder_names):  # 从最近的父级开始
            if fname:
                segments.append((fname, "folder"))

        stats["total_segments"] += len(segments)

        for text, source in segments:
            # 过滤过长片段
            if len(text) > args.max_seq_len:
                stats["discarded_toolong"] += 1
                continue

            result = labeler.label(
                text=text,
                exif_dt=exif_dt,
                min_conf_for_positive=args.min_conf,
                item_conf=item_conf,
                source=source,
            )

            if result is None:
                # 区分是"噪声丢弃"还是"置信度不足"
                # 判断方法：看是否有正则匹配
                has_match = any(
                    p.search(text)
                    for _, p, _ in _RAW_PATTERNS
                )
                if has_match:
                    if item_conf < args.min_conf or exif_dt is None:
                        stats["discarded_no_conf"] += 1
                    else:
                        stats["discarded_noise"] += 1
                # 无匹配且被丢弃 → 不应该发生（无匹配应返回全O），记录一下
                continue

            # 补充 meta 字段（调试用）
            result["item_id"] = item_id
            result["relpath"] = relpath
            if exif_dt:
                result["exif_dt"] = exif_dt.isoformat()

            # 统计
            if result["clf_label"] == 1:
                stats["pos_samples"] += 1
            else:
                stats["neg_samples"] += 1
            stats["source_freq"][source] += 1
            for lbl in result["labels"]:
                stats["label_freq"][lbl] += 1

            all_samples.append(result)

    print(f"\n生成样本: {len(all_samples)}")
    print(f"  正样本: {stats['pos_samples']}")
    print(f"  负样本: {stats['neg_samples']}")
    print(f"  丢弃（与EXIF不一致）: {stats['discarded_noise']}")
    print(f"  丢弃（置信度不足）  : {stats['discarded_no_conf']}")
    print(f"  丢弃（片段过长）    : {stats['discarded_toolong']}")

    if not all_samples:
        print("[ERROR] 没有生成任何样本，请检查输入数据", file=sys.stderr)
        sys.exit(1)

    # ── 负样本下采样（可选）────────────────────────────────────────
    # 正负比 1:3.3 已经不平衡，不再额外采样，保留全部
    # 如果未来数据集变大导致负样本过多，可以在这里按 args.neg_ratio 采样

    # ── 打乱顺序 ────────────────────────────────────────────────────
    random.shuffle(all_samples)

    # ── 按 item_id 划分数据集 ────────────────────────────────────────
    train, val, test = split_by_item(
        all_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"\n数据集划分:")
    print(f"  train: {len(train)}")
    print(f"  val  : {len(val)}")
    print(f"  test : {len(test)}")

    # ── 构建词表（只用训练集，避免测试集信息泄露）─────────────────
    vocab = build_vocab(train, min_freq=1)
    print(f"\n词表大小: {len(vocab)}  (含 <PAD> <UNK>)")

    # ── 写入文件 ─────────────────────────────────────────────────────
    def write_jsonl(samples: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val,   out_dir / "val.jsonl")
    write_jsonl(test,  out_dir / "test.jsonl")

    # vocab.json：同时保存 label 体系，推理时一起加载
    vocab_data = {
        "char2id":  vocab,
        "id2char":  {str(v): k for k, v in vocab.items()},
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "special_tokens": SPECIAL_TOKENS,
        "pad_id":  vocab["<PAD>"],
        "unk_id":  vocab["<UNK>"],
        "num_labels": len(LABELS),
    }
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    # stats.json
    stats_data = {
        "total_items":       stats["total_items"],
        "total_segments":    stats["total_segments"],
        "pos_samples":       stats["pos_samples"],
        "neg_samples":       stats["neg_samples"],
        "discarded_noise":   stats["discarded_noise"],
        "discarded_no_conf": stats["discarded_no_conf"],
        "discarded_toolong": stats["discarded_toolong"],
        "train_size":        len(train),
        "val_size":          len(val),
        "test_size":         len(test),
        "vocab_size":        len(vocab),
        "label_freq":        dict(stats["label_freq"]),
        "source_freq":       dict(stats["source_freq"]),
        "pos_neg_ratio":     (
            stats["neg_samples"] / stats["pos_samples"]
            if stats["pos_samples"] > 0 else 0
        ),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)

    # ── 打印标签分布 ─────────────────────────────────────────────────
    print("\n标签频次分布（训练集视角）:")
    total_chars = sum(stats["label_freq"].values()) or 1
    for lbl in LABELS:
        cnt = stats["label_freq"].get(lbl, 0)
        bar = "█" * int(cnt / total_chars * 30)
        print(f"  {lbl:<10}: {cnt:>8}  ({cnt/total_chars*100:.2f}%)  {bar}")

    # ── 打印几条正样本样例（肉眼验证标注是否正确）─────────────────
    print("\n── 正样本样例（前5条）──")
    shown = 0
    for s in all_samples:
        if s["clf_label"] == 1 and shown < 5:
            print(f"  text  : {s['text']!r}")
            print(f"  labels: {s['labels']}")
            if "exif_dt" in s:
                print(f"  exif  : {s['exif_dt']}")
            print()
            shown += 1

    print(f"输出目录: {out_dir.resolve()}")
    print("完成。")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()