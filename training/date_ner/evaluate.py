"""
tools/train_date_ner/evaluate.py

在测试集上评估已训练的 DateNER 模型，输出详细指标和错误样本分析。

用法（以包方式运行，项目根目录下执行）：
    python -m training.date_ner.evaluate
    python -m training.date_ner.evaluate \
        --weights weights/date_ner.pt \
        --vocab   weights/vocab.json \
        --test    training/date_ner/data/test.jsonl \
        --errors  100        # 打印前 N 条错误样本
        --out     weights/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# 绝对包导入：要求以 `python -m training.date_ner.evaluate` 方式运行。
from models.date_ner import (
    DateNERModel, ModelConfig, VocabHelper, load_model, _decode_bio_spans,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="评估 DateNER 模型")
    p.add_argument("--weights",    default="weights/date_ner.pt")
    p.add_argument("--vocab",      default="weights/vocab.json")
    p.add_argument("--test",       default="/home/marx/code/AlbumTimeFixSystem/training/date_ner/data/test.jsonl")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--errors",     type=int, default=30,
                   help="打印多少条错误样本（0=不打印）")
    p.add_argument("--out",        default="weights/eval_report.json",
                   help="将完整评估报告写入此 JSON 文件")
    p.add_argument("--device",     default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# span 提取（与 train.py 保持一致）
# ---------------------------------------------------------------------------

def extract_spans(label_ids: list[int], id2label: dict[int, str]) -> set[tuple]:
    spans = set()
    cur_field: Optional[str] = None
    cur_start: int = 0

    for pos, lid in enumerate(label_ids):
        lbl = id2label.get(lid, "O")
        if lbl.startswith("B-"):
            if cur_field is not None:
                spans.add((cur_start, pos, cur_field))
            cur_field = lbl[2:]
            cur_start = pos
        elif lbl.startswith("I-"):
            field = lbl[2:]
            if field != cur_field:
                if cur_field is not None:
                    spans.add((cur_start, pos, cur_field))
                cur_field = field
                cur_start = pos
        else:
            if cur_field is not None:
                spans.add((cur_start, pos, cur_field))
            cur_field = None

    if cur_field is not None:
        spans.add((cur_start, len(label_ids), cur_field))
    return spans


# ---------------------------------------------------------------------------
# 批量推理
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:      DateNERModel,
    samples:    list[dict],
    vocab:      VocabHelper,
    device:     torch.device,
    batch_size: int,
) -> list[dict]:
    """
    对样本列表做批量推理，返回每条样本附加预测结果的字典列表。
    结果字段：
        pred_clf      : int   0/1
        pred_clf_prob : float
        pred_labels   : list[str]  NER 预测标签序列（与 chars 等长）
        pred_spans    : set[tuple] (start, end, field)
        gold_spans    : set[tuple]
    """
    model.eval()
    results = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i: i + batch_size]

        texts   = ["".join(s["chars"]) for s in batch]
        lengths = [min(len(s["chars"]), len(s["labels"])) for s in batch]
        max_len = max(lengths)

        padded = torch.full((len(batch), max_len), vocab.pad_id, dtype=torch.long)
        for j, s in enumerate(batch):
            L = lengths[j]
            ids = [vocab.char2id.get(c, vocab.unk_id) for c in s["chars"][:L]]
            padded[j, :L] = torch.tensor(ids, dtype=torch.long)

        len_t = torch.tensor(lengths, dtype=torch.long)
        clf_logits, ner_logits = model(padded.to(device), len_t.to(device))

        clf_probs  = F.softmax(clf_logits, dim=-1)[:, 1].cpu().tolist()
        ner_preds  = ner_logits.argmax(dim=-1).cpu().tolist()  # [B, T]

        for j, s in enumerate(batch):
            L          = lengths[j]
            pred_ids   = ner_preds[j][:L]
            gold_ids   = [vocab.label2id.get(lbl, 0) for lbl in s["labels"][:L]]

            pred_labels = [vocab.id2label.get(lid, "O") for lid in pred_ids]
            pred_spans  = extract_spans(pred_ids, vocab.id2label)
            gold_spans  = extract_spans(gold_ids, vocab.id2label)

            results.append({
                **s,
                "pred_clf":      int(clf_probs[j] >= 0.4),
                "pred_clf_prob": round(clf_probs[j], 4),
                "pred_labels":   pred_labels,
                "pred_spans":    pred_spans,
                "gold_spans":    gold_spans,
                "text":          texts[j],
            })

    return results


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

ALL_FIELDS = ["YEAR", "MONTH", "DAY", "HOUR", "MIN", "SEC"]

def compute_metrics(results: list[dict]) -> dict:
    # ── NER span-level ────────────────────────────────────────────
    tp_by  = Counter()
    fp_by  = Counter()
    fn_by  = Counter()

    for r in results:
        pred = r["pred_spans"]
        gold = r["gold_spans"]
        for span in pred:
            field = span[2]
            if span in gold:
                tp_by[field] += 1
            else:
                fp_by[field] += 1
        for span in gold:
            field = span[2]
            if span not in pred:
                fn_by[field] += 1

    def prf(tp, fp, fn):
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        return prec, rec, f1

    field_metrics = {}
    for f in ALL_FIELDS:
        tp, fp, fn = tp_by[f], fp_by[f], fn_by[f]
        p, r, f1   = prf(tp, fp, fn)
        field_metrics[f] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
        }

    all_tp = sum(tp_by.values())
    all_fp = sum(fp_by.values())
    all_fn = sum(fn_by.values())
    micro_p, micro_r, micro_f1 = prf(all_tp, all_fp, all_fn)

    # ── 分类头 ────────────────────────────────────────────────────
    clf_tp = clf_fp = clf_fn = clf_tn = 0
    for r in results:
        pred = r["pred_clf"]
        gold = r["clf_label"]
        if pred == 1 and gold == 1:   clf_tp += 1
        elif pred == 1 and gold == 0: clf_fp += 1
        elif pred == 0 and gold == 1: clf_fn += 1
        else:                         clf_tn += 1

    clf_p, clf_r, clf_f1 = prf(clf_tp, clf_fp, clf_fn)
    clf_acc = (clf_tp + clf_tn) / max(len(results), 1)

    # ── 端到端：datetime 正确解析率 ──────────────────────────────
    # 定义：gold 有 YEAR+MONTH+DAY 的样本里，pred 也全部正确
    n_full_gold = n_full_correct = 0
    for r in results:
        gold_fields = {s[2] for s in r["gold_spans"]}
        if {"YEAR", "MONTH", "DAY"}.issubset(gold_fields):
            n_full_gold += 1
            pred_fields = {s[2] for s in r["pred_spans"]}
            gold_ymd = {s for s in r["gold_spans"] if s[2] in {"YEAR","MONTH","DAY"}}
            pred_ymd = {s for s in r["pred_spans"] if s[2] in {"YEAR","MONTH","DAY"}}
            if gold_ymd == pred_ymd:
                n_full_correct += 1

    e2e_acc = n_full_correct / max(n_full_gold, 1)

    return {
        "ner_micro": {
            "precision": round(micro_p, 4),
            "recall":    round(micro_r, 4),
            "f1":        round(micro_f1, 4),
            "tp": all_tp, "fp": all_fp, "fn": all_fn,
        },
        "ner_by_field":  field_metrics,
        "clf": {
            "accuracy":  round(clf_acc, 4),
            "precision": round(clf_p, 4),
            "recall":    round(clf_r, 4),
            "f1":        round(clf_f1, 4),
            "tp": clf_tp, "fp": clf_fp, "fn": clf_fn, "tn": clf_tn,
        },
        "e2e": {
            "ymd_correct":    n_full_correct,
            "ymd_total_gold": n_full_gold,
            "ymd_accuracy":   round(e2e_acc, 4),
        },
        "total_samples": len(results),
    }


# ---------------------------------------------------------------------------
# 错误分析
# ---------------------------------------------------------------------------

def collect_errors(results: list[dict]) -> list[dict]:
    """收集所有预测与 gold 不一致的样本。"""
    errors = []
    for r in results:
        if r["pred_spans"] != r["gold_spans"] or r["pred_clf"] != r["clf_label"]:
            # 分类错误类型
            error_types = []
            if r["pred_clf"] != r["clf_label"]:
                error_types.append(
                    "clf_FP" if r["pred_clf"] == 1 else "clf_FN"
                )
            extra  = r["pred_spans"] - r["gold_spans"]
            missed = r["gold_spans"] - r["pred_spans"]
            if extra:
                error_types.append(f"ner_FP:{[s[2] for s in extra]}")
            if missed:
                error_types.append(f"ner_FN:{[s[2] for s in missed]}")

            errors.append({
                "text":        r["text"],
                "source":      r.get("source", "?"),
                "gold_labels": r["labels"],
                "pred_labels": r["pred_labels"],
                "gold_spans":  sorted(r["gold_spans"]),
                "pred_spans":  sorted(r["pred_spans"]),
                "gold_clf":    r["clf_label"],
                "pred_clf":    r["pred_clf"],
                "pred_clf_prob": r["pred_clf_prob"],
                "error_types": error_types,
                "relpath":     r.get("relpath", ""),
            })
    return errors


def print_errors(errors: list[dict], n: int) -> None:
    if not errors:
        print("  （无错误样本）")
        return

    # 按错误类型分组统计
    type_counter: Counter = Counter()
    for e in errors:
        for t in e["error_types"]:
            type_counter[t] += 1
    print(f"  错误类型分布:")
    for t, cnt in type_counter.most_common():
        print(f"    {t}: {cnt}")

    print(f"\n  错误样本（前 {min(n, len(errors))} 条）:")
    for e in errors[:n]:
        print(f"  ── {'stem' if e['source']=='stem' else 'folder'} ──")
        print(f"    text      : {e['text']!r}")
        print(f"    gold_spans: {e['gold_spans']}")
        print(f"    pred_spans: {e['pred_spans']}")
        if e["gold_clf"] != e["pred_clf"]:
            print(f"    clf: gold={e['gold_clf']} pred={e['pred_clf']} "
                  f"(prob={e['pred_clf_prob']:.4f})")
        print(f"    类型: {e['error_types']}")
        # 对齐展示标签差异
        chars      = list(e["text"])
        gold_lbls  = e["gold_labels"]
        pred_lbls  = e["pred_labels"]
        diff_pos   = [i for i in range(min(len(gold_lbls), len(pred_lbls)))
                      if gold_lbls[i] != pred_lbls[i]]
        if diff_pos:
            print(f"    标签差异位置: {diff_pos}")
            for i in diff_pos[:5]:
                c = chars[i] if i < len(chars) else "?"
                print(f"      [{i}] '{c}'  gold={gold_lbls[i]}  pred={pred_lbls[i]}")
        print()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    # ── 加载模型 ───────────────────────────────────────────────────
    weights_path = Path(args.weights)
    vocab_path   = Path(args.vocab)
    test_path    = Path(args.test)

    for p in [weights_path, vocab_path, test_path]:
        if not p.exists():
            print(f"[ERROR] 找不到: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"加载模型: {weights_path}")
    model, vocab = load_model(weights_path, vocab_path, device)
    print(f"加载完成。词表大小: {vocab.vocab_size()}  标签数: {vocab.num_labels}")

    # ── 加载测试集 ────────────────────────────────────────────────
    print(f"\n加载测试集: {test_path}")
    samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"测试样本: {len(samples)} 条")

    # ── 推理 ────────────────────────────────────────────────────
    print("\n推理中...")
    results = run_inference(model, samples, vocab, device, args.batch_size)

    # ── 指标 ────────────────────────────────────────────────────
    metrics = compute_metrics(results)

    print("\n" + "=" * 56)
    print("  NER span-level（micro）")
    print("=" * 56)
    m = metrics["ner_micro"]
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    print("\n  各字段 F1:")
    print(f"  {'Field':<8}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Support':>8}")
    print("  " + "─" * 42)
    for f in ALL_FIELDS:
        fm = metrics["ner_by_field"].get(f, {})
        if fm.get("support", 0) == 0:
            continue
        print(f"  {f:<8}  {fm['precision']:>6.4f}  {fm['recall']:>6.4f}  "
              f"{fm['f1']:>6.4f}  {fm['support']:>8}")

    print("\n" + "=" * 56)
    print("  分类头（有无时间信息）")
    print("=" * 56)
    c = metrics["clf"]
    print(f"  Accuracy  : {c['accuracy']:.4f}")
    print(f"  Precision : {c['precision']:.4f}")
    print(f"  Recall    : {c['recall']:.4f}")
    print(f"  F1        : {c['f1']:.4f}")
    print(f"  TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")

    print("\n" + "=" * 56)
    print("  端到端（年月日全部正确）")
    print("=" * 56)
    e = metrics["e2e"]
    print(f"  YMD 正确率 : {e['ymd_accuracy']:.4f}  "
          f"({e['ymd_correct']} / {e['ymd_total_gold']})")

    # ── 错误分析 ─────────────────────────────────────────────────
    errors = collect_errors(results)
    print(f"\n{'=' * 56}")
    print(f"  错误样本分析  （共 {len(errors)} 条，占 "
          f"{len(errors)/max(len(results),1)*100:.2f}%）")
    print("=" * 56)
    if args.errors > 0:
        print_errors(errors, args.errors)

    # ── 写报告 ──────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics": metrics,
        "errors":  [
            {k: list(v) if isinstance(v, set) else v for k, v in e.items()}
            for e in errors
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已写入: {out_path.resolve()}")


if __name__ == "__main__":
    main()
