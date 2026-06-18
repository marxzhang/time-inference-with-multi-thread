"""
tools/train_date_ner/train.py

MTL DateNER 训练入口。

用法：
    # 基本训练
    python tools/train_date_ner/train.py

    # 指定路径和超参数
    python tools/train_date_ner/train.py \
        --data-dir tools/train_date_ner/data \
        --out-dir  weights \
        --epochs   30 \
        --batch-size 256 \
        --lr 1e-3

    # 快速验证流程（少量数据跑通）
    python tools/train_date_ner/train.py --smoke-test

训练产物（weights/）：
    date_ner.pt      模型权重 + config（用 load_model() 加载）
    vocab.json       从 data/ 复制过来（推理时与 .pt 同目录）
    training_log.jsonl  每个 epoch 的指标记录
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 同目录导入
sys.path.insert(0, str(Path(__file__).parent))
from model import DateNERModel, ModelConfig, VocabHelper


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="训练 DateNER MTL 模型")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--out-dir",    default="weights")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--embed-dim",  type=int,   default=64)
    p.add_argument("--hidden-dim", type=int,   default=128)
    p.add_argument("--num-layers", type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--clf-weight", type=float, default=0.3,
                   help="联合损失中分类头权重（NER 头权重 = 1 - clf_weight）")
    p.add_argument("--patience",   type=int,   default=5,
                   help="早停：验证集 NER F1 连续 N 个 epoch 无提升则停止")
    p.add_argument("--cjk-min-freq", type=int, default=3,
                   help="CJK 字符最低频次，低于此值归入 <UNK>（压缩词表）")
    p.add_argument("--no-class-weight", action="store_true",
                   help="不对 NER 标签做类权重（默认启用权重）")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="只用 500 条数据跑 2 个 epoch，验证流程")
    p.add_argument("--device",     default="auto",
                   help="auto / cpu / cuda / mps")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """
    从 .jsonl 文件加载样本，将字符序列和标签序列转换为 tensor。

    每个样本：
        char_ids  : LongTensor [T]
        ner_labels: LongTensor [T]
        clf_label : LongTensor []
        length    : int
    """

    def __init__(
        self,
        path:      Path,
        vocab:     VocabHelper,
        max_len:   int = 64,
        limit:     Optional[int] = None,   # smoke-test 用
    ):
        self.samples: list[dict] = []
        self.vocab   = vocab
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        chars  = s["chars"]
        labels = s["labels"]

        # 防御：chars 和 labels 长度必须一致（build_dataset 偶发不对齐时截到短的）
        min_len = min(len(chars), len(labels))
        chars  = chars[:min_len][:self.max_len]
        labels = labels[:min_len][:self.max_len]
        length = len(chars)

        char_ids   = [self.vocab.char2id.get(c, self.vocab.unk_id) for c in chars]
        label_ids  = [self.vocab.label2id.get(l, 0) for l in labels]
        clf_label  = int(s["clf_label"])

        return {
            "char_ids":   char_ids,
            "ner_labels": label_ids,
            "clf_label":  clf_label,
            "length":     length,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """将变长序列 padding 到 batch 内最长。"""
    max_len = max(b["length"] for b in batch)
    pad_id = 0  # <PAD>

    char_ids_padded   = []
    ner_labels_padded = []
    clf_labels        = []
    lengths           = []

    for b in batch:
        L = b["length"]
        # char_ids / ner_labels 已在 __getitem__ 里截断到 max_len，
        # 但 b["length"] 记录的是截断后的实际长度，以它为准再次截断防止不一致
        char_ids_b   = b["char_ids"][:L]
        ner_labels_b = b["ner_labels"][:L]
        pad = max_len - L

        char_ids_padded.append(char_ids_b   + [pad_id] * pad)
        ner_labels_padded.append(ner_labels_b + [0]     * pad)  # O=0 for pad
        clf_labels.append(b["clf_label"])
        lengths.append(L)

    return {
        "char_ids":   torch.tensor(char_ids_padded,   dtype=torch.long),
        "ner_labels": torch.tensor(ner_labels_padded, dtype=torch.long),
        "clf_labels": torch.tensor(clf_labels,        dtype=torch.long),
        "lengths":    torch.tensor(lengths,            dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 词表（在 train 阶段重建，支持 cjk_min_freq 压缩）
# ---------------------------------------------------------------------------

def build_vocab_from_data(
    train_path:   Path,
    orig_vocab:   VocabHelper,
    cjk_min_freq: int,
) -> VocabHelper:
    """
    从训练集重新统计字符频次，应用 cjk_min_freq 过滤，生成压缩词表。
    标签体系直接复用 orig_vocab（不变）。

    返回 VocabHelper。序列化交给 VocabHelper.to_dict()（唯一来源），
    调用方写盘时直接调用 vocab.to_dict()，不在此处重复构造字典。
    """
    import unicodedata

    freq: Counter = Counter()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                freq.update(s["chars"])
            except json.JSONDecodeError:
                pass

    char2id = {"<PAD>": 0, "<UNK>": 1}
    kept = dropped = 0
    for char, cnt in freq.most_common():
        is_cjk = "CJK" in unicodedata.name(char, "") if len(char) == 1 else False
        threshold = cjk_min_freq if is_cjk else 1
        if cnt >= threshold:
            if char not in char2id:
                char2id[char] = len(char2id)
            kept += 1
        else:
            dropped += 1

    print(f"  词表压缩: 保留 {kept} 个字符，丢弃 {dropped} 个低频CJK → "
          f"词表大小 {len(char2id)}（含 <PAD> <UNK>）")

    return VocabHelper.from_dict({
        "char2id":  char2id,
        "id2char":  {str(v): k for k, v in char2id.items()},
        "label2id": orig_vocab.label2id,
        "id2label": {str(k): v for k, v in orig_vocab.id2label.items()},
        "special_tokens": ["<PAD>", "<UNK>"],
        "pad_id":  0,
        "unk_id":  1,
        "num_labels": orig_vocab.num_labels,
    })


# ---------------------------------------------------------------------------
# NER 类权重计算
# ---------------------------------------------------------------------------

def compute_ner_class_weights(
    train_path: Path,
    num_labels: int,
    label2id:   dict[str, int],
    device:     torch.device,
) -> torch.Tensor:
    """
    统计训练集标签频次，返回逆频率权重张量 [num_labels]。

    O 标签权重固定为 1.0，非 O 标签权重 = total / (num_non_O_classes * count)，
    并 clip 到 [1.0, 5.0] 防止极端值。
    """
    freq: Counter = Counter()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                freq.update(s["labels"])
            except json.JSONDecodeError:
                pass

    weights = torch.ones(num_labels)
    total_non_O = sum(cnt for lbl, cnt in freq.items() if lbl != "O")
    num_non_O   = sum(1   for lbl in freq       if lbl != "O")

    if num_non_O > 0 and total_non_O > 0:
        for lbl, cnt in freq.items():
            if lbl == "O":
                continue
            lid = label2id.get(lbl)
            if lid is None:
                continue
            w = total_non_O / (num_non_O * cnt)
            weights[lid] = max(1.0, min(5.0, w))

    print("  NER 类权重:")
    id2label = {v: k for k, v in label2id.items()}
    for lid in range(num_labels):
        lbl = id2label.get(lid, str(lid))
        cnt = freq.get(lbl, 0)
        print(f"    {lbl:<12}: weight={weights[lid]:.3f}  freq={cnt}")

    return weights.to(device)


# ---------------------------------------------------------------------------
# 评估：NER span-level F1
# ---------------------------------------------------------------------------

def evaluate(
    model:      DateNERModel,
    loader:     DataLoader,
    vocab:      VocabHelper,
    device:     torch.device,
    clf_weight: float,
    ner_weight: float,
    ner_class_weights: Optional[torch.Tensor],
) -> dict:
    """
    在给定 DataLoader 上评估，返回指标字典。

    NER 指标：严格 span-level F1
        - span = (start, end, label_type) 完全相同才算命中
    CLF 指标：accuracy + F1(正类)
    """
    model.eval()

    total_loss = clf_loss_sum = ner_loss_sum = 0.0
    n_batches  = 0

    # 分类统计
    clf_tp = clf_fp = clf_fn = 0

    # NER span 统计（按字段分开统计）
    ner_tp_by_field: Counter = Counter()
    ner_fp_by_field: Counter = Counter()
    ner_fn_by_field: Counter = Counter()

    id2label = vocab.id2label

    with torch.no_grad():
        for batch in loader:
            char_ids   = batch["char_ids"].to(device)
            ner_labels = batch["ner_labels"].to(device)
            clf_labels = batch["clf_labels"].to(device)
            lengths    = batch["lengths"].to(device)

            clf_logits, ner_logits = model(char_ids, lengths)

            loss, cl, nl = model.compute_loss(
                clf_logits, ner_logits, clf_labels, ner_labels, lengths,
                clf_weight=clf_weight, ner_weight=ner_weight,
                ner_class_weights=ner_class_weights,
            )
            total_loss    += loss.item()
            clf_loss_sum  += cl.item()
            ner_loss_sum  += nl.item()
            n_batches += 1

            # ── 分类指标 ──────────────────────────────────────────
            clf_preds = clf_logits.argmax(dim=-1)    # [B]
            for pred, gold in zip(clf_preds.tolist(), clf_labels.tolist()):
                if pred == 1 and gold == 1:
                    clf_tp += 1
                elif pred == 1 and gold == 0:
                    clf_fp += 1
                elif pred == 0 and gold == 1:
                    clf_fn += 1

            # ── NER span 指标 ────────────────────────────────────
            ner_preds = ner_logits.argmax(dim=-1)    # [B, T]
            B = char_ids.size(0)
            for i in range(B):
                L       = int(lengths[i])
                pred_seq = ner_preds[i, :L].tolist()
                gold_seq = ner_labels[i, :L].tolist()

                pred_spans = _extract_spans(pred_seq, id2label)
                gold_spans = _extract_spans(gold_seq, id2label)

                for span in pred_spans:
                    field = span[2]
                    if span in gold_spans:
                        ner_tp_by_field[field] += 1
                    else:
                        ner_fp_by_field[field] += 1
                for span in gold_spans:
                    field = span[2]
                    if span not in pred_spans:
                        ner_fn_by_field[field] += 1

    n_batches = max(n_batches, 1)

    # ── 汇总分类指标 ──────────────────────────────────────────────
    clf_prec = clf_tp / max(clf_tp + clf_fp, 1)
    clf_rec  = clf_tp / max(clf_tp + clf_fn, 1)
    clf_f1   = 2 * clf_prec * clf_rec / max(clf_prec + clf_rec, 1e-9)

    # ── 汇总 NER 指标（micro 平均） ──────────────────────────────
    all_tp = sum(ner_tp_by_field.values())
    all_fp = sum(ner_fp_by_field.values())
    all_fn = sum(ner_fn_by_field.values())
    ner_prec = all_tp / max(all_tp + all_fp, 1)
    ner_rec  = all_tp / max(all_tp + all_fn, 1)
    ner_f1   = 2 * ner_prec * ner_rec / max(ner_prec + ner_rec, 1e-9)

    # 各字段 F1
    field_f1 = {}
    all_fields = set(ner_tp_by_field) | set(ner_fp_by_field) | set(ner_fn_by_field)
    for f in all_fields:
        tp = ner_tp_by_field[f]
        fp = ner_fp_by_field[f]
        fn = ner_fn_by_field[f]
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        field_f1[f] = 2 * p * r / max(p + r, 1e-9)

    return {
        "loss":     total_loss    / n_batches,
        "clf_loss": clf_loss_sum  / n_batches,
        "ner_loss": ner_loss_sum  / n_batches,
        "clf_f1":   clf_f1,
        "clf_prec": clf_prec,
        "clf_rec":  clf_rec,
        "ner_f1":   ner_f1,
        "ner_prec": ner_prec,
        "ner_rec":  ner_rec,
        "field_f1": field_f1,
    }


def _extract_spans(
    label_ids: list[int],
    id2label:  dict[int, str],
) -> set[tuple[int, int, str]]:
    """
    从标签 ID 序列中提取 (start, end, field) span 集合。
    end 是 exclusive。
    """
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
                # I- 与当前字段不符，关闭旧的，开启新的
                if cur_field is not None:
                    spans.add((cur_start, pos, cur_field))
                cur_field = field
                cur_start = pos
        else:  # O
            if cur_field is not None:
                spans.add((cur_start, pos, cur_field))
            cur_field = None

    if cur_field is not None:
        spans.add((cur_start, len(label_ids), cur_field))

    return spans


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       DateNERModel,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    device:      torch.device,
    clf_weight:  float,
    ner_weight:  float,
    ner_class_weights: Optional[torch.Tensor],
) -> dict:
    model.train()
    total_loss = clf_loss_sum = ner_loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        char_ids   = batch["char_ids"].to(device)
        ner_labels = batch["ner_labels"].to(device)
        clf_labels = batch["clf_labels"].to(device)
        lengths    = batch["lengths"].to(device)

        optimizer.zero_grad()
        clf_logits, ner_logits = model(char_ids, lengths)
        loss, cl, nl = model.compute_loss(
            clf_logits, ner_logits, clf_labels, ner_labels, lengths,
            clf_weight=clf_weight, ner_weight=ner_weight,
            ner_class_weights=ner_class_weights,
        )
        loss.backward()

        # 梯度裁剪（防止 LSTM 梯度爆炸）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss    += loss.item()
        clf_loss_sum  += cl.item()
        ner_loss_sum  += nl.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    n_batches = max(n_batches, 1)
    return {
        "loss":     total_loss   / n_batches,
        "clf_loss": clf_loss_sum / n_batches,
        "ner_loss": ner_loss_sum / n_batches,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 设备 ─────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    val_path   = data_dir / "val.jsonl"
    vocab_path = data_dir / "vocab.json"

    for p in [train_path, val_path, vocab_path]:
        if not p.exists():
            print(f"[ERROR] 找不到: {p}", file=sys.stderr)
            sys.exit(1)

    # ── 词表（应用 cjk_min_freq 压缩） ──────────────────────────────
    print("\n构建词表...")
    orig_vocab = VocabHelper(vocab_path)
    vocab = build_vocab_from_data(train_path, orig_vocab, args.cjk_min_freq)

    # 把压缩后的 vocab.json 写到 out_dir（推理时需要与 .pt 配套）
    # to_dict() 是唯一的序列化来源，避免在多处手动拼同样的字典结构
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, ensure_ascii=False, indent=2)

    # ── NER 类权重 ───────────────────────────────────────────────────
    ner_class_weights = None
    if not args.no_class_weight:
        print("\n计算 NER 类权重...")
        ner_class_weights = compute_ner_class_weights(
            train_path, vocab.num_labels, vocab.label2id, device
        )

    # ── 数据加载 ─────────────────────────────────────────────────────
    print("\n加载数据集...")
    limit = 500 if args.smoke_test else None
    train_ds = NERDataset(train_path, vocab, limit=limit)
    val_ds   = NERDataset(val_path,   vocab, limit=limit)
    print(f"  train: {len(train_ds)} 条")
    print(f"  val  : {len(val_ds)} 条")

    # 有效 batch_size：smoke-test 时自动缩小
    bs = min(args.batch_size, len(train_ds))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── 模型 ─────────────────────────────────────────────────────────
    config = ModelConfig(
        vocab_size=vocab.vocab_size(),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_labels=vocab.num_labels,
        pad_id=vocab.pad_id,
    )
    model = DateNERModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # ── 优化器 + 调度器 ──────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # CosineAnnealing：前 5 epoch warmup，之后余弦衰减
    warmup_epochs = min(5, args.epochs // 4)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    clf_weight = args.clf_weight
    ner_weight = 1.0 - clf_weight

    # ── 早停状态 ─────────────────────────────────────────────────────
    best_ner_f1   = -1.0
    best_epoch    = 0
    patience_left = args.patience
    log_records: list[dict] = []
    best_ckpt_path = out_dir / "date_ner_best.pt"

    epochs = 2 if args.smoke_test else args.epochs
    print(f"\n开始训练（{'smoke-test: 2 epochs' if args.smoke_test else f'{epochs} epochs'}）\n")
    print(f"{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>8}  "
          f"{'ClfF1':>6}  {'NerF1':>6}  {'LR':>8}  {'Time':>6}")
    print("─" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            clf_weight, ner_weight, ner_class_weights,
        )
        val_metrics = evaluate(
            model, val_loader, vocab, device,
            clf_weight, ner_weight, ner_class_weights,
        )

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>5}  "
            f"{train_metrics['loss']:>9.4f}  "
            f"{val_metrics['loss']:>8.4f}  "
            f"{val_metrics['clf_f1']:>6.4f}  "
            f"{val_metrics['ner_f1']:>6.4f}  "
            f"{cur_lr:>8.2e}  "
            f"{elapsed:>5.1f}s"
        )

        record = {
            "epoch": epoch,
            "lr":    cur_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }
        log_records.append(record)

        # ── 保存最优模型 ──────────────────────────────────────────
        if val_metrics["ner_f1"] > best_ner_f1:
            best_ner_f1   = val_metrics["ner_f1"]
            best_epoch    = epoch
            patience_left = args.patience

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config":      config.__dict__,
                    "best_epoch":  best_epoch,
                    "best_ner_f1": best_ner_f1,
                },
                best_ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"\n早停：验证集 NER F1 连续 {args.patience} 个 epoch 无提升")
                break

    # ── 训练结束：把最优 checkpoint 复制为正式产物 ───────────────
    final_path = out_dir / "date_ner.pt"
    if best_ckpt_path.exists():
        shutil.copy(best_ckpt_path, final_path)

    print(f"\n训练完成。最优 epoch: {best_epoch}，val NER F1: {best_ner_f1:.4f}")
    print(f"模型已保存: {final_path.resolve()}")
    print(f"词表已保存: {(out_dir / 'vocab.json').resolve()}")

    # ── 写训练日志 ────────────────────────────────────────────────
    log_path = out_dir / "training_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in log_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 打印各字段最终 F1 ────────────────────────────────────────
    if log_records:
        best_rec = max(log_records, key=lambda r: r["val_ner_f1"])
        print(f"\n最优 epoch {best_rec['epoch']} 各字段 F1:")
        for field, f1 in sorted(best_rec.get("val_field_f1", {}).items()):
            print(f"  {field:<8}: {f1:.4f}")

    # smoke-test 快速验证提示
    if args.smoke_test:
        print("\n[smoke-test 完成] 流程跑通，可用全量数据正式训练：")
        print("  python tools/train_date_ner/train.py")


if __name__ == "__main__":
    main()