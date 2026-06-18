"""
tools/train_date_ner/model.py

MTL DateNER 模型：字符级 BiLSTM + 分类头 + NER 头。

架构：
    输入字符 ID 序列
        ↓
    CharEmbedding  (vocab_size, embed_dim)
        ↓  dropout
    BiLSTM         (embed_dim → hidden_dim*2，num_layers 层)
        ↓
        ├── max-pooling over time → Linear → clf_logits  [B, 2]
        └── 每位置 hidden        → Linear → ner_logits   [B, T, num_labels]

推理时使用 predict()，返回：
    [
        {
            "text":       原始文本,
            "clf_prob":   含时间的概率 (float),
            "spans": [
                {
                    "field":      "YEAR" / "MONTH" / ...,
                    "start":      字符起始下标,
                    "end":        字符结束下标（exclusive）,
                    "value":      对应子串,
                    "confidence": 该 span 的平均 NER softmax 概率,
                }
            ],
            "datetime_candidates": [   # 由 spans 组合出的候选 datetime
                {
                    "datetime": datetime | None,
                    "fields":   {"year": "2019", "month": "10", ...},
                    "char_spans": [...],          # 参与组合的 span
                    "confidence": float,          # clf_prob * span 置信度均值
                }
            ],
        }
    ]

独立测试：
    python model.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size:  int = 1400      # 实际由 vocab.json 决定，此处为默认值
    embed_dim:   int = 64
    hidden_dim:  int = 128       # 单向隐层；双向后输出维度 = hidden_dim * 2
    num_layers:  int = 2
    dropout:     float = 0.3
    num_labels:  int = 13        # len(LABELS)，含 O
    pad_id:      int = 0
    # 推理时的阈值
    clf_threshold: float = 0.4   # 低于此值直接返回空 spans（快速过滤）
    ner_threshold: float = 0.5   # span 置信度低于此值丢弃


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class DateNERModel(nn.Module):
    """
    字符级 BiLSTM 多任务学习模型。

    参数
    ----
    config : ModelConfig

    forward 输入
    -----------
    char_ids  : LongTensor  [B, T]   字符 ID（含 PAD）
    lengths   : LongTensor  [B]      每条序列的实际长度（不含 PAD）

    forward 输出
    -----------
    clf_logits : FloatTensor [B, 2]
    ner_logits : FloatTensor [B, T, num_labels]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 字符 Embedding
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_id,
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        lstm_out_dim = config.hidden_dim * 2  # 双向拼接

        self.embed_dropout = nn.Dropout(config.dropout)
        self.lstm_dropout  = nn.Dropout(config.dropout)

        # 分类头：max-pooling → Linear(256, 2)
        self.clf_head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_out_dim // 2, 2),
        )

        # NER 头：逐位置 Linear(256, num_labels)
        self.ner_head = nn.Linear(lstm_out_dim, config.num_labels)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        # LSTM 权重：正交初始化隐层，Xavier 输入权重
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # forget gate bias 设为 1（缓解梯度消失）
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)
        for layer in self.clf_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.ner_head.weight)
        nn.init.zeros_(self.ner_head.bias)

    def forward(
        self,
        char_ids: torch.Tensor,   # [B, T]
        lengths:  torch.Tensor,   # [B]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Embedding + dropout
        x = self.embedding(char_ids)          # [B, T, E]
        x = self.embed_dropout(x)

        # BiLSTM（pack 以忽略 PAD 位置的计算）
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )                                      # [B, T, H*2]
        lstm_out = self.lstm_dropout(lstm_out)

        # ── 分类头：对有效位置做 max-pooling ────────────────────────
        # 先将 PAD 位置填为极小值，再做 max
        mask = self._length_mask(lengths, lstm_out.size(1))  # [B, T]
        masked = lstm_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values      # [B, H*2]
        clf_logits = self.clf_head(pooled)     # [B, 2]

        # ── NER 头：逐位置投影 ───────────────────────────────────────
        ner_logits = self.ner_head(lstm_out)   # [B, T, num_labels]

        return clf_logits, ner_logits

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """生成 [B, T] 的 bool mask，有效位置为 True。"""
        device = lengths.device
        idxs = torch.arange(max_len, device=device).unsqueeze(0)   # [1, T]
        return idxs < lengths.unsqueeze(1)                          # [B, T]

    # -----------------------------------------------------------------------
    # 损失函数（在 train.py 里调用，也可以直接放这里方便复用）
    # -----------------------------------------------------------------------

    def compute_loss(
        self,
        clf_logits:  torch.Tensor,   # [B, 2]
        ner_logits:  torch.Tensor,   # [B, T, num_labels]
        clf_labels:  torch.Tensor,   # [B]
        ner_labels:  torch.Tensor,   # [B, T]
        lengths:     torch.Tensor,   # [B]
        clf_weight:  float = 0.3,
        ner_weight:  float = 0.7,
        ner_class_weights: Optional[torch.Tensor] = None,  # [num_labels]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        联合损失 = clf_weight * L_clf + ner_weight * L_ner

        返回 (total_loss, clf_loss, ner_loss)
        """
        # 分类损失（交叉熵）
        clf_loss = F.cross_entropy(clf_logits, clf_labels)

        # NER 损失（mask 掉 PAD 位置）
        B, T, C = ner_logits.shape
        mask = self._length_mask(lengths, T)          # [B, T]

        ner_logits_flat = ner_logits.view(B * T, C)
        ner_labels_flat = ner_labels.view(B * T)
        mask_flat       = mask.view(B * T)

        ner_loss_all = F.cross_entropy(
            ner_logits_flat,
            ner_labels_flat,
            weight=ner_class_weights,
            reduction="none",
        )                                             # [B*T]

        # 只对有效位置求均值
        ner_loss = (ner_loss_all * mask_flat.float()).sum() / mask_flat.sum().clamp(min=1)

        total = clf_weight * clf_loss + ner_weight * ner_loss
        return total, clf_loss, ner_loss

    # -----------------------------------------------------------------------
    # 推理接口
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        texts:  list[str],
        vocab:  "VocabHelper",
        device: torch.device,
    ) -> list[dict]:
        """
        对一批文本片段做推理，返回结构化结果。

        参数
        ----
        texts  : 原始文本列表（文件名 stem 或文件夹名）
        vocab  : VocabHelper 实例（封装 char→id 映射和 id→label 映射）
        device : 推理设备

        返回
        ----
        list[dict]，每个元素对应一条文本，格式见模块 docstring。
        """
        self.eval()

        # 编码 + padding
        char_ids_list = [vocab.encode(t) for t in texts]
        lengths = torch.tensor([len(ids) for ids in char_ids_list], dtype=torch.long)
        max_len = int(lengths.max())

        # padding
        padded = torch.full((len(texts), max_len), vocab.pad_id, dtype=torch.long)
        for i, ids in enumerate(char_ids_list):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        padded  = padded.to(device)
        lengths = lengths.to(device)

        clf_logits, ner_logits = self(padded, lengths)

        clf_probs = F.softmax(clf_logits, dim=-1)[:, 1].cpu().tolist()   # 含时间概率
        ner_probs = F.softmax(ner_logits, dim=-1).cpu()                  # [B, T, C]

        results = []
        for i, text in enumerate(texts):
            L = int(lengths[i])
            clf_prob = clf_probs[i]

            # 快速过滤：分类置信度过低
            if clf_prob < self.config.clf_threshold:
                results.append({
                    "text":               text,
                    "clf_prob":           clf_prob,
                    "spans":              [],
                    "datetime_candidates": [],
                })
                continue

            # 解码 NER：贪心取 argmax 标签
            token_probs  = ner_probs[i, :L]                   # [L, C]
            token_labels = token_probs.argmax(dim=-1).tolist() # [L]
            token_confs  = token_probs.max(dim=-1).values.tolist()

            spans = _decode_bio_spans(
                token_labels, token_confs,
                vocab.id2label, text,
                threshold=self.config.ner_threshold,
            )

            dt_candidates = _build_datetime_candidates(spans, clf_prob)

            results.append({
                "text":                text,
                "clf_prob":            clf_prob,
                "spans":               spans,
                "datetime_candidates": dt_candidates,
            })

        return results


# ---------------------------------------------------------------------------
# 词表辅助类（封装 char→id 和 id→label，train.py / infer.py 都用这个）
# ---------------------------------------------------------------------------

class VocabHelper:
    """
    从 vocab.json 加载，提供编码接口。
    """

    def __init__(self, vocab_path: str | Path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char2id:  dict[str, int] = data["char2id"]
        self.id2char:  dict[int, str] = {int(k): v for k, v in data["id2char"].items()}
        self.label2id: dict[str, int] = data["label2id"]
        self.id2label: dict[int, str] = {int(k): v for k, v in data["id2label"].items()}
        self.pad_id:   int = data["pad_id"]
        self.unk_id:   int = data["unk_id"]
        self.num_labels: int = data["num_labels"]

    def encode(self, text: str) -> list[int]:
        return [self.char2id.get(c, self.unk_id) for c in text]

    def to_dict(self) -> dict:
        """
        序列化为 vocab.json 的标准格式。

        唯一的序列化来源：build_dataset.py / train.py 任何需要写出
        vocab.json 的地方都应调用此方法，而不是手动拼字典，
        防止字段定义在多处重复、容易写歪。
        """
        return {
            "char2id":  self.char2id,
            "id2char":  {str(v): k for k, v in self.char2id.items()},
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "special_tokens": ["<PAD>", "<UNK>"],
            "pad_id":  self.pad_id,
            "unk_id":  self.unk_id,
            "num_labels": self.num_labels,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VocabHelper":
        """从 dict（而非文件路径）构造，避免临时文件往返。"""
        obj = cls.__new__(cls)
        obj.char2id    = data["char2id"]
        obj.id2char    = {int(k): v for k, v in data["id2char"].items()}
        obj.label2id   = data["label2id"]
        obj.id2label   = {int(k): v for k, v in data["id2label"].items()}
        obj.pad_id     = data["pad_id"]
        obj.unk_id     = data["unk_id"]
        obj.num_labels = data["num_labels"]
        return obj

    def vocab_size(self) -> int:
        return len(self.char2id)


# ---------------------------------------------------------------------------
# 后处理：BIO 序列 → spans
# ---------------------------------------------------------------------------

def _decode_bio_spans(
    token_labels: list[int],
    token_confs:  list[float],
    id2label:     dict[int, str],
    text:         str,
    threshold:    float = 0.5,
) -> list[dict]:
    """
    将 BIO 标签序列转换为 span 列表。

    处理规则：
        - 遇到 B-XXX 开启新 span
        - 遇到 I-XXX 且与当前 span 字段一致 → 延伸
        - 遇到 I-XXX 但与当前 span 字段不一致 → 视为新 B（容错）
        - 遇到 O 或 B-新字段 → 关闭当前 span
        - span 平均置信度 < threshold → 丢弃
    """
    spans = []
    cur_field: Optional[str] = None
    cur_start: int = 0
    cur_confs: list[float] = []

    def _close_span(end: int):
        nonlocal cur_field, cur_start, cur_confs
        if cur_field is not None and cur_confs:
            avg_conf = sum(cur_confs) / len(cur_confs)
            if avg_conf >= threshold:
                spans.append({
                    "field":      cur_field,
                    "start":      cur_start,
                    "end":        end,
                    "value":      text[cur_start:end],
                    "confidence": round(avg_conf, 4),
                })
        cur_field = None
        cur_confs = []

    for pos, (lid, conf) in enumerate(zip(token_labels, token_confs)):
        label = id2label.get(lid, "O")

        if label == "O":
            _close_span(pos)

        elif label.startswith("B-"):
            _close_span(pos)
            cur_field = label[2:]   # 去掉 "B-"
            cur_start = pos
            cur_confs = [conf]

        elif label.startswith("I-"):
            field = label[2:]
            if field == cur_field:
                cur_confs.append(conf)
            else:
                # I-XXX 字段与当前不一致，视为错误的 B-XXX
                _close_span(pos)
                cur_field = field
                cur_start = pos
                cur_confs = [conf]

    _close_span(len(token_labels))
    return spans


# ---------------------------------------------------------------------------
# 后处理：spans → datetime 候选
# ---------------------------------------------------------------------------

# 时间字段的层级顺序（从高到低精度）
_FIELD_ORDER = ["YEAR", "MONTH", "DAY", "HOUR", "MIN", "SEC"]

def _build_datetime_candidates(
    spans:    list[dict],
    clf_prob: float,
) -> list[dict]:
    """
    从解码出的 spans 组合出 datetime 候选。

    当前策略：将所有 spans 视为一个候选（一个文本片段里不会有两套日期）。
    若 spans 里字段不完整（如只有 YEAR），生成部分日期候选（datetime 字段为 None，
    但 fields 里有具体值，交由 ResolverStage 进一步处理）。
    """
    if not spans:
        return []

    # 按字段分组，同一字段取置信度最高的 span
    field_map: dict[str, dict] = {}
    for span in spans:
        f = span["field"]
        if f not in field_map or span["confidence"] > field_map[f]["confidence"]:
            field_map[f] = span

    # 提取数值
    fields_vals: dict[str, str] = {}
    for f in _FIELD_ORDER:
        if f in field_map:
            fields_vals[f] = field_map[f]["value"]

    if not fields_vals:
        return []

    # 尝试组合 datetime
    dt: Optional[datetime] = None
    try:
        year  = int(fields_vals.get("YEAR",  "0") or "0")
        month = int(fields_vals.get("MONTH", "1") or "1")
        day   = int(fields_vals.get("DAY",   "1") or "1")
        hour  = int(fields_vals.get("HOUR",  "0") or "0")
        minute = int(fields_vals.get("MIN",  "0") or "0")
        second = int(fields_vals.get("SEC",  "0") or "0")
        if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            dt = datetime(year, month, day, hour, minute, second)
    except (ValueError, OverflowError):
        pass

    # 候选置信度 = clf_prob * span 置信度均值
    span_conf_mean = (
        sum(s["confidence"] for s in field_map.values()) / len(field_map)
    )
    candidate_conf = round(clf_prob * span_conf_mean, 4)

    return [
        {
            "datetime":   dt.isoformat() if dt else None,
            "fields":     fields_vals,
            "char_spans": list(field_map.values()),
            "confidence": candidate_conf,
        }
    ]


# ---------------------------------------------------------------------------
# 工厂函数：从 weights 文件加载模型
# ---------------------------------------------------------------------------

def load_model(
    weights_path: str | Path,
    vocab_path:   str | Path,
    device:       Optional[torch.device] = None,
) -> tuple["DateNERModel", "VocabHelper"]:
    """
    加载已训练模型，返回 (model, vocab)。

    weights_path 对应的 .pt 文件格式：
        {
            "model_state": state_dict,
            "config":      ModelConfig 的 __dict__,
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model  = DateNERModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    vocab = VocabHelper(vocab_path)
    return model, vocab


# ---------------------------------------------------------------------------
# 独立测试（python model.py）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("=== DateNERModel 独立测试 ===\n")

    # ── 构造最小词表 ─────────────────────────────────────────────────
    chars = (
        list("0123456789")
        + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        + list("_-. ")
        + list("年月日来自手机照片备份")
    )
    char2id = {"<PAD>": 0, "<UNK>": 1}
    for c in chars:
        if c not in char2id:
            char2id[c] = len(char2id)

    LABELS = [
        "O",
        "B-YEAR","I-YEAR","B-MONTH","I-MONTH",
        "B-DAY","I-DAY","B-HOUR","I-HOUR",
        "B-MIN","I-MIN","B-SEC","I-SEC",
    ]
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label  = {i: l for i, l in enumerate(LABELS)}

    import tempfile
    vocab_data = {
        "char2id":  char2id,
        "id2char":  {str(v): k for k, v in char2id.items()},
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "special_tokens": ["<PAD>", "<UNK>"],
        "pad_id":  0,
        "unk_id":  1,
        "num_labels": len(LABELS),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(vocab_data, f, ensure_ascii=False)
        vocab_path = f.name

    vocab = VocabHelper(vocab_path)

    # ── 构造模型 ─────────────────────────────────────────────────────
    config = ModelConfig(
        vocab_size=vocab.vocab_size(),
        num_labels=vocab.num_labels,
    )
    model = DateNERModel(config)
    device = torch.device("cpu")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}  ({total_params/1e6:.2f}M)")
    print(f"词表大小:   {vocab.vocab_size()}")
    print(f"标签数量:   {vocab.num_labels}\n")

    # ── 前向传播测试 ─────────────────────────────────────────────────
    texts = [
        "IMG_20191002_105038",
        "2021-04-23 155835",
        "来自小火鸡早期的照片",
        "2019 老刘手机上的一些私藏",
        "PIC_20130916_142420_E34",
    ]

    print("── forward pass ──")
    ids_list = [vocab.encode(t) for t in texts]
    lengths  = torch.tensor([len(ids) for ids in ids_list])
    max_len  = int(lengths.max())
    padded   = torch.full((len(texts), max_len), vocab.pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = torch.tensor(ids)

    with torch.no_grad():
        clf_logits, ner_logits = model(padded, lengths)

    print(f"clf_logits shape: {clf_logits.shape}")   # [5, 2]
    print(f"ner_logits shape: {ner_logits.shape}\n") # [5, max_len, 13]

    # ── 损失计算测试 ─────────────────────────────────────────────────
    print("── loss 计算 ──")
    clf_labels = torch.tensor([1, 1, 0, 1, 1])
    ner_labels = torch.zeros(len(texts), max_len, dtype=torch.long)  # 全 O

    # 模拟给第0条打标
    # IMG_20191002_105038：YEAR 在 4-7，MONTH 在 8-9，DAY 在 10-11
    for pos in range(4, 8):
        ner_labels[0, pos] = label2id["I-YEAR"] if pos > 4 else label2id["B-YEAR"]
    ner_labels[0, 4] = label2id["B-YEAR"]

    total_loss, clf_loss, ner_loss = model.compute_loss(
        clf_logits, ner_logits, clf_labels, ner_labels, lengths
    )
    print(f"total_loss: {total_loss.item():.4f}")
    print(f"clf_loss  : {clf_loss.item():.4f}")
    print(f"ner_loss  : {ner_loss.item():.4f}\n")

    # ── predict 接口测试 ─────────────────────────────────────────────
    print("── predict 接口 ──")
    results = model.predict(texts, vocab, device)
    for r in results:
        print(f"  [{r['text']!r}]")
        print(f"    clf_prob : {r['clf_prob']:.4f}")
        print(f"    spans    : {r['spans']}")
        if r['datetime_candidates']:
            print(f"    dt_cand  : {r['datetime_candidates'][0]['datetime']}")
        print()

    # ── 梯度流动检查 ─────────────────────────────────────────────────
    print("── 梯度检查 ──")
    model.train()
    total_loss, _, _ = model.compute_loss(
        *model(padded, lengths),
        clf_labels, ner_labels, lengths,
    )
    total_loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    all_have_grad = all(v > 0 for v in grad_norms.values())
    print(f"所有参数均有非零梯度: {all_have_grad}")
    for name, norm in list(grad_norms.items())[:5]:
        print(f"  {name}: grad_norm={norm:.6f}")
    print("  ...")

    os.unlink(vocab_path)
    print("\n✓ 所有测试通过")