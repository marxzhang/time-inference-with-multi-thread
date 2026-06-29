"""
models/date_ner.py — DateNER MTL 模型定义 + 推理封装

与 models/clip.py 同级、同语义：定义"如何加载和调用这个模型"，
不包含任何训练逻辑（训练脚本在 training/date_ner/ 下，反向 import 本文件）。

架构：字符级 BiLSTM + 分类头 + NER 头。详见 DateNERModel 类文档。

独立测试：
    python -m models.date_ner
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

    架构：
        输入字符 ID 序列
            ↓
        CharEmbedding  (vocab_size, embed_dim)
            ↓ dropout
        BiLSTM         (embed_dim → hidden_dim*2，num_layers 层)
            ↓
            ├── max-pooling over time → Linear → clf_logits  [B, 2]
            └── 每位置 hidden        → Linear → ner_logits   [B, T, num_labels]

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

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_id,
        )

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

        self.clf_head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_out_dim // 2, 2),
        )

        self.ner_head = nn.Linear(lstm_out_dim, config.num_labels)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
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

        x = self.embedding(char_ids)          # [B, T, E]
        x = self.embed_dropout(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )                                      # [B, T, H*2]
        lstm_out = self.lstm_dropout(lstm_out)

        mask = self._length_mask(lengths, lstm_out.size(1))  # [B, T]
        masked = lstm_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values      # [B, H*2]
        clf_logits = self.clf_head(pooled)     # [B, 2]

        ner_logits = self.ner_head(lstm_out)   # [B, T, num_labels]

        return clf_logits, ner_logits

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """生成 [B, T] 的 bool mask，有效位置为 True。"""
        device = lengths.device
        idxs = torch.arange(max_len, device=device).unsqueeze(0)   # [1, T]
        return idxs < lengths.unsqueeze(1)                          # [B, T]

    # -----------------------------------------------------------------------
    # 损失函数（训练时调用，定义放在这里方便训练脚本直接复用）
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
        clf_loss = F.cross_entropy(clf_logits, clf_labels)

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
        对一批文本片段做推理，返回结构化结果：
        [
            {
                "text":       原始文本,
                "clf_prob":   含时间的概率 (float),
                "spans": [
                    {"field": "YEAR"/"MONTH"/..., "start": int, "end": int,
                     "value": str, "confidence": float}
                ],
                "datetime_candidates": [
                    {"datetime": str|None, "fields": dict,
                     "char_spans": [...], "confidence": float}
                ],
            }
        ]
        """
        self.eval()

        char_ids_list = [vocab.encode(t) for t in texts]
        lengths = torch.tensor([len(ids) for ids in char_ids_list], dtype=torch.long)
        max_len = int(lengths.max())

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

            if clf_prob < self.config.clf_threshold:
                results.append({
                    "text":               text,
                    "clf_prob":           clf_prob,
                    "spans":              [],
                    "datetime_candidates": [],
                })
                continue

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
# 词表辅助类
# ---------------------------------------------------------------------------

class VocabHelper:
    """从 vocab.json 加载（或从 dict 构造），提供编码接口。"""

    def __init__(self, vocab_path: str | Path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._load_from_dict(data)

    def _load_from_dict(self, data: dict) -> None:
        self.char2id:  dict[str, int] = data["char2id"]
        self.id2char:  dict[int, str] = {int(k): v for k, v in data["id2char"].items()}
        self.label2id: dict[str, int] = data["label2id"]
        self.id2label: dict[int, str] = {int(k): v for k, v in data["id2label"].items()}
        self.pad_id:   int = data["pad_id"]
        self.unk_id:   int = data["unk_id"]
        self.num_labels: int = data["num_labels"]

    @classmethod
    def from_dict(cls, data: dict) -> "VocabHelper":
        """从 dict（而非文件路径）构造，避免训练时的临时文件往返。"""
        obj = cls.__new__(cls)
        obj._load_from_dict(data)
        return obj

    def encode(self, text: str) -> list[int]:
        return [self.char2id.get(c, self.unk_id) for c in text]

    def to_dict(self) -> dict:
        """
        序列化为 vocab.json 的标准格式。

        唯一的序列化来源：任何需要写出 vocab.json 的地方都应调用此方法，
        而不是手动拼字典，防止字段定义在多处重复、容易写歪。
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

    规则：
        - 遇到 B-XXX 开启新 span
        - 遇到 I-XXX 且字段一致 → 延伸
        - 遇到 I-XXX 但字段不一致 → 视为新 B（容错）
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
            cur_field = label[2:]
            cur_start = pos
            cur_confs = [conf]
        elif label.startswith("I-"):
            field = label[2:]
            if field == cur_field:
                cur_confs.append(conf)
            else:
                _close_span(pos)
                cur_field = field
                cur_start = pos
                cur_confs = [conf]

    _close_span(len(token_labels))
    return spans


# ---------------------------------------------------------------------------
# 后处理：spans → datetime 候选
# ---------------------------------------------------------------------------

_FIELD_ORDER = ["YEAR", "MONTH", "DAY", "HOUR", "MIN", "SEC"]

def _build_datetime_candidates(
    spans:    list[dict],
    clf_prob: float,
) -> list[dict]:
    """
    从解码出的 spans 组合出 datetime 候选。

    当前策略：所有 spans 视为一个候选（一个文本片段不会有两套日期）。
    字段不完整时（如只有 YEAR）仍生成候选，datetime 字段为 None，
    但 fields 里有具体值，交由 ResolverStage 进一步处理。
    """
    if not spans:
        return []

    field_map: dict[str, dict] = {}
    for span in spans:
        f = span["field"]
        if f not in field_map or span["confidence"] > field_map[f]["confidence"]:
            field_map[f] = span

    fields_vals: dict[str, str] = {}
    for f in _FIELD_ORDER:
        if f in field_map:
            fields_vals[f] = field_map[f]["value"]

    if not fields_vals:
        return []

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
        {"model_state": state_dict, "config": ModelConfig 的 __dict__}
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
# 独立测试（python -m models.date_ner）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    print("=== DateNERModel 独立测试 ===\n")

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
    vocab = VocabHelper.from_dict(vocab_data)

    config = ModelConfig(vocab_size=vocab.vocab_size(), num_labels=vocab.num_labels)
    model = DateNERModel(config)
    device = torch.device("cpu")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}  ({total_params/1e6:.2f}M)")
    print(f"词表大小:   {vocab.vocab_size()}")
    print(f"标签数量:   {vocab.num_labels}\n")

    texts = [
        "IMG_20191002_105038",
        "2021-04-23 155835",
        "来自小火鸡早期的照片",
        "2019 老刘手机上的一些私藏",
        "PIC_20130916_142420_E34",
    ]

    print("── predict 接口 ──")
    results = model.predict(texts, vocab, device)
    for r in results:
        print(f"  [{r['text']!r}]")
        print(f"    clf_prob : {r['clf_prob']:.4f}")
        print(f"    spans    : {r['spans']}")
        if r['datetime_candidates']:
            print(f"    dt_cand  : {r['datetime_candidates'][0]['datetime']}")
        print()

    print("✓ 测试通过")