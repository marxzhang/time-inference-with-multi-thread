"""
stages/timeline/tokens.py — 三层语义时间 Token 数据结构

Token 是 Tokenizer 的输出单元，也是各 Resolver 的输入。
本文件只定义数据结构，不包含任何解析或业务逻辑。

三层含义：
    DirectToken   第一层，来自已有 Evidence，时间确定，无歧义
    StructToken   第二层，来自文件夹名中有歧义的数字段，候选集待消歧
    SemanticToken 第三层，来自语义文本（地名/节日/季节），概率分布（D阶段实现）

设计原则：
    - 所有字段有明确默认值，不强制构造时传全部参数
    - StructToken.resolved_* 初始为 None，由 StructResolver 填入
    - SemanticToken 整体初始为空，D 阶段前不使用
    - to_dict / from_dict 支持调试时序列化输出
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# 第一层：直接时间 Token
# ---------------------------------------------------------------------------

@dataclass
class DirectToken:
    """
    来自 item.evidence 中已有确定时间的记录。

    Tokenizer 直接读 item.evidence，不重新解析任何文本。
    作用：让 SessionBuilder 知道哪些 item 是可信锚点，
         同时为 StructResolver 提供 session 年份范围。

    字段说明：
        source_evidence_id  追溯到哪条原始 Evidence（调试用）
        text                Evidence 匹配到的原始文本片段（调试用，可为空）
    """

    layer: Literal["direct"] = field(default="direct", init=False)

    dt: datetime = None                  # 确定的时间点
    precision: str = "unknown"           # second / minute / hour / day / month / year
    confidence: float = 0.0             # 直接继承自源 Evidence

    source_evidence_id: str = ""        # 源 Evidence.id
    source_name: str = ""               # 源 Evidence.source（"exif"/"filename"/…）
    text: str = ""                      # 匹配到的原始文本，如 "20190213"

    def __repr__(self) -> str:
        dt_str = self.dt.isoformat() if self.dt else "None"
        return (
            f"DirectToken(dt={dt_str}, "
            f"precision={self.precision!r}, "
            f"conf={self.confidence:.2f}, "
            f"from={self.source_name!r})"
        )

    def to_dict(self) -> dict:
        return {
            "layer": "direct",
            "dt": self.dt.isoformat() if self.dt else None,
            "precision": self.precision,
            "confidence": self.confidence,
            "source_evidence_id": self.source_evidence_id,
            "source_name": self.source_name,
            "text": self.text,
        }


# ---------------------------------------------------------------------------
# 第二层：结构时间 Token
# ---------------------------------------------------------------------------

@dataclass
class StructToken:
    """
    来自文件夹名中有歧义的数字段。

    PatternMiner 处理不到的两类情况：
        A. 低频结构：只出现一次，频次未达 PatternMiner 门槛
        B. 跨文件歧义：单看文件名无法确定分段方案，需要 session 时间上下文

    生命周期：
        Tokenizer 创建时：candidates 已列出，resolved_* 为 None
        StructResolver 处理后：resolved_dt / resolved_precision /
                               resolved_confidence 填入

    字段说明：
        folder_text     原始文件夹名（如 "都江堰_0213"）
        digit_runs      从 folder_text 中提取的各数字段字符串（如 ["0213"]）
        digit_lengths   各数字段的长度（如 (4,)）
        candidates      所有合法的 datetime 解释（至少 2 个才生成 StructToken）
        source_folder   该 token 来自 relpath 的第几层文件夹（0=直接父级）
    """

    layer: Literal["struct"] = field(default="struct", init=False)

    folder_text: str = ""               # 原始文件夹名
    digit_runs: list[str] = field(default_factory=list)     # ["02", "13"]
    digit_lengths: tuple = ()           # (2, 2)
    candidates: list[datetime] = field(default_factory=list)  # 所有合法解释
    source_folder: int = 0             # 0=直接父级, 1=祖父级, 2=更高层

    # StructResolver 填入（消歧前为 None）
    resolved_dt: Optional[datetime] = None
    resolved_precision: str = "unknown"
    resolved_confidence: float = 0.0
    resolved_reason: str = ""           # 消歧依据（调试用）

    @property
    def is_resolved(self) -> bool:
        return self.resolved_dt is not None

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def __repr__(self) -> str:
        cands = [d.strftime("%Y-%m-%d") for d in self.candidates]
        resolved = (
            f" → {self.resolved_dt.strftime('%Y-%m-%d')}"
            f"({self.resolved_precision}, conf={self.resolved_confidence:.2f})"
            if self.is_resolved else " [unresolved]"
        )
        return (
            f"StructToken({self.folder_text!r}, "
            f"candidates={cands}{resolved})"
        )

    def to_dict(self) -> dict:
        return {
            "layer": "struct",
            "folder_text": self.folder_text,
            "digit_runs": self.digit_runs,
            "digit_lengths": list(self.digit_lengths),
            "candidates": [d.isoformat() for d in self.candidates],
            "source_folder": self.source_folder,
            "resolved_dt": self.resolved_dt.isoformat() if self.resolved_dt else None,
            "resolved_precision": self.resolved_precision,
            "resolved_confidence": self.resolved_confidence,
            "resolved_reason": self.resolved_reason,
        }


# ---------------------------------------------------------------------------
# 第三层：语义时间 Token（D 阶段实现，本阶段只占位）
# ---------------------------------------------------------------------------

@dataclass
class SemanticToken:
    """
    来自语义文本（地名/节日/季节词）的概率分布。

    本阶段（A/B/C）：Tokenizer._extract_semantic() 返回空列表，
                     此类不被实例化。
    D 阶段实现后：SemanticResolver 用 session 锚点做后验更新，
                  产出 Evidence(source="timeline")。

    字段说明：
        prior_month_range   先验月份约束，如 (3, 5) 表示 3~5 月
                            None 表示无月份约束（纯地名等）
        prior_confidence    词典匹配的基础置信度
        resolved_*          SemanticResolver 填入（D 阶段）
    """

    layer: Literal["semantic"] = field(default="semantic", init=False)

    text: str = ""                          # 原始词/短语，如 "春游"
    semantic_type: str = "unknown"          # place / festival / season / activity

    # 先验（词典查询结果，D 阶段填入）
    prior_month_range: Optional[tuple[int, int]] = None
    prior_year_hint: Optional[int] = None
    prior_confidence: float = 0.0

    # 后验（SemanticResolver 填入，D 阶段前始终为 None）
    resolved_dt: Optional[datetime] = None
    resolved_confidence: float = 0.0

    @property
    def is_resolved(self) -> bool:
        return self.resolved_dt is not None

    def __repr__(self) -> str:
        return (
            f"SemanticToken({self.text!r}, "
            f"type={self.semantic_type!r}, "
            f"prior_conf={self.prior_confidence:.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "layer": "semantic",
            "text": self.text,
            "semantic_type": self.semantic_type,
            "prior_month_range": list(self.prior_month_range) if self.prior_month_range else None,
            "prior_year_hint": self.prior_year_hint,
            "prior_confidence": self.prior_confidence,
            "resolved_dt": self.resolved_dt.isoformat() if self.resolved_dt else None,
            "resolved_confidence": self.resolved_confidence,
        }


# ---------------------------------------------------------------------------
# 类型别名（供其他模块 import）
# ---------------------------------------------------------------------------

AnyToken = DirectToken | StructToken | SemanticToken
TokenLayer = Literal["direct", "struct", "semantic"]


# ---------------------------------------------------------------------------
# 独立测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime

    print("=" * 50)
    print("测试 DirectToken")
    print("=" * 50)

    dt1 = DirectToken(
        dt=datetime(2019, 2, 13, 14, 30, 0),
        precision="second",
        confidence=1.0,
        source_evidence_id="ev-001",
        source_name="exif",
        text="2019:02:13 14:30:00",
    )
    print(dt1)
    print("to_dict:", dt1.to_dict())

    dt2 = DirectToken(
        dt=datetime(2019, 2, 13),
        precision="day",
        confidence=0.75,
        source_name="filename",
        text="20190213",
    )
    print(dt2)

    print()
    print("=" * 50)
    print("测试 StructToken（消歧前）")
    print("=" * 50)

    st1 = StructToken(
        folder_text="0213",
        digit_runs=["0213"],
        digit_lengths=(4,),
        candidates=[
            datetime(2019, 2, 13),   # 解释为 MMDD（需要外部年份）
            datetime(2013, 2, 1),    # 解释为 YYMM（2013年2月）
        ],
        source_folder=0,
    )
    print(st1)
    print("is_resolved:", st1.is_resolved)
    print("n_candidates:", st1.n_candidates)

    print()
    print("模拟 StructResolver 填入结果：")
    st1.resolved_dt = datetime(2019, 2, 13)
    st1.resolved_precision = "day"
    st1.resolved_confidence = 0.65
    st1.resolved_reason = "session year_candidates=[2019] 排除了 2013 年解释"
    print(st1)
    print("is_resolved:", st1.is_resolved)

    print()
    print("=" * 50)
    print("测试 StructToken（多段数字歧义）")
    print("=" * 50)

    st2 = StructToken(
        folder_text="190213",
        digit_runs=["190213"],
        digit_lengths=(6,),
        candidates=[
            datetime(2019, 2, 13),   # YYMMDD
            datetime(1902, 1, 3),    # YYYYMM 的错误解释（会被年份范围过滤）
        ],
        source_folder=1,
    )
    print(st2)

    print()
    print("=" * 50)
    print("测试 SemanticToken（D 阶段占位）")
    print("=" * 50)

    sem = SemanticToken(
        text="春游",
        semantic_type="season",
        prior_month_range=(3, 5),
        prior_confidence=0.45,
    )
    print(sem)
    print("is_resolved:", sem.is_resolved)
    print("to_dict:", sem.to_dict())

    print()
    print("=" * 50)
    print("测试 to_dict / 字段完整性")
    print("=" * 50)

    for token in [dt1, dt2, st1, st2, sem]:
        d = token.to_dict()
        assert "layer" in d, f"missing layer in {type(token).__name__}"
        print(f"  {type(token).__name__}: layer={d['layer']!r} ✓")

    print()
    print("所有测试通过 ✓")