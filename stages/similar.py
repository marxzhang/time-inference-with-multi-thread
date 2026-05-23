"""
stages/similar.py — 相似图时间推断（BatchStage）

职责：
    ClipStage 计算了所有 item 的 embedding 之后，本 stage 批量运行：
    1. 把所有 embedding 建入 faiss VectorIndex
    2. 对每个 embedding 为 None 的 item，跳过
    3. 对每个有 embedding 的 item，检索最相似的 K 个邻居
    4. 筛选出邻居中 time_result 已 resolved 且 confidence 足够高的
    5. 用邻居的时间产出 Evidence（source="similar"），追加到 item.evidence
    6. 对有新 evidence 的 item 重新运行 ResolverStage

关键设计：
    - 只对 time_result 未 resolved 或 confidence 低的 item 尝试借用时间
    - 相似度阈值（min_score）和时间借用置信度需要保守，避免误传播
    - evidence 的 confidence 按相似度线性映射，相似度越高置信度越高
    - 重新 resolve：直接调用 ResolverStage.process()，不重走整个 pipeline

相似度 → confidence 映射：
    cosine >= 0.98  → 0.75  （几乎一样的图，如连拍）
    cosine >= 0.95  → 0.65
    cosine >= 0.90  → 0.55
    cosine >= 0.85  → 0.45  （最低阈值）
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from core.stage_base import BatchStage
from storage.index import VectorIndex

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item


# 相似度阈值（低于此值的邻居不参与时间推断）
_MIN_COSINE = 0.85

# 借用时间的邻居最低 confidence 要求
# 避免用低可信度的邻居时间去推断其他 item
_MIN_NEIGHBOR_CONFIDENCE = 0.6

# 每张图最多参考的相似邻居数
_TOP_K = 5

# 相似度 → evidence confidence 的分段映射
_SCORE_CONFIDENCE_MAP = [
    (0.98, 0.75),
    (0.95, 0.65),
    (0.90, 0.55),
    (0.85, 0.45),
]


def _score_to_confidence(score: float) -> float:
    for threshold, confidence in _SCORE_CONFIDENCE_MAP:
        if score >= threshold:
            return confidence
    return 0.4


class SimilarStage(BatchStage):
    """
    基于 CLIP embedding 相似度的批量时间推断。
    在所有逐 item Stage 完成后，由 Scheduler 调用 run_batch()。
    """

    name = "SimilarStage"

    def __init__(
        self,
        min_score: float = _MIN_COSINE,
        top_k: int = _TOP_K,
        min_neighbor_confidence: float = _MIN_NEIGHBOR_CONFIDENCE,
    ) -> None:
        self.min_score = min_score
        self.top_k = top_k
        self.min_neighbor_confidence = min_neighbor_confidence

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run_batch(self, items: list["Item"], ctx: "Context") -> None:
        t0 = time.monotonic()

        # 1. 建立 id → item 映射，方便检索后快速取 item
        item_map: dict[str, "Item"] = {it.id: it for it in items}

        # 2. 筛出有 embedding 的 item
        has_embed = [it for it in items if it.clip_embedding is not None]
        if not has_embed:
            ctx.logger.info(f"[{self.name}] no embeddings available, skip")
            return

        ctx.logger.info(
            f"[{self.name}] building index with {len(has_embed)} embeddings..."
        )

        # 3. 建 faiss 索引
        dim = len(has_embed[0].clip_embedding)
        index = VectorIndex(dim=dim)
        index.add_batch([(it.id, it.clip_embedding) for it in has_embed])

        ctx.logger.info(f"[{self.name}] index built, searching neighbors...")

        # 4. 对每个"需要帮助"的 item 进行检索
        improved = 0
        for item in has_embed:
            if not self._needs_help(item, ctx):
                continue

            neighbors = index.search(
                item.clip_embedding,
                k=self.top_k,
                min_score=self.min_score,
            )
            # 排除自身
            neighbors = [(nid, sc) for nid, sc in neighbors if nid != item.id]

            if not neighbors:
                continue

            added = self._apply_neighbors(item, neighbors, item_map, ctx)
            if added:
                # 有新 evidence → 重新 resolve
                self._re_resolve(item, ctx)
                improved += 1
                item.mark_stage_done(self.name)

        elapsed = time.monotonic() - t0
        ctx.logger.info(
            f"[{self.name}] done — improved {improved} items "
            f"({elapsed:.2f}s)"
        )

    # ------------------------------------------------------------------
    # 判断是否需要相似图帮助
    # ------------------------------------------------------------------

    def _needs_help(self, item: "Item", ctx: "Context") -> bool:
        """
        只对以下情况尝试相似图推断：
        1. 完全没有 time_result（unresolved）
        2. 置信度低于阈值
        """
        tr = item.time_result
        if not tr.is_resolved:
            return True
        return tr.confidence < ctx.config.confidence_threshold

    # ------------------------------------------------------------------
    # 把邻居时间转为 Evidence
    # ------------------------------------------------------------------

    def _apply_neighbors(
        self,
        item: "Item",
        neighbors: list[tuple[str, float]],
        item_map: dict[str, "Item"],
        ctx: "Context",
    ) -> int:
        """
        对每个有可信时间的邻居，产出一条 similar evidence。
        返回实际添加的 evidence 数量。
        """
        from stages.resolver import ResolverStage
        resolver = ResolverStage()

        added = 0
        for neighbor_id, score in neighbors:
            neighbor = item_map.get(neighbor_id)
            if neighbor is None:
                continue

            tr = neighbor.time_result
            if not tr.is_resolved:
                continue
            if tr.confidence < self.min_neighbor_confidence:
                continue
            if tr.final_datetime is None:
                continue

            ev_confidence = _score_to_confidence(score)

            # 构建 evidence（手动构建，SimilarStage 不继承 Stage）
            from core.evidence import Evidence
            import uuid
            ev = Evidence(
                source="similar",
                stage=self.name,
                dt=tr.final_datetime,
                precision=tr.precision,
                confidence=ev_confidence,
                is_direct=False,
                reason=(
                    f"similar image {neighbor.filename!r} "
                    f"(cosine={score:.3f}, "
                    f"neighbor_conf={tr.confidence:.2f})"
                ),
                metadata={
                    "matched_item_id": neighbor_id,
                    "matched_filename": neighbor.filename,
                    "cosine_score": round(score, 4),
                    "neighbor_confidence": round(tr.confidence, 3),
                    "neighbor_source": tr.primary_source,
                },
                related_item_ids=[neighbor_id],
            )
            item.add_evidence(ev)
            added += 1

        return added

    # ------------------------------------------------------------------
    # 重新 resolve
    # ------------------------------------------------------------------

    def _re_resolve(self, item: "Item", ctx: "Context") -> None:
        """
        有新 evidence 后重新运行 ResolverStage。
        直接调用 process()，绕过幂等检查（因为 evidence 已更新）。
        """
        from stages.resolver import ResolverStage
        # 清除旧的 resolve 完成记录，允许重新运行
        if ResolverStage.name in item.completed_stages:
            item.completed_stages.remove(ResolverStage.name)
        resolver = ResolverStage()
        resolver.run(item, ctx)