"""
storage/index.py — 向量索引（faiss）和 hash 索引

职责：
    1. VectorIndex   : faiss 向量近邻检索（用于 CLIP embedding 相似图）
    2. HashIndex     : phash / sha1 精确匹配（用于去重和精确相似）

依赖：
    pip install faiss-cpu   （CPU 版，无 GPU 依赖）
    pip install faiss-gpu   （GPU 版，可选）

设计：
    VectorIndex 使用 IndexFlatIP（内积 = 余弦相似度，要求向量已 L2 归一化）。
    对于 >100 万向量的场景，可换 IndexIVFFlat（需要训练），接口不变。

    HashIndex 用 Python dict，O(1) 查找，内存占用极小。
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# VectorIndex — CLIP embedding 近邻检索
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    faiss 向量索引，支持增量添加和近邻检索。

    使用示例（SimilarStage 中）
    --------------------------
    index = VectorIndex(dim=512)
    for item in items:
        if item.clip_embedding:
            index.add(item.id, item.clip_embedding)
    index.build()  # 当前 IndexFlatIP 无需显式 build，保留接口兼容性

    results = index.search(query_embedding, k=5)
    # results = [(item_id, score), ...]，score 越大越相似（余弦相似度）
    """

    def __init__(self, dim: int = 512) -> None:
        """
        参数
        ----
        dim : embedding 向量维度，必须与 ClipModel.dim 一致
        """
        self.dim = dim
        self._index = None          # faiss index，懒初始化
        self._id_map: list[str] = []  # faiss 内部整数 id → item_id 的映射

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def add(self, item_id: str, embedding: list[float]) -> None:
        """
        添加一个向量。可在 build() 前多次调用。
        embedding 必须已 L2 归一化（ClipModel 的输出默认已归一化）。
        """
        self._ensure_index()
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        self._index.add(vec)
        self._id_map.append(item_id)

    def add_batch(self, items: list[tuple[str, list[float]]]) -> None:
        """
        批量添加，比逐个 add() 快（减少 faiss C++ 调用次数）。
        items: [(item_id, embedding), ...]
        """
        if not items:
            return
        self._ensure_index()
        ids, vecs = zip(*items)
        matrix = np.array(vecs, dtype=np.float32)
        self._index.add(matrix)
        self._id_map.extend(ids)

    def build(self) -> None:
        """
        为将来换用 IndexIVFFlat 等需要训练的索引预留接口。
        IndexFlatIP 无需训练，此方法为空操作。
        """
        pass

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        embedding: list[float],
        k: int = 10,
        min_score: float = 0.85,
    ) -> list[tuple[str, float]]:
        """
        近邻检索，返回最相似的 k 个结果。

        参数
        ----
        embedding  : 查询向量（已 L2 归一化）
        k          : 返回结果数量
        min_score  : 最低余弦相似度阈值（低于此值的结果过滤掉）

        返回
        ----
        [(item_id, score), ...]，按 score 降序，不含查询向量自身
        （调用方负责排除 item_id == query_item_id 的结果）
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        actual_k = min(k + 1, self._index.ntotal)  # +1 是为了排除自身
        scores, indices = self._index.search(vec, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # faiss 用 -1 表示无效结果
                continue
            if float(score) < min_score:
                continue
            results.append((self._id_map[idx], float(score)))

        return results

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """索引中的向量数量。"""
        return len(self._id_map)

    def is_empty(self) -> bool:
        return self.size == 0

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError:
            raise RuntimeError(
                "faiss not installed. Run: pip install faiss-cpu"
            )
        # IndexFlatIP：精确内积检索，要求向量已 L2 归一化
        self._index = faiss.IndexFlatIP(self.dim)

    def __repr__(self) -> str:
        return f"VectorIndex(dim={self.dim}, size={self.size})"


# ---------------------------------------------------------------------------
# HashIndex — phash / sha1 精确匹配
# ---------------------------------------------------------------------------

class HashIndex:
    """
    基于 dict 的 hash 精确匹配索引。
    用于：
    - phash 相似匹配（汉明距离 <= 阈值）
    - sha1 精确去重

    使用示例
    --------
    index = HashIndex()
    for item in items:
        index.add_phash(item.id, item.phash)

    # 查找与 query_phash 汉明距离 <= 8 的所有 item
    matches = index.search_phash(query_phash, max_distance=8)
    """

    def __init__(self) -> None:
        self._phash_map: dict[str, str] = {}  # item_id → phash hex str
        self._sha1_map:  dict[str, str] = {}  # sha1 → item_id（第一个见到的）

    # ------------------------------------------------------------------
    # phash
    # ------------------------------------------------------------------

    def add_phash(self, item_id: str, phash: str) -> None:
        if phash:
            self._phash_map[item_id] = phash

    def search_phash(
        self,
        query_phash: str,
        max_distance: int = 8,
        exclude_id: Optional[str] = None,
    ) -> list[tuple[str, int]]:
        """
        查找汉明距离 <= max_distance 的所有 item。

        返回
        ----
        [(item_id, hamming_distance), ...]，按距离升序
        """
        if not query_phash:
            return []

        results = []
        for item_id, phash in self._phash_map.items():
            if item_id == exclude_id:
                continue
            dist = _hamming_distance(query_phash, phash)
            if dist <= max_distance:
                results.append((item_id, dist))

        results.sort(key=lambda x: x[1])
        return results

    # ------------------------------------------------------------------
    # sha1（精确去重）
    # ------------------------------------------------------------------

    def add_sha1(self, item_id: str, sha1: str) -> None:
        if sha1 and sha1 not in self._sha1_map:
            self._sha1_map[sha1] = item_id

    def find_duplicate(self, sha1: str) -> Optional[str]:
        """
        查找相同 sha1 的 item_id（第一个见到的）。
        返回 None 表示无重复。
        """
        return self._sha1_map.get(sha1)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    @property
    def phash_count(self) -> int:
        return len(self._phash_map)

    @property
    def sha1_count(self) -> int:
        return len(self._sha1_map)

    def __repr__(self) -> str:
        return (
            f"HashIndex(phash={self.phash_count}, sha1={self.sha1_count})"
        )


# ---------------------------------------------------------------------------
# 工具：汉明距离
# ---------------------------------------------------------------------------

def _hamming_distance(a: str, b: str) -> int:
    """
    计算两个十六进制 phash 字符串的汉明距离（不同位数）。
    长度不同时取较短者长度计算（容错）。
    """
    if not a or not b:
        return 64  # 视为完全不同
    try:
        int_a = int(a, 16)
        int_b = int(b, 16)
        return bin(int_a ^ int_b).count("1")
    except ValueError:
        return 64