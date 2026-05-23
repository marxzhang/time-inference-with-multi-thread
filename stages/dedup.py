"""
stages/dedup.py — 去重分组（BatchStage）

去重强度档位（dedup_mode）
--------------------------
    "no"    不做任何去重分析，直接跳过
    "sha1"  字节级完全一致（默认）
    "phash" 包含 sha1，且合并视觉相同的图（汉明距离 <= phash_threshold）
    "clip"  包含 phash，且标记肉眼相似的图（余弦相似度 >= clip_min_score）

每一档都包含上一档的效果：clip ⊃ phash ⊃ sha1。

写入 item 的字段
-----------------
    duplicate_group_id : 重复组 ID；空字符串 = 无重复
    duplicate_rank     : 组内质量排名，0 = 最优（保留），1+ = 丢弃候选
                         clip 级仅标记，不排名，统一为 -1
    duplicate_kind     : 触发去重的环节："sha1" / "phash" / "clip" / ""

重复组内排名评分（sha1 / phash 级）
-------------------------------------
    +3.0   有 DateTimeOriginal EXIF
    +2.0   时间 confidence >= 0.9
    +1.0   时间 confidence >= 0.7
    +1.0   format_real 与 ext 匹配（无需转换）
    +0.5   有 GPS 信息
    -1.0   is_screenshot
    -0.5   is_scan
    tiebreaker：文件越大越好

clip 级
--------
    只打 NEEDS_REVIEW flag，不合并 group，不排名。
    依赖 item.clip_embedding 已计算（需要 --clip 启用）。
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from core.stage_base import BatchStage
from storage.index import _hamming_distance

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item


DedupMode = Literal["no", "sha1", "phash", "clip"]

_DEFAULT_PHASH_THRESHOLD = 4    # 汉明距离 <= 4，覆盖 JPEG 重压缩和格式转换


class DedupStage(BatchStage):
    """
    全量去重分组。强度由 dedup_mode 控制。
    """

    name = "DedupStage"

    def __init__(
        self,
        dedup_mode: DedupMode = "sha1",
        phash_threshold: int = _DEFAULT_PHASH_THRESHOLD,
        clip_min_score: float = 0.85,
    ) -> None:
        """
        参数
        ----
        dedup_mode      : 去重强度，见模块文档
        phash_threshold : phash 汉明距离阈值（仅 mode >= phash 时生效）
        clip_min_score  : CLIP 余弦相似度阈值（仅 mode == clip 时生效）
                          建议与 config.clip_min_score 保持一致
        """
        self.dedup_mode      = dedup_mode
        self.phash_threshold = phash_threshold
        self.clip_min_score  = clip_min_score

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run_batch(self, items: list["Item"], ctx: "Context") -> None:
        if self.dedup_mode == "no":
            ctx.logger.info(f"[{self.name}] dedup_mode=no, skipping")
            return

        t0 = time.monotonic()
        ctx.logger.info(
            f"[{self.name}] mode={self.dedup_mode}, "
            f"{len(items)} items..."
        )

        # ── sha1 分组（所有非 no 模式都执行）─────────────────────────
        groups = self._group_by_sha1(items)

        # ── phash 合并（phash / clip 模式）────────────────────────────
        if self.dedup_mode in ("phash", "clip"):
            groups = self._merge_by_phash(groups, ctx)

        # ── 组内排名，写入 item 字段 ──────────────────────────────────
        n_groups, n_items = 0, 0
        for group_id, group_items in groups.items():
            if len(group_items) < 2:
                continue
            n_groups += 1
            n_items  += len(group_items)
            self._rank_group(group_id, group_items)

        # ── clip 标记（仅 clip 模式，不合并 group）────────────────────
        n_clip_pairs = 0
        if self.dedup_mode == "clip":
            n_clip_pairs = self._flag_clip_similar(items, ctx)

        elapsed = time.monotonic() - t0
        ctx.logger.info(
            f"[{self.name}] done — "
            f"{n_groups} group(s), {n_items} grouped items"
            + (f", {n_clip_pairs} clip-similar pair(s)" if n_clip_pairs else "")
            + f", elapsed {elapsed:.2f}s"
        )

    # ------------------------------------------------------------------
    # sha1 分组
    # ------------------------------------------------------------------

    def _group_by_sha1(
        self,
        items: list["Item"],
    ) -> dict[str, list["Item"]]:
        sha1_map: dict[str, str] = {}
        groups: dict[str, list["Item"]] = defaultdict(list)

        for item in items:
            if not item.sha1:
                groups[f"nosha1_{item.id}"].append(item)
                continue
            if item.sha1 not in sha1_map:
                sha1_map[item.sha1] = item.sha1
            groups[sha1_map[item.sha1]].append(item)

        return dict(groups)

    # ------------------------------------------------------------------
    # phash 合并（Union-Find）
    # ------------------------------------------------------------------

    def _merge_by_phash(
        self,
        groups: dict[str, list["Item"]],
        ctx: "Context",
    ) -> dict[str, list["Item"]]:
        """
        对 sha1 不同但 phash 相近的组，用并查集合并。
        复杂度 O(n²)，n < 1 万时可接受。
        """
        # 收集每个 group 的代表 key 和 phash 代表 item
        # UnionFind 必须包含所有 group 的 key（含无 phash 的），否则 find() KeyError
        all_keys: list[str] = []
        rep_by_key: dict[str, "Item"] = {}  # 只有有 phash 的才进这里

        for gid, group in groups.items():
            rep_with_phash = next((it for it in group if it.phash), None)
            if rep_with_phash:
                key = rep_with_phash.sha1 or f"nosha1_{rep_with_phash.id}"
            else:
                # 无 phash：用第一个 item 的 key，不参与 phash 比较
                key = group[0].sha1 or f"nosha1_{group[0].id}"

            all_keys.append(key)
            if rep_with_phash:
                rep_by_key[key] = rep_with_phash

        if len(rep_by_key) < 2:
            return groups

        # 用全部 key 初始化，保证 find() 不会 KeyError
        uf = _UnionFind(all_keys)
        merged = 0

        phash_keys = list(rep_by_key.keys())
        for i in range(len(phash_keys)):
            for j in range(i + 1, len(phash_keys)):
                ka, kb = phash_keys[i], phash_keys[j]
                pa = rep_by_key[ka].phash
                pb = rep_by_key[kb].phash
                if pa and pb and _hamming_distance(pa, pb) <= self.phash_threshold:
                    uf.union(ka, kb)
                    merged += 1

        if merged == 0:
            return groups

        new_groups: dict[str, list["Item"]] = defaultdict(list)
        for gid, group in groups.items():
            rep_with_phash = next((it for it in group if it.phash), None)
            if rep_with_phash:
                key = rep_with_phash.sha1 or f"nosha1_{rep_with_phash.id}"
            else:
                key = group[0].sha1 or f"nosha1_{group[0].id}"
            new_groups[uf.find(key)].extend(group)

        ctx.logger.debug(f"[{self.name}] phash merged {merged} pair(s)")
        return dict(new_groups)

    # ------------------------------------------------------------------
    # 组内排名
    # ------------------------------------------------------------------

    def _rank_group(
        self,
        group_id: str,
        group_items: list["Item"],
    ) -> None:
        """
        评分排名，写入 duplicate_* 字段，打 DUPLICATE flag。
        duplicate_kind = 触发合并的最低级别环节。
        """
        # 判断触发环节：所有 sha1 相同 → sha1，否则 → phash
        sha1s = {it.sha1 for it in group_items if it.sha1}
        kind: str = "sha1" if len(sha1s) == 1 else "phash"

        scored = sorted(
            [(self._score(it), it.filesize, it) for it in group_items],
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )

        for rank, (score, _, item) in enumerate(scored):
            item.duplicate_group_id = group_id
            item.duplicate_rank     = rank
            item.duplicate_kind     = kind

            if rank == 0:
                item.log(
                    self.name,
                    f"kept in {kind} group "
                    f"({len(group_items)} items, score={score:.2f})"
                )
            else:
                item.add_flag("DUPLICATE")
                item.log(
                    self.name,
                    f"duplicate [{kind}] of {scored[0][2].filename!r} "
                    f"(rank={rank}, score={score:.2f})"
                )

            item.mark_stage_done(self.name)

    # ------------------------------------------------------------------
    # clip 相似标记（不合并 group，不排名）
    # ------------------------------------------------------------------

    def _flag_clip_similar(
        self,
        items: list["Item"],
        ctx: "Context",
    ) -> int:
        """
        对 CLIP 余弦相似度 >= clip_min_score 的 item 对：
        - 打 NEEDS_REVIEW flag
        - 写入 duplicate_kind="clip"，duplicate_rank=-1
        - 不合并到同一 group（group_id 保持各自独立）

        只处理尚未被 sha1/phash 归组的 item（已有 group_id 的已经处理过了）。
        返回标记的 pair 数量。
        """
        has_embed = [it for it in items
                     if it.clip_embedding is not None
                     and not it.duplicate_group_id]

        if len(has_embed) < 2:
            return 0

        try:
            import numpy as np
        except ImportError:
            ctx.logger.debug(f"[{self.name}] numpy not available, skip clip dedup")
            return 0

        # 构建矩阵，批量计算余弦相似度（向量已 L2 归一化，点积 = 余弦）
        matrix = np.array(
            [it.clip_embedding for it in has_embed],
            dtype=np.float32,
        )
        # shape: (n, n) 相似度矩阵
        sim_matrix = matrix @ matrix.T

        flagged_ids: set[str] = set()
        pair_count = 0

        n = len(has_embed)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score < self.clip_min_score:
                    continue

                ia, ib = has_embed[i], has_embed[j]
                pair_count += 1

                for item in (ia, ib):
                    item.add_flag("NEEDS_REVIEW")
                    if item.duplicate_kind != "clip":
                        item.duplicate_kind = "clip"
                        item.duplicate_rank = -1

                ia.warn(
                    self.name,
                    f"clip-similar to {ib.filename!r} (cosine={score:.3f})"
                )
                ib.warn(
                    self.name,
                    f"clip-similar to {ia.filename!r} (cosine={score:.3f})"
                )
                flagged_ids.add(ia.id)
                flagged_ids.add(ib.id)

        if flagged_ids:
            ctx.logger.debug(
                f"[{self.name}] clip: {pair_count} pair(s), "
                f"{len(flagged_ids)} items flagged"
            )

        return pair_count

    # ------------------------------------------------------------------
    # 评分
    # ------------------------------------------------------------------

    @staticmethod
    def _score(item: "Item") -> float:
        score = 0.0
        tr = item.time_result
        if tr.is_resolved:
            score += 2.0 if tr.confidence >= 0.9 else (1.0 if tr.confidence >= 0.7 else 0)
        if item.exif.datetime_original is not None:
            score += 3.0
        if item.has_gps:
            score += 0.5
        _EXT_FMT = {
            "jpg": "JPEG", "jpeg": "JPEG", "png": "PNG",
            "heic": "HEIF", "heif": "HEIF",
            "tif": "TIFF", "tiff": "TIFF", "webp": "WEBP",
        }
        if item.format_real and _EXT_FMT.get(item.ext, "").upper() == item.format_real.upper():
            score += 1.0
        if item.is_screenshot:
            score -= 1.0
        if item.is_scan:
            score -= 0.5
        return score


# ---------------------------------------------------------------------------
# 并查集
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, elements) -> None:
        self._parent = {e: e for e in elements}
        self._rank   = {e: 0  for e in elements}

    def find(self, x) -> str:
        if x not in self._parent:
            # 未知 key：自动注册为自身根节点（防御性兜底）
            self._parent[x] = x
            self._rank[x]   = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x, y) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1