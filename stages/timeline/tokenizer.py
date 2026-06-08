"""
stages/timeline/tokenizer.py — 三层时间 Token 提取

职责
----
对单个 item 提取三类 TimeToken，供 SessionBuilder / StructResolver 消费。

三层分工
--------
    _extract_direct   第一层：直接读 item.evidence，不重新解析任何文本
    _extract_struct   第二层：识别文件夹名中 PatternMiner 未覆盖的歧义数字结构
    _extract_semantic 第三层：占位，D 阶段实现，当前返回空列表

盲区判断标准（_extract_struct）
--------------------------------
PatternMiner 已覆盖的情况（跳过，不生成 StructToken）：
    - item.evidence 里已有 source="foldername" 且 metadata["folder_name"] 匹配的条目
      → FilenameStage 已经处理过该文件夹名，无需重复
    - 数字总位数 >= 8：DATE8 / DATETIME8_6 等预置正则已全覆盖
    - 数字总位数 == 0 或 == 1：无时间信息

需要 StructToken 的盲区：
    - 总位数 2~7，且未被 foldername evidence 覆盖
    - 候选 >= 2 → StructToken（真正有歧义，需要 session 上下文消歧）
    - 候选 == 1 → DirectToken（低置信，确定但缺乏交叉验证）

设计约束
--------
    - Tokenizer 是无状态的，extract() 只看当前 item，不需要其他 item 的信息
    - 不重复 PatternMiner / FilenameStage 已完成的工作
    - 复用 utils.miner 里的 _try_all_schemes / _extract_digit_values
    - 复用 utils.patterns 里的 _build_datetime
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

# from stages.timeline.tokens import DirectToken, StructToken, SemanticToken
from tokens import DirectToken, StructToken, SemanticToken

if TYPE_CHECKING:
    from core.item import Item


# 数字段总位数落在此范围内才考虑作为 StructToken 候选
# 上限 7：8 位及以上已被 DATE8 / DATETIME 系列覆盖
# 下限 2：单位数无法构成日期，两位数（如"13"）才有可能是月或日
_STRUCT_MIN_DIGITS = 2
_STRUCT_MAX_DIGITS = 7

# 单段数字的正则（提取 folder_text 中所有连续数字串）
_DIGIT_RE = re.compile(r'\d+')

# 单段非数字字母词（供 _extract_semantic 占位用）
_ALPHA_RE = re.compile(r'[^\d\W_]+', re.UNICODE)

# StructToken 候选为 1 个时，直接转 DirectToken 的置信度
# 比正常 foldername confidence (0.6~0.75) 低，因为缺乏 PatternMiner 的频次验证
_SINGLE_CANDIDATE_CONFIDENCE = 0.45


class Tokenizer:
    """
    从单个 item 提取三层 TimeToken。

    使用示例
    --------
    t = Tokenizer()
    direct, struct, semantic = t.extract(item)
    """

    def extract(
        self,
        item: "Item",
    ) -> tuple[list[DirectToken], list[StructToken], list[SemanticToken]]:
        """
        对单个 item 提取全部 TimeToken。

        返回
        ----
        (direct_tokens, struct_tokens, semantic_tokens)
        三个列表独立，调用方按需使用。
        semantic_tokens 在 D 阶段前始终为空列表。
        """
        direct          = self._extract_direct(item)
        extra_d, struct = self._extract_struct(item)
        direct.extend(extra_d)
        direct.sort(key=lambda t: t.confidence, reverse=True)
        semantic        = self._extract_semantic(item)
        return direct, struct, semantic

    # ------------------------------------------------------------------
    # 第一层：直接读 item.evidence
    # ------------------------------------------------------------------

    def _extract_direct(self, item: "Item") -> list[DirectToken]:
        """
        把 item.evidence 里所有有确定时间的条目包装为 DirectToken。

        不重新解析任何文本——时间值直接来自 Evidence.dt，
        置信度直接来自 Evidence.confidence。
        这保证 DirectToken 与第一阶段的决策完全一致。
        """
        tokens: list[DirectToken] = []
        for ev in item.evidence:
            if ev.dt is None:
                continue
            tokens.append(DirectToken(
                dt=ev.dt,
                precision=ev.precision,
                confidence=ev.confidence,
                source_evidence_id=ev.id,
                source_name=ev.source,
                text=ev.metadata.get("matched_text", ""),
            ))
        # 按置信度降序，让 SessionBuilder 优先看到最可信的
        tokens.sort(key=lambda t: t.confidence, reverse=True)
        return tokens

    # ------------------------------------------------------------------
    # 第二层：文件夹名中的歧义数字结构
    # ------------------------------------------------------------------

    def _extract_struct(
        self,
        item: "Item",
    ) -> tuple[list[DirectToken], list[StructToken]]:
        """
        从 item.relpath 各级文件夹名中提取 PatternMiner 未覆盖的数字结构。

        返回两个列表：
            extra_direct  单候选（无歧义）→ 低置信 DirectToken
            struct        多候选（有歧义）→ StructToken，等待 StructResolver 消歧

        只处理文件夹名（不含文件名 stem）——文件名已由 FilenameStage 充分处理。
        """
        extra_direct: list[DirectToken] = []
        struct_tokens: list[StructToken] = []

        folder_parts = list(Path(item.relpath).parts[:-1])
        covered_folders = self._covered_folder_names(item)

        for depth, part in enumerate(reversed(folder_parts)):
            if depth >= 3:
                break
            singles, multis = self._analyze_folder_part(part, depth, covered_folders)
            extra_direct.extend(singles)
            struct_tokens.extend(multis)

        return extra_direct, struct_tokens

    def _covered_folder_names(self, item: "Item") -> set[str]:
        """
        收集已被 FilenameStage 产出 evidence 覆盖的文件夹名。

        FilenameStage 在 evidence.metadata["folder_name"] 里记录了
        产生该 evidence 的文件夹名，此处直接读取。
        """
        covered: set[str] = set()
        for ev in item.evidence:
            if ev.source == "foldername":
                folder_name = ev.metadata.get("folder_name", "")
                if folder_name:
                    covered.add(folder_name)
        return covered

    def _analyze_folder_part(
        self,
        part: str,
        depth: int,
        covered_folders: set[str],
    ) -> tuple[list[DirectToken], list[StructToken]]:
        """
        分析单个文件夹名。

        返回 (single_candidate_directs, multi_candidate_structs)：
            single_candidate_directs : 唯一解释 → 低置信 DirectToken
            multi_candidate_structs  : 有歧义   → StructToken
        两个列表均可为空。
        """
        empty: tuple[list, list] = ([], [])

        if part in covered_folders:
            return empty

        digit_runs = _DIGIT_RE.findall(part)
        if not digit_runs:
            return empty

        total_digits = sum(len(d) for d in digit_runs)
        if total_digits < _STRUCT_MIN_DIGITS or total_digits > _STRUCT_MAX_DIGITS:
            return empty

        candidates = self._get_candidates(digit_runs)
        if not candidates:
            return empty

        digit_lengths = tuple(len(d) for d in digit_runs)

        if len(candidates) == 1:
            return ([DirectToken(
                dt=candidates[0],
                precision=self._infer_precision(digit_lengths),
                confidence=_SINGLE_CANDIDATE_CONFIDENCE,
                source_name="foldername_unambiguous",
                text=part,
            )], [])

        return ([], [StructToken(
            folder_text=part,
            digit_runs=list(digit_runs),
            digit_lengths=digit_lengths,
            candidates=candidates,
            source_folder=depth,
        )])

    def _get_candidates(self, digit_runs: list[str]) -> list:
        """
        列出所有合法的 datetime 解释。

        两条路径：
          A. _try_all_schemes（miner 现有逻辑）：覆盖总位数 >= 4 且年份 >= 4 位的情况
          B. _get_short_candidates（本模块补充）：覆盖 _try_all_schemes 的盲区
               - 两位年份（YYMMDD / YYMM）
               - 纯 MMDD（无年份，用占位年 2000 表示"年份待定"）
               - 多段短数字（如 (2,2) 结构的 MM_DD）

        去重：以 isoformat 字符串为 key，保留先出现的。
        """
        from utils.miner import _try_all_schemes
        from utils.patterns import _build_datetime

        digit_lengths = tuple(len(d) for d in digit_runs)
        seen: set[str] = set()
        candidates: list = []

        def _add(dt):
            if dt is None:
                return
            key = dt.isoformat()
            if key not in seen:
                seen.add(key)
                candidates.append(dt)

        # 路径 A：miner 现有逻辑（覆盖 YYYY 开头的格式）
        for scheme in _try_all_schemes(digit_lengths, digit_runs):
            _add(_build_datetime(scheme.fields))

        # 路径 B：短数字段补充（_try_all_schemes 的盲区）
        for dt in self._get_short_candidates(digit_runs, digit_lengths):
            _add(dt)

        return candidates

    def _get_short_candidates(
        self,
        digit_runs: list[str],
        digit_lengths: tuple,
    ) -> list:
        """
        _try_all_schemes 盲区补充：处理两位年份和纯 MMDD 结构。

        覆盖的格式：
            单段 6 位 → YYMMDD（19 0213 → 2019-02-13）
            单段 4 位 → MMDD（0213 → 2000-02-13，年份用 2000 占位）
                      → YYMM（1902 → 2019-02，但月份要合法）
            多段 (2,2)→ MM_DD（02_13 → 2000-02-13）
            多段 (2,4)/(4,2) → MMYYYY / YYYYMM（已被 miner 覆盖，跳过）

        两位年份补全策略：
            00~30 → 2000~2030，31~99 → 1931~1999
        """
        from datetime import datetime

        results = []
        joined = "".join(digit_runs)
        total = len(joined)

        def _expand_yy(yy: int) -> int:
            return 2000 + yy if yy <= 30 else 1900 + yy

        def _try_dt(*args) -> datetime | None:
            try:
                return datetime(*args)
            except ValueError:
                return None

        # ── 单段 ──────────────────────────────────────────────────────
        if len(digit_runs) == 1:
            s = joined

            if total == 6:
                # YYMMDD：190213 → 2019-02-13
                yy, mm, dd = int(s[0:2]), int(s[2:4]), int(s[4:6])
                results.append(_try_dt(_expand_yy(yy), mm, dd))

                # YYYYM_：1902_13 已被 miner 的 YEARMONTH 覆盖，跳过

            elif total == 4:
                mm, dd = int(s[0:2]), int(s[2:4])
                # MMDD：0213 → 月=02, 日=13，年份用 2000 占位（表示"年份待定"）
                results.append(_try_dt(2000, mm, dd))

                # YYMM：0213 → 年=2002, 月=13（月非法，会被 _try_dt 过滤）
                yy2 = int(s[0:2])
                mm2 = int(s[2:4])
                results.append(_try_dt(_expand_yy(yy2), mm2, 1))

            elif total == 5:
                # YMMDD：不是标准格式，但"2_0213"这类分段会在多段里处理
                # 单段 5 位极罕见，保守处理：不生成候选
                pass

        # ── 多段 ──────────────────────────────────────────────────────
        elif len(digit_runs) == 2:
            a, b = digit_runs
            la, lb = len(a), len(b)

            if la == 2 and lb == 2:
                # MM_DD：02_13 → 2000-02-13（年份占位）
                results.append(_try_dt(2000, int(a), int(b)))
                # DD_MM：13_02（欧式格式）→ 同样生成，让 StructResolver 消歧
                results.append(_try_dt(2000, int(b), int(a)))

            elif la == 2 and lb == 4:
                # MM_YYYY → 已被 miner YEARMONTH_SEP 覆盖，跳过
                pass

            elif la == 4 and lb == 2:
                # YYYY_MM → 已被 miner YEARMONTH6 / YEARMONTH_SEP 覆盖，跳过
                pass

            elif la == 2 and lb == 6:
                # YY_MMDDXX：少见，跳过
                pass

        # 过滤 None
        return [dt for dt in results if dt is not None]

    def _infer_precision(self, digit_lengths: tuple) -> str:
        """根据数字段结构推断时间精度（单候选时使用）。"""
        total = sum(digit_lengths)
        if total >= 14:
            return "second"
        if total >= 12:
            return "minute"
        if total >= 8:
            return "day"
        if total >= 6:
            return "month"
        if total >= 4:
            return "year"
        return "unknown"

    # ------------------------------------------------------------------
    # 第三层：语义文本提取（D 阶段实现，当前占位）
    # ------------------------------------------------------------------

    def _extract_semantic(self, item: "Item") -> list[SemanticToken]:
        """
        从文件夹名中提取语义词（地名/节日/季节词）。

        D 阶段实现，当前返回空列表。
        接口签名固定，D 阶段只需填入实现，调用方不需要改动。
        """
        # [D阶段] 实现：
        # tokens = []
        # for part in Path(item.relpath).parts[:-1]:
        #     words = _ALPHA_RE.findall(part)
        #     for word in words:
        #         if len(word) >= 2:
        #             from stages.timeline.semantic import _lookup_semantic
        #             token = _lookup_semantic(word)
        #             if token:
        #                 tokens.append(token)
        # return tokens
        return []


# ---------------------------------------------------------------------------
# 独立测试（__main__）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # ── 路径设置，在项目根目录下运行 ──────────────────────────────────
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from tokenizer import Tokenizer

    print("=" * 60)
    print("模式 A：用假数据验证逻辑")
    print("=" * 60)

    # 构造最小假 Item，不依赖真实图片
    from core.item import Item, TimeResult
    from core.evidence import Evidence
    from datetime import datetime

    def _make_item(relpath: str, evidences: list = None) -> Item:
        item = Item(path=f"/fake/{relpath}", relpath=relpath)
        item.filename = Path(relpath).name
        item.ext = Path(relpath).suffix.lstrip(".")
        if evidences:
            for ev in evidences:
                item.add_evidence(ev)
        return item

    t = Tokenizer()

    # 测试 1：有 EXIF evidence 的 item → DirectToken
    print("\n[测试1] 有 EXIF evidence → DirectToken")
    ev_exif = Evidence(
        source="exif", stage="ExifStage",
        dt=datetime(2019, 2, 13, 14, 30, 0),
        precision="second", confidence=1.0,
        metadata={"matched_text": "2019:02:13 14:30:00"}
    )
    item1 = _make_item("旅行/成都/IMG_001.jpg", [ev_exif])
    d, s, sem = t.extract(item1)
    print(f"  DirectToken 数量: {len(d)}")
    for tok in d:
        print(f"    {tok}")
    print(f"  StructToken 数量: {len(s)}")

    # 测试 2：文件夹名 "0213" → 唯一合法解释 MMDD → DirectToken（低置信）
    # 注：YYMM 解释 month=13 非法，被过滤，只剩 MMDD 一个候选
    print("\n[测试2] 文件夹名 '0213' → 单候选 MMDD → DirectToken（年份占位 2000）")
    item2 = _make_item("旅行/0213/IMG_002.jpg")
    d, s, sem = t.extract(item2)
    extra = [tok for tok in d if tok.source_name == "foldername_unambiguous"]
    print(f"  foldername DirectToken 数量: {len(extra)}（期望 1）")
    for tok in extra:
        print(f"    {tok}")
    print(f"  StructToken 数量: {len(s)}（期望 0，单候选不歧义）")

    # 测试 3：foldername evidence 已覆盖的文件夹 → 完全跳过
    print("\n[测试3] 已有 foldername evidence 覆盖 → 跳过该文件夹")
    ev_folder = Evidence(
        source="foldername", stage="FilenameStage",
        dt=datetime(2019, 2, 13),
        precision="day", confidence=0.72,
        metadata={"folder_name": "0213", "matched_text": "0213"}
    )
    item3 = _make_item("旅行/0213/IMG_003.jpg", [ev_folder])
    d, s, sem = t.extract(item3)
    extra3 = [tok for tok in d if tok.source_name == "foldername_unambiguous"]
    print(f"  foldername DirectToken 数量: {len(extra3)}（期望 0）")
    print(f"  StructToken 数量: {len(s)}（期望 0）")

    # 测试 4：文件夹名 "190213"（YYMMDD，唯一合法解释）→ DirectToken
    print("\n[测试4] 文件夹名 '190213'（YYMMDD → 唯一合法解释 2019-02-13）")
    item4 = _make_item("相册/190213/photo.jpg")
    d, s, sem = t.extract(item4)
    extra4 = [tok for tok in d if tok.source_name == "foldername_unambiguous"]
    print(f"  foldername DirectToken: {len(extra4)}（期望 1）")
    for tok in extra4:
        print(f"    {tok}")

    # 测试 5：文件夹名 "20190213"（8位）→ 超出盲区，跳过
    print("\n[测试5] 文件夹名 '20190213'（8位 DATE8）→ 跳过（超出盲区范围）")
    item5 = _make_item("相册/20190213/photo.jpg")
    d, s, sem = t.extract(item5)
    extra5 = [tok for tok in d if tok.source_name == "foldername_unambiguous"]
    print(f"  foldername DirectToken: {len(extra5)}（期望 0）")
    print(f"  StructToken 数量: {len(s)}（期望 0）")

    # 测试 6：文件夹名 "0901"（两种合法解释：MMDD 和 YYMM）→ StructToken
    # "0901" → MMDD: 2000-09-01, YYMM: 2009-01-01，两者都合法
    print("\n[测试6] 文件夹名 '0901'（MMDD 和 YYMM 均合法）→ StructToken")
    item6 = _make_item("日记/0901/photo.jpg")
    d, s, sem = t.extract(item6)
    print(f"  StructToken 数量: {len(s)}（期望 1）")
    for tok in s:
        cands = [c.strftime("%Y-%m-%d") for c in tok.candidates]
        print(f"    {tok.folder_text!r} → 候选: {cands}")

    # 测试 7：多层文件夹，各层独立分析
    print("\n[测试7] 多层文件夹 'backup/2019/0213/photo.jpg'")
    item7 = _make_item("backup/2019/0213/photo.jpg")
    d, s, sem = t.extract(item7)
    extra7 = [tok for tok in d if tok.source_name == "foldername_unambiguous"]
    print(f"  foldername DirectToken 数量: {len(extra7)}（'0213'→MMDD, '2019'被年份正则覆盖应=0或1）")
    for tok in extra7:
        print(f"    {tok}")
    print(f"  StructToken 数量: {len(s)}")
    for tok in s:
        print(f"    depth={tok.source_folder} {tok.folder_text!r} → {[c.strftime('%Y-%m-%d') for c in tok.candidates]}")

    print()
    print("=" * 60)

    # ── 模式 B：读真实 snapshot（可选）──────────────────────────────────
    if len(sys.argv) > 1:
        snap_path = Path(sys.argv[1])
        print(f"\n模式 B：真实 snapshot — {snap_path}")
        print("=" * 60)

        from storage.snapshot import load
        items = load(snap_path)
        print(f"加载 {len(items)} 个 item")

        t = Tokenizer()
        total_direct = total_struct = 0
        struct_examples: list[StructToken] = []

        for item in items:
            d, s, _ = t.extract(item)
            total_direct += len(d)
            total_struct += len(s)
            struct_examples.extend(s)

        print(f"\n统计：")
        print(f"  DirectToken 总数: {total_direct}")
        print(f"  StructToken 总数: {total_struct}")

        if struct_examples:
            print(f"\n前 20 个 StructToken（按候选数降序）：")
            struct_examples.sort(key=lambda tok: tok.n_candidates, reverse=True)
            for tok in struct_examples[:20]:
                cands = [d.strftime("%Y-%m-%d") for d in tok.candidates]
                print(
                    f"  [{tok.source_folder}层] {tok.folder_text!r}"
                    f"  → {tok.n_candidates} 个候选: {cands}"
                )
        else:
            print("\n未发现 StructToken（所有文件夹名均已被 PatternMiner 覆盖，或无歧义数字）")