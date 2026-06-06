"""
storage/snapshot.py — 阶段间快照

解决的问题
----------
第一阶段（Scan → Exif → Filename → Resolver → Dedup）耗时较长，
开发第二阶段（Timeline）时不应该每次都重跑第一阶段。

与 Checkpoint 的区别
--------------------
    Checkpoint : 中断恢复，任务完成后 clear() 删除，不保证存在
    Snapshot   : 阶段间接口，主动写出，长期保留，供第二阶段独立消费

格式
----
    jsonl，每行一条 item.to_dict() 的 JSON 序列化。
    与 checkpoint.jsonl 完全相同的格式，直接复用 Item.from_dict()。
    同一 item_id 出现多行时取最后一条（与 Checkpoint._load_saved 一致）。

典型用法
--------
    # 第一阶段完成后保存（main.py 里）
    from storage.snapshot import save
    save(items, Path(".cache/phase1.jsonl"))

    # 第二阶段开发时加载（dev_timeline.py 里）
    from storage.snapshot import load
    items = load(Path(".cache/phase1.jsonl"))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.item import Item


def save(items: list["Item"], path: Path) -> int:
    """
    将 items 完整状态写入 snapshot 文件。

    参数
    ----
    items : 第一阶段处理完毕的 item 列表
    path  : 输出路径，父目录不存在时自动创建

    返回
    ----
    实际写入的条目数（去重后）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            if item.id in seen:
                continue
            seen.add(item.id)
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            written += 1

    return written


def load(path: Path) -> list["Item"]:
    """
    从 snapshot 文件恢复 items，完整还原所有字段。

    包含：evidence, time_result, completed_stages, flags,
          exif, clip_embedding, sha1, phash 等第一阶段产出的全部信息。

    同一 item_id 出现多行时取最后一条。

    参数
    ----
    path : snapshot 文件路径

    返回
    ----
    list[Item]，顺序与写入时一致（同 id 去重后）
    """
    from core.item import Item

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"snapshot not found: {path}")

    # 同 id 取最后一条
    seen: dict[str, dict] = {}
    order: list[str] = []          # 保持写入顺序

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                item_id = d.get("id", "")
                if not item_id:
                    continue
                if item_id not in seen:
                    order.append(item_id)
                seen[item_id] = d
            except json.JSONDecodeError as e:
                # 损坏的行跳过，不中断加载
                import warnings
                warnings.warn(f"snapshot line {lineno} malformed, skipped: {e}")

    return [Item.from_dict(seen[iid]) for iid in order]


def stats(path: Path) -> dict:
    """
    读取 snapshot 的统计信息，不加载完整 item（适合快速检查）。

    返回
    ----
    {
        "total": int,           # 条目数（去重后）
        "resolved": int,        # time_result 已解析的数量
        "has_exif": int,        # 有 EXIF 时间的数量
        "has_clip": int,        # 有 clip_embedding 的数量
        "flags": dict,          # 各 flag 的出现次数
        "stages": dict,         # 各 completed_stage 的出现次数
        "file_size_mb": float,
    }
    """
    from collections import Counter

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"snapshot not found: {path}")

    seen: set[str] = set()
    total = resolved = has_exif = has_clip = 0
    flag_counter: Counter = Counter()
    stage_counter: Counter = Counter()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                iid = d.get("id", "")
                if not iid or iid in seen:
                    continue
                seen.add(iid)
                total += 1

                tr = d.get("time_result", {})
                if tr.get("final_datetime"):
                    resolved += 1

                exif = d.get("exif", {})
                if exif.get("datetime_original"):
                    has_exif += 1

                if d.get("clip_embedding"):
                    has_clip += 1

                for flag in d.get("flags", []):
                    flag_counter[flag] += 1

                for stage in d.get("completed_stages", []):
                    stage_counter[stage] += 1

            except json.JSONDecodeError:
                pass

    file_size_mb = path.stat().st_size / (1024 * 1024)

    return {
        "total": total,
        "resolved": resolved,
        "has_exif": has_exif,
        "has_clip": has_clip,
        "flags": dict(flag_counter),
        "stages": dict(stage_counter),
        "file_size_mb": round(file_size_mb, 2),
    }


# ---------------------------------------------------------------------------
# 独立测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile
    from datetime import datetime
    from pathlib import Path

    print("=" * 50)
    print("测试 snapshot save / load 往返")
    print("=" * 50)

    # 构造最小可用的假 Item（不依赖真实图片）
    # 直接用 Item.from_dict 构造，避免 import 链问题
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.item import Item, ExifData, TimeResult
    from core.evidence import Evidence

    def _make_fake_item(idx: int) -> Item:
        item = Item(
            path=f"/fake/photos/img_{idx:03d}.jpg",
            relpath=f"photos/img_{idx:03d}.jpg",
        )
        item.filename = f"img_{idx:03d}.jpg"
        item.ext = "jpg"
        item.filesize = 1024 * idx
        item.sha1 = f"sha1_{idx:040d}"
        item.phash = f"phash_{idx}"
        item.width, item.height = 4032, 3024
        item.format_real = "JPEG"

        dt = datetime(2019, 2, idx % 28 + 1, 10, 0, 0)
        item.exif = ExifData(datetime_original=dt)

        ev = Evidence(
            source="exif",
            stage="ExifStage",
            dt=dt,
            precision="second",
            confidence=1.0,
        )
        item.add_evidence(ev)

        item.time_result = TimeResult(
            final_datetime=dt,
            confidence=1.0,
            precision="second",
            primary_source="exif",
        )
        item.mark_stage_done("ExifStage")
        item.mark_stage_done("ResolverStage")

        return item

    # 生成假数据
    fake_items = [_make_fake_item(i) for i in range(1, 6)]
    print(f"生成 {len(fake_items)} 个假 Item")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        snap_path = Path(f.name)

    try:
        # 测试 save
        n = save(fake_items, snap_path)
        print(f"save() 写入 {n} 条  →  {snap_path}")
        assert n == len(fake_items), f"期望 {len(fake_items)}，实际 {n}"

        # 测试 load
        loaded = load(snap_path)
        print(f"load() 恢复 {len(loaded)} 条")
        assert len(loaded) == len(fake_items)

        # 验证字段往返
        for orig, restored in zip(fake_items, loaded):
            assert orig.id == restored.id, "id 不一致"
            assert orig.path == restored.path, "path 不一致"
            assert orig.sha1 == restored.sha1, "sha1 不一致"
            assert len(restored.evidence) == len(orig.evidence), "evidence 数量不一致"
            assert restored.time_result.is_resolved, "time_result 丢失"
            assert restored.time_result.confidence == orig.time_result.confidence
            assert "ExifStage" in restored.completed_stages, "completed_stages 丢失"

        print("往返验证通过 ✓")

        # 测试 stats
        print()
        s = stats(snap_path)
        print("stats():")
        for k, v in s.items():
            print(f"  {k}: {v}")
        assert s["total"] == len(fake_items)
        assert s["resolved"] == len(fake_items)
        assert s["has_exif"] == len(fake_items)
        print("stats 验证通过 ✓")

        # 测试重复 id 取最后一条
        print()
        print("测试重复 id 处理...")
        dup_items = fake_items + [fake_items[0]]   # 第一个重复
        n2 = save(dup_items, snap_path)
        loaded2 = load(snap_path)
        assert n2 == len(fake_items), f"重复 id 应被去除，期望 {len(fake_items)}，实际 {n2}"
        assert len(loaded2) == len(fake_items)
        print("重复 id 去除验证通过 ✓")

    finally:
        snap_path.unlink(missing_ok=True)

    print()
    print("所有测试通过 ✓")

    # 如果传入了真实 snapshot 路径，打印其统计信息
    if len(sys.argv) > 1:
        real_path = Path(sys.argv[1])
        print()
        print("=" * 50)
        print(f"真实 snapshot 统计: {real_path}")
        print("=" * 50)
        s = stats(real_path)
        for k, v in s.items():
            print(f"  {k}: {v}")