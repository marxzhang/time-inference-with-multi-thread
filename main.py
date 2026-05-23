"""
main.py — 程序入口

用法：
    python main.py <input_dir> [选项]

示例：
    python main.py /Volumes/Photos
    python main.py /Volumes/Photos --output /Volumes/Out --dry-run
    python main.py /Volumes/Photos --workers 8 --log-level DEBUG

流程：
    1. 解析命令行参数 → Config
    2. 构建 Context（含 logger）
    3. ScanStage：扫描目录 → items
    4. PatternMiner：从文件名挖掘动态模式（可选）
    5. 构建 Pipeline
    6. Scheduler.run()：执行所有 stage
    7. 输出结果摘要 / JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="time_inference",
        description="从照片文件名、EXIF 等信息推断拍摄时间",
    )
    parser.add_argument(
        "input_dir",
        help="扫描的根目录",
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="输出目录（默认原地写回）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只推断，不写回任何文件",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并发线程数（当前串行版本暂未生效，预留）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="日志文件路径（默认只输出到终端）",
    )
    parser.add_argument(
        "--no-mine",
        action="store_true",
        help="禁用文件名模式挖掘（PatternMiner）",
    )
    parser.add_argument(
        "--dump-json",
        default="",
        help="将所有 item 结果导出为 JSON 文件（调试用）",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="低于此值标记 LOW_CONFIDENCE（默认 0.6）",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="启用计算结果缓存（CLIP embedding 等跨运行复用）",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="启用断点续传（中断后重启可跳过已完成 item）",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="cache / checkpoint 存放目录（默认 .cache）",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="启用 CLIP embedding + 相似图时间推断",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="CLIP 模型名称（默认 ViT-B-32）",
    )
    parser.add_argument(
        "--clip-min-score",
        type=float,
        default=0.85,
        help="相似图检索余弦相似度阈值（默认 0.85）",
    )
    parser.add_argument(
        "--export-dir",
        default="",
        help="导出根目录（confident / review / unsupported 的父目录）",
    )
    parser.add_argument(
        "--dedup",
        default="sha1",
        choices=["no", "sha1", "phash", "clip"],
        help="去重强度：no / sha1 / phash / clip（默认 sha1，每档包含上一档）",
    )
    parser.add_argument(
        "--dedup-phash-threshold",
        type=int,
        default=4,
        help="phash 汉明距离阈值（dedup=phash/clip 时生效，默认 4）",
    )
    parser.add_argument(
        "--skip-ext",
        default="",
        help=(
            "追加到默认 skip_extensions 的扩展名，逗号分隔，不含点。"
            "例：--skip-ext pdf,txt,log。"
            "若以 = 开头则完全覆盖默认值，例：--skip-ext =pdf,txt"
        ),
    )
    parser.add_argument(
        "--livp",
        action="store_true",
        help="解压 .livp 文件（Live Photo），将静态图参与 pipeline",
    )
    parser.add_argument(
        "--export-plan",
        action="store_true",
        help="运行完 pipeline 后生成 export_plan.jsonl 和 summary.txt",
    )
    parser.add_argument(
        "--export-write",
        action="store_true",
        help="执行 export_plan.jsonl 中的文件操作（需先运行 --export-plan）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    """返回 exit code：0 成功，1 失败。"""
    args = parse_args()

    # ── 1. Config ─────────────────────────────────────────────────────
    from config import Config
    cfg = Config(
        input_dir=args.input_dir,
        output_dir=args.output,
        dry_run=args.dry_run,
        max_workers=args.workers,
        log_level=args.log_level,
        log_file=args.log_file,
        confidence_threshold=args.confidence_threshold,
        cache_dir=args.cache_dir,
        enable_clip=args.clip,
        enable_similar=args.clip,
        clip_model=args.clip_model,
        clip_min_score=args.clip_min_score,
        dedup_mode=args.dedup,
        dedup_phash_threshold=args.dedup_phash_threshold,
        skip_extensions=_parse_skip_ext(args.skip_ext),
    )

    # ── 2. Context ────────────────────────────────────────────────────
    from core.context import Context
    ctx = Context(cfg)
    ctx.logger.info(f"time_inference starting")
    ctx.logger.info(f"input_dir : {cfg.input_dir}")
    ctx.logger.info(f"dry_run   : {cfg.dry_run}")

    # ── 2b. Cache / Checkpoint（可选）───────────────────────────────────
    if args.cache:
        ctx.setup_cache()

    if args.checkpoint:
        ctx.setup_checkpoint()

    if args.clip:
        ctx.setup_models()

    # ── 3. Live Photo 预处理（可选）─────────────────────────────────
    livp_map = None
    if args.livp:
        from stages.livephoto import LivePhotoStage
        livp_stage = LivePhotoStage(ctx)
        livp_map = livp_stage.unpack_all()

    # ── 4. Scan ───────────────────────────────────────────────────────
    from stages.scan import ScanStage
    scanner = ScanStage()
    try:
        items = scanner.scan(ctx, extra_livp_map=livp_map)
    except ValueError as e:
        ctx.logger.error(f"scan failed: {e}")
        return 1

    if not items:
        ctx.logger.warning("no supported files found, exiting")
        return 0

    # ── 5. PatternMiner ───────────────────────────────────────────────
    extra_patterns = None
    if not args.no_mine:
        from utils.miner import PatternMiner
        miner = PatternMiner()
        extra_patterns = miner.mine(items, ctx)
        ctx.logger.info(
            f"PatternMiner: {len(extra_patterns)} dynamic pattern(s) discovered"
        )

    # ── 6. Pipeline ───────────────────────────────────────────────────
    from core.pipeline import Pipeline
    pipeline = Pipeline.default(ctx, extra_patterns=extra_patterns)
    ctx.logger.info(f"pipeline:\n{pipeline}")

    # ── 7. Scheduler ──────────────────────────────────────────────────
    from core.scheduler import Scheduler
    scheduler = Scheduler(pipeline)
    scheduler.run(items, ctx)

    # ── 8. 标记 LOW_CONFIDENCE ────────────────────────────────────────
    _mark_low_confidence(items, cfg.confidence_threshold)

    # ── 9. 输出 JSON（可选）──────────────────────────────────────────
    if args.dump_json:
        _dump_json(items, args.dump_json, ctx)

    # ── 10. Export Plan（可选）────────────────────────────────────────
    if args.export_plan:
        if not args.export_dir:
            ctx.logger.error("--export-plan requires --export-dir")
            return 1
        _run_planner(items, args.export_dir, ctx)

    # ── 11. Export Write（可选）──────────────────────────────────────
    if args.export_write:
        if not args.export_dir:
            ctx.logger.error("--export-write requires --export-dir")
            return 1
        _run_writer(args.export_dir, ctx)

    return 0


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _mark_low_confidence(items: list, threshold: float) -> None:
    """对置信度低于阈值的 item 打 LOW_CONFIDENCE flag。"""
    for item in items:
        if (item.time_result.is_resolved
                and item.time_result.confidence < threshold):
            item.add_flag("LOW_CONFIDENCE")
        elif not item.time_result.is_resolved:
            item.add_flag("LOW_CONFIDENCE")


def _parse_skip_ext(raw: str):
    """
    解析 --skip-ext 参数。
    空字符串 → None（使用默认值）。
    "pdf,txt" → 在默认值基础上追加。
    "=pdf,txt" → 完全覆盖默认值。
    """
    from stages.scan import DEFAULT_SKIP_EXTENSIONS
    if not raw:
        return None
    if raw.startswith("="):
        return frozenset(x.strip().lower() for x in raw[1:].split(",") if x.strip())
    extra = frozenset(x.strip().lower() for x in raw.split(",") if x.strip())
    return DEFAULT_SKIP_EXTENSIONS | extra


def _run_planner(items: list, export_dir: str, ctx) -> None:
    """运行 ExportPlanner，生成 plan 文件和 summary。"""
    from export.planner import ExportPlanner
    planner = ExportPlanner(export_dir, input_dir=ctx.config.input_dir)
    actions = planner.plan(items, ctx)
    planner.save(actions, ctx)


def _run_writer(export_dir: str, ctx) -> None:
    """执行 *_plan.jsonl 中的文件操作。"""
    from export.writer import ExportWriter
    from export.planner import ExportPlanner
    writer = ExportWriter(export_dir, input_dir=ctx.config.input_dir)
    writer.run(ctx)


def _dump_json(items: list, path: str, ctx) -> None:
    """将所有 item 序列化为 JSON 文件（调试用）。"""
    out = Path(path)
    try:
        data = [it.to_dict() for it in items]
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.logger.info(f"JSON dumped to {out}  ({len(data)} items)")
    except Exception as e:
        ctx.logger.error(f"JSON dump failed: {e}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())