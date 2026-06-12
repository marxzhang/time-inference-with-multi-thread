"""
ScanStage — 目录扫描，生成 Item 列表

不支持的文件格式（非 ALL_EXTENSIONS，非 skip_extensions）：
    生成 Item 并打 UNSUPPORTED flag，供 Planner 统计并输出到
    unsupported/ bucket，不参与后续 pipeline。

    livp 文件：
    - --livp 启用时：被 LivePhotoStage 展开为 jpg/heic，不计入 unsupported
    - --livp 未启用时：计入 unsupported（无法处理）

skip_extensions：
    完全跳过，不建立 item，不计入任何统计。
    默认：ds_store / db / ini / nomedia / json / xml / txt / md / pdf / xmp / thm
    用户可通过 --skip-ext 追加或覆盖。
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item


# ---------------------------------------------------------------------------
# 支持的文件格式
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif",
    "webp", "heic", "heif", "avif",
    "raw", "cr2", "cr3", "nef", "arw", "orf", "rw2", "dng",
})

VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    "mp4", "mov", "avi", "mkv", "m4v", "3gp", "wmv", "flv", "ts",
})

ALL_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# 默认跳过的系统 / 元数据文件（完全不建 item）
# DEFAULT_SKIP_EXTENSIONS: frozenset[str] = frozenset({
#     "ds_store", "db", "ini", "nomedia",
#     "json", "xml", "txt", "md",
#     "pdf", "xmp", "thm",
# })
DEFAULT_SKIP_EXTENSIONS: frozenset[str] = frozenset({

})

_SCREENSHOT_RATIO_THRESHOLD = 2.2

_COMMON_SCREEN_SIZES: frozenset[tuple[int, int]] = frozenset({
    (750, 1334),
    (1080, 1920),
    (1170, 2532),
    (1284, 2778),
    (1440, 3088),
    (1080, 2340),
    (2560, 1440),
    (1920, 1080),
    (2560, 1600),
    (2880, 1800),
    (3024, 1964),
})


# ---------------------------------------------------------------------------
# ScanStage
# ---------------------------------------------------------------------------

class ScanStage:
    """目录扫描器，pipeline 的数据入口。"""

    name: str = "ScanStage"

    def scan(
        self,
        ctx: "Context",
        extra_livp_map: "dict[str, Path] | None" = None,
    ) -> list["Item"]:
        """
        扫描 ctx.config.input_dir，返回 Item 列表。

        多线程：
            cache miss 的文件（需要计算 sha1 / phash）用线程池并发处理。
            cache hit 的文件在主线程直接恢复，无需并发。
            workers 来自 config.max_workers，设为 1 退化为串行。

        断点续传（scan cache）：
            已扫描过的文件（relpath + filesize + mtime 不变）直接从缓存恢复。
            每 FLUSH_INTERVAL 个新 item 写盘一次，兼顾性能和崩溃安全。

        统计规则：
            total_input_files = confident + review + unsupported + duplicate
        """
        from core.item import Item
        from storage.scan_cache import ScanCache


        input_dir = Path(ctx.config.input_dir).resolve()
        if not input_dir.is_dir():
            raise ValueError(
                f"input_dir does not exist or is not a directory: {input_dir}"
            )

        skip_ext = getattr(ctx.config, "skip_extensions", None) or DEFAULT_SKIP_EXTENSIONS
        livp_enabled = extra_livp_map is not None
        workers = getattr(ctx.config, "max_workers", 1)

        scan_cache = ScanCache(ctx.config.cache_dir)
        n_preloaded = scan_cache.preload()
        # atexit / SIGINT 兜底：程序任何形式退出时都写盘
        import atexit, signal as _signal
        atexit.register(scan_cache.flush)
        def _flush_on_signal(sig, frame):
            scan_cache.flush()
            raise SystemExit(0)
        try:
            _signal.signal(_signal.SIGTERM, _flush_on_signal)
        except (OSError, ValueError):
            pass  # 非主线程注册 signal 会失败，忽略

        ctx.logger.info(
            f"[{self.name}] scanning: {input_dir}"
            + (f" (scan cache: {n_preloaded} entries)" if n_preloaded else "")
            + (f", workers={workers}" if workers > 1 else "")
        )
        t0 = time.monotonic()

        all_paths = sorted(self._walk(input_dir, skip_ext))

        # ── 分拣：cache hit（主线程直接恢复）vs miss（子线程 _build_item）──
        # 结构：(path, relpath, stat, is_supported, cached_dict | None)
        hit_items: list[tuple[Path, str, bool, dict]] = []   # (path, relpath, is_supported, d)
        miss_paths: list[tuple[Path, str, bool]]       = []  # (path, relpath, is_supported)
        total_count = 0

        for path in all_paths:
            ext = path.suffix.lstrip(".").lower()

            if ext == "livp" and livp_enabled:
                continue

            total_count += 1
            is_supported = ext in ALL_EXTENSIONS

            try:
                st = path.stat()
                relpath = str(path.relative_to(input_dir))
                cached = scan_cache.get(relpath, st.st_size, st.st_mtime)
                if cached:
                    hit_items.append((path, relpath, is_supported, cached))
                else:
                    miss_paths.append((path, relpath, is_supported))
            except Exception as exc:
                ctx.logger.warning(f"[{self.name}] stat failed {path.name}: {exc}")
                total_count -= 1   # stat 失败不计入 total

        ctx.logger.debug(
            f"[{self.name}] cache hits={len(hit_items)}, "
            f"miss={len(miss_paths)} (will _build_item)"
        )

        # ── 收集结果用的共享容器（主线程写，无需加锁）──────────────────────
        # key = relpath，保证后续排序一致
        result_map: dict[str, "Item"] = {}
        skipped = 0
        unsupported_count = 0

        # ── 处理 cache hit（主线程，极快）───────────────────────────────────
        for path, relpath, is_supported, cached in hit_items:
            cached["path"] = str(path)
            item = Item.from_dict(cached)
            if not is_supported:
                item.add_flag("UNSUPPORTED")
                unsupported_count += 1
            result_map[relpath] = item

        # ── 处理 cache miss（线程池）────────────────────────────────────────
        if miss_paths:
            # tqdm：主线程创建，通过 lock 在主线程 update
            progress = self._make_progress_total(len(miss_paths), desc="scanning")
            counts_lock = threading.Lock()

            def _build_and_cache(
                path: Path,
                relpath: str,
                is_supported: bool,
            ) -> tuple[str, "Item | None"]:
                """子线程执行：_build_item + scan_cache.set（已加锁）。"""
                try:
                    item = self._build_item(path, input_dir, ctx)
                    st = path.stat()
                    scan_cache.set(relpath, st.st_size, st.st_mtime, item.to_dict())
                    return relpath, item
                except Exception as exc:
                    ctx.logger.warning(f"[{self.name}] skip {path.name}: {exc}")
                    return relpath, None

            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {
                    executor.submit(_build_and_cache, p, rp, sup): (p, rp, sup)
                    for p, rp, sup in miss_paths
                }
                for future in as_completed(futures):
                    _, _, is_supported = futures[future]
                    try:
                        # 超时保护：Pillow 读损坏文件可能永久挂起
                        # 60s 足够处理任何正常图片（含大型 RAW）
                        relpath, item = future.result(timeout=60)
                    except TimeoutError:
                        path_info = futures[future][0]
                        ctx.logger.warning(
                            f"[{self.name}] timeout (>60s), skip: {path_info.name}"
                        )
                        skipped += 1
                        if progress:
                            progress.update(1)
                        continue
                    except Exception as exc:
                        ctx.logger.warning(f"[{self.name}] unexpected: {exc}")
                        skipped += 1
                        if progress:
                            progress.update(1)
                        continue

                    if item is None:
                        skipped += 1
                    else:
                        if not is_supported:
                            item.add_flag("UNSUPPORTED")
                        # as_completed 回调在主线程执行，result_map 写入无需加锁
                        result_map[relpath] = item
                        if not is_supported:
                            unsupported_count += 1
                        # flush 由 scan_cache.set() 内部自动触发（每 500 条）
                        # 主线程不再手动计数

                    if progress:
                        progress.update(1)

            if progress:
                progress.close()

        # 最终 flush（剩余未写盘的）
        scan_cache.flush()
        # scan 完成后释放 scan_cache 内存，后续不再需要
        scan_cache._store.clear()

        # ── 按原始路径顺序组装 items ────────────────────────────────────────
        # sorted(all_paths) 决定了顺序，result_map 用 relpath 作 key 保证对应
        items: list[Item] = []
        for path in all_paths:
            ext = path.suffix.lstrip(".").lower()
            if ext == "livp" and livp_enabled:
                continue
            relpath = str(path.relative_to(input_dir))
            if relpath in result_map:
                items.append(result_map[relpath])

        # 解压出的 livp 静态图（替代原 livp 文件）
        livp_count = 0
        if extra_livp_map:
            livp_paths = list(extra_livp_map.items())
            livp_progress = self._make_progress(livp_paths, desc="scanning livp")
            livp_iter = livp_progress if livp_progress is not None else livp_paths

            for livp_relpath, unpacked_path in livp_iter:
                try:
                    item = self._build_item(unpacked_path, unpacked_path.parent, ctx)
                    item.relpath = str(
                        Path(livp_relpath).with_suffix(unpacked_path.suffix)
                    )
                    item.livp_source_path = str(
                        (input_dir / livp_relpath).resolve()
                    )
                    items.append(item)
                    livp_count += 1
                except Exception as exc:
                    ctx.logger.warning(
                        f"[{self.name}] skip livp unpacked {unpacked_path.name}: {exc}"
                    )
                    skipped += 1

        # 写入 ctx 供 Planner 读取
        ctx.total_input_files = total_count + livp_count

        elapsed = time.monotonic() - t0
        supported_count = len(items) - unsupported_count
        ctx.logger.info(
            f"[{self.name}] total={ctx.total_input_files} files — "
            f"supported={supported_count}, "
            f"unsupported={unsupported_count}"
            + (f", livp={livp_count}" if livp_count else "")
            + (f", cache_hits={len(hit_items)}" if hit_items else "")
            + f", skipped={skipped}, elapsed={elapsed:.2f}s"
        )
        return items

    # ------------------------------------------------------------------
    # 文件遍历
    # ------------------------------------------------------------------

    def _walk(self, root: Path, skip_ext: frozenset[str]) -> Iterator[Path]:
        """
        递归遍历目录，跳过：
          - 隐藏文件 / 目录（. 开头）
          - skip_ext 中的扩展名
        其余全部产出（包括不支持的格式，由 scan() 打 UNSUPPORTED flag）。
        """
        for entry in root.rglob("*"):
            if not entry.is_file():
                continue
            if any(part.startswith(".") for part in entry.parts):
                continue
            if entry.suffix.lstrip(".").lower() in skip_ext:
                continue
            yield entry

    # ------------------------------------------------------------------
    # 构建单个 Item
    # ------------------------------------------------------------------

    def _build_item(self, path: Path, root: Path, ctx: "Context") -> "Item":
        from core.item import Item

        stat = path.stat()
        item = Item(
            path=str(path),
            relpath=str(path.relative_to(root)),
            filename=path.name,
            ext=path.suffix.lstrip(".").lower(),
            filesize=stat.st_size,
            ctime=stat.st_ctime,
            mtime=stat.st_mtime,
        )
        item.sha1 = self._sha1(path)

        if item.ext in IMAGE_EXTENSIONS:
            self._fill_image_fields(item, path, ctx)

        ctx.logger.debug(f"[{self.name}] scanned: {item.relpath}")
        return item

    # ------------------------------------------------------------------
    # 图片字段填充
    # ------------------------------------------------------------------

    def _fill_image_fields(self, item: "Item", path: Path, ctx: "Context") -> None:
        try:
            from PIL import Image
        except ImportError:
            ctx.logger.warning(f"[{self.name}] Pillow not installed, skipping image fields")
            return

        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            if item.ext in ("heic", "heif", "avif"):
                item.warnings.append(
                    f"[{self.name}] pillow-heif not installed, "
                    f"cannot read {item.ext.upper()}. Run: pip install pillow-heif"
                )
                return

        # 截断/损坏文件：Pillow 默认只发 UserWarning 然后挂起。
        # LOAD_TRUNCATED_IMAGES=True 让 Pillow 在截断处停止而不是等待，
        # 同时把 warnings 转为可捕获的异常。
        from PIL import ImageFile
        import warnings
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # UserWarning → 可捕获异常
                with Image.open(path) as img:
                    item.width, item.height = img.size
                    item.format_real = img.format or ""
                    item.phash, item.dhash = self._compute_hashes(img, path, ctx)
                    item.is_screenshot = self._detect_screenshot(img, item)
        except Exception as exc:
            item.warnings.append(f"[{self.name}] image open failed: {exc}")

    def _compute_hashes(self, img, path: Path, ctx: "Context") -> tuple[str, str]:
        try:
            import imagehash
            return str(imagehash.phash(img)), str(imagehash.dhash(img))
        except ImportError:
            ctx.logger.debug(f"[{self.name}] imagehash not installed")
            return "", ""
        except Exception as exc:
            ctx.logger.debug(f"[{self.name}] hash failed for {path.name}: {exc}")
            return "", ""

    @staticmethod
    def _detect_screenshot(img, item: "Item") -> bool:
        w, h = item.width, item.height
        if not w or not h:
            return False
        if (w, h) in _COMMON_SCREEN_SIZES or (h, w) in _COMMON_SCREEN_SIZES:
            return True
        if max(w, h) / min(w, h) > _SCREENSHOT_RATIO_THRESHOLD:
            return True
        return False

    @staticmethod
    def _sha1(path: Path) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _make_progress(items: list, desc: str = ""):
        """tqdm 进度条（包裹 iterable），未安装时返回 None。"""
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc, unit="file", dynamic_ncols=True)
        except ImportError:
            return None

    @staticmethod
    def _make_progress_total(total: int, desc: str = ""):
        """
        tqdm 手动模式（已知总数，通过 update() 推进）。
        多线程场景下由主线程创建，子线程完成后在主线程调用 update()。
        未安装 tqdm 时返回 None。
        """
        try:
            from tqdm import tqdm
            return tqdm(total=total, desc=desc, unit="file", dynamic_ncols=True)
        except ImportError:
            return None

    def __repr__(self) -> str:
        return "ScanStage()"