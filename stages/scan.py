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
import time
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
DEFAULT_SKIP_EXTENSIONS: frozenset[str] = frozenset({
    "ds_store", "db", "ini", "nomedia",
    "json", "xml", "txt", "md",
    "xmp", "thm",
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

        断点续传：
            若 ctx.cache 已启用，自动使用 ScanCache。
            已扫描过的文件（relpath + filesize + mtime 不变）直接从缓存恢复，
            跳过 sha1 计算、Pillow 解码、phash 等慢操作。

        统计规则：
            total_input_files = 所有非 skip_extensions 的文件数
                              = confident + review + unsupported + duplicate（最终）
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

        # 初始化 scan cache（只要 cache_dir 存在就启用，不依赖 --cache 开关）
        scan_cache = ScanCache(ctx.config.cache_dir)
        n_preloaded = scan_cache.preload()

        ctx.logger.info(
            f"[{self.name}] scanning: {input_dir}"
            + (f" (scan cache: {n_preloaded} entries)" if n_preloaded else "")
        )
        t0 = time.monotonic()

        all_paths = sorted(self._walk(input_dir, skip_ext))

        items: list[Item] = []
        skipped = 0
        unsupported_count = 0
        total_count = 0
        cache_hits = 0

        progress = self._make_progress(all_paths, desc="scanning")
        path_iter = progress if progress is not None else all_paths

        for path in path_iter:
            ext = path.suffix.lstrip(".").lower()
            is_supported = ext in ALL_EXTENSIONS

            if ext == "livp" and livp_enabled:
                # livp 已被展开为 jpg/heic，原文件不计入 total
                # total 只统计解压后的图像（由 livp_count 计入）
                continue

            total_count += 1
            try:
                stat = path.stat()
                relpath = str(path.relative_to(input_dir))

                # cache 命中：直接恢复，跳过慢操作
                cached = scan_cache.get(relpath, stat.st_size, stat.st_mtime)
                if cached:
                    # 恢复时更新 path（绝对路径可能因移动而变化）
                    cached["path"] = str(path)
                    item = Item.from_dict(cached)
                    cache_hits += 1
                else:
                    item = self._build_item(path, input_dir, ctx)
                    scan_cache.set(relpath, stat.st_size, stat.st_mtime, item.to_dict())

                if not is_supported:
                    item.add_flag("UNSUPPORTED")
                    unsupported_count += 1
                items.append(item)

            except Exception as exc:
                ctx.logger.warning(f"[{self.name}] skip {path.name}: {exc}")
                skipped += 1

        # 写盘
        scan_cache.flush()

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
            + (f", cache_hits={cache_hits}" if cache_hits else "")
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

        try:
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
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc, unit="file", dynamic_ncols=True)
        except ImportError:
            return None

    def __repr__(self) -> str:
        return "ScanStage()"