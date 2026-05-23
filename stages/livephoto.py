"""
stages/livephoto.py — Live Photo (.livp) 预处理

职责：
    在 ScanStage 之前运行，遍历 input_dir 找到所有 .livp 文件，
    将其解压到 cache_dir/livp_unpacked/，让解压出的图片以普通
    jpg/heic 文件的身份参与后续完整 pipeline。

.livp 文件结构
--------------
    本质是 ZIP 压缩包，内含：
        XXXX.jpg  或  XXXX.heic   ← 静态图（主体，我们要的）
        XXXX.mov                  ← 短视频片段（忽略）

解压策略
--------
    - 只提取第一个 jpg 或 heic 文件（即静态主图）
    - 输出到 cache_dir/livp_unpacked/<relpath_stem>.jpg/heic
    - 保持原始相对路径结构，方便回溯来源
    - 若 cache 中已有（文件存在 + 大小 > 0），跳过解压（幂等）

与其他模块的关系
----------------
    LivePhotoStage.unpack_all() 返回 {livp_relpath: unpacked_path}，
    main.py 把这个 mapping 传给 ScanStage 的 extra_paths 参数，
    ScanStage 扫描时把解压路径也纳入 item 列表。
    解压出的 item 的 livp_source_path 会被填充，
    Writer 据此知道输出时要对应到原始 .livp 的位置。

临时文件生命周期
----------------
    livp_unpacked/ 在 Writer 全部完成后可以清理。
    因为支持 checkpoint，建议 Writer 完成后手动调用 cleanup()，
    或在 main.py 的 --export-write 之后自动清理。
"""

from __future__ import annotations

import zipfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.context import Context


# livp 内部认为是静态图的扩展名（按优先级）
_PHOTO_EXTENSIONS = (".heic", ".heif", ".jpg", ".jpeg", ".png")


class LivePhotoStage:
    """
    .livp 预处理器，在 ScanStage 之前运行。

    使用示例（main.py 中）
    ----------------------
    livp_stage = LivePhotoStage(ctx)
    livp_map = livp_stage.unpack_all()
    # livp_map: {livp_relpath: unpacked_abs_path}
    # 传给 ScanStage，让解压出的 jpg/heic 参与 scan
    items = scanner.scan(ctx, extra_livp_map=livp_map)
    """

    def __init__(self, ctx: "Context") -> None:
        self._ctx       = ctx
        self._input_dir = Path(ctx.config.input_dir).resolve()
        self._unpack_dir = Path(ctx.config.cache_dir) / "livp_unpacked"
        self._unpack_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def unpack_all(self) -> dict[str, Path]:
        """
        扫描 input_dir，解压所有 .livp 文件。

        返回
        ----
        dict[str, Path]：
            key  = livp 相对于 input_dir 的 relpath（str）
            value = 解压出的静态图的绝对路径（Path）

        已存在且大小 > 0 的解压文件直接复用（幂等）。
        """
        ctx = self._ctx
        livp_files = list(self._input_dir.rglob("*.livp"))
        if not livp_files:
            return {}

        ctx.logger.info(
            f"[LivePhotoStage] found {len(livp_files)} .livp file(s), unpacking..."
        )
        t0 = time.monotonic()

        result: dict[str, Path] = {}
        skipped = done = failed = 0

        for livp_path in sorted(livp_files):
            relpath = str(livp_path.relative_to(self._input_dir))
            try:
                unpacked = self._unpack_one(livp_path, relpath)
                if unpacked is None:
                    failed += 1
                    ctx.logger.warning(
                        f"[LivePhotoStage] no photo found in {relpath}"
                    )
                    continue

                if unpacked == "cached":
                    skipped += 1
                    # 还是要把 cached 路径加进 result
                    cached_path = self._expected_unpack_path(livp_path, relpath)
                    if cached_path:
                        result[relpath] = cached_path
                else:
                    result[relpath] = unpacked
                    done += 1

            except Exception as e:
                failed += 1
                ctx.logger.warning(
                    f"[LivePhotoStage] failed to unpack {relpath}: {e}"
                )

        elapsed = time.monotonic() - t0
        ctx.logger.info(
            f"[LivePhotoStage] done — "
            f"unpacked={done}, cached={skipped}, failed={failed}, "
            f"elapsed={elapsed:.2f}s"
        )
        return result

    # ------------------------------------------------------------------
    # 解压单个 livp
    # ------------------------------------------------------------------

    def _unpack_one(
        self,
        livp_path: Path,
        relpath: str,
    ) -> Path | str | None:
        """
        解压单个 .livp，返回：
            Path   → 解压成功，返回静态图路径
            "cached" → 已存在，跳过
            None   → livp 内没有找到静态图
        """
        if not zipfile.is_zipfile(livp_path):
            raise ValueError(f"not a valid zip/livp file")

        with zipfile.ZipFile(livp_path, "r") as zf:
            # 找静态图：优先 heic > jpg，忽略 mov
            photo_entry = self._find_photo_entry(zf.namelist())
            if photo_entry is None:
                return None

            suffix = Path(photo_entry).suffix.lower()
            dest = self._build_dest_path(relpath, suffix)

            # 幂等检查
            if dest.exists() and dest.stat().st_size > 0:
                return "cached"

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(photo_entry) as src, open(dest, "wb") as f:
                f.write(src.read())

            self._ctx.logger.debug(
                f"[LivePhotoStage] unpacked {relpath} → {dest.name}"
            )
            return dest

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _find_photo_entry(self, namelist: list[str]) -> str | None:
        """
        在 livp 内部名称列表里找静态图，按 _PHOTO_EXTENSIONS 优先级。
        """
        for ext in _PHOTO_EXTENSIONS:
            for name in namelist:
                if name.lower().endswith(ext) and not name.startswith("__"):
                    return name
        return None

    def _build_dest_path(self, livp_relpath: str, photo_suffix: str) -> Path:
        """
        计算解压目标路径，保持目录结构，替换扩展名。

        示例：
            livp_relpath = "2020/Japan/IMG_001.livp"
            photo_suffix = ".heic"
            → unpack_dir / "2020/Japan/IMG_001.heic"
        """
        rel = Path(livp_relpath)
        return self._unpack_dir / rel.with_suffix(photo_suffix)

    def _expected_unpack_path(
        self,
        livp_path: Path,
        relpath: str,
    ) -> Path | None:
        """
        当缓存命中时，推断已解压文件的路径（不打开 zip）。
        尝试所有支持的后缀，返回第一个存在的。
        """
        rel = Path(relpath)
        for ext in _PHOTO_EXTENSIONS:
            candidate = self._unpack_dir / rel.with_suffix(ext)
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        return None

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        删除 livp_unpacked/ 目录（Writer 完成后调用）。
        """
        import shutil
        if self._unpack_dir.exists():
            shutil.rmtree(self._unpack_dir)
            self._ctx.logger.info(
                f"[LivePhotoStage] cleaned up {self._unpack_dir}"
            )
