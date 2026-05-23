"""
export/writer.py — 导出执行器

职责：
    读取 ExportPlanner 生成的 export_plan.jsonl，
    逐条执行文件操作，输出到 export_dir 下的三个 bucket 子目录。

目录结构
--------
    export_dir/
    ├── confident/          置信度高，可直接使用
    │   └── (原始目录结构)
    ├── review/             需要人工审核
    │   └── (原始目录结构)
    │       ├── photo.jpg
    │       └── photo.jpg.review.txt   ← 说明为何需要 review
    ├── unsupported/        不支持的格式，原样复制
    │   └── (原始目录结构)
    ├── export_plan.jsonl   plan 文件（Planner 生成，Writer 更新 status）
    └── export_plan_summary.txt

支持的操作
----------
    copy             : shutil.copy2（保留文件时间戳和元数据）
    convert          : Pillow 重新编码 + piexif 写回 EXIF 时间
    skip_duplicate   : 不输出文件，只记录日志
    skip_unsupported : 复制到 unsupported/，不做任何处理

断点续传
--------
    启动时加载 plan，过滤 status="pending" 的 action 执行。
    每条 action 完成后立即更新 status 并追加写回 plan 文件。
    重启后已完成的 action 自动跳过。

幂等性保证
----------
    复制/转换前检查目标文件是否已存在：
    - 存在且 sha1 一致 → 跳过（status 标为 "skipped"）
    - 存在但 sha1 不同 → 覆盖（可能是上次转换不完整）
    - 不存在 → 正常执行

安全检查
--------
    Writer 启动时验证 export_dir 不是 input_dir 的子目录，
    防止误操作覆盖源文件。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from export.planner import ExportAction, ExportPlanner

if TYPE_CHECKING:
    from core.context import Context


# ---------------------------------------------------------------------------
# ExportWriter
# ---------------------------------------------------------------------------

class ExportWriter:
    """
    执行 export_plan.jsonl 中的文件操作。

    使用示例
    --------
    writer = ExportWriter(export_dir="/path/to/export")
    writer.run(ctx)
    """

    def __init__(self, export_dir: str, input_dir: str = "") -> None:
        self.export_dir = Path(export_dir)
        self.input_dir  = input_dir
        # plan 文件路径与 planner 保持一致的命名规则
        # 不重新生成时间戳（plan/write 是两次独立运行）
        # 始终找目录下最新的 *_plan.jsonl
        candidates = sorted(self.export_dir.glob("*_plan.jsonl"))
        self.plan_path = candidates[-1] if candidates else self.export_dir / "export_plan.jsonl"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, ctx: "Context") -> None:
        """
        执行所有 pending action。支持断点续传。
        """
        if not self.plan_path.exists():
            ctx.logger.error(
                f"[ExportWriter] plan file not found: {self.plan_path}\n"
                f"  Run with --export-plan first."
            )
            return

        # 安全检查
        self._safety_check(ctx)

        # 加载 plan，只处理 pending
        all_actions = ExportPlanner.load(str(self.export_dir), self.input_dir)
        pending = [a for a in all_actions if a.status == "pending"]
        done    = len(all_actions) - len(pending)

        ctx.logger.info(
            f"[ExportWriter] {len(all_actions)} total actions, "
            f"{done} already done, {len(pending)} pending"
        )

        if not pending:
            ctx.logger.info("[ExportWriter] nothing to do, all actions completed")
            return

        t0 = time.monotonic()
        counts = {"done": 0, "skipped": 0, "failed": 0}

        for action in pending:
            result = self._execute(action, ctx)
            self._update_status(action, result, ctx)
            counts[action.status if action.status in counts else "done"] += 1

        elapsed = time.monotonic() - t0
        ctx.logger.info(
            f"[ExportWriter] complete — "
            f"done={counts['done']}, "
            f"skipped={counts['skipped']}, "
            f"failed={counts['failed']}, "
            f"elapsed={elapsed:.2f}s"
        )

    # ------------------------------------------------------------------
    # 单条 action 执行
    # ------------------------------------------------------------------

    def _execute(
        self,
        action: ExportAction,
        ctx: "Context",
    ) -> tuple[str, str]:
        """
        执行单条 action。
        返回 (status, error_message)：
            ("done",    "")
            ("skipped", "reason")
            ("failed",  "error detail")
        """
        try:
            if action.action == "duplicate":
                return self._do_duplicate(action, ctx)

            if action.action == "skip_unsupported":
                # 不支持的格式：复制到 unsupported/ 原样保留
                return self._do_copy(action, ctx)

            if action.action == "copy":
                return self._do_copy(action, ctx)

            if action.action == "convert":
                return self._do_convert(action, ctx)

            return "failed", f"unknown action: {action.action}"

        except Exception as e:
            import traceback
            return "failed", f"{e}\n{traceback.format_exc()}"

    # ------------------------------------------------------------------
    # duplicate（复制到 duplicate/ 并附说明文件）
    # ------------------------------------------------------------------

    def _do_duplicate(
        self,
        action: ExportAction,
        ctx: "Context",
    ) -> tuple[str, str]:
        """
        把重复文件复制到 duplicate/ bucket，并生成 .duplicate.txt 说明。
        """
        dest = self._dest_path(action)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # 幂等检查
        if dest.exists() and _sha1_file(dest) == _sha1_file(Path(action.source_path)):
            self._write_duplicate_note(action, dest)
            return "skipped", "already exists with same content"

        shutil.copy2(action.source_path, dest)
        self._write_duplicate_note(action, dest)
        ctx.logger.debug(f"[ExportWriter] duplicate → {action.dest_relpath}")
        return "done", ""

    # ------------------------------------------------------------------
    # copy
    # ------------------------------------------------------------------

    def _do_copy(
        self,
        action: ExportAction,
        ctx: "Context",
    ) -> tuple[str, str]:
        """
        shutil.copy2 复制文件，保留时间戳。
        如果 bucket=review，额外生成 .review.txt。
        """
        dest = self._dest_path(action)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # 幂等检查
        if dest.exists():
            if _sha1_file(dest) == _sha1_file(Path(action.source_path)):
                if action.bucket == "review":
                    self._write_review_note(action, dest)
                return "skipped", "already exists with same content"
            # sha1 不同：上次可能写了一半，覆盖
            ctx.logger.debug(
                f"[ExportWriter] overwriting {dest.name} (sha1 mismatch)"
            )

        shutil.copy2(action.source_path, dest)
        ctx.logger.debug(f"[ExportWriter] copy → {action.dest_relpath}")

        if action.bucket == "review":
            self._write_review_note(action, dest)

        return "done", ""

    # ------------------------------------------------------------------
    # convert
    # ------------------------------------------------------------------

    def _do_convert(
        self,
        action: ExportAction,
        ctx: "Context",
    ) -> tuple[str, str]:
        """
        用 Pillow 转换格式，piexif 写回 EXIF 时间戳。
        """
        dest = self._dest_path(action)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # 幂等检查（目标文件已存在）
        if dest.exists() and dest.stat().st_size > 0:
            return "skipped", "already converted"

        # 注册 HEIC 支持
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass

        try:
            from PIL import Image
        except ImportError:
            return "failed", "Pillow not installed"

        # 读取源文件
        with Image.open(action.source_path) as img:
            # 保留原始 EXIF bytes（Pillow 能读的话）
            raw_exif: Optional[bytes] = None
            try:
                raw_exif = img.info.get("exif")
            except Exception:
                pass

            # 转换为 RGB（部分格式需要）
            save_img = img
            target_format = action.target_ext.upper()
            if target_format == "JPG":
                target_format = "JPEG"
            if target_format == "HEIC":
                target_format = "HEIF"

            # JPEG 不支持透明通道
            if target_format == "JPEG" and save_img.mode in ("RGBA", "LA", "P"):
                save_img = save_img.convert("RGB")

            # 保存
            save_kwargs: dict = {}
            if raw_exif:
                save_kwargs["exif"] = raw_exif
            if target_format == "JPEG":
                save_kwargs["quality"] = 95
                save_kwargs["subsampling"] = 0  # 最高质量色度采样

            save_img.save(dest, format=target_format, **save_kwargs)

        # 如果 raw_exif 丢失，但 action 有 final_datetime，用 piexif 写入
        if not raw_exif and action.final_datetime:
            self._write_exif_datetime(dest, action.final_datetime)

        ctx.logger.debug(
            f"[ExportWriter] convert .{action.source_ext}→.{action.target_ext} "
            f"→ {action.dest_relpath}"
        )

        if action.bucket == "review":
            self._write_review_note(action, dest)

        return "done", ""

    # ------------------------------------------------------------------
    # review 说明文件
    # ------------------------------------------------------------------

    def _write_review_note(
        self,
        action: ExportAction,
        dest: Path,
    ) -> None:
        """
        在 dest 旁边生成 .review.txt，说明为什么这个文件需要 review。
        """
        note_path = dest.with_suffix(dest.suffix + ".review.txt")
        dt_str = action.final_datetime or "unknown"
        conf_str = f"{action.confidence:.2f}"

        lines = [
            "time_inference review note",
            "─" * 40,
            f"file      : {action.source_relpath}",
            f"reason    : {action.reason}",
            "",
            "time inference result:",
            f"  datetime : {dt_str}",
            f"  confidence: {conf_str}",
            f"  source   : {action.primary_source or '-'}",
        ]
        if action.duplicate_group_id:
            lines += [
                "",
                "duplicate info:",
                f"  group  : {action.duplicate_group_id}",
                f"  rank   : {action.duplicate_rank}",
                f"  kind   : {action.duplicate_kind}",
            ]

        note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # duplicate 说明文件
    # ------------------------------------------------------------------

    @staticmethod
    def _write_duplicate_note(action: ExportAction, dest: Path) -> None:
        """
        在 dest 旁边生成 .duplicate.txt，说明该文件与哪张图重复。
        """
        note_path = dest.with_suffix(dest.suffix + ".duplicate.txt")
        kept = action.duplicate_kept_relpath or "(unknown)"
        kind = action.duplicate_kind or "unknown"
        lines = [
            "time_inference: duplicate file",
            "─" * 40,
            f"This file is a duplicate and was not selected for export.",
            "",
            f"file     : {action.source_relpath}",
            f"kept as  : {kept}",
            f"kind     : {kind}",
            f"group    : {action.duplicate_group_id}",
            f"rank     : {action.duplicate_rank}",
            "",
            "You can safely delete this file.",
        ]
        note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # EXIF 时间写回
    # ------------------------------------------------------------------

    @staticmethod
    def _write_exif_datetime(dest: Path, iso_datetime: str) -> None:
        """
        用 piexif 把推断出的时间写入输出文件的 EXIF。
        只写 DateTimeOriginal 和 DateTime，不修改其他字段。
        失败时静默忽略（时间信息不影响文件可用性）。
        """
        try:
            import piexif
            from datetime import datetime

            dt = datetime.fromisoformat(iso_datetime)
            exif_dt = dt.strftime("%Y:%m:%d %H:%M:%S")

            try:
                exif_dict = piexif.load(str(dest))
            except Exception:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

            exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_dt.encode()
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_dt.encode()
            exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_dt.encode()

            piexif.insert(piexif.dump(exif_dict), str(dest))
        except Exception:
            pass  # EXIF 写回失败不影响文件本身

    # ------------------------------------------------------------------
    # 状态更新（追加写入 plan 文件）
    # ------------------------------------------------------------------

    def _update_status(
        self,
        action: ExportAction,
        result: tuple[str, str],
        ctx: "Context",
    ) -> None:
        """
        更新 action.status 并追加写入 plan 文件。
        追加模式：load() 时同一 item_id 取最后一条，天然支持断点续传。
        """
        status, error = result
        action.status = status  # type: ignore[assignment]
        action.error  = error

        if status == "failed":
            ctx.logger.warning(
                f"[ExportWriter] FAILED {action.source_relpath}: {error[:120]}"
            )

        try:
            with open(self.plan_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(action.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            ctx.logger.error(f"[ExportWriter] failed to update plan: {e}")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _dest_path(self, action: ExportAction) -> Path:
        """计算目标文件的绝对路径。"""
        return self.export_dir / action.bucket / action.dest_relpath

    def _safety_check(self, ctx: "Context") -> None:
        """
        确保 export_dir 不是 input_dir 的子目录，防止覆盖源文件。
        """
        export = self.export_dir.resolve()
        input_dir = Path(ctx.config.input_dir).resolve()
        if export == input_dir or input_dir in export.parents:
            raise ValueError(
                f"[ExportWriter] SAFETY: export_dir ({export}) must not be "
                f"inside input_dir ({input_dir}). Aborting."
            )
        # 反向检查：export 不能包含 input（也会造成混乱）
        if export in input_dir.parents:
            ctx.logger.warning(
                f"[ExportWriter] WARNING: export_dir is a parent of input_dir. "
                f"This is unusual but allowed."
            )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _sha1_file(path: Path) -> str:
    """流式计算文件 SHA-1，用于幂等检查。"""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""