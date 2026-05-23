"""
export/planner.py — 导出规划器

输出 Bucket
-----------
    confident   : 置信度高，可直接使用
    review      : 时间置信度低 / 有冲突 / 格式问题，需人工审查
    duplicate   : 被去重识别为冗余的文件（仅 dedup_mode != "no" 时产生）
                  每个文件旁边有一个 .duplicate.txt 说明与哪张图重复
    unsupported : 不支持的格式 / 损坏文件

Summary 格式
------------
    export_plan_summary.html — 带超链接，点击文件名可直接打开图片
"""

from __future__ import annotations

import json
import html as html_module
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

ActionType = Literal[
    "copy",
    "convert",
    "duplicate",       # 输出到 duplicate/ bucket（附说明文件）
    "skip_unsupported",
]

BucketType = Literal[
    "confident",
    "review",
    "duplicate",
    "unsupported",
]

_EXT_FORMAT_MAP: dict[str, str] = {
    "jpg": "JPEG", "jpeg": "JPEG",
    "png": "PNG",
    "heic": "HEIF", "heif": "HEIF",
    "tiff": "TIFF", "tif": "TIFF",
    "webp": "WEBP",
    "gif": "GIF",
    "bmp": "BMP",
}

_FORMAT_EXT_MAP: dict[str, str] = {
    "JPEG": "jpg", "PNG": "png", "HEIF": "heic",
    "TIFF": "tiff", "WEBP": "webp", "GIF": "gif", "BMP": "bmp",
}

_RAW_EXTENSIONS = frozenset({
    "raw", "cr2", "cr3", "nef", "arw", "orf", "rw2", "dng",
})
_VIDEO_EXTENSIONS = frozenset({
    "mp4", "mov", "avi", "mkv", "m4v", "3gp", "wmv", "flv", "ts",
})
_UNSUPPORTED_EXTENSIONS = frozenset({"livp"})

# duplicate_kind → 人类可读描述
_KIND_LABEL: dict[str, str] = {
    "sha1":  "byte-identical (sha1)",
    "phash": "visually identical (phash)",
    "clip":  "visually similar (clip)",
    "":      "unknown",
}


# ---------------------------------------------------------------------------
# ExportAction
# ---------------------------------------------------------------------------

@dataclass
class ExportAction:
    item_id:        str
    source_path:    str
    source_relpath: str

    action:  ActionType
    reason:  str

    bucket:       BucketType
    dest_relpath: str   # 在 bucket 内的相对路径

    source_ext: str
    target_ext: str

    final_datetime: Optional[str]
    confidence:     float
    primary_source: Optional[str]

    duplicate_group_id: str
    duplicate_rank:     int
    duplicate_kind:     str
    # 保留者的 relpath，duplicate bucket 说明文件用
    duplicate_kept_relpath: str = ""

    status: Literal["pending", "done", "failed", "skipped"] = "pending"
    error:  str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExportAction":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# ExportPlanner
# ---------------------------------------------------------------------------

class ExportPlanner:

    def __init__(self, output_dir: str, input_dir: str = "") -> None:
        self.output_dir = Path(output_dir)
        # 文件名前缀：时间戳_输入文件夹名
        # 例：/home/data/school/primary → 20260519150830_primary
        prefix = _make_filename_prefix(input_dir)
        self.plan_path    = self.output_dir / f"{prefix}_plan.jsonl"
        self.summary_path = self.output_dir / f"{prefix}_summary.html"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def plan(self, items: list["Item"], ctx: "Context") -> list[ExportAction]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 先把所有 item 的 action 建出来，再回填 duplicate_kept_relpath
        actions = [self._decide(item, ctx) for item in items]

        # 建立 group_id → 保留者 action 的映射，回填 kept_relpath
        kept_map: dict[str, ExportAction] = {}
        for a in actions:
            if a.duplicate_group_id and a.duplicate_rank == 0:
                kept_map[a.duplicate_group_id] = a
        for a in actions:
            if a.action == "duplicate" and a.duplicate_group_id in kept_map:
                a.duplicate_kept_relpath = kept_map[a.duplicate_group_id].source_relpath

        ctx.logger.info(
            f"[ExportPlanner] planned {len(actions)} actions — "
            + self._stats_str(actions)
        )
        return actions

    def save(self, actions: list[ExportAction], ctx: "Context") -> None:
        with open(self.plan_path, "w", encoding="utf-8") as f:
            for a in actions:
                f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")

        html = self._build_summary_html(actions, ctx)
        self.summary_path.write_text(html, encoding="utf-8")

        ctx.logger.info(
            f"[ExportPlanner] plan saved:\n"
            f"  {self.plan_path}\n"
            f"  {self.summary_path}"
        )

    @classmethod
    def load(cls, output_dir: str, input_dir: str = "") -> list[ExportAction]:
        """
        加载已有 plan 文件。
        若指定 input_dir，按命名规则找对应文件；
        否则找目录下最新的 *_plan.jsonl。
        """
        out = Path(output_dir)
        if input_dir:
            prefix = _make_filename_prefix(input_dir)
            path = out / f"{prefix}_plan.jsonl"
        else:
            # 找最新的 *_plan.jsonl（按文件名排序，时间戳前缀保证最新在最后）
            candidates = sorted(out.glob("*_plan.jsonl"))
            path = candidates[-1] if candidates else out / "export_plan.jsonl"
        if not path.exists():
            return []
        seen: dict[str, ExportAction] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    a = ExportAction.from_dict(d)
                    seen[a.item_id] = a
                except (json.JSONDecodeError, TypeError):
                    pass
        return list(seen.values())

    # ------------------------------------------------------------------
    # 单 item 决策
    # ------------------------------------------------------------------

    def _decide(self, item: "Item", ctx: "Context") -> ExportAction:
        threshold = ctx.config.confidence_threshold

        # 1. 不支持格式（UNSUPPORTED flag 由 ScanStage 打上，
        #    涵盖所有非 ALL_EXTENSIONS 的文件，包括 livp）
        if item.has_flag("UNSUPPORTED") or item.ext in _UNSUPPORTED_EXTENSIONS:
            return self._make(item, "skip_unsupported", "unsupported",
                              f"unsupported format: .{item.ext}")

        # 2. 重复组非保留者 → duplicate bucket
        if item.has_flag("DUPLICATE"):
            kind_label = _KIND_LABEL.get(item.duplicate_kind, item.duplicate_kind)
            return self._make(
                item, "duplicate", "duplicate",
                f"{kind_label} duplicate "
                f"(group={item.duplicate_group_id[:8]}…, rank={item.duplicate_rank})",
            )

        # 3. 格式不匹配
        needs_convert, target_ext, fmt_reason = self._check_format(item)

        # 4. 决定 bucket
        review_reasons: list[str] = []
        if not item.time_result.is_resolved:
            review_reasons.append("time unresolved")
        elif item.time_result.confidence < threshold:
            review_reasons.append(
                f"low confidence ({item.time_result.confidence:.2f} < {threshold})"
            )
        if item.has_flag("CONFLICT_TIME"):
            review_reasons.append("time conflict detected")
        if item.has_flag("NEEDS_REVIEW"):
            review_reasons.append("visually similar to another photo")
        if needs_convert:
            review_reasons.append(f"format mismatch: {fmt_reason}")

        bucket: BucketType = "review" if review_reasons else "confident"
        reason = "; ".join(review_reasons) if review_reasons else "ok"
        return self._make(
            item,
            "convert" if needs_convert else "copy",
            bucket,
            reason,
            target_ext=target_ext,
        )

    def _check_format(self, item: "Item") -> tuple[bool, str, str]:
        ext  = item.ext.lower()
        real = (item.format_real or "").upper()
        if ext in _VIDEO_EXTENSIONS or ext in _RAW_EXTENSIONS or not real:
            return False, ext, ""
        expected = _EXT_FORMAT_MAP.get(ext, "").upper()
        if not expected or real == expected:
            return False, ext, ""
        target_ext = _FORMAT_EXT_MAP.get(real, ext)
        return True, target_ext, f".{ext} but actually {real}"

    def _make(
        self,
        item: "Item",
        action: ActionType,
        bucket: BucketType,
        reason: str,
        target_ext: Optional[str] = None,
    ) -> ExportAction:
        source_ext = item.ext
        if target_ext is None:
            target_ext = source_ext

        relpath = Path(item.relpath)
        if target_ext != source_ext:
            relpath = relpath.with_suffix(f".{target_ext}")
        dest_relpath = str(relpath)

        tr = item.time_result
        return ExportAction(
            item_id=item.id,
            source_path=item.path,
            source_relpath=item.relpath,
            action=action,
            reason=reason,
            bucket=bucket,
            dest_relpath=dest_relpath,
            source_ext=source_ext,
            target_ext=target_ext,
            final_datetime=(tr.final_datetime.isoformat() if tr.final_datetime else None),
            confidence=tr.confidence,
            primary_source=tr.primary_source,
            duplicate_group_id=item.duplicate_group_id,
            duplicate_rank=item.duplicate_rank,
            duplicate_kind=item.duplicate_kind,
        )

    # ------------------------------------------------------------------
    # HTML Summary
    # ------------------------------------------------------------------

    def _build_summary_html(
        self,
        actions: list[ExportAction],
        ctx: "Context",
    ) -> str:
        now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        threshold = ctx.config.confidence_threshold
        total_input_files = getattr(ctx, "total_input_files", 0)
        stats     = self._stats(actions, total_input_files)
        dedup_mode = getattr(ctx.config, "dedup_mode", "sha1")

        e = html_module.escape  # 字符串转义

        def file_link(source_path: str, display: str) -> str:
            """生成可点击的文件链接，点击在系统默认程序中打开原始文件。
            as_uri() 要求绝对路径，用 resolve() 兜底。
            """
            try:
                uri = Path(source_path).resolve().as_uri()
            except ValueError:
                return e(display)
            return f'<a href="{uri}" title="{e(source_path)}">{e(display)}</a>'

        # ── 分组数据准备 ──────────────────────────────────────────────
        dup_groups: dict[str, list[ExportAction]] = defaultdict(list)
        for a in actions:
            if a.duplicate_group_id:
                dup_groups[a.duplicate_group_id].append(a)

        # ── 按 bucket 分类 ────────────────────────────────────────────
        by_bucket: dict[str, list[ExportAction]] = defaultdict(list)
        for a in actions:
            by_bucket[a.bucket].append(a)

        # ── 构建 HTML ─────────────────────────────────────────────────
        sections: list[str] = []

        def section(title: str, color: str, body: str) -> None:
            sections.append(f"""
<details open>
<summary style="background:{color};padding:8px 12px;border-radius:6px;
  cursor:pointer;font-size:1.05em;font-weight:600;">{e(title)}</summary>
<div style="padding:12px 16px;">{body}</div>
</details>""")

        # ── Overview ──────────────────────────────────────────────────
        has_dup = dedup_mode != "no"
        ov_rows = [
            ("Total files",   stats["total"]),
            ("→ confident",   stats.get("confident", 0)),
            ("→ review",      stats.get("review", 0)),
            ("→ unsupported", stats.get("unsupported", 0)),
        ]
        if has_dup:
            ov_rows.append(("→ duplicate", stats.get("duplicate", 0)))
        # Needs conversion 不在表头，体现在 review 的 reason 里

        ov_html = "<table style='border-collapse:collapse;width:360px'>"
        for label, val in ov_rows:
            ov_html += (
                f"<tr><td style='padding:3px 12px 3px 0;color:#555'>{e(label)}</td>"
                f"<td style='font-weight:600'>{val}</td></tr>"
            )
        ov_html += "</table>"
        ov_html += (
            f"<p style='margin-top:8px;color:#666;font-size:.9em'>"
            f"Generated: {e(now)} &nbsp;|&nbsp; "
            f"Confidence threshold: {threshold} &nbsp;|&nbsp; "
            f"Dedup mode: {e(dedup_mode)}</p>"
        )
        section("📊 Overview", "#e8f4fd", ov_html)

        # ── Confident ─────────────────────────────────────────────────
        conf_actions = by_bucket.get("confident", [])
        if conf_actions:
            rows = "".join(
                f"<tr>"
                f"<td>{file_link(a.source_path, a.source_relpath)}</td>"
                f"<td style='color:#888'>{e(a.final_datetime or '—')}</td>"
                f"<td style='color:#888'>{a.confidence:.2f}</td>"
                f"<td style='color:#888'>{e(a.primary_source or '—')}</td>"
                f"{'<td style=color:#e67>convert</td>' if a.action=='convert' else '<td></td>'}"
                f"</tr>"
                for a in conf_actions
            )
            body = (
                "<table style='border-collapse:collapse;width:100%;font-size:.9em'>"
                "<tr style='background:#f0f0f0'>"
                "<th style='text-align:left;padding:4px 8px'>File</th>"
                "<th style='padding:4px 8px'>Datetime</th>"
                "<th style='padding:4px 8px'>Conf</th>"
                "<th style='padding:4px 8px'>Source</th>"
                "<th style='padding:4px 8px'>Note</th>"
                f"</tr>{rows}</table>"
            )
            section(f"✅ Confident ({len(conf_actions)})", "#eafaf1", body)

        # ── Review ────────────────────────────────────────────────────
        review_actions = by_bucket.get("review", [])
        if review_actions:
            rows = "".join(
                f"<tr style='border-bottom:1px solid #eee'>"
                f"<td style='padding:6px 8px'>{file_link(a.source_path, a.source_relpath)}</td>"
                f"<td style='padding:6px 8px;color:#888'>{e(a.final_datetime or '—')}</td>"
                f"<td style='padding:6px 8px;color:#e67'>{a.confidence:.2f}</td>"
                f"<td style='padding:6px 8px;color:#555;font-size:.85em'>{e(a.reason)}</td>"
                f"</tr>"
                for a in review_actions
            )
            body = (
                "<p style='color:#666;font-size:.9em'>Each file will have a "
                "<code>.review.txt</code> companion explaining the issue.</p>"
                "<table style='border-collapse:collapse;width:100%;font-size:.9em'>"
                "<tr style='background:#f0f0f0'>"
                "<th style='text-align:left;padding:4px 8px'>File</th>"
                "<th style='padding:4px 8px'>Datetime</th>"
                "<th style='padding:4px 8px'>Conf</th>"
                "<th style='text-align:left;padding:4px 8px'>Reason</th>"
                f"</tr>{rows}</table>"
            )
            section(f"🔍 Review ({len(review_actions)})", "#fef9e7", body)

        # ── Duplicate ─────────────────────────────────────────────────
        dup_actions = by_bucket.get("duplicate", [])
        if dup_actions:
            # 按 group 展示
            groups_html = ""
            for gid, members in sorted(dup_groups.items()):
                if len(members) < 2:
                    continue
                kind = _KIND_LABEL.get(members[0].duplicate_kind, "")
                kept = next((m for m in members if m.duplicate_rank == 0), None)
                kept_link = (
                    file_link(kept.source_path, kept.source_relpath)
                    if kept else "—"
                )
                groups_html += (
                    f"<div style='margin:12px 0;padding:10px;border:1px solid #ddd;"
                    f"border-radius:6px;background:#fafafa'>"
                    f"<div style='font-size:.8em;color:#888;margin-bottom:6px'>"
                    f"Group <code>{e(gid[:16])}…</code> · {e(kind)}</div>"
                    f"<div style='margin-bottom:4px'>✅ KEEP: {kept_link}</div>"
                )
                dropped = sorted(
                    (m for m in members if m.duplicate_rank != 0),
                    key=lambda x: x.duplicate_rank,
                )
                for m in dropped:
                    groups_html += (
                        f"<div style='color:#c0392b;margin-left:16px'>"
                        f"🗑 DROP: {file_link(m.source_path, m.source_relpath)}"
                        f"<span style='font-size:.8em;color:#888'>"
                        f" rank={m.duplicate_rank}</span></div>"
                    )
                groups_html += "</div>"

            body = (
                "<p style='color:#666;font-size:.9em'>Each dropped file will have a "
                "<code>.duplicate.txt</code> explaining which file it duplicates.</p>"
                + groups_html
            )
            section(
                f"🗑 Duplicate ({len(dup_actions)} dropped, "
                f"{len(dup_groups)} group(s))",
                "#fdecea", body,
            )

        # ── Unsupported ───────────────────────────────────────────────
        unsup_actions = by_bucket.get("unsupported", [])
        if unsup_actions:
            items_html = "".join(
                f"<li>{file_link(a.source_path, a.source_relpath)}"
                f" <span style='color:#888;font-size:.85em'>({e(a.reason)})</span></li>"
                for a in unsup_actions
            )
            body = (
                "<p style='color:#666;font-size:.9em'>"
                "Copied unchanged to <code>unsupported/</code>.</p>"
                f"<ul style='margin:0;padding-left:20px'>{items_html}</ul>"
            )
            section(f"⚠️ Unsupported ({len(unsup_actions)})", "#f5f5f5", body)

        # ── 组装完整 HTML ─────────────────────────────────────────────
        body_html = "\n".join(sections)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Export Plan Summary</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        max-width:960px;margin:32px auto;padding:0 24px;color:#222;}}
  h1   {{font-size:1.4em;margin-bottom:4px}}
  details {{margin:12px 0;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
  summary {{outline:none;user-select:none}}
  summary::-webkit-details-marker {{display:none}}
  a {{color:#2980b9;text-decoration:none}}
  a:hover {{text-decoration:underline}}
  table {{border-collapse:collapse;width:100%}}
  th,td {{padding:4px 8px;text-align:left;vertical-align:top}}
  tr:nth-child(even) {{background:#f9f9f9}}
  code {{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:.9em}}
</style>
</head>
<body>
<h1>📁 Export Plan Summary</h1>
<p style="color:#666;margin-top:0">Output: <code>{e(str(self.output_dir))}</code></p>
{body_html}
<p style="margin-top:32px;color:#aaa;font-size:.85em">
  To execute: <code>python main.py ... --export-write</code>
</p>
</body>
</html>
"""

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def _stats(
        self,
        actions: list[ExportAction],
        total_input_files: int = 0,
    ) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        # total = 磁盘实际文件数（由 ScanStage 写入 ctx.total_input_files）
        # 等于 confident + review + unsupported + duplicate
        for a in actions:
            d[a.bucket] += 1
        bucket_sum = (
            d.get("confident", 0) + d.get("review", 0)
            + d.get("unsupported", 0) + d.get("duplicate", 0)
        )
        d["total"] = total_input_files if total_input_files else bucket_sum
        return dict(d)

    def _stats_str(self, actions: list[ExportAction]) -> str:
        s = self._stats(actions)  # logger 用，不传 total_input_files
        return (
            f"confident={s.get('confident',0)}, "
            f"review={s.get('review',0)}, "
            f"duplicate={s.get('duplicate',0)}, "
            f"unsupported={s.get('unsupported',0)}"
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _make_filename_prefix(input_dir: str) -> str:
    """
    生成文件名前缀：时间戳_输入文件夹名。
    例：input_dir="/home/data/school/primary" → "20260519150830_primary"
    input_dir 为空时只用时间戳。
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if input_dir:
        folder_name = Path(input_dir).resolve().name or "root"
        # 清理文件名中的非法字符
        safe_name = "".join(
            c if c.isalnum() or c in "-_." else "_"
            for c in folder_name
        )
        return f"{timestamp}_{safe_name}"
    return timestamp