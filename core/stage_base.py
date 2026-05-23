"""
Stage 基类

所有 Stage 继承自 Stage，并实现 process() 方法。
基类负责：
  - 统一的 run() 入口（含异常捕获、耗时统计、stage 状态记录）
  - 跳过已完成 stage（幂等保障）
  - 统一的 Evidence 构建快捷方法
  - skip / 条件判断的标准化钩子

子类只需关心：
  - name      —— stage 的唯一名称（类属性）
  - process() —— 核心逻辑，直接操作 item
  - should_run()（可选）—— 自定义跳过条件

约定：
  - process() 内抛出的异常会被 run() 捕获，记录到 item.failed_stages
  - process() 不需要自己调用 mark_stage_done，基类处理
  - process() 通过 self.make_evidence() 构建 Evidence，保持字段一致
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from core.evidence import Evidence, EvidenceSource, TimePrecision

if TYPE_CHECKING:
    from core.context import Context
    from core.item import Item


class Stage(ABC):
    """
    所有 Stage 的抽象基类。

    最小子类示例
    ------------
    class ExifStage(Stage):
        name = "ExifStage"

        def process(self, item: Item, ctx: Context) -> None:
            dt = extract_exif_datetime(item.path)
            if dt:
                item.add_evidence(self.make_evidence(
                    source="exif",
                    dt=dt,
                    precision="second",
                    confidence=1.0,
                    reason="DateTimeOriginal from EXIF",
                    is_direct=True,
                ))
    """

    # ------------------------------------------------------------------
    # 子类必须定义
    # ------------------------------------------------------------------

    name: str = ""
    """Stage 唯一名称，用于日志、checkpoint、completed_stages 记录。
    子类必须覆盖为非空字符串。"""

    # ------------------------------------------------------------------
    # 可选配置（子类可覆盖）
    # ------------------------------------------------------------------

    idempotent: bool = True
    """
    True（默认）：已完成的 stage 直接跳过，保障幂等。
    False：每次都重新执行（适用于 WriterStage 等输出型 stage）。
    """

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def run(self, item: "Item", ctx: "Context") -> None:
        """
        统一执行入口，由 pipeline / scheduler 调用。
        子类不要覆盖此方法，只实现 process()。

        流程：
          1. 检查 name 是否设置
          2. 幂等检查（已完成则跳过）
          3. should_run() 前置条件检查
          4. 执行 process()，捕获异常
          5. 记录耗时、更新 stage 状态
        """
        if not self.name:
            raise ValueError(
                f"{self.__class__.__name__} must define a non-empty `name` class attribute."
            )

        # 幂等：已完成则跳过
        if self.idempotent and item.stage_done(self.name):
            item.log(self.name, "skipped (already completed)")
            return

        # 不支持的格式：跳过所有业务 stage（ScanStage 已打 flag）
        if item.has_flag("UNSUPPORTED"):
            item.mark_stage_done(self.name)
            return

        # 前置条件检查
        reason = self._should_run_reason(item, ctx)
        if reason is not None:
            item.log(self.name, f"skipped: {reason}")
            # 跳过也算"完成"，防止 scheduler 反复重试
            item.mark_stage_done(self.name)
            return

        t0 = time.monotonic()
        try:
            self.process(item, ctx)
            elapsed = time.monotonic() - t0
            item.mark_stage_done(self.name)
            item.log(self.name, f"done ({elapsed:.3f}s)")

        except StageSkip as e:
            # process() 内主动跳过（不算失败）
            elapsed = time.monotonic() - t0
            item.mark_stage_done(self.name)
            item.log(self.name, f"skipped by stage: {e} ({elapsed:.3f}s)")

        except Exception:
            elapsed = time.monotonic() - t0
            tb = traceback.format_exc()
            item.mark_stage_failed(self.name)
            item.warn(self.name, f"FAILED ({elapsed:.3f}s):\n{tb}")

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, item: "Item", ctx: "Context") -> None:
        """
        核心处理逻辑。

        参数
        ----
        item : Item
            当前正在处理的照片，直接修改其属性。
        ctx : Context
            全局共享资源（config、cache、models、indexes 等）。

        约定
        ----
        - 想跳过：raise StageSkip("原因")，不算失败
        - 想记录：item.log(self.name, "...")
        - 想警告：item.warn(self.name, "...")
        - 添加证据：item.add_evidence(self.make_evidence(...))
        - 不要手动调用 item.mark_stage_done()，基类处理
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 可选钩子（子类可覆盖）
    # ------------------------------------------------------------------

    def should_run(self, item: "Item", ctx: "Context") -> bool:
        """
        前置条件检查。返回 False 则跳过此 stage（不算失败）。

        默认返回 True（始终运行）。子类可覆盖以添加条件：

        示例::

            def should_run(self, item, ctx):
                # 已有高可信度 EXIF 则跳过 OCR
                return not item.has_exif_time or item.time_result.confidence < 0.8
        """
        return True

    def skip_reason(self, item: "Item", ctx: "Context") -> Optional[str]:
        """
        返回跳过原因字符串（用于日志），None 表示不跳过。

        比 should_run() 更具表达力，两者选其一覆盖即可：
        - 只需 True/False → 覆盖 should_run()
        - 需要解释原因   → 覆盖 skip_reason()
        """
        return None

    # ------------------------------------------------------------------
    # Evidence 构建快捷方法
    # ------------------------------------------------------------------

    def make_evidence(
        self,
        *,
        source: EvidenceSource,
        dt: Optional[datetime] = None,
        precision: TimePrecision = "unknown",
        confidence: float = 0.5,
        range_start: Optional[datetime] = None,
        range_end: Optional[datetime] = None,
        reason: str = "",
        metadata: Optional[dict[str, Any]] = None,
        related_item_ids: Optional[list[str]] = None,
        is_direct: bool = True,
        debug: Optional[dict[str, Any]] = None,
    ) -> Evidence:
        """
        构建 Evidence，自动填入 stage 名称。

        所有参数均为 keyword-only，防止位置参数顺序错误。

        示例::

            ev = self.make_evidence(
                source="filename",
                dt=parsed_dt,
                precision="day",
                confidence=0.7,
                reason='matched "IMG_20200101.jpg"',
                metadata={"matched_text": "20200101", "pattern": "DATE8"},
            )
        """
        return Evidence(
            source=source,
            stage=self.name,
            dt=dt,
            precision=precision,
            confidence=confidence,
            range_start=range_start,
            range_end=range_end,
            reason=reason,
            metadata=metadata or {},
            related_item_ids=related_item_ids or [],
            is_direct=is_direct,
            debug=debug or {},
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _should_run_reason(self, item: "Item", ctx: "Context") -> Optional[str]:
        """整合 should_run() 和 skip_reason() 的内部逻辑。"""
        explicit = self.skip_reason(item, ctx)
        if explicit is not None:
            return explicit
        if not self.should_run(item, ctx):
            return "should_run() returned False"
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# StageSkip：process() 内主动跳过的信号
# ---------------------------------------------------------------------------

class StageSkip(Exception):
    """
    在 process() 内部抛出以主动跳过，不算失败。

    示例::

        def process(self, item, ctx):
            if item.has_exif_time:
                raise StageSkip("already has EXIF time")
            ...
    """
    pass


# ---------------------------------------------------------------------------
# BatchStage：需要跨 item 协作的 stage 基类（相似图、时间轴等）
# ---------------------------------------------------------------------------

class BatchStage(ABC):
    """
    批处理 Stage 基类，用于需要整批 item 才能运行的逻辑。
    例如：SimilarStage（需要先建完 phash 索引）、TimelineStage（需要排序）。

    与 Stage 的区别：
    - run_batch() 接收全部 item 列表，而不是单个 item
    - 内部可以自行决定处理顺序和并发方式
    - 每处理完一个 item 后，手动调用 item.mark_stage_done(self.name)

    pipeline 中 BatchStage 排在普通 Stage 之后，由 scheduler 特殊调度。
    """

    name: str = ""

    @abstractmethod
    def run_batch(self, items: list["Item"], ctx: "Context") -> None:
        """
        批量处理入口。

        参数
        ----
        items : list[Item]
            所有已完成前置 stage 的 item。
        ctx : Context
            全局共享资源。
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"