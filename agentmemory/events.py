"""事件系统：记忆操作的回调钩子。

支持在以下操作前后触发自定义回调：
- remember: 添加记忆前后
- forget: 删除记忆前后
- search: 搜索前后
- update: 更新记忆前后
- consolidate: 整合操作前后
- snapshot: 快照操作前后

回调可以是同步或异步函数，支持优先级排序和错误处理。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class EventType(Enum):
    """事件类型枚举"""
    BEFORE_REMEMBER = "before_remember"
    AFTER_REMEMBER = "after_remember"
    BEFORE_FORGET = "before_forget"
    AFTER_FORGET = "after_forget"
    BEFORE_SEARCH = "before_search"
    AFTER_SEARCH = "after_search"
    BEFORE_UPDATE = "before_update"
    AFTER_UPDATE = "after_update"
    BEFORE_CONSOLIDATE = "before_consolidate"
    AFTER_CONSOLIDATE = "after_consolidate"
    BEFORE_SNAPSHOT = "before_snapshot"
    AFTER_SNAPSHOT = "after_snapshot"
    ON_ERROR = "on_error"


@dataclass
class EventContext:
    """事件上下文，传递给回调函数。

    Attributes:
        event_type: 事件类型
        timestamp: 事件时间戳
        data: 事件相关数据
        cancelled: 是否取消操作（before 回调可设置）
        error: 错误信息（on_error 事件）
    """

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    error: Optional[Exception] = None

    def cancel(self) -> None:
        """取消当前操作（仅 before 事件有效）"""
        self.cancelled = True

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": {k: str(v)[:100] for k, v in self.data.items()},
            "cancelled": self.cancelled,
            "error": str(self.error) if self.error else None,
        }


@dataclass
class EventHandler:
    """事件处理器包装。

    Attributes:
        callback: 回调函数
        event_type: 监听的事件类型
        priority: 优先级（越小越先执行），默认 0
        name: 处理器名称（用于调试/删除）
        once: 是否只触发一次
    """

    callback: Callable[[EventContext], None]
    event_type: EventType
    priority: int = 0
    name: str = ""
    once: bool = False
    _triggered: bool = field(default=False, repr=False)


class EventBus:
    """事件总线。

    管理事件注册、触发和错误处理。

    Example:
        >>> bus = EventBus()
        >>> def on_add(ctx: EventContext):
        ...     print(f"Added: {ctx.data}")
        >>> bus.on(EventType.AFTER_REMEMBER, on_add, name="logger")
        >>> bus.emit(EventType.AFTER_REMEMBER, {"memory_id": "123"})
        >>> bus.get_history()  # 查看事件历史
    """

    def __init__(self, max_history: int = 100) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._history: list[EventContext] = []
        self._max_history = max_history
        self._enabled = True
        self._error_handler: Optional[Callable[[Exception, EventContext], None]] = None

    def on(
        self,
        event_type: EventType,
        callback: Callable[[EventContext], None],
        priority: int = 0,
        name: str = "",
        once: bool = False,
    ) -> EventHandler:
        """注册事件处理器。

        Args:
            event_type: 事件类型
            callback: 回调函数
            priority: 优先级（越小越先执行）
            name: 处理器名称
            once: 是否只触发一次

        Returns:
            EventHandler 实例（可用于后续移除）
        """
        handler = EventHandler(
            callback=callback,
            event_type=event_type,
            priority=priority,
            name=name,
            once=once,
        )

        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)
        self._handlers[event_type].sort(key=lambda h: h.priority)

        return handler

    def off(
        self,
        event_type: EventType,
        handler_or_name: Optional[EventHandler | str] = None,
    ) -> int:
        """移除事件处理器。

        Args:
            event_type: 事件类型
            handler_or_name: 要移除的处理器或名称，None 表示移除该事件的所有处理器

        Returns:
            移除的处理器数量
        """
        if event_type not in self._handlers:
            return 0

        if handler_or_name is None:
            count = len(self._handlers[event_type])
            self._handlers[event_type] = []
            return count

        handlers = self._handlers[event_type]
        before_count = len(handlers)

        if isinstance(handler_or_name, str):
            self._handlers[event_type] = [
                h for h in handlers if h.name != handler_or_name
            ]
        else:
            self._handlers[event_type] = [
                h for h in handlers if h is not handler_or_name
            ]

        return before_count - len(self._handlers[event_type])

    def emit(
        self,
        event_type: EventType,
        data: Optional[dict[str, Any]] = None,
    ) -> EventContext:
        """触发事件。

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            EventContext 实例（可检查 cancelled 状态）
        """
        ctx = EventContext(event_type=event_type, data=data or {})

        if not self._enabled:
            return ctx

        # 记录历史
        self._history.append(ctx)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # 执行处理器
        to_remove: list[EventHandler] = []

        for handler in self._handlers.get(event_type, []):
            try:
                handler.callback(ctx)
                if handler.once:
                    to_remove.append(handler)
            except Exception as e:
                if self._error_handler:
                    self._error_handler(e, ctx)
                else:
                    # 默认：记录错误但不中断
                    error_ctx = EventContext(
                        event_type=EventType.ON_ERROR,
                        data={"original_event": event_type.value, "error": str(e)},
                        error=e,
                    )
                    self._history.append(error_ctx)

        # 清理一次性处理器
        for handler in to_remove:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

        return ctx

    def set_error_handler(
        self,
        handler: Callable[[Exception, EventContext], None],
    ) -> None:
        """设置全局错误处理器。

        Args:
            handler: 错误处理器函数 (exception, context) -> None
        """
        self._error_handler = handler

    def enable(self) -> None:
        """启用事件系统"""
        self._enabled = True

    def disable(self) -> None:
        """禁用事件系统（事件仍然记录但不触发处理器）"""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """是否启用"""
        return self._enabled

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 50,
    ) -> list[EventContext]:
        """获取事件历史。

        Args:
            event_type: 过滤的事件类型
            limit: 最大返回数量

        Returns:
            EventContext 列表
        """
        if event_type is not None:
            filtered = [ctx for ctx in self._history if ctx.event_type == event_type]
        else:
            filtered = list(self._history)

        return filtered[-limit:]

    def clear_history(self) -> int:
        """清空事件历史。

        Returns:
            清除的历史记录数
        """
        count = len(self._history)
        self._history.clear()
        return count

    def handler_count(self, event_type: Optional[EventType] = None) -> int:
        """获取处理器数量。

        Args:
            event_type: 事件类型，None 表示所有事件

        Returns:
            处理器数量
        """
        if event_type is not None:
            return len(self._handlers.get(event_type, []))
        return sum(len(handlers) for handlers in self._handlers.values())

    def list_handlers(self) -> list[dict[str, Any]]:
        """列出所有注册的处理器。

        Returns:
            处理器信息列表
        """
        result: list[dict[str, Any]] = []
        for event_type, handlers in self._handlers.items():
            for h in handlers:
                result.append({
                    "event_type": event_type.value,
                    "name": h.name or "<anonymous>",
                    "priority": h.priority,
                    "once": h.once,
                })
        return result


# 全局事件总线实例
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线单例。

    Returns:
        EventBus 实例
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """重置全局事件总线（用于测试）"""
    global _global_bus
    _global_bus = None
