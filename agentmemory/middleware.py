"""可组合中间件管道：pre/post hooks，可变换数据/阻断操作。

中间件可以在操作执行前后拦截、变换或阻断操作，用于：
- 日志记录
- 输入验证/变换
- 权限检查
- 性能监控
- 重试逻辑
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class HookType(Enum):
    """Hook 类型"""
    PRE = "pre"
    POST = "post"


@dataclass
class HookContext:
    """Hook 上下文，传递操作信息。

    Attributes:
        operation: 操作名称 (如 'remember', 'search', 'forget')
        data: 操作数据（pre hook 中可修改）
        result: 操作结果（post hook 中可用）
        metadata: 附加元数据
        blocked: 是否被阻断
        block_reason: 阻断原因
    """

    operation: str
    data: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    block_reason: Optional[str] = None

    def block(self, reason: str) -> None:
        """阻断操作。

        Args:
            reason: 阻断原因
        """
        self.blocked = True
        self.block_reason = reason


# Hook 函数类型：接收 HookContext，可返回修改后的 context
HookFn = Callable[[HookContext], Optional[HookContext]]


@dataclass
class MiddlewareEntry:
    """中间件条目。

    Attributes:
        name: 中间件名称
        hook_type: Hook 类型 (PRE/POST)
        fn: Hook 函数
        priority: 优先级（越小越先执行）
        enabled: 是否启用
    """

    name: str
    hook_type: HookType
    fn: HookFn
    priority: int = 100
    enabled: bool = True


class MiddlewarePipeline:
    """可组合中间件管道。

    支持 pre/post hooks，按优先级排序执行。
    Hook 可以修改数据、阻断操作、记录日志。

    Example:
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add_pre("validate", lambda ctx: ctx if ctx.data.get("content") else None)
        >>> pipeline.add_post("log", lambda ctx: print(f"Done: {ctx.operation}"))
        >>> ctx = HookContext(operation="remember", data={"content": "hello"})
        >>> ctx = pipeline.run_pre(ctx)
    """

    def __init__(self) -> None:
        self._entries: list[MiddlewareEntry] = []
        self._operation_hooks: dict[str, list[MiddlewareEntry]] = {}

    def add_pre(
        self,
        name: str,
        fn: HookFn,
        priority: int = 100,
        operation: Optional[str] = None,
    ) -> None:
        """添加 pre-hook（操作前执行）。

        Args:
            name: 中间件名称
            fn: Hook 函数
            priority: 优先级（越小越先执行）
            operation: 限定操作名称，None 表示对所有操作生效
        """
        entry = MiddlewareEntry(name=name, hook_type=HookType.PRE, fn=fn, priority=priority)
        if operation:
            self._operation_hooks.setdefault(operation, []).append(entry)
        else:
            self._entries.append(entry)

    def add_post(
        self,
        name: str,
        fn: HookFn,
        priority: int = 100,
        operation: Optional[str] = None,
    ) -> None:
        """添加 post-hook（操作后执行）。

        Args:
            name: 中间件名称
            fn: Hook 函数
            priority: 优先级（越小越先执行）
            operation: 限定操作名称，None 表示对所有操作生效
        """
        entry = MiddlewareEntry(name=name, hook_type=HookType.POST, fn=fn, priority=priority)
        if operation:
            self._operation_hooks.setdefault(operation, []).append(entry)
        else:
            self._entries.append(entry)

    def remove(self, name: str) -> bool:
        """移除中间件。

        Args:
            name: 中间件名称

        Returns:
            是否成功移除
        """
        original_len = len(self._entries)
        self._entries = [e for e in self._entries if e.name != name]
        for op_entries in self._operation_hooks.values():
            op_entries[:] = [e for e in op_entries if e.name != name]
        return len(self._entries) < original_len or any(
            len(v) > 0 for v in self._operation_hooks.values()
        )

    def enable(self, name: str) -> bool:
        """启用中间件。

        Args:
            name: 中间件名称

        Returns:
            是否找到并启用
        """
        return self._set_enabled(name, True)

    def disable(self, name: str) -> bool:
        """禁用中间件。

        Args:
            name: 中间件名称

        Returns:
            是否找到并禁用
        """
        return self._set_enabled(name, False)

    def _set_enabled(self, name: str, enabled: bool) -> bool:
        """设置中间件启用状态。"""
        found = False
        for entry in self._entries:
            if entry.name == name:
                entry.enabled = enabled
                found = True
        for op_entries in self._operation_hooks.values():
            for entry in op_entries:
                if entry.name == name:
                    entry.enabled = enabled
                    found = True
        return found

    def _get_entries(self, hook_type: HookType, operation: Optional[str] = None) -> list[MiddlewareEntry]:
        """获取指定类型和操作的中间件条目，按优先级排序。"""
        entries: list[MiddlewareEntry] = []
        for entry in self._entries:
            if entry.hook_type == hook_type and entry.enabled:
                entries.append(entry)
        if operation and operation in self._operation_hooks:
            for entry in self._operation_hooks[operation]:
                if entry.hook_type == hook_type and entry.enabled:
                    entries.append(entry)
        entries.sort(key=lambda e: e.priority)
        return entries

    def run_pre(self, ctx: HookContext) -> HookContext:
        """执行所有 pre-hooks。

        Args:
            ctx: Hook 上下文

        Returns:
            可能被修改的 Hook 上下文
        """
        for entry in self._get_entries(HookType.PRE, ctx.operation):
            result = entry.fn(ctx)
            if result is not None:
                ctx = result
            if ctx.blocked:
                break
        return ctx

    def run_post(self, ctx: HookContext) -> HookContext:
        """执行所有 post-hooks。

        Args:
            ctx: Hook 上下文

        Returns:
            可能被修改的 Hook 上下文
        """
        for entry in self._get_entries(HookType.POST, ctx.operation):
            result = entry.fn(ctx)
            if result is not None:
                ctx = result
        return ctx

    def list_middleware(self) -> list[dict[str, Any]]:
        """列出所有中间件。

        Returns:
            中间件信息列表
        """
        items: list[dict[str, Any]] = []
        for entry in self._entries:
            items.append({
                "name": entry.name,
                "type": entry.hook_type.value,
                "priority": entry.priority,
                "enabled": entry.enabled,
                "operation": None,
            })
        for op, entries in self._operation_hooks.items():
            for entry in entries:
                items.append({
                    "name": entry.name,
                    "type": entry.hook_type.value,
                    "priority": entry.priority,
                    "enabled": entry.enabled,
                    "operation": op,
                })
        items.sort(key=lambda x: x["priority"])
        return items

    def clear(self) -> None:
        """清空所有中间件。"""
        self._entries.clear()
        self._operation_hooks.clear()

    def __len__(self) -> int:
        """中间件总数。"""
        count = len(self._entries)
        for entries in self._operation_hooks.values():
            count += len(entries)
        return count


# 内置中间件
class BuiltinMiddleware:
    """内置中间件工厂。"""

    @staticmethod
    def timing() -> tuple[str, HookFn, HookFn]:
        """计时中间件，记录操作耗时。

        Returns:
            (name, pre_hook, post_hook) 元组
        """
        def pre(ctx: HookContext) -> None:
            ctx.metadata["start_time"] = time.time()

        def post(ctx: HookContext) -> None:
            start = ctx.metadata.get("start_time")
            if start:
                ctx.metadata["elapsed_ms"] = (time.time() - start) * 1000

        return "timing", pre, post

    @staticmethod
    def content_validator(min_length: int = 1, max_length: int = 10000) -> tuple[str, HookFn]:
        """内容验证中间件。

        Args:
            min_length: 最小内容长度
            max_length: 最大内容长度

        Returns:
            (name, pre_hook) 元组
        """
        def pre(ctx: HookContext) -> None:
            content = ctx.data.get("content", "")
            if isinstance(content, str):
                if len(content) < min_length:
                    ctx.block(f"内容长度 {len(content)} < 最小要求 {min_length}")
                elif len(content) > max_length:
                    ctx.block(f"内容长度 {len(content)} > 最大限制 {max_length}")

        return "content_validator", pre

    @staticmethod
    def rate_limiter(max_per_second: float = 100.0) -> tuple[str, HookFn]:
        """速率限制中间件。

        Args:
            max_per_second: 每秒最大请求数

        Returns:
            (name, pre_hook) 元组
        """
        timestamps: list[float] = []

        def pre(ctx: HookContext) -> None:
            now = time.time()
            # 清理 1 秒前的时间戳
            timestamps[:] = [t for t in timestamps if now - t < 1.0]
            if len(timestamps) >= max_per_second:
                ctx.block(f"速率限制: {max_per_second}/秒")
            else:
                timestamps.append(now)

        return "rate_limiter", pre

    @staticmethod
    def audit_log(callback: Callable[[str, HookContext], None]) -> tuple[str, HookFn, HookFn]:
        """审计日志中间件。

        Args:
            callback: 日志回调函数，接收 (message, ctx)

        Returns:
            (name, pre_hook, post_hook) 元组
        """
        def pre(ctx: HookContext) -> None:
            callback(f"[PRE] {ctx.operation} data_keys={list(ctx.data.keys())}", ctx)

        def post(ctx: HookContext) -> None:
            elapsed = ctx.metadata.get("elapsed_ms", "?")
            callback(f"[POST] {ctx.operation} elapsed={elapsed}ms", ctx)

        return "audit_log", pre, post
