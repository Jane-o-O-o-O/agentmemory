"""内存垃圾回收器：基于 TTL/重要性/访问模式自动清理记忆。

提供自动和手动两种模式的垃圾回收：
- 基于 TTL 过期
- 基于重要性阈值
- 基于访问频率/空闲时间
- 基于最大存活时间
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agentmemory.models import Memory
from agentmemory.lifecycle import MemoryLifecycle


@dataclass
class GCPolicy:
    """垃圾回收策略。

    Attributes:
        min_importance: 最低重要性阈值（0~1），低于此值的记忆将被清理
        max_age: 最大存活时间（秒），超过此时间的记忆将被清理
        min_access_count: 最低访问次数
        max_idle_time: 最大空闲时间（秒），超过此时间未访问的记忆将被清理
        batch_size: 每次 GC 清理的最大条目数
        preserve_tags: 保留带有这些标签的记忆（不清理）
    """

    min_importance: float = 0.0
    max_age: Optional[float] = None
    min_access_count: int = 0
    max_idle_time: Optional[float] = None
    batch_size: int = 100
    preserve_tags: list[str] = field(default_factory=list)


@dataclass
class GCResult:
    """垃圾回收结果。

    Attributes:
        collected: 被回收的记忆 ID 列表
        retained: 保留的记忆 ID 列表
        reasons: 每个被回收记忆的回收原因
        elapsed_ms: GC 执行耗时（毫秒）
    """

    collected: list[str] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def total_collected(self) -> int:
        """被回收的记忆数量。"""
        return len(self.collected)

    @property
    def total_retained(self) -> int:
        """保留的记忆数量。"""
        return len(self.retained)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict。"""
        return {
            "collected": self.collected,
            "retained": self.retained,
            "reasons": self.reasons,
            "elapsed_ms": self.elapsed_ms,
            "total_collected": self.total_collected,
            "total_retained": self.total_retained,
        }


class GarbageCollector:
    """记忆垃圾回收器。

    基于多种策略自动识别和清理低价值记忆：
    - TTL 过期（通过 MemoryLifecycle）
    - 重要性低于阈值
    - 超过最大存活时间
    - 长期未访问（空闲时间）

    Args:
        lifecycle: MemoryLifecycle 实例，用于获取生命周期信息
        policy: GC 策略
    """

    def __init__(
        self,
        lifecycle: Optional[MemoryLifecycle] = None,
        policy: Optional[GCPolicy] = None,
    ) -> None:
        self._lifecycle = lifecycle or MemoryLifecycle()
        self._policy = policy or GCPolicy()
        self._history: list[GCResult] = []

    @property
    def policy(self) -> GCPolicy:
        """当前 GC 策略。"""
        return self._policy

    @policy.setter
    def policy(self, value: GCPolicy) -> None:
        """设置 GC 策略。"""
        self._policy = value

    @property
    def history(self) -> list[GCResult]:
        """GC 历史记录。"""
        return list(self._history)

    def should_collect(self, memory: Memory, now: Optional[float] = None) -> Optional[str]:
        """判断记忆是否应该被回收。

        Args:
            memory: Memory 对象
            now: 当前时间戳（用于测试），None 使用 time.time()

        Returns:
            回收原因字符串，None 表示不应回收
        """
        current_time = now if now is not None else time.time()
        policy = self._policy

        # 保留标签检查
        if policy.preserve_tags:
            for tag in policy.preserve_tags:
                if memory.has_tag(tag):
                    return None

        # TTL 过期检查
        if self._lifecycle.is_expired(memory):
            return "ttl_expired"

        # 最大存活时间检查
        if policy.max_age is not None:
            age = current_time - memory.created_at
            if age > policy.max_age:
                return f"max_age_exceeded({age:.0f}s > {policy.max_age:.0f}s)"

        # 空闲时间检查
        if policy.max_idle_time is not None:
            last_access = self._lifecycle._last_accessed.get(memory.id)
            if last_access is not None:
                idle = current_time - last_access
                if idle > policy.max_idle_time:
                    return f"idle_too_long({idle:.0f}s > {policy.max_idle_time:.0f}s)"
            else:
                # 从未访问，用创建时间
                idle = current_time - memory.created_at
                if idle > policy.max_idle_time:
                    return f"never_accessed_and_idle({idle:.0f}s > {policy.max_idle_time:.0f}s)"

        # 访问次数检查（与空闲时间组合）
        if policy.min_access_count > 0:
            access_count = self._lifecycle.get_access_count(memory.id)
            if access_count < policy.min_access_count:
                # 只有同时满足空闲条件才回收
                if policy.max_idle_time is not None:
                    last_access = self._lifecycle._last_accessed.get(memory.id, memory.created_at)
                    idle = current_time - last_access
                    if idle > policy.max_idle_time:
                        return f"low_access({access_count} < {policy.min_access_count})"

        # 重要性检查
        if policy.min_importance > 0:
            importance = self._lifecycle.compute_importance_score(memory)
            if importance < policy.min_importance:
                return f"low_importance({importance:.3f} < {policy.min_importance:.3f})"

        return None

    def collect(
        self,
        memories: list[Memory],
        now: Optional[float] = None,
    ) -> GCResult:
        """执行垃圾回收。

        Args:
            memories: 记忆列表
            now: 当前时间戳（用于测试）

        Returns:
            GCResult 回收结果
        """
        start = time.time()
        current_time = now if now is not None else time.time()

        result = GCResult()
        collected_count = 0

        for memory in memories:
            if collected_count >= self._policy.batch_size:
                # 达到批次限制，剩余的标记为保留
                result.retained.append(memory.id)
                continue

            reason = self.should_collect(memory, now=current_time)
            if reason:
                result.collected.append(memory.id)
                result.reasons[memory.id] = reason
                # 清理生命周期数据
                self._lifecycle._access_counts.pop(memory.id, None)
                self._lifecycle._last_accessed.pop(memory.id, None)
                self._lifecycle._custom_ttls.pop(memory.id, None)
                self._lifecycle._importance.pop(memory.id, None)
                collected_count += 1
            else:
                result.retained.append(memory.id)

        result.elapsed_ms = (time.time() - start) * 1000
        self._history.append(result)
        return result

    def preview(
        self,
        memories: list[Memory],
        now: Optional[float] = None,
    ) -> GCResult:
        """预览垃圾回收结果（不实际清理数据）。

        Args:
            memories: 记忆列表
            now: 当前时间戳（用于测试）

        Returns:
            GCResult 预览结果
        """
        start = time.time()
        current_time = now if now is not None else time.time()

        result = GCResult()
        collected_count = 0

        for memory in memories:
            if collected_count >= self._policy.batch_size:
                result.retained.append(memory.id)
                continue

            reason = self.should_collect(memory, now=current_time)
            if reason:
                result.collected.append(memory.id)
                result.reasons[memory.id] = reason
                collected_count += 1
            else:
                result.retained.append(memory.id)

        result.elapsed_ms = (time.time() - start) * 1000
        return result

    def stats(self, memories: list[Memory], now: Optional[float] = None) -> dict[str, Any]:
        """统计记忆状态。

        Args:
            memories: 记忆列表
            now: 当前时间戳

        Returns:
            统计信息字典
        """
        current_time = now if now is not None else time.time()
        total = len(memories)
        expired = sum(1 for m in memories if self._lifecycle.is_expired(m))
        ages = [current_time - m.created_at for m in memories]
        importance_scores = [self._lifecycle.compute_importance_score(m) for m in memories]

        return {
            "total_memories": total,
            "expired": expired,
            "would_collect": len([m for m in memories if self.should_collect(m, current_time)]),
            "avg_age_seconds": sum(ages) / total if total else 0,
            "max_age_seconds": max(ages) if ages else 0,
            "avg_importance": sum(importance_scores) / total if total else 0,
            "min_importance": min(importance_scores) if importance_scores else 0,
            "gc_runs": len(self._history),
            "total_collected": sum(r.total_collected for r in self._history),
        }
