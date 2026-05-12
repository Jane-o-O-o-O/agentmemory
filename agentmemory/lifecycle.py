"""记忆生命周期管理：TTL 过期、重要性评分、衰减机制。

为 Memory 对象提供时间衰减和重要性排序能力，
支持自动过期清理和基于多因素的记忆优先级管理。
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

from agentmemory.models import Memory


class MemoryLifecycle:
    """记忆生命周期管理器。

    提供以下能力：
    - TTL（Time-To-Live）：记忆过期时间
    - 重要性评分：基于多因素的记忆优先级
    - 时间衰减：越久远的记忆权重越低
    - 访问频率追踪：经常被检索的记忆权重更高

    Args:
        default_ttl: 默认 TTL（秒），None 表示永不过期
        decay_rate: 衰减速率（越大衰减越快），默认 0.001
        recency_weight: 时间新鲜度权重（0~1），默认 0.3
        frequency_weight: 访问频率权重（0~1），默认 0.3
        relevance_weight: 原始相关性权重（0~1），默认 0.4
    """

    def __init__(
        self,
        default_ttl: Optional[float] = None,
        decay_rate: float = 0.001,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.3,
        relevance_weight: float = 0.4,
    ) -> None:
        self._default_ttl = default_ttl
        self._decay_rate = decay_rate
        self._recency_weight = recency_weight
        self._frequency_weight = frequency_weight
        self._relevance_weight = relevance_weight

        # 访问计数：memory_id -> count
        self._access_counts: dict[str, int] = {}
        # 最后访问时间：memory_id -> timestamp
        self._last_accessed: dict[str, float] = {}
        # 自定义 TTL 覆盖：memory_id -> ttl_seconds
        self._custom_ttls: dict[str, float] = {}
        # 自定义重要性：memory_id -> importance (0~1)
        self._importance: dict[str, float] = {}

    def set_ttl(self, memory_id: str, ttl_seconds: float) -> None:
        """为特定记忆设置自定义 TTL。

        Args:
            memory_id: 记忆 ID
            ttl_seconds: TTL 秒数
        """
        self._custom_ttls[memory_id] = ttl_seconds

    def set_importance(self, memory_id: str, importance: float) -> None:
        """设置记忆的重要性评分。

        Args:
            memory_id: 记忆 ID
            importance: 重要性评分（0~1）
        """
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"importance 必须在 0~1 之间，got {importance}")
        self._importance[memory_id] = importance

    def record_access(self, memory_id: str) -> None:
        """记录一次记忆访问。

        Args:
            memory_id: 记忆 ID
        """
        self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1
        self._last_accessed[memory_id] = time.time()

    def get_access_count(self, memory_id: str) -> int:
        """获取记忆的访问次数。

        Args:
            memory_id: 记忆 ID

        Returns:
            访问次数
        """
        return self._access_counts.get(memory_id, 0)

    def is_expired(self, memory: Memory) -> bool:
        """检查记忆是否已过期。

        Args:
            memory: Memory 对象

        Returns:
            是否已过期
        """
        ttl = self._custom_ttls.get(memory.id, self._default_ttl)
        if ttl is None:
            return False
        return (time.time() - memory.created_at) > ttl

    def time_remaining(self, memory: Memory) -> Optional[float]:
        """获取记忆剩余存活时间。

        Args:
            memory: Memory 对象

        Returns:
            剩余秒数，永不过期返回 None
        """
        ttl = self._custom_ttls.get(memory.id, self._default_ttl)
        if ttl is None:
            return None
        elapsed = time.time() - memory.created_at
        return max(0.0, ttl - elapsed)

    def compute_decay_factor(self, memory: Memory) -> float:
        """计算时间衰减因子。

        使用指数衰减：factor = exp(-decay_rate * age_seconds)

        Args:
            memory: Memory 对象

        Returns:
            衰减因子（0~1），1 表示完全新鲜，趋近 0 表示非常陈旧
        """
        age = time.time() - memory.created_at
        return math.exp(-self._decay_rate * age)

    def compute_importance_score(
        self,
        memory: Memory,
        base_relevance: float = 1.0,
    ) -> float:
        """计算综合重要性评分。

        综合考虑：
        1. 时间新鲜度（recency）
        2. 访问频率（frequency）
        3. 原始相关性/手动设置的重要性（relevance）

        Args:
            memory: Memory 对象
            base_relevance: 基础相关性分数（通常来自搜索分数）

        Returns:
            综合重要性评分（0~1）
        """
        # 新鲜度分量
        recency = self.compute_decay_factor(memory)

        # 频率分量（对数归一化）
        count = self._access_counts.get(memory.id, 0)
        max_count = max(self._access_counts.values(), default=1) if self._access_counts else 1
        frequency = math.log1p(count) / math.log1p(max_count) if max_count > 0 else 0.0

        # 相关性分量
        custom_imp = self._importance.get(memory.id)
        relevance = custom_imp if custom_imp is not None else min(1.0, base_relevance)

        return (
            self._recency_weight * recency
            + self._frequency_weight * frequency
            + self._relevance_weight * relevance
        )

    def filter_expired(self, memories: list[Memory]) -> list[Memory]:
        """过滤掉已过期的记忆。

        Args:
            memories: 记忆列表

        Returns:
            未过期的记忆列表
        """
        return [m for m in memories if not self.is_expired(m)]

    def rank_by_importance(
        self,
        memories: list[Memory],
        relevance_scores: Optional[dict[str, float]] = None,
    ) -> list[tuple[Memory, float]]:
        """按重要性排序记忆。

        Args:
            memories: 记忆列表
            relevance_scores: 可选的 memory_id -> relevance 分数映射

        Returns:
            (Memory, importance_score) 元组列表，按重要性降序
        """
        scores: dict[str, float] = relevance_scores or {}
        ranked = [
            (m, self.compute_importance_score(m, scores.get(m.id, 1.0)))
            for m in memories
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def cleanup(self, memories: list[Memory]) -> list[Memory]:
        """清理过期记忆并清除其生命周期数据。

        Args:
            memories: 记忆列表

        Returns:
            未过期的记忆列表
        """
        expired = [m for m in memories if self.is_expired(m)]
        for m in expired:
            self._access_counts.pop(m.id, None)
            self._last_accessed.pop(m.id, None)
            self._custom_ttls.pop(m.id, None)
            self._importance.pop(m.id, None)
        return [m for m in memories if not self.is_expired(m)]

    def get_lifecycle_info(self, memory: Memory) -> dict[str, Any]:
        """获取记忆的生命周期信息。

        Args:
            memory: Memory 对象

        Returns:
            包含生命周期信息的字典
        """
        return {
            "memory_id": memory.id,
            "age_seconds": time.time() - memory.created_at,
            "is_expired": self.is_expired(memory),
            "time_remaining": self.time_remaining(memory),
            "decay_factor": self.compute_decay_factor(memory),
            "access_count": self._access_counts.get(memory.id, 0),
            "importance": self._importance.get(memory.id),
            "ttl": self._custom_ttls.get(memory.id, self._default_ttl),
        }
