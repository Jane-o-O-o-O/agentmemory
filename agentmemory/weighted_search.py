"""加权搜索排序 — 融合向量相似度、重要性、时间衰减、访问频率。

提供智能排序策略，让搜索结果不仅考虑语义相关性，
还考虑记忆的重要程度、新鲜度和使用频率。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentmemory.models import Memory, SearchResult


@dataclass
class ScoringWeights:
    """搜索评分权重配置。

    Attributes:
        similarity: 向量相似度权重（默认 0.5）
        importance: 重要性权重（默认 0.2）
        recency: 时间新鲜度权重（默认 0.2）
        frequency: 访问频率权重（默认 0.1）
    """

    similarity: float = 0.5
    importance: float = 0.2
    recency: float = 0.2
    frequency: float = 0.1

    def normalize(self) -> ScoringWeights:
        """归一化权重使总和为 1。

        Returns:
            归一化后的新 ScoringWeights
        """
        total = self.similarity + self.importance + self.recency + self.frequency
        if total == 0:
            return ScoringWeights()
        return ScoringWeights(
            similarity=self.similarity / total,
            importance=self.importance / total,
            recency=self.recency / total,
            frequency=self.frequency / total,
        )

    def validate(self) -> None:
        """验证权重值。

        Raises:
            ValueError: 权重值不在有效范围内
        """
        for name in ("similarity", "importance", "recency", "frequency"):
            val = getattr(self, name)
            if val < 0 or val > 1:
                raise ValueError(f"权重 {name} 必须在 [0, 1] 范围内，got {val}")


class WeightedScorer:
    """加权搜索评分器。

    根据多维因素对搜索结果进行重新排序，包括：
    - 向量相似度（余弦相似度）
    - 记忆重要性（用户设置 0~1）
    - 时间新鲜度（指数衰减）
    - 访问频率（对数衰减）

    Args:
        weights: 评分权重配置
        decay_rate: 时间衰减速率（越大衰减越快）
        half_life_hours: 半衰期（小时），超过此时间的记忆重要性减半
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        decay_rate: float = 0.001,
        half_life_hours: float = 168.0,  # 7 天
    ) -> None:
        self._weights = (weights or ScoringWeights()).normalize()
        self._decay_rate = decay_rate
        self._half_life = half_life_hours * 3600  # 转换为秒
        # 访问次数记录：memory_id -> access_count
        self._access_counts: dict[str, int] = {}

    @property
    def weights(self) -> ScoringWeights:
        """当前权重配置"""
        return self._weights

    def set_weights(self, weights: ScoringWeights) -> None:
        """设置新的权重配置。

        Args:
            weights: 新的权重配置
        """
        weights.validate()
        self._weights = weights.normalize()

    def record_access(self, memory_id: str) -> None:
        """记录一次访问。

        Args:
            memory_id: 记忆 ID
        """
        self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1

    def get_access_count(self, memory_id: str) -> int:
        """获取访问次数。

        Args:
            memory_id: 记忆 ID

        Returns:
            访问次数
        """
        return self._access_counts.get(memory_id, 0)

    def _compute_importance_score(self, memory: Memory) -> float:
        """计算重要性得分。

        优先使用 metadata 中的 importance 字段，否则使用标签数作为代理。

        Args:
            memory: 记忆对象

        Returns:
            重要性得分 (0~1)
        """
        # 显式设置的重要性
        if "importance" in memory.metadata:
            return max(0.0, min(1.0, float(memory.metadata["importance"])))

        # 标签数量作为重要性代理（更多标签 = 更重要）
        tag_score = min(1.0, len(memory.tags) / 5.0)

        # 元数据丰富度作为代理
        meta_score = min(1.0, len(memory.metadata) / 5.0)

        return (tag_score + meta_score) / 2

    def _compute_recency_score(self, memory: Memory) -> float:
        """计算时间新鲜度得分。

        使用指数衰减：score = exp(-decay_rate * age_hours)

        Args:
            memory: 记忆对象

        Returns:
            新鲜度得分 (0~1)
        """
        age_seconds = time.time() - memory.created_at
        if self._half_life > 0:
            # 使用半衰期公式
            return math.pow(0.5, age_seconds / self._half_life)
        # 使用指数衰减
        return math.exp(-self._decay_rate * age_seconds / 3600)

    def _compute_frequency_score(self, memory: Memory) -> float:
        """计算访问频率得分。

        使用对数衰减：score = log(1 + count) / log(1 + max_count)

        Args:
            memory: 记忆对象

        Returns:
            频率得分 (0~1)
        """
        count = self._access_counts.get(memory.id, 0)
        if count == 0:
            return 0.0
        max_count = max(self._access_counts.values()) if self._access_counts else 1
        return math.log(1 + count) / math.log(1 + max_count) if max_count > 0 else 0.0

    def compute_score(
        self,
        memory: Memory,
        similarity_score: float,
    ) -> float:
        """计算综合得分。

        Args:
            memory: 记忆对象
            similarity_score: 向量相似度得分 (0~1)

        Returns:
            加权综合得分
        """
        w = self._weights
        importance = self._compute_importance_score(memory)
        recency = self._compute_recency_score(memory)
        frequency = self._compute_frequency_score(memory)

        return (
            w.similarity * similarity_score
            + w.importance * importance
            + w.recency * recency
            + w.frequency * frequency
        )

    def rerank(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """对搜索结果进行加权重排序。

        Args:
            results: 原始搜索结果列表

        Returns:
            重排序后的结果列表
        """
        scored: list[tuple[float, SearchResult]] = []
        for result in results:
            weighted_score = self.compute_score(result.memory, result.score)
            scored.append((weighted_score, result))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 更新 score 为加权分数
        reranked: list[SearchResult] = []
        for weighted_score, result in scored:
            reranked.append(SearchResult(
                memory=result.memory,
                score=weighted_score,
                context=result.context,
            ))
        return reranked


def weighted_search(
    results: list[SearchResult],
    weights: Optional[ScoringWeights] = None,
    scorer: Optional[WeightedScorer] = None,
) -> list[SearchResult]:
    """便捷函数：对搜索结果进行加权重排序。

    Args:
        results: 原始搜索结果
        weights: 权重配置（与 scorer 二选一）
        scorer: 自定义评分器（优先使用）

    Returns:
        重排序后的结果列表
    """
    if scorer is None:
        scorer = WeightedScorer(weights=weights)
    return scorer.rerank(results)
