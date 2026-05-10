"""向量存储与余弦相似度搜索"""

from __future__ import annotations

import math
from typing import Optional

from agentmemory.models import Memory, SearchResult


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    Args:
        a: 第一个向量
        b: 第二个向量

    Returns:
        余弦相似度值 (-1 ~ 1)

    Raises:
        ValueError: 向量维度不同或为零向量
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不匹配: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量不支持余弦相似度计算")

    return dot / (norm_a * norm_b)


class EmbeddingStore:
    """基于内存的向量存储，支持余弦相似度搜索。

    Args:
        dimension: 向量维度
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._memories: dict[str, Memory] = {}

    @property
    def dimension(self) -> int:
        """向量维度"""
        return self._dimension

    def add(self, memory: Memory) -> None:
        """添加 Memory 到存储。

        Args:
            memory: 待添加的 Memory，必须有 embedding

        Raises:
            ValueError: embedding 为空或维度不匹配
        """
        if memory.embedding is None:
            raise ValueError(f"Memory {memory.id} 没有 embedding")
        if len(memory.embedding) != self._dimension:
            raise ValueError(
                f"embedding 维度不匹配: 期望 {self._dimension}, 实际 {len(memory.embedding)}"
            )
        self._memories[memory.id] = memory

    def remove(self, memory_id: str) -> None:
        """根据 id 删除 Memory。

        Args:
            memory_id: 待删除的 Memory ID

        Raises:
            KeyError: 该 ID 不存在
        """
        if memory_id not in self._memories:
            raise KeyError(f"Memory {memory_id} 不存在")
        del self._memories[memory_id]

    def get(self, memory_id: str) -> Optional[Memory]:
        """根据 id 获取 Memory。

        Args:
            memory_id: Memory ID

        Returns:
            对应的 Memory，不存在返回 None
        """
        return self._memories.get(memory_id)

    def count(self) -> int:
        """返回存储中的 Memory 数量"""
        return len(self._memories)

    def list_all(self) -> list[Memory]:
        """返回所有 Memory 列表"""
        return list(self._memories.values())

    def search(
        self,
        query: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        """向量相似度搜索。

        Args:
            query: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值，低于此值的结果将被过滤

        Returns:
            按相似度降序排列的 SearchResult 列表
        """
        if len(self._memories) == 0:
            return []

        results: list[SearchResult] = []
        for mem in self._memories.values():
            score = cosine_similarity(query, mem.embedding)  # type: ignore[arg-type]
            if score >= threshold:
                results.append(SearchResult(memory=mem, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
