"""Embedding 缓存层 — LRU 缓存避免重复计算 embedding。

包装 EmbeddingProvider，对相同文本的嵌入计算结果进行缓存，
减少重复计算开销。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional

from agentmemory.embedding_provider import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """带 LRU 缓存的 EmbeddingProvider 包装器。

    对相同文本的嵌入计算结果进行缓存，避免重复计算。
    适合批量操作中存在大量重复文本的场景。

    Args:
        provider: 被包装的 EmbeddingProvider 实例
        max_cache_size: 最大缓存条目数（默认 1024）
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        max_cache_size: int = 1024,
    ) -> None:
        self._provider = provider
        self._max_cache_size = max_cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def dimension(self) -> int:
        """返回嵌入维度。"""
        return self._provider.dimension()

    def embed(self, text: str) -> list[float]:
        """计算文本的嵌入向量（带 LRU 缓存）。

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        cache_key = self._make_key(text)

        if cache_key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        self._misses += 1
        embedding = self._provider.embed(text)

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_cache_size:
            self._cache.popitem(last=False)

        self._cache[cache_key] = embedding
        return embedding

    def _make_key(self, text: str) -> str:
        """生成缓存键 — 对长文本使用 hash 以节省内存。"""
        if len(text) <= 256:
            return text
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @property
    def cache_stats(self) -> dict[str, int]:
        """返回缓存统计信息。

        Returns:
            包含 hits, misses, size, max_size 的字典
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_cache_size,
            "hit_rate": (
                round(self._hits / (self._hits + self._misses), 4)
                if (self._hits + self._misses) > 0
                else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """清空缓存。"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def __repr__(self) -> str:
        stats = self.cache_stats
        return (
            f"CachedEmbeddingProvider("
            f"provider={self._provider!r}, "
            f"cache_size={stats['size']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
