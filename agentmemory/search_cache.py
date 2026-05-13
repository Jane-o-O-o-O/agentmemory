"""搜索结果缓存 — LRU 缓存加速频繁查询。

对高频重复查询提供缓存加速，避免重复的向量相似度计算。
支持基于查询文本和查询向量的缓存键，带 TTL 过期机制。
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Optional, Union

from agentmemory.models import SearchResult


class SearchCache:
    """搜索结果 LRU 缓存。

    使用 OrderedDict 实现 LRU 淘汰策略，支持：
    - 基于查询文本/向量的缓存键
    - TTL 过期机制
    - 缓存命中统计
    - 可配置最大容量

    Args:
        max_size: 最大缓存条目数（默认 256）
        ttl_seconds: 缓存条目 TTL（秒），None 表示永不过期

    Example:
        >>> cache = SearchCache(max_size=128, ttl_seconds=300)
        >>> cache.put("hello world", top_k=5, results=results)
        >>> cached = cache.get("hello world", top_k=5)
    """

    def __init__(
        self,
        max_size: int = 256,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size 必须 >= 1, got {max_size}")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, list[SearchResult]]] = OrderedDict()
        # 统计
        self._hits: int = 0
        self._misses: int = 0

    @property
    def max_size(self) -> int:
        """最大缓存容量"""
        return self._max_size

    @property
    def size(self) -> int:
        """当前缓存条目数"""
        self._evict_expired()
        return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        """缓存统计信息"""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }

    def _make_key(
        self,
        query: Union[str, list[float]],
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
        extra: Optional[str] = None,
    ) -> str:
        """生成缓存键。

        Args:
            query: 查询文本或向量
            top_k: 返回数量
            threshold: 相似度阈值
            tags: 标签过滤
            extra: 附加标识

        Returns:
            缓存键字符串
        """
        if isinstance(query, str):
            query_part = query
        else:
            # 向量：取 hash
            query_part = hashlib.md5(
                ",".join(f"{v:.6f}" for v in query[:16]).encode()
            ).hexdigest()

        tag_part = ",".join(sorted(tags)) if tags else ""
        extra_part = f"|{extra}" if extra else ""
        return f"{query_part}|k={top_k}|t={threshold:.4f}|tags={tag_part}{extra_part}"

    def _evict_expired(self) -> None:
        """移除过期条目"""
        if self._ttl is None:
            return
        now = time.time()
        expired_keys = [
            k for k, (ts, _) in self._cache.items()
            if now - ts > self._ttl
        ]
        for k in expired_keys:
            del self._cache[k]

    def get(
        self,
        query: Union[str, list[float]],
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
        extra: Optional[str] = None,
    ) -> Optional[list[SearchResult]]:
        """查询缓存。

        Args:
            query: 查询文本或向量
            top_k: 返回数量
            threshold: 相似度阈值
            tags: 标签过滤
            extra: 附加标识

        Returns:
            缓存的搜索结果列表，未命中返回 None
        """
        key = self._make_key(query, top_k, threshold, tags, extra)
        if key not in self._cache:
            self._misses += 1
            return None

        ts, results = self._cache[key]
        # 检查 TTL
        if self._ttl is not None and time.time() - ts > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        # 移到末尾（LRU）
        self._cache.move_to_end(key)
        self._hits += 1
        return results

    def put(
        self,
        query: Union[str, list[float]],
        results: list[SearchResult],
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
        extra: Optional[str] = None,
    ) -> None:
        """存入缓存。

        Args:
            query: 查询文本或向量
            results: 搜索结果列表
            top_k: 返回数量
            threshold: 相似度阈值
            tags: 标签过滤
            extra: 附加标识
        """
        key = self._make_key(query, top_k, threshold, tags, extra)
        self._cache[key] = (time.time(), results)
        self._cache.move_to_end(key)

        # LRU 淘汰
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, query: Union[str, list[float]], **kwargs: Any) -> bool:
        """使指定查询的缓存失效。

        Args:
            query: 查询文本或向量
            **kwargs: 传给 _make_key 的额外参数

        Returns:
            是否找到并移除
        """
        key = self._make_key(query, **kwargs)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """清空缓存。

        Returns:
            被清除的条目数
        """
        count = len(self._cache)
        self._cache.clear()
        return count
