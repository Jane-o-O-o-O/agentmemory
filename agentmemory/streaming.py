"""流式搜索：异步生成器逐步返回搜索结果。

支持：
- 异步搜索结果流（AsyncIterator）
- 带进度回调的搜索
- 可中断搜索（通过回调返回 False 停止）
- 多源并行搜索结果合并流
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from agentmemory.embedding_store import EmbeddingStore, cosine_similarity
from agentmemory.models import Memory, SearchResult


@dataclass
class SearchProgress:
    """搜索进度信息。

    Attributes:
        total_candidates: 总候选数
        processed: 已处理数
        found: 已找到的结果数
        elapsed_ms: 已用时间（毫秒）
        current_score: 当前结果的分数
    """

    total_candidates: int = 0
    processed: int = 0
    found: int = 0
    elapsed_ms: float = 0.0
    current_score: float = 0.0

    @property
    def progress_ratio(self) -> float:
        """进度比例（0~1）"""
        if self.total_candidates == 0:
            return 1.0
        return self.processed / self.total_candidates

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "total_candidates": self.total_candidates,
            "processed": self.processed,
            "found": self.found,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "progress_ratio": round(self.progress_ratio, 4),
            "current_score": round(self.current_score, 4),
        }


@dataclass
class StreamConfig:
    """流式搜索配置。

    Attributes:
        batch_size: 每批返回的结果数
        yield_progress: 是否定期返回进度信息
        progress_interval_ms: 进度回调间隔（毫秒）
        min_score: 最低分数阈值
        max_results: 最大结果数（0 表示不限）
    """

    batch_size: int = 1
    yield_progress: bool = False
    progress_interval_ms: float = 100.0
    min_score: float = 0.0
    max_results: int = 0


class StreamingSearcher:
    """流式搜索器。

    提供逐步返回搜索结果的能力，支持同步和异步两种模式。

    Args:
        store: EmbeddingStore 实例
        on_progress: 进度回调函数
        on_result: 结果回调函数（返回 False 可中断搜索）
    """

    def __init__(
        self,
        store: EmbeddingStore,
        on_progress: Optional[Callable[[SearchProgress], None]] = None,
        on_result: Optional[Callable[[SearchResult], bool | None]] = None,
    ) -> None:
        self._store = store
        self._on_progress = on_progress
        self._on_result = on_result

    def search_iter(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
        config: Optional[StreamConfig] = None,
    ) -> Iterator[SearchResult]:
        """同步迭代式搜索。

        逐步计算相似度并返回结果，支持中途停止。

        Args:
            query_embedding: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值
            tags: 标签过滤
            config: 流式配置

        Yields:
            SearchResult 对象，按相似度降序
        """
        cfg = config or StreamConfig()
        t0 = time.time()

        # 收集候选
        all_memories = self._store.list_all()
        candidates: list[Memory] = []

        for mem in all_memories:
            if tags and not all(mem.has_tag(t) for t in tags):
                continue
            if mem.embedding is None:
                continue
            candidates.append(mem)

        total = len(candidates)
        progress = SearchProgress(total_candidates=total)
        results: list[SearchResult] = []
        should_stop = False

        for i, mem in enumerate(candidates):
            if should_stop:
                break

            score = cosine_similarity(query_embedding, mem.embedding)  # type: ignore[arg-type]
            progress.processed = i + 1
            progress.current_score = score
            progress.elapsed_ms = (time.time() - t0) * 1000

            if score >= threshold:
                result = SearchResult(memory=mem, score=score)
                results.append(result)
                results.sort(key=lambda r: r.score, reverse=True)

                # 保持 top_k
                if len(results) > top_k:
                    results = results[:top_k]

                # 回调通知
                if self._on_result:
                    cont = self._on_result(result)
                    if cont is False:
                        should_stop = True

                # 批量 yield
                if len(results) >= cfg.batch_size:
                    for r in results:
                        yield r
                    results = []

            # 进度回调
            if self._on_progress and (i + 1) % max(1, total // 10) == 0:
                progress.found = len(results) + sum(1 for _ in [])
                self._on_progress(progress)

        # yield 剩余结果
        for r in results:
            yield r

    async def search_aiter(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
        config: Optional[StreamConfig] = None,
    ) -> AsyncIterator[SearchResult]:
        """异步迭代式搜索。

        逐步计算相似度并返回结果，每批之间让出事件循环。

        Args:
            query_embedding: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值
            tags: 标签过滤
            config: 流式配置

        Yields:
            SearchResult 对象，按相似度降序
        """
        cfg = config or StreamConfig()
        t0 = time.time()

        all_memories = self._store.list_all()
        candidates: list[Memory] = []

        for mem in all_memories:
            if tags and not all(mem.has_tag(t) for t in tags):
                continue
            if mem.embedding is None:
                continue
            candidates.append(mem)

        total = len(candidates)
        progress = SearchProgress(total_candidates=total)
        results: list[SearchResult] = []
        should_stop = False
        last_progress_time = t0

        for i, mem in enumerate(candidates):
            if should_stop:
                break

            score = cosine_similarity(query_embedding, mem.embedding)  # type: ignore[arg-type]
            progress.processed = i + 1
            progress.current_score = score
            progress.elapsed_ms = (time.time() - t0) * 1000

            if score >= threshold:
                result = SearchResult(memory=mem, score=score)
                results.append(result)
                results.sort(key=lambda r: r.score, reverse=True)

                if len(results) > top_k:
                    results = results[:top_k]

                if self._on_result:
                    cont = self._on_result(result)
                    if cont is False:
                        should_stop = True

                # 批量 yield
                if len(results) >= cfg.batch_size:
                    for r in results:
                        yield r
                    results = []

            # 进度回调 + 让出事件循环
            now = time.time()
            if (now - last_progress_time) * 1000 >= cfg.progress_interval_ms:
                progress.found = len(results)
                if self._on_progress:
                    self._on_progress(progress)
                last_progress_time = now
                await asyncio.sleep(0)  # 让出事件循环

        for r in results:
            yield r

    def search_progressive(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """渐进式搜索：边计算边维护 top-k 排序。

        适用于大批量数据的实时搜索，可以在任意时刻中断获取当前最优结果。

        Args:
            query_embedding: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值
            tags: 标签过滤

        Returns:
            SearchResult 列表（即使中途中断也有部分结果）
        """
        all_memories = self._store.list_all()
        results: list[SearchResult] = []
        min_score_in_results = float("-inf")

        for mem in all_memories:
            if tags and not all(mem.has_tag(t) for t in tags):
                continue
            if mem.embedding is None:
                continue

            score = cosine_similarity(query_embedding, mem.embedding)  # type: ignore[arg-type]

            if score < threshold:
                continue

            # 如果结果未满，直接添加
            if len(results) < top_k:
                results.append(SearchResult(memory=mem, score=score))
                results.sort(key=lambda r: r.score, reverse=True)
                min_score_in_results = results[-1].score if results else float("-inf")
            elif score > min_score_in_results:
                # 替换最低分
                results[-1] = SearchResult(memory=mem, score=score)
                results.sort(key=lambda r: r.score, reverse=True)
                min_score_in_results = results[-1].score

            if self._on_result:
                cont = self._on_result(SearchResult(memory=mem, score=score))
                if cont is False:
                    break

        return results


async def stream_search(
    store: EmbeddingStore,
    query_embedding: list[float],
    top_k: int = 10,
    threshold: float = 0.0,
    tags: Optional[list[str]] = None,
    batch_size: int = 1,
) -> list[SearchResult]:
    """便捷异步搜索函数。

    Args:
        store: EmbeddingStore 实例
        query_embedding: 查询向量
        top_k: 返回前 k 个结果
        threshold: 相似度阈值
        tags: 标签过滤
        batch_size: 每批大小

    Returns:
        搜索结果列表
    """
    searcher = StreamingSearcher(store)
    results: list[SearchResult] = []
    async for result in searcher.search_aiter(
        query_embedding=query_embedding,
        top_k=top_k,
        threshold=threshold,
        tags=tags,
        config=StreamConfig(batch_size=batch_size),
    ):
        results.append(result)
    return results
