"""异步 API 封装 — 为 HybridMemory 提供 async 方法。

支持 asyncio.gather 并发操作，适合高并发 RAG 场景。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from agentmemory.models import Memory, SearchResult
from agentmemory.hybrid_memory import HybridMemory


class AsyncHybridMemory:
    """HybridMemory 的异步封装。

    在独立线程中执行同步操作，通过 asyncio 接口暴露，
    支持 asyncio.gather 等并发模式。

    Args:
        memory: HybridMemory 实例
        max_workers: 线程池最大线程数
    """

    def __init__(
        self,
        memory: HybridMemory,
        max_workers: int = 4,
    ) -> None:
        self._memory = memory
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def memory(self) -> HybridMemory:
        """底层 HybridMemory 实例"""
        return self._memory

    async def _run(self, func, *args, **kwargs):
        """在线程池中执行同步函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs),
        )

    # --- 记忆管理 ---

    async def aremember(
        self,
        content: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        ttl: Optional[float] = None,
    ) -> Memory:
        """异步添加记忆。"""
        return await self._run(
            self._memory.remember,
            content, embedding, metadata, tags, importance, ttl,
        )

    async def aforget(self, memory_id: str) -> None:
        """异步删除记忆。"""
        return await self._run(self._memory.forget, memory_id)

    async def aget_memory(self, memory_id: str) -> Optional[Memory]:
        """异步获取记忆。"""
        return await self._run(self._memory.get_memory, memory_id)

    async def aupdate_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Memory:
        """异步更新记忆。"""
        return await self._run(
            self._memory.update_memory, memory_id, content, metadata, tags,
        )

    async def alist_all(self) -> list[Memory]:
        """异步获取所有记忆。"""
        return await self._run(self._memory.list_all)

    # --- 搜索 ---

    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """异步向量搜索。"""
        return await self._run(
            self._memory.search, query_embedding, top_k, threshold, tags,
        )

    async def asearch_text(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """异步文本搜索。"""
        return await self._run(
            self._memory.search_text, query, top_k, threshold, tags,
        )

    async def ahybrid_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        graph_depth: int = 1,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """异步混合搜索。"""
        return await self._run(
            self._memory.hybrid_search,
            query_embedding, top_k, threshold, graph_depth, tags,
        )

    async def ahybrid_search_text(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        graph_depth: int = 1,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """异步文本混合搜索。"""
        return await self._run(
            self._memory.hybrid_search_text,
            query, top_k, threshold, graph_depth, tags,
        )

    async def abatch_search(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[list[SearchResult]]:
        """异步批量搜索 — 使用 asyncio.gather 并发执行。"""
        tasks = [
            self.asearch(emb, top_k=top_k, threshold=threshold, tags=tags)
            for emb in query_embeddings
        ]
        return await asyncio.gather(*tasks)

    async def abatch_remember(
        self,
        contents: list[str],
        embeddings: Optional[list[list[float]]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        tagss: Optional[list[list[str]]] = None,
    ) -> list[Memory]:
        """异步批量添加记忆 — 使用 asyncio.gather 并发执行。"""
        n = len(contents)
        tasks = []
        for i in range(n):
            emb = embeddings[i] if embeddings else None
            meta = metadatas[i] if metadatas else None
            tags = tagss[i] if tagss else None
            tasks.append(self.aremember(contents[i], emb, meta, tags))
        return await asyncio.gather(*tasks)

    # --- 知识图谱 ---

    async def aadd_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[dict[str, Any]] = None,
    ):
        """异步添加实体。"""
        return await self._run(
            self._memory.add_entity, name, entity_type, properties,
        )

    async def aadd_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ):
        """异步添加关系。"""
        return await self._run(
            self._memory.add_relation,
            source_id, target_id, relation_type, weight,
        )

    # --- 持久化 ---

    async def asave(self) -> None:
        """异步保存。"""
        return await self._run(self._memory.save)

    async def aload(self) -> None:
        """异步加载。"""
        return await self._run(self._memory.load)

    # --- 生命周期 ---

    async def acleanup_expired(self) -> list[str]:
        """异步清理过期记忆。"""
        return await self._run(self._memory.cleanup_expired)

    async def aget_lifecycle_info(self, memory_id: str) -> Optional[dict[str, Any]]:
        """异步获取生命周期信息。"""
        return await self._run(self._memory.get_lifecycle_info, memory_id)

    # --- 统计 ---

    async def astats(self) -> dict[str, Any]:
        """异步获取统计信息。"""
        return await self._run(self._memory.stats)

    # --- 上下文管理 ---

    async def __aenter__(self) -> "AsyncHybridMemory":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._executor.shutdown(wait=False)
