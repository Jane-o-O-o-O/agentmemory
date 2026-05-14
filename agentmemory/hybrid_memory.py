"""混合记忆框架：统一向量搜索与知识图谱的高级 API。

支持 LSH 加速搜索、记忆生命周期管理、高级查询等。
"""

from __future__ import annotations

import csv
import json
import time
from io import StringIO
from typing import Any, Callable, Optional

from agentmemory.models import Entity, Memory, Relation, SearchResult
from agentmemory.embedding_store import EmbeddingStore
from agentmemory.knowledge_graph import KnowledgeGraph
from agentmemory.embedding_provider import EmbeddingProvider
from agentmemory.lifecycle import MemoryLifecycle
from agentmemory.search_filter import SearchFilter, filter_search_results
from agentmemory.weighted_search import WeightedScorer, ScoringWeights
from agentmemory.search_cache import SearchCache
from agentmemory.metrics import (
    MetricsCollector,
    HealthChecker,
    HealthStatus,
    HealthCheck,
    check_memory_health,
    check_lsh_health,
)


class HybridMemory:
    """混合记忆系统，结合向量搜索与知识图谱。

    提供统一的记忆存储、检索和知识关联 API。
    支持批量操作、标签过滤、数据导出/导入、生命周期管理。

    Args:
        dimension: 向量维度。如果提供 embedding_provider，可以从 provider 自动推断。
        embedding_provider: Embedding 提供者，设置后 remember/search_text 可自动计算向量。
        storage_path: 持久化存储路径（目录或文件），None 表示不持久化
        storage_backend: 存储后端类型，'json' 或 'sqlite'
        auto_save: 每次 remember/forget 后自动保存
        auto_load: 初始化时自动加载已有数据
        use_lsh: 是否启用 LSH 近似搜索索引（默认 False）
        lsh_tables: LSH 哈希表数量（仅 use_lsh=True 时生效）
        lsh_hyperplanes: LSH 超平面数量（仅 use_lsh=True 时生效）
        default_ttl: 默认记忆 TTL（秒），None 表示永不过期
        decay_rate: 时间衰减速率
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage_path: Optional[str] = None,
        storage_backend: str = "json",
        auto_save: bool = False,
        auto_load: bool = False,
        use_lsh: bool = False,
        lsh_tables: int = 8,
        lsh_hyperplanes: int = 16,
        default_ttl: Optional[float] = None,
        decay_rate: float = 0.001,
        weighted_scoring: bool = False,
        scoring_weights: Optional[ScoringWeights] = None,
        cache_size: int = 0,
        cache_ttl: Optional[float] = None,
    ) -> None:
        self._embedding_provider = embedding_provider

        # 搜索过滤器（可全局设置）
        self._default_filter: Optional[SearchFilter] = None
        # 加权评分器
        self._scorer: Optional[WeightedScorer] = None
        if weighted_scoring:
            self._scorer = WeightedScorer(weights=scoring_weights, decay_rate=decay_rate)

        # 搜索缓存
        self._cache: Optional[SearchCache] = None
        if cache_size > 0:
            self._cache = SearchCache(max_size=cache_size, ttl_seconds=cache_ttl)

        # 推断维度
        if dimension is not None and embedding_provider is not None:
            if dimension != embedding_provider.dimension():
                raise ValueError(
                    f"维度不匹配: 手动指定 {dimension}, provider 为 {embedding_provider.dimension()}"
                )
            self._dimension = dimension
        elif dimension is not None:
            self._dimension = dimension
        elif embedding_provider is not None:
            self._dimension = embedding_provider.dimension()
        else:
            raise ValueError("必须指定 dimension 或 embedding_provider")

        self.embedding_store = EmbeddingStore(
            dimension=self._dimension,
            use_lsh=use_lsh,
            lsh_tables=lsh_tables,
            lsh_hyperplanes=lsh_hyperplanes,
        )
        self.knowledge_graph = KnowledgeGraph()
        self.lifecycle = MemoryLifecycle(
            default_ttl=default_ttl,
            decay_rate=decay_rate,
        )
        self._storage_path = storage_path
        self._storage_backend = storage_backend
        self._auto_save = auto_save
        self._backend = self._create_backend() if storage_path else None

        # 可观测性：指标收集器
        self._metrics = MetricsCollector(namespace="agentmemory")
        self._metrics_timer_search = self._metrics.timer("search_latency_ms", "搜索延迟（毫秒）")
        self._metrics_timer_remember = self._metrics.timer("remember_latency_ms", "记忆添加延迟（毫秒）")
        self._metrics_counter_remember = self._metrics.counter("remember_count", "记忆添加次数")
        self._metrics_counter_search = self._metrics.counter("search_count", "搜索次数")
        self._metrics_counter_forget = self._metrics.counter("forget_count", "删除次数")
        self._metrics_gauge_memories = self._metrics.gauge("memory_count", "当前记忆数量")

        if auto_load and self._backend:
            self.load()
            self._metrics_gauge_memories.set(self.embedding_store.count())

    def _create_backend(self) -> Any:
        """创建持久化后端实例"""
        from agentmemory.persistence import JSONBackend, SQLiteBackend

        if self._storage_backend == "json":
            return JSONBackend(self._storage_path)
        elif self._storage_backend == "sqlite":
            return SQLiteBackend(self._storage_path)
        else:
            raise ValueError(f"不支持的存储后端: {self._storage_backend}")

    @property
    def dimension(self) -> int:
        """向量维度"""
        return self._dimension

    # --- 持久化 ---

    # --- 上下文管理器 ---

    def session(self) -> "MemorySession":
        """创建记忆会话上下文管理器。

        使用 `with hm.session() as s:` 自动在退出时保存数据，
        进入时可选加载。

        Returns:
            MemorySession 实例

        Example:
            >>> with hm.session() as s:
            ...     s.remember("hello")
            ...     s.remember("world")
            # 自动保存
        """
        return MemorySession(self)

    def save(self) -> None:
        """保存当前数据到磁盘。

        Raises:
            ValueError: 未配置 storage_path
        """
        if self._backend is None:
            raise ValueError("未配置 storage_path，无法保存")
        self._backend.save_embedding_store(self.embedding_store)
        self._backend.save_knowledge_graph(self.knowledge_graph)

    def load(self) -> None:
        """从磁盘加载数据。

        Raises:
            ValueError: 未配置 storage_path
        """
        if self._backend is None:
            raise ValueError("未配置 storage_path，无法加载")
        self._backend.load_embedding_store(self.embedding_store)
        self._backend.load_knowledge_graph(self.knowledge_graph)

    def _auto_save_if_enabled(self) -> None:
        """如果启用了自动保存，则执行保存"""
        if self._auto_save and self._backend:
            self.save()

    # --- 记忆管理 ---

    def remember(
        self,
        content: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        ttl: Optional[float] = None,
    ) -> Memory:
        """添加一条记忆到存储。

        如果配置了 embedding_provider 且未手动传入 embedding，
        将自动使用 provider 计算向量。

        Args:
            content: 文本内容
            embedding: 向量表示（可选，有 provider 时自动计算）
            metadata: 附加元数据
            tags: 标签列表
            importance: 重要性评分（0~1）
            ttl: 自定义 TTL（秒）

        Returns:
            创建的 Memory 对象

        Raises:
            ValueError: 没有 embedding 也没有 provider
        """
        _t = time.time()
        if embedding is None and self._embedding_provider is not None:
            embedding = self._embedding_provider.embed(content)
        mem = Memory(content=content, embedding=embedding, metadata=metadata or {}, tags=tags or [])
        self.embedding_store.add(mem)

        if importance is not None:
            self.lifecycle.set_importance(mem.id, importance)
        if ttl is not None:
            self.lifecycle.set_ttl(mem.id, ttl)

        self._auto_save_if_enabled()
        elapsed_ms = (time.time() - _t) * 1000
        self._metrics_timer_remember.record(elapsed_ms)
        self._metrics_counter_remember.increment()
        self._metrics_gauge_memories.set(self.embedding_store.count())
        return mem

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Memory:
        """更新已有记忆。

        如果更新了内容且有 embedding_provider，会自动重新计算向量。

        Args:
            memory_id: 记忆 ID
            content: 新内容（可选）
            metadata: 新元数据（与现有合并）
            tags: 新标签列表（替换现有）

        Returns:
            更新后的 Memory

        Raises:
            KeyError: 记忆不存在
        """
        if content is not None and self._embedding_provider is not None:
            new_embedding = self._embedding_provider.embed(content)
            mem = self.embedding_store.get(memory_id)
            if mem is None:
                raise KeyError(f"Memory {memory_id} 不存在")
            mem.embedding = new_embedding

        mem = self.embedding_store.update(memory_id, content=content, metadata=metadata, tags=tags)
        self._auto_save_if_enabled()
        return mem

    def merge_memories(self, memory_ids: list[str], new_content: Optional[str] = None) -> Memory:
        """合并多条记忆为一条。

        将多条记忆的内容合并，元数据合并，标签去重合并，
        保留第一条记忆的 ID，删除其余记忆。

        Args:
            memory_ids: 要合并的记忆 ID 列表
            new_content: 合并后的新内容（可选，默认拼接所有内容）

        Returns:
            合并后的 Memory

        Raises:
            ValueError: 记忆列表为空
            KeyError: 任何记忆 ID 不存在
        """
        if not memory_ids:
            raise ValueError("记忆列表不能为空")

        memories: list[Memory] = []
        for mid in memory_ids:
            mem = self.embedding_store.get(mid)
            if mem is None:
                raise KeyError(f"Memory {mid} 不存在")
            memories.append(mem)

        # 合并内容
        if new_content is None:
            new_content = "\n".join(m.content for m in memories)

        # 合并元数据
        merged_metadata: dict[str, Any] = {}
        for m in memories:
            merged_metadata.update(m.metadata)

        # 合并标签
        merged_tags: list[str] = []
        seen: set[str] = set()
        for m in memories:
            for tag in m.tags:
                if tag.lower() not in seen:
                    merged_tags.append(tag)
                    seen.add(tag.lower())

        # 更新第一条记忆
        primary = memories[0]
        primary.content = new_content
        primary.metadata = merged_metadata
        primary.tags = merged_tags

        # 重新计算 embedding
        if self._embedding_provider is not None:
            primary.embedding = self._embedding_provider.embed(new_content)

        # 删除其他记忆
        for m in memories[1:]:
            try:
                self.embedding_store.remove(m.id)
            except KeyError:
                pass

        self._auto_save_if_enabled()
        return primary

    def batch_remember(
        self,
        contents: list[str],
        embeddings: Optional[list[list[float]]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        tagss: Optional[list[list[str]]] = None,
    ) -> list[Memory]:
        """批量添加记忆。

        Args:
            contents: 文本内容列表
            embeddings: 向量列表（可选，有 provider 时自动计算）
            metadatas: 元数据列表（可选）
            tagss: 标签列表的列表（可选）

        Returns:
            创建的 Memory 对象列表

        Raises:
            ValueError: 列表长度不一致或缺少 embedding 和 provider
        """
        n = len(contents)
        if embeddings is not None and len(embeddings) != n:
            raise ValueError(f"contents 长度 {n} 与 embeddings 长度 {len(embeddings)} 不一致")
        if metadatas is not None and len(metadatas) != n:
            raise ValueError(f"contents 长度 {n} 与 metadatas 长度 {len(metadatas)} 不一致")
        if tagss is not None and len(tagss) != n:
            raise ValueError(f"contents 长度 {n} 与 tagss 长度 {len(tagss)} 不一致")

        memories: list[Memory] = []
        for i, content in enumerate(contents):
            emb = embeddings[i] if embeddings else None
            meta = metadatas[i] if metadatas else None
            tags = tagss[i] if tagss else None
            memories.append(self.remember(content, embedding=emb, metadata=meta, tags=tags))
        return memories

    def forget(self, memory_id: str) -> None:
        """删除一条记忆。

        Args:
            memory_id: 记忆 ID

        Raises:
            KeyError: 记忆不存在
        """
        self.embedding_store.remove(memory_id)
        self._auto_save_if_enabled()
        self._metrics_counter_forget.increment()
        self._metrics_gauge_memories.set(self.embedding_store.count())

    def forget_where(self, predicate: Callable[[Memory], bool]) -> list[str]:
        """按条件删除记忆。

        Args:
            predicate: 判断函数，返回 True 的记忆将被删除

        Returns:
            删除的记忆 ID 列表
        """
        to_delete = [m for m in self.embedding_store.list_all() if predicate(m)]
        deleted: list[str] = []
        for m in to_delete:
            try:
                self.embedding_store.remove(m.id)
                deleted.append(m.id)
            except KeyError:
                pass
        if deleted:
            self._auto_save_if_enabled()
        return deleted

    def batch_forget(self, memory_ids: list[str]) -> list[str]:
        """批量删除记忆。

        Args:
            memory_ids: 记忆 ID 列表

        Returns:
            成功删除的 ID 列表
        """
        deleted: list[str] = []
        for mid in memory_ids:
            try:
                self.embedding_store.remove(mid)
                deleted.append(mid)
            except KeyError:
                pass
        if deleted:
            self._auto_save_if_enabled()
        return deleted

    def list_all(self) -> list[Memory]:
        """返回所有记忆列表。

        Returns:
            所有 Memory 对象的列表
        """
        return self.embedding_store.list_all()

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """根据 ID 获取记忆。

        Args:
            memory_id: 记忆 ID

        Returns:
            对应的 Memory，不存在返回 None
        """
        mem = self.embedding_store.get(memory_id)
        if mem is not None:
            self.lifecycle.record_access(memory_id)
        return mem

    def get_lifecycle_info(self, memory_id: str) -> Optional[dict[str, Any]]:
        """获取记忆的生命周期信息。

        Args:
            memory_id: 记忆 ID

        Returns:
            生命周期信息字典，记忆不存在返回 None
        """
        mem = self.embedding_store.get(memory_id)
        if mem is None:
            return None
        return self.lifecycle.get_lifecycle_info(mem)

    def cleanup_expired(self) -> list[str]:
        """清理所有过期记忆。

        Returns:
            被清理的记忆 ID 列表
        """
        all_memories = self.embedding_store.list_all()
        expired_ids: list[str] = []
        for mem in all_memories:
            if self.lifecycle.is_expired(mem):
                try:
                    self.embedding_store.remove(mem.id)
                    expired_ids.append(mem.id)
                except KeyError:
                    pass
        if expired_ids:
            self._auto_save_if_enabled()
        return expired_ids

    # --- 标签管理 ---

    def add_tag(self, memory_id: str, tag: str) -> None:
        """为记忆添加标签。

        Args:
            memory_id: 记忆 ID
            tag: 标签名称

        Raises:
            KeyError: 记忆不存在
        """
        mem = self.embedding_store.get(memory_id)
        if mem is None:
            raise KeyError(f"Memory {memory_id} 不存在")
        if not mem.has_tag(tag):
            mem.tags.append(tag)
        self._auto_save_if_enabled()

    def remove_tag(self, memory_id: str, tag: str) -> None:
        """移除记忆的标签。

        Args:
            memory_id: 记忆 ID
            tag: 标签名称

        Raises:
            KeyError: 记忆不存在
        """
        mem = self.embedding_store.get(memory_id)
        if mem is None:
            raise KeyError(f"Memory {memory_id} 不存在")
        # 不区分大小写移除
        mem.tags = [t for t in mem.tags if t.lower() != tag.lower()]
        self._auto_save_if_enabled()

    def get_all_tags(self) -> dict[str, int]:
        """获取所有标签及其使用次数。

        Returns:
            标签名称到使用次数的映射
        """
        return self.embedding_store.get_all_tags()

    # --- 知识图谱管理 ---

    def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> Entity:
        """添加实体到知识图谱。

        Args:
            name: 实体名称
            entity_type: 实体类型
            properties: 附加属性

        Returns:
            创建的 Entity 对象
        """
        entity = Entity(name=name, entity_type=entity_type, properties=properties or {})
        self.knowledge_graph.add_entity(entity)
        self._auto_save_if_enabled()
        return entity

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> Relation:
        """添加关系到知识图谱。

        Args:
            source_id: 源实体 ID
            target_id: 目标实体 ID
            relation_type: 关系类型
            weight: 关系权重

        Returns:
            创建的 Relation 对象
        """
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
        )
        self.knowledge_graph.add_relation(relation)
        self._auto_save_if_enabled()
        return relation

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
    ) -> list[Entity]:
        """获取实体的邻居节点。

        Args:
            entity_id: 实体 ID
            relation_type: 按关系类型过滤

        Returns:
            邻居实体列表
        """
        return self.knowledge_graph.get_neighbors(entity_id, relation_type=relation_type)

    # --- 搜索 ---

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """纯向量相似度搜索，支持标签过滤。

        Args:
            query_embedding: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值
            tags: 标签过滤列表（AND 逻辑）

        Returns:
            按相似度降序排列的 SearchResult 列表
        """
        results = self.embedding_store.search(
            query=query_embedding,
            top_k=top_k,
            threshold=threshold,
            tags=tags,
        )
        # 应用默认过滤器
        if self._default_filter is not None:
            results = filter_search_results(results, self._default_filter, self.lifecycle)
        # 记录搜索访问
        for r in results:
            self.lifecycle.record_access(r.memory.id)
            if self._scorer is not None:
                self._scorer.record_access(r.memory.id)
        # 加权重排序
        if self._scorer is not None and results:
            results = self._scorer.rerank(results)
        return results

    def batch_search(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[list[SearchResult]]:
        """批量向量搜索。

        Args:
            query_embeddings: 查询向量列表
            top_k: 每个查询返回前 k 个结果
            threshold: 相似度阈值
            tags: 标签过滤列表

        Returns:
            每个查询对应的 SearchResult 列表
        """
        return [
            self.search(emb, top_k=top_k, threshold=threshold, tags=tags)
            for emb in query_embeddings
        ]

    def hybrid_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        graph_depth: int = 1,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """混合搜索：结合向量相似度 + 知识图谱上下文。

        1. 先通过向量搜索找到最相关的记忆
        2. 对每条记忆，尝试通过知识图谱找到关联的实体上下文
        3. 为搜索结果附加图谱上下文

        Args:
            query_embedding: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值
            graph_depth: 图谱遍历深度（0 表示不使用图谱）
            tags: 标签过滤列表

        Returns:
            带有图谱上下文的 SearchResult 列表
        """
        # Step 1: 向量搜索
        results = self.embedding_store.search(
            query=query_embedding,
            top_k=top_k,
            threshold=threshold,
            tags=tags,
        )

        if graph_depth <= 0 or len(results) == 0:
            return results

        # Step 2: 为每条结果查找图谱上下文
        for result in results:
            context_memories = self._find_graph_context(result.memory, graph_depth)
            result.context = context_memories

        # 应用默认过滤器
        if self._default_filter is not None:
            results = filter_search_results(results, self._default_filter, self.lifecycle)
        return results

    def _find_graph_context(
        self,
        memory: Memory,
        depth: int,
    ) -> list[Memory]:
        """为一条记忆查找知识图谱上下文。

        通过记忆内容中的关键词匹配实体，再从图谱中找到相关实体关联的记忆。

        Args:
            memory: 目标记忆
            depth: 图谱遍历深度

        Returns:
            关联的上下文记忆列表
        """
        content_lower = memory.content.lower()
        context_memories: list[Memory] = []
        seen_memory_ids: set[str] = {memory.id}

        # 找到记忆内容中提及的实体
        for entity in self.knowledge_graph.find_entities():
            if entity.name.lower() in content_lower:
                # 通过 BFS 找到相关实体
                neighbors = self.knowledge_graph.bfs(
                    entity.id, max_depth=depth
                )
                # 用邻居实体的名称在记忆库中搜索相关内容
                for neighbor in neighbors:
                    for stored_mem in self.embedding_store.list_all():
                        if (
                            stored_mem.id not in seen_memory_ids
                            and neighbor.name.lower() in stored_mem.content.lower()
                        ):
                            context_memories.append(stored_mem)
                            seen_memory_ids.add(stored_mem.id)

        return context_memories

    # --- 文本搜索 ---

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """文本相似度搜索（需要 embedding_provider）。

        自动将查询文本转为向量后执行搜索。

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            threshold: 相似度阈值
            tags: 标签过滤列表

        Returns:
            按相似度降序排列的 SearchResult 列表

        Raises:
            ValueError: 未配置 embedding_provider
        """
        if self._embedding_provider is None:
            raise ValueError(
                "使用 search_text 需要配置 embedding_provider，"
                "或使用 search(query_embedding=[...]) 直接搜索向量"
            )
        _t = time.time()
        # 检查缓存
        if self._cache is not None:
            cached = self._cache.get(query, top_k=top_k, threshold=threshold, tags=tags)
            if cached is not None:
                self._metrics_counter_search.increment()
                return cached
        query_embedding = self._embedding_provider.embed(query)
        results = self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
            tags=tags,
        )
        # 写入缓存
        if self._cache is not None:
            self._cache.put(query, results, top_k=top_k, threshold=threshold, tags=tags)
        elapsed_ms = (time.time() - _t) * 1000
        self._metrics_timer_search.record(elapsed_ms)
        self._metrics_counter_search.increment()
        return results

    def hybrid_search_text(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        graph_depth: int = 1,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """文本混合搜索（需要 embedding_provider）。

        自动将查询文本转为向量后执行混合搜索。

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            threshold: 相似度阈值
            graph_depth: 图谱遍历深度
            tags: 标签过滤列表

        Returns:
            带有图谱上下文的 SearchResult 列表

        Raises:
            ValueError: 未配置 embedding_provider
        """
        if self._embedding_provider is None:
            raise ValueError(
                "使用 hybrid_search_text 需要配置 embedding_provider，"
                "或使用 hybrid_search(query_embedding=[...]) 直接搜索向量"
            )
        query_embedding = self._embedding_provider.embed(query)
        return self.hybrid_search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
            graph_depth=graph_depth,
            tags=tags,
        )

    # --- 统计 ---

    def stats(self) -> dict[str, Any]:
        """返回系统统计信息。

        Returns:
            包含 memory_count, entity_count, relation_count 等的字典
        """
        return {
            "memory_count": self.embedding_store.count(),
            "entity_count": self.knowledge_graph.entity_count(),
            "relation_count": self.knowledge_graph.relation_count(),
            "use_lsh": self.embedding_store.use_lsh,
            "dimension": self._dimension,
        }

    # --- 搜索过滤器 ---

    def set_default_filter(self, search_filter: Optional[SearchFilter]) -> None:
        """设置全局默认搜索过滤器。

        设置后所有 search/hybrid_search 方法都会自动应用过滤。

        Args:
            search_filter: SearchFilter 实例，None 表示清除
        """
        self._default_filter = search_filter

    def set_scorer(self, scorer: Optional[WeightedScorer]) -> None:
        """设置加权评分器。

        Args:
            scorer: WeightedScorer 实例，None 表示禁用加权评分
        """
        self._scorer = scorer

    def get_scorer(self) -> Optional[WeightedScorer]:
        """获取当前加权评分器。

        Returns:
            WeightedScorer 实例，未设置返回 None
        """
        return self._scorer

    # --- 搜索缓存管理 ---

    def get_cache_stats(self) -> Optional[dict[str, Any]]:
        """获取搜索缓存统计信息。

        Returns:
            缓存统计字典，未启用缓存返回 None
        """
        if self._cache is None:
            return None
        return self._cache.stats

    def clear_cache(self) -> int:
        """清空搜索缓存。

        Returns:
            被清除的条目数（未启用缓存返回 0）
        """
        if self._cache is None:
            return 0
        return self._cache.clear()

    # --- 图谱推理 ---

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
    ) -> Optional[list[Entity]]:
        """查找两个实体之间的最短路径。

        Args:
            source_id: 起始实体 ID
            target_id: 目标实体 ID
            max_depth: 最大搜索深度

        Returns:
            路径上的实体列表（含起始和目标），不可达返回 None
        """
        return self.knowledge_graph.shortest_path(source_id, target_id, max_depth=max_depth)

    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        max_paths: int = 10,
    ) -> list[list[Entity]]:
        """查找两个实体之间的所有路径。

        Args:
            source_id: 起始实体 ID
            target_id: 目标实体 ID
            max_depth: 最大路径深度
            max_paths: 最大路径数

        Returns:
            所有路径的列表
        """
        return self.knowledge_graph.find_all_paths(
            source_id, target_id, max_depth=max_depth, max_paths=max_paths
        )

    def common_neighbors(
        self,
        entity_id_1: str,
        entity_id_2: str,
    ) -> list[Entity]:
        """查找两个实体的共同邻居。

        Args:
            entity_id_1: 第一个实体 ID
            entity_id_2: 第二个实体 ID

        Returns:
            共同邻居实体列表
        """
        return self.knowledge_graph.common_neighbors(entity_id_1, entity_id_2)

    def connected_components(self) -> list[list[Entity]]:
        """查找图谱中的所有连通分量。

        Returns:
            每个连通分量包含的实体列表
        """
        return self.knowledge_graph.connected_components()

    def subgraph(
        self,
        entity_ids: set[str],
    ) -> dict[str, list]:
        """提取子图。

        Args:
            entity_ids: 子图包含的实体 ID 集合

        Returns:
            包含 entities 和 relations 的字典
        """
        return self.knowledge_graph.subgraph(entity_ids)

    def export_dot(self, title: str = "Knowledge Graph") -> str:
        """导出知识图为 Graphviz DOT 格式。

        Args:
            title: 图表标题

        Returns:
            DOT 格式字符串
        """
        from agentmemory.graph_viz import export_dot
        return export_dot(self.knowledge_graph, title=title)

    def export_html(self, title: str = "Knowledge Graph") -> str:
        """导出知识图为交互式 HTML。

        Args:
            title: 页面标题

        Returns:
            HTML 字符串
        """
        from agentmemory.graph_viz import export_html
        return export_html(self.knowledge_graph, title=title)

    # --- 导出/导入 ---

    def export_json(self, pretty: bool = True) -> str:
        """将所有数据导出为 JSON 字符串。

        Args:
            pretty: 是否格式化输出

        Returns:
            JSON 字符串
        """
        data = {
            "version": "2.0",
            "stats": self.stats(),
            "memories": [m.to_dict() for m in self.embedding_store.list_all()],
            "entities": [e.to_dict() for e in self.knowledge_graph.find_entities()],
            "relations": [r.to_dict() for r in self.knowledge_graph.find_relations()],
        }
        indent = 2 if pretty else None
        return json.dumps(data, ensure_ascii=False, indent=indent)

    def import_json(self, json_str: str, overwrite: bool = False) -> dict[str, int]:
        """从 JSON 字符串导入数据。

        Args:
            json_str: JSON 字符串
            overwrite: 是否清空现有数据后导入

        Returns:
            导入统计 {"memories": N, "entities": N, "relations": N}
        """
        data = json.loads(json_str)

        if overwrite:
            self.embedding_store = EmbeddingStore(dimension=self._dimension)
            self.knowledge_graph = KnowledgeGraph()

        counts = {"memories": 0, "entities": 0, "relations": 0}

        # 先导入实体
        for entity_data in data.get("entities", []):
            try:
                entity = Entity.from_dict(entity_data)
                self.knowledge_graph.add_entity(entity)
                counts["entities"] += 1
            except ValueError:
                pass  # 重复实体跳过

        # 导入关系
        for rel_data in data.get("relations", []):
            try:
                relation = Relation.from_dict(rel_data)
                self.knowledge_graph.add_relation(relation)
                counts["relations"] += 1
            except ValueError:
                pass  # 无效关系跳过

        # 导入记忆
        for mem_data in data.get("memories", []):
            try:
                mem = Memory.from_dict(mem_data)
                self.embedding_store.add(mem)
                counts["memories"] += 1
            except ValueError:
                pass  # 无效记忆跳过

        self._auto_save_if_enabled()
        return counts

    def export_csv(self) -> str:
        """将记忆数据导出为 CSV 字符串。

        Returns:
            CSV 格式字符串（包含 id, content, created_at, metadata, tags 列）
        """
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "content", "created_at", "metadata", "tags"])
        for mem in self.embedding_store.list_all():
            writer.writerow([
                mem.id,
                mem.content,
                mem.created_at,
                json.dumps(mem.metadata, ensure_ascii=False),
                json.dumps(mem.tags, ensure_ascii=False),
            ])
        return output.getvalue()

    def import_csv(self, csv_str: str) -> int:
        """从 CSV 字符串导入记忆数据。

        CSV 中不包含向量，如果有 embedding_provider 会自动计算。

        Args:
            csv_str: CSV 格式字符串

        Returns:
            成功导入的记忆数量
        """
        reader = csv.DictReader(StringIO(csv_str))
        count = 0
        for row in reader:
            try:
                tags = json.loads(row.get("tags", "[]"))
                embedding = None
                if self._embedding_provider is not None:
                    embedding = self._embedding_provider.embed(row["content"])
                mem = Memory(
                    id=row["id"],
                    content=row["content"],
                    created_at=float(row["created_at"]),
                    metadata=json.loads(row.get("metadata", "{}")),
                    tags=tags,
                    embedding=embedding,
                )
                self.embedding_store.add(mem)
                count += 1
            except (ValueError, KeyError):
                pass
        self._auto_save_if_enabled()
        return count


    # --- 向量量化 ---

    def compress_vectors(
        self,
        method: str = "sq8",
        num_subspaces: int = 8,
    ) -> dict[str, Any]:
        """使用量化压缩所有已存储的向量。

        压缩后可通过 compressed_search() 进行近似最近邻搜索，
        显著减少内存占用。

        Args:
            method: 量化方法，'sq8'（标量量化 4x 压缩）或 'pq'（乘积量化）
            num_subspaces: PQ 子空间数量（仅 method='pq' 时有效）

        Returns:
            压缩统计信息字典
        """
        from agentmemory.vector_quantizer import (
            ScalarQuantizer,
            ProductQuantizer,
            CompressedVectorStore,
        )

        all_mems = self.embedding_store.list_all()
        vectors = [(m.id, m.embedding) for m in all_mems if m.embedding is not None]

        if not vectors:
            return {"error": "没有可压缩的向量"}

        dim = self._dimension
        ids_list = [vid for vid, _ in vectors]
        vecs_list = [v for _, v in vectors]

        if method == "sq8":
            quantizer = ScalarQuantizer(dim)
            quantizer.fit(vecs_list)
        elif method == "pq":
            quantizer = ProductQuantizer(dim, num_subspaces=num_subspaces)
            quantizer.fit(vecs_list)
        else:
            raise ValueError(f"不支持的量化方法: {method}，可选: 'sq8', 'pq'")

        compressed_store = CompressedVectorStore(quantizer)
        for vid, vec in zip(ids_list, vecs_list):
            compressed_store.add(vid, vec)

        self._compressed_store = compressed_store
        self._compressed_method = method

        stats = compressed_store.stats()
        return {
            "method": stats["method"],
            "num_vectors": stats["num_vectors"],
            "compression_ratio": round(stats["compression_ratio"], 2),
            "compressed_bytes_per_vector": stats["compressed_bytes_per_vector"],
            "total_compressed_bytes": stats["total_compressed_bytes"],
        }

    def compressed_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """在压缩向量上执行近似最近邻搜索。

        需先调用 compress_vectors() 创建压缩索引。

        Args:
            query_embedding: 查询向量
            top_k: 返回前 k 个结果

        Returns:
            SearchResult 列表（近似分数）

        Raises:
            RuntimeError: 未调用 compress_vectors()
        """
        if not hasattr(self, "_compressed_store") or self._compressed_store is None:
            raise RuntimeError("请先调用 compress_vectors() 创建压缩索引")

        # 将查询向量量化再反量化以对齐精度
        quantizer = self._compressed_store._quantizer
        compressed_query = quantizer.quantize(query_embedding)
        approx_query = quantizer.dequantize(compressed_query)

        from agentmemory.embedding_store import cosine_similarity

        scored: list[tuple[float, str]] = []
        for vid in self._compressed_store.list_ids():
            approx_vec = self._compressed_store.get(vid)
            if approx_vec is not None:
                sim = cosine_similarity(approx_query, approx_vec)
                scored.append((sim, vid))

        scored.sort(reverse=True)
        results: list[SearchResult] = []
        for score, vid in scored[:top_k]:
            mem = self.embedding_store.get(vid)
            if mem is not None:
                results.append(SearchResult(memory=mem, score=score))
        return results

    # --- 可观测性 ---

    def metrics_snapshot(self) -> dict[str, Any]:
        """获取运行时指标快照。

        Returns:
            包含 counters、timers、gauges 的完整指标字典
        """
        return self._metrics.snapshot()

    def metrics_json(self, indent: int = 2) -> str:
        """导出指标为 JSON 字符串。

        Args:
            indent: 缩进空格数

        Returns:
            JSON 格式的指标字符串
        """
        return self._metrics.export_json(indent=indent)

    def metrics_prometheus(self) -> str:
        """导出指标为 Prometheus 文本格式。

        Returns:
            Prometheus 文本格式的指标
        """
        return self._metrics.export_prometheus()

    def health_check(self) -> dict[str, Any]:
        """执行综合健康检查。

        检查记忆存储和 LSH 索引的健康状态。

        Returns:
            包含 overall_status 和 checks 的健康报告字典
        """
        checker = HealthChecker(name="agentmemory")
        checker.add_check(check_memory_health(self))
        checker.add_check(check_lsh_health(self))
        report = checker.report()
        return {
            "overall_status": report.overall_status.value,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in report.checks
            ],
            "timestamp": report.timestamp,
        }

    # --- RAG 管道 ---

    def rag(
        self,
        query: str,
        top_k: int = 5,
        max_context_tokens: int = 2000,
        tags: Optional[list[str]] = None,
        use_hybrid: bool = False,
    ) -> dict[str, Any]:
        """执行 RAG（检索增强生成）管道。

        检索相关记忆 → 重排序 → 上下文组装 → Prompt 生成。

        Args:
            query: 用户查询
            top_k: 检索结果数量
            max_context_tokens: 上下文最大 token 数
            tags: 标签过滤
            use_hybrid: 是否使用混合检索（向量+图谱）

        Returns:
            包含 prompt、context、sources、timing 的结果字典
        """
        from agentmemory.rag_pipeline import RAGPipeline, Reranker

        pipeline = RAGPipeline(
            memory=self,
            max_context_tokens=max_context_tokens,
            top_k=top_k,
            reranker=Reranker(freshness_weight=0.2),
        )
        result = pipeline.run(query=query, top_k=top_k, tags=tags, use_hybrid=use_hybrid)
        return {
            "prompt": result.prompt,
            "context_text": result.context.text,
            "sources": [
                {"id": s.id, "content": s.content[:100]}
                for s in result.context.sources
            ],
            "total_tokens": result.context.total_tokens,
            "truncated": result.context.truncated,
            "reranked": result.reranked,
            "pipeline_time_ms": round(result.pipeline_time_ms, 2),
        }


class MemorySession:
    """记忆会话上下文管理器。

    进入时可选加载数据，退出时自动保存。
    代理 HybridMemory 的主要方法以方便使用。

    Args:
        memory: HybridMemory 实例
        load_on_enter: 进入时是否自动加载（仅当有 storage_path 时生效）
    """

    def __init__(
        self,
        memory: HybridMemory,
        load_on_enter: bool = True,
    ) -> None:
        self._memory = memory
        self._load_on_enter = load_on_enter
        self._operations_count: int = 0

    def __enter__(self) -> "MemorySession":
        if self._load_on_enter and self._memory._backend:
            try:
                self._memory.load()
            except Exception:
                pass  # 首次运行可能没有文件
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._memory._backend:
            self._memory.save()

    @property
    def operations_count(self) -> int:
        """本次会话的操作次数"""
        return self._operations_count

    def remember(self, *args, **kwargs):
        """代理 HybridMemory.remember"""
        self._operations_count += 1
        return self._memory.remember(*args, **kwargs)

    def search(self, *args, **kwargs):
        """代理 HybridMemory.search"""
        return self._memory.search(*args, **kwargs)

    def search_text(self, *args, **kwargs):
        """代理 HybridMemory.search_text"""
        return self._memory.search_text(*args, **kwargs)

    def hybrid_search(self, *args, **kwargs):
        """代理 HybridMemory.hybrid_search"""
        return self._memory.hybrid_search(*args, **kwargs)

    def hybrid_search_text(self, *args, **kwargs):
        """代理 HybridMemory.hybrid_search_text"""
        return self._memory.hybrid_search_text(*args, **kwargs)

    def forget(self, *args, **kwargs):
        """代理 HybridMemory.forget"""
        self._operations_count += 1
        return self._memory.forget(*args, **kwargs)

    def get_memory(self, *args, **kwargs):
        """代理 HybridMemory.get_memory"""
        return self._memory.get_memory(*args, **kwargs)

    def update_memory(self, *args, **kwargs):
        """代理 HybridMemory.update_memory"""
        self._operations_count += 1
        return self._memory.update_memory(*args, **kwargs)

    def add_entity(self, *args, **kwargs):
        """代理 HybridMemory.add_entity"""
        self._operations_count += 1
        return self._memory.add_entity(*args, **kwargs)

    def add_relation(self, *args, **kwargs):
        """代理 HybridMemory.add_relation"""
        self._operations_count += 1
        return self._memory.add_relation(*args, **kwargs)

    def stats(self):
        """代理 HybridMemory.stats + 会话统计"""
        base_stats = self._memory.stats()
        base_stats["session_operations"] = self._operations_count
        return base_stats

    def metrics_snapshot(self):
        """代理 HybridMemory.metrics_snapshot"""
        return self._memory.metrics_snapshot()

    def metrics_json(self, *args, **kwargs):
        """代理 HybridMemory.metrics_json"""
        return self._memory.metrics_json(*args, **kwargs)

    def metrics_prometheus(self):
        """代理 HybridMemory.metrics_prometheus"""
        return self._memory.metrics_prometheus()

    def health_check(self):
        """代理 HybridMemory.health_check"""
        return self._memory.health_check()

    def rag(self, *args, **kwargs):
        """代理 HybridMemory.rag"""
        return self._memory.rag(*args, **kwargs)

    def compress_vectors(self, *args, **kwargs):
        """代理 HybridMemory.compress_vectors"""
        return self._memory.compress_vectors(*args, **kwargs)

    def compressed_search(self, *args, **kwargs):
        """代理 HybridMemory.compressed_search"""
        return self._memory.compressed_search(*args, **kwargs)
