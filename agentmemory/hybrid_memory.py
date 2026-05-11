"""混合记忆框架：统一向量搜索与知识图谱的高级 API"""

from __future__ import annotations

from typing import Any, Optional

from agentmemory.models import Entity, Memory, Relation, SearchResult
from agentmemory.embedding_store import EmbeddingStore
from agentmemory.knowledge_graph import KnowledgeGraph


class HybridMemory:
    """混合记忆系统，结合向量搜索与知识图谱。

    提供统一的记忆存储、检索和知识关联 API。

    Args:
        dimension: 向量维度
        storage_path: 持久化存储路径（目录或文件），None 表示不持久化
        storage_backend: 存储后端类型，'json' 或 'sqlite'
        auto_save: 每次 remember/forget 后自动保存
        auto_load: 初始化时自动加载已有数据
    """

    def __init__(
        self,
        dimension: int,
        storage_path: Optional[str] = None,
        storage_backend: str = "json",
        auto_save: bool = False,
        auto_load: bool = False,
    ) -> None:
        self._dimension = dimension
        self.embedding_store = EmbeddingStore(dimension=dimension)
        self.knowledge_graph = KnowledgeGraph()
        self._storage_path = storage_path
        self._storage_backend = storage_backend
        self._auto_save = auto_save
        self._backend = self._create_backend() if storage_path else None

        if auto_load and self._backend:
            self.load()

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
    ) -> Memory:
        """添加一条记忆到存储。

        Args:
            content: 文本内容
            embedding: 向量表示
            metadata: 附加元数据

        Returns:
            创建的 Memory 对象

        Raises:
            ValueError: embedding 为空或维度不匹配
        """
        mem = Memory(content=content, embedding=embedding, metadata=metadata or {})
        self.embedding_store.add(mem)
        self._auto_save_if_enabled()
        return mem

    def forget(self, memory_id: str) -> None:
        """删除一条记忆。

        Args:
            memory_id: 记忆 ID

        Raises:
            KeyError: 记忆不存在
        """
        self.embedding_store.remove(memory_id)
        self._auto_save_if_enabled()

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
        return self.embedding_store.get(memory_id)

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
    ) -> list[SearchResult]:
        """纯向量相似度搜索。

        Args:
            query_embedding: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值

        Returns:
            按相似度降序排列的 SearchResult 列表
        """
        return self.embedding_store.search(
            query=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

    def hybrid_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        graph_depth: int = 1,
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

        Returns:
            带有图谱上下文的 SearchResult 列表
        """
        # Step 1: 向量搜索
        results = self.embedding_store.search(
            query=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

        if graph_depth <= 0 or len(results) == 0:
            return results

        # Step 2: 为每条结果查找图谱上下文
        for result in results:
            context_memories = self._find_graph_context(result.memory, graph_depth)
            result.context = context_memories

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

    # --- 统计 ---

    def stats(self) -> dict[str, int]:
        """返回系统统计信息。

        Returns:
            包含 memory_count, entity_count, relation_count 的字典
        """
        return {
            "memory_count": self.embedding_store.count(),
            "entity_count": self.knowledge_graph.entity_count(),
            "relation_count": self.knowledge_graph.relation_count(),
        }
