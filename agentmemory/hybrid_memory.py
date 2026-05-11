"""混合记忆框架：统一向量搜索与知识图谱的高级 API"""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any, Optional

from agentmemory.models import Entity, Memory, Relation, SearchResult
from agentmemory.embedding_store import EmbeddingStore
from agentmemory.knowledge_graph import KnowledgeGraph
from agentmemory.embedding_provider import EmbeddingProvider


class HybridMemory:
    """混合记忆系统，结合向量搜索与知识图谱。

    提供统一的记忆存储、检索和知识关联 API。
    支持批量操作、标签过滤、数据导出/导入。

    Args:
        dimension: 向量维度。如果提供 embedding_provider，可以从 provider 自动推断。
        embedding_provider: Embedding 提供者，设置后 remember/search_text 可自动计算向量。
        storage_path: 持久化存储路径（目录或文件），None 表示不持久化
        storage_backend: 存储后端类型，'json' 或 'sqlite'
        auto_save: 每次 remember/forget 后自动保存
        auto_load: 初始化时自动加载已有数据
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage_path: Optional[str] = None,
        storage_backend: str = "json",
        auto_save: bool = False,
        auto_load: bool = False,
    ) -> None:
        self._embedding_provider = embedding_provider

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

        self.embedding_store = EmbeddingStore(dimension=self._dimension)
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
        tags: Optional[list[str]] = None,
    ) -> Memory:
        """添加一条记忆到存储。

        如果配置了 embedding_provider 且未手动传入 embedding，
        将自动使用 provider 计算向量。

        Args:
            content: 文本内容
            embedding: 向量表示（可选，有 provider 时自动计算）
            metadata: 附加元数据
            tags: 标签列表

        Returns:
            创建的 Memory 对象

        Raises:
            ValueError: 没有 embedding 也没有 provider
        """
        if embedding is None and self._embedding_provider is not None:
            embedding = self._embedding_provider.embed(content)
        mem = Memory(content=content, embedding=embedding, metadata=metadata or {}, tags=tags or [])
        self.embedding_store.add(mem)
        self._auto_save_if_enabled()
        return mem

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
        return self.embedding_store.get(memory_id)

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
        return self.embedding_store.search(
            query=query_embedding,
            top_k=top_k,
            threshold=threshold,
            tags=tags,
        )

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
        query_embedding = self._embedding_provider.embed(query)
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
            tags=tags,
        )

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

    # --- 导出/导入 ---

    def export_json(self, pretty: bool = True) -> str:
        """将所有数据导出为 JSON 字符串。

        Args:
            pretty: 是否格式化输出

        Returns:
            JSON 字符串
        """
        data = {
            "version": "1.0",
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
