"""命名空间/多租户隔离：为不同 Agent 提供独立的记忆空间。

每个命名空间拥有独立的 EmbeddingStore 和 KnowledgeGraph，
共享同一个底层存储后端和配置。

支持：
- 创建/删除命名空间
- 切换当前命名空间
- 跨命名空间搜索
- 命名空间级别的统计和清理
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agentmemory.embedding_store import EmbeddingStore
from agentmemory.knowledge_graph import KnowledgeGraph
from agentmemory.models import Memory, Entity, Relation, SearchResult


class Namespace:
    """单个命名空间，包含独立的存储。

    Attributes:
        name: 命名空间名称
        created_at: 创建时间
        description: 描述
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        description: str = "",
        use_lsh: bool = False,
        lsh_tables: int = 8,
        lsh_hyperplanes: int = 16,
    ) -> None:
        if not name:
            raise ValueError("命名空间名称不能为空")

        self.name = name
        self.dimension = dimension
        self.description = description
        self.created_at = time.time()
        self.last_accessed = time.time()

        self.embedding_store = EmbeddingStore(
            dimension=dimension,
            use_lsh=use_lsh,
            lsh_tables=lsh_tables,
            lsh_hyperplanes=lsh_hyperplanes,
        )
        self.knowledge_graph = KnowledgeGraph()
        self._access_count = 0

    def record_access(self) -> None:
        """记录一次访问"""
        self._access_count += 1
        self.last_accessed = time.time()

    @property
    def access_count(self) -> int:
        """访问次数"""
        return self._access_count

    def stats(self) -> dict[str, Any]:
        """命名空间统计"""
        return {
            "name": self.name,
            "description": self.description,
            "memory_count": self.embedding_store.count(),
            "entity_count": self.knowledge_graph.entity_count(),
            "relation_count": self.knowledge_graph.relation_count(),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self._access_count,
        }


class NamespaceManager:
    """命名空间管理器。

    管理多个命名空间，提供创建、删除、切换和跨命名空间操作。

    Args:
        dimension: 向量维度（所有命名空间共享）
        default_namespace: 默认命名空间名称
        use_lsh: 是否为新命名空间启用 LSH
        lsh_tables: LSH 哈希表数量
        lsh_hyperplanes: LSH 超平面数量
    """

    def __init__(
        self,
        dimension: int,
        default_namespace: str = "default",
        use_lsh: bool = False,
        lsh_tables: int = 8,
        lsh_hyperplanes: int = 16,
    ) -> None:
        self._dimension = dimension
        self._use_lsh = use_lsh
        self._lsh_tables = lsh_tables
        self._lsh_hyperplanes = lsh_hyperplanes
        self._namespaces: dict[str, Namespace] = {}
        self._current_name = default_namespace

        # 自动创建默认命名空间
        self.create(default_namespace, description="默认命名空间")

    def create(
        self,
        name: str,
        description: str = "",
    ) -> Namespace:
        """创建命名空间。

        Args:
            name: 命名空间名称
            description: 描述

        Returns:
            创建的 Namespace 实例

        Raises:
            ValueError: 命名空间已存在
        """
        if name in self._namespaces:
            raise ValueError(f"命名空间 '{name}' 已存在")

        ns = Namespace(
            name=name,
            dimension=self._dimension,
            description=description,
            use_lsh=self._use_lsh,
            lsh_tables=self._lsh_tables,
            lsh_hyperplanes=self._lsh_hyperplanes,
        )
        self._namespaces[name] = ns
        return ns

    def delete(self, name: str) -> bool:
        """删除命名空间。

        Args:
            name: 命名空间名称

        Returns:
            是否成功删除

        Raises:
            ValueError: 不能删除默认命名空间或当前命名空间
            KeyError: 命名空间不存在
        """
        if name == "default":
            raise ValueError("不能删除默认命名空间")
        if name == self._current_name:
            raise ValueError("不能删除当前正在使用的命名空间")
        if name not in self._namespaces:
            raise KeyError(f"命名空间 '{name}' 不存在")

        del self._namespaces[name]
        return True

    def switch(self, name: str) -> Namespace:
        """切换当前命名空间。

        Args:
            name: 命名空间名称

        Returns:
            切换到的 Namespace

        Raises:
            KeyError: 命名空间不存在
        """
        if name not in self._namespaces:
            raise KeyError(f"命名空间 '{name}' 不存在")

        self._current_name = name
        ns = self._namespaces[name]
        ns.record_access()
        return ns

    @property
    def current(self) -> Namespace:
        """当前命名空间"""
        return self._namespaces[self._current_name]

    @property
    def current_name(self) -> str:
        """当前命名空间名称"""
        return self._current_name

    def get(self, name: str) -> Optional[Namespace]:
        """获取指定命名空间。

        Args:
            name: 命名空间名称

        Returns:
            Namespace 实例，不存在返回 None
        """
        return self._namespaces.get(name)

    def list_namespaces(self) -> list[dict[str, Any]]:
        """列出所有命名空间。

        Returns:
            命名空间统计列表
        """
        return [ns.stats() for ns in self._namespaces.values()]

    def exists(self, name: str) -> bool:
        """检查命名空间是否存在。

        Args:
            name: 命名空间名称

        Returns:
            是否存在
        """
        return name in self._namespaces

    @property
    def count(self) -> int:
        """命名空间总数"""
        return len(self._namespaces)

    def global_stats(self) -> dict[str, Any]:
        """跨命名空间全局统计。

        Returns:
            全局统计字典
        """
        total_memories = 0
        total_entities = 0
        total_relations = 0

        for ns in self._namespaces.values():
            total_memories += ns.embedding_store.count()
            total_entities += ns.knowledge_graph.entity_count()
            total_relations += ns.knowledge_graph.relation_count()

        return {
            "namespace_count": self.count,
            "total_memories": total_memories,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "namespaces": self.list_namespaces(),
        }

    def cross_namespace_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        namespaces: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """跨命名空间搜索。

        Args:
            query_embedding: 查询向量
            top_k: 每个命名空间返回的结果数
            threshold: 相似度阈值
            namespaces: 要搜索的命名空间列表，None 表示全部

        Returns:
            合并后的搜索结果，按相似度降序
        """
        target_ns = namespaces or list(self._namespaces.keys())
        all_results: list[SearchResult] = []

        for ns_name in target_ns:
            ns = self._namespaces.get(ns_name)
            if ns is None:
                continue
            ns.record_access()

            results = ns.embedding_store.search(
                query=query_embedding,
                top_k=top_k,
                threshold=threshold,
            )
            # 添加命名空间来源信息
            for r in results:
                r.memory.metadata["_namespace"] = ns_name
            all_results.extend(results)

        # 全局排序去重
        seen: set[str] = set()
        unique_results: list[SearchResult] = []
        all_results.sort(key=lambda r: r.score, reverse=True)
        for r in all_results:
            if r.memory.id not in seen:
                seen.add(r.memory.id)
                unique_results.append(r)

        return unique_results[:top_k]

    def merge_into(
        self,
        source_name: str,
        target_name: str,
    ) -> dict[str, int]:
        """将一个命名空间的内容合并到另一个。

        Args:
            source_name: 源命名空间
            target_name: 目标命名空间

        Returns:
            合并统计

        Raises:
            KeyError: 命名空间不存在
        """
        source = self._namespaces.get(source_name)
        target = self._namespaces.get(target_name)

        if source is None:
            raise KeyError(f"命名空间 '{source_name}' 不存在")
        if target is None:
            raise KeyError(f"命名空间 '{target_name}' 不存在")

        counts = {"memories": 0, "entities": 0, "relations": 0}

        # 合并记忆
        for mem in source.embedding_store.list_all():
            try:
                target.embedding_store.add(mem)
                counts["memories"] += 1
            except ValueError:
                pass  # 已存在

        # 合并实体
        for entity in source.knowledge_graph.find_entities():
            try:
                target.knowledge_graph.add_entity(entity)
                counts["entities"] += 1
            except ValueError:
                pass

        # 合并关系
        for relation in source.knowledge_graph.find_relations():
            try:
                target.knowledge_graph.add_relation(relation)
                counts["relations"] += 1
            except ValueError:
                pass

        return counts

    def clear(self, name: str) -> dict[str, int]:
        """清空指定命名空间的所有数据。

        Args:
            name: 命名空间名称

        Returns:
            清除的数据统计

        Raises:
            KeyError: 命名空间不存在
        """
        ns = self._namespaces.get(name)
        if ns is None:
            raise KeyError(f"命名空间 '{name}' 不存在")

        mem_count = ns.embedding_store.count()
        ent_count = ns.knowledge_graph.entity_count()
        rel_count = ns.knowledge_graph.relation_count()

        # 重建存储
        ns.embedding_store = EmbeddingStore(
            dimension=self._dimension,
            use_lsh=self._use_lsh,
            lsh_tables=self._lsh_tables,
            lsh_hyperplanes=self._lsh_hyperplanes,
        )
        ns.knowledge_graph = KnowledgeGraph()

        return {
            "cleared_memories": mem_count,
            "cleared_entities": ent_count,
            "cleared_relations": rel_count,
        }
