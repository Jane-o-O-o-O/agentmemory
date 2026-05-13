"""ChromaDB 持久化后端 — 可选的向量数据库后端。

通过插件架构注册为可选后端，需要安装 chromadb 包：
    pip install chromadb

提供向量存储和持久化能力，适合需要原生向量检索的场景。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from agentmemory.models import Memory, Entity, Relation
from agentmemory.embedding_store import EmbeddingStore
from agentmemory.knowledge_graph import KnowledgeGraph


def _check_chromadb() -> None:
    """检查 chromadb 是否可用"""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        raise ImportError(
            "ChromaDB 后端需要安装 chromadb: pip install chromadb"
        ) from None


class ChromaDBBackend:
    """ChromaDB 持久化后端。

    使用 ChromaDB 作为向量存储后端，支持：
    - 向量相似度搜索（原生 ANN）
    - 元数据过滤
    - 持久化存储

    注意：知识图谱数据仍使用 JSON/SQLite 存储，ChromaDB 仅处理向量部分。

    Args:
        path: ChromaDB 持久化目录路径
        collection_name: ChromaDB 集合名称（默认 "agentmemory"）

    Example:
        >>> backend = ChromaDBBackend("./chroma_data")
        >>> backend.save_embedding_store(store)
        >>> backend.load_embedding_store(store)
    """

    def __init__(
        self,
        path: str,
        collection_name: str = "agentmemory",
    ) -> None:
        _check_chromadb()
        import chromadb

        self._path = path
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # 用 JSON 存储知识图谱（ChromaDB 不支持图关系）
        self._graph_path = os.path.join(path, "knowledge_graph.json")
        os.makedirs(path, exist_ok=True)

    @property
    def collection(self) -> Any:
        """ChromaDB 集合对象"""
        return self._collection

    def save_embedding_store(self, store: EmbeddingStore) -> None:
        """将 EmbeddingStore 保存到 ChromaDB。

        Args:
            store: 待保存的 EmbeddingStore
        """
        # 清空现有数据
        try:
            existing = self._collection.get()
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        except Exception:
            pass

        memories = store.list_all()
        if not memories:
            return

        # 批量添加
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for mem in memories:
            if mem.embedding is None:
                continue
            ids.append(mem.id)
            embeddings.append(mem.embedding)
            documents.append(mem.content)
            metadatas.append({
                "created_at": mem.created_at,
                "metadata_json": json.dumps(mem.metadata, ensure_ascii=False),
                "tags_json": json.dumps(mem.tags, ensure_ascii=False),
            })

        if ids:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    def load_embedding_store(self, store: EmbeddingStore) -> None:
        """从 ChromaDB 加载 EmbeddingStore。

        Args:
            store: 目标 EmbeddingStore
        """
        try:
            result = self._collection.get(
                include=["embeddings", "documents", "metadatas"],
            )
        except Exception:
            return

        if not result["ids"]:
            return

        for i, mem_id in enumerate(result["ids"]):
            embedding = result["embeddings"][i] if result["embeddings"] else None
            content = result["documents"][i] if result["documents"] else ""
            metadata_raw = result["metadatas"][i] if result["metadatas"] else {}

            mem = Memory(
                id=mem_id,
                content=content,
                created_at=metadata_raw.get("created_at", 0),
                metadata=json.loads(metadata_raw.get("metadata_json", "{}")),
                embedding=list(embedding) if embedding is not None else None,
                tags=json.loads(metadata_raw.get("tags_json", "[]")),
            )
            store.add(mem)

    def save_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """将知识图谱保存到 JSON（ChromaDB 不支持图关系）。

        Args:
            kg: 待保存的 KnowledgeGraph
        """
        entities = [e.to_dict() for e in kg.find_entities()]
        relations = [r.to_dict() for r in kg.find_relations()]
        data = {"entities": entities, "relations": relations}
        with open(self._graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """从 JSON 加载知识图谱。

        Args:
            kg: 目标 KnowledgeGraph
        """
        if not os.path.exists(self._graph_path):
            return
        with open(self._graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entity_data in data.get("entities", []):
            entity = Entity.from_dict(entity_data)
            kg.add_entity(entity)

        for rel_data in data.get("relations", []):
            relation = Relation.from_dict(rel_data)
            kg.add_relation(relation)

    def chroma_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """直接使用 ChromaDB 原生向量搜索。

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数
            where: ChromaDB 过滤条件

        Returns:
            搜索结果列表
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        results: list[dict[str, Any]] = []
        if result["ids"] and result["ids"][0]:
            for i, mem_id in enumerate(result["ids"][0]):
                results.append({
                    "id": mem_id,
                    "document": result["documents"][0][i] if result["documents"] else "",
                    "distance": result["distances"][0][i] if result["distances"] else 0,
                    "metadata": result["metadatas"][0][i] if result["metadatas"] else {},
                })
        return results


def register_chromadb_plugin() -> None:
    """将 ChromaDB 后端注册到全局插件注册表。

    在已安装 chromadb 的环境中调用此函数即可启用。
    """
    from agentmemory.plugins import get_registry
    registry = get_registry()
    if "chromadb" not in registry.list_backends():
        registry.register_backend("chromadb", ChromaDBBackend)
