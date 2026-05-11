"""持久化存储后端：JSON 和 SQLite

支持将 EmbeddingStore 和 KnowledgeGraph 的数据持久化到磁盘，
重启后可完整恢复。
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from agentmemory.models import Memory, Entity, Relation
from agentmemory.embedding_store import EmbeddingStore
from agentmemory.knowledge_graph import KnowledgeGraph


class JSONBackend:
    """JSON 文件持久化后端。

    将 EmbeddingStore 和 KnowledgeGraph 数据分别存储为 JSON 文件。
    适合小规模数据（<10k 条记忆），文件可读、易调试。

    Args:
        base_path: 存储目录路径，会自动创建
    """

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _store_path(self) -> Path:
        return self._base_path / "memories.json"

    def _graph_path(self) -> Path:
        return self._base_path / "knowledge_graph.json"

    def save_embedding_store(self, store: EmbeddingStore) -> None:
        """将 EmbeddingStore 保存到 JSON 文件。

        Args:
            store: 待保存的 EmbeddingStore
        """
        data = [m.to_dict() for m in store.list_all()]
        with open(self._store_path(), "w", encoding="utf-8") as f:
            json.dump({"dimension": store.dimension, "memories": data}, f, ensure_ascii=False, indent=2)

    def load_embedding_store(self, store: EmbeddingStore) -> None:
        """从 JSON 文件加载 EmbeddingStore。

        Args:
            store: 目标 EmbeddingStore（数据将被追加）
        """
        path = self._store_path()
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for mem_data in data.get("memories", []):
            mem = Memory.from_dict(mem_data)
            store.add(mem)

    def save_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """将 KnowledgeGraph 保存到 JSON 文件。

        Args:
            kg: 待保存的 KnowledgeGraph
        """
        entities = [e.to_dict() for e in kg.find_entities()]
        relations = [r.to_dict() for r in kg.find_relations()]
        data = {"entities": entities, "relations": relations}
        with open(self._graph_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """从 JSON 文件加载 KnowledgeGraph。

        Args:
            kg: 目标 KnowledgeGraph（数据将被追加）
        """
        path = self._graph_path()
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 先加载实体
        for entity_data in data.get("entities", []):
            entity = Entity.from_dict(entity_data)
            kg.add_entity(entity)

        # 再加载关系（依赖实体存在）
        for rel_data in data.get("relations", []):
            relation = Relation.from_dict(rel_data)
            kg.add_relation(relation)


class SQLiteBackend:
    """SQLite 持久化后端。

    使用 SQLite 数据库存储，适合大规模数据。
    所有数据存在一个 .db 文件中。

    Args:
        db_path: 数据库文件路径
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self, conn: sqlite3.Connection) -> None:
        """创建表结构"""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT
            );
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            );
        """)

    def save_embedding_store(self, store: EmbeddingStore) -> None:
        """将 EmbeddingStore 保存到 SQLite。

        Args:
            store: 待保存的 EmbeddingStore
        """
        conn = self._connect()
        try:
            self._init_tables(conn)
            # 清空旧数据
            conn.execute("DELETE FROM memories")
            # 插入新数据
            for mem in store.list_all():
                embedding_json = json.dumps(mem.embedding) if mem.embedding else None
                conn.execute(
                    "INSERT INTO memories (id, content, created_at, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                    (mem.id, mem.content, mem.created_at, json.dumps(mem.metadata, ensure_ascii=False), embedding_json),
                )
            conn.commit()
        finally:
            conn.close()

    def load_embedding_store(self, store: EmbeddingStore) -> None:
        """从 SQLite 加载 EmbeddingStore。

        Args:
            store: 目标 EmbeddingStore
        """
        if not os.path.exists(self._db_path):
            return

        conn = self._connect()
        try:
            # 检查表是否存在
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            )
            if cursor.fetchone() is None:
                return

            cursor = conn.execute("SELECT id, content, created_at, metadata, embedding FROM memories")
            for row in cursor:
                mem = Memory(
                    id=row[0],
                    content=row[1],
                    created_at=row[2],
                    metadata=json.loads(row[3]),
                    embedding=json.loads(row[4]) if row[4] else None,
                )
                store.add(mem)
        finally:
            conn.close()

    def save_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """将 KnowledgeGraph 保存到 SQLite。

        Args:
            kg: 待保存的 KnowledgeGraph
        """
        conn = self._connect()
        try:
            self._init_tables(conn)
            conn.execute("DELETE FROM relations")
            conn.execute("DELETE FROM entities")

            for entity in kg.find_entities():
                conn.execute(
                    "INSERT INTO entities (id, name, entity_type, properties) VALUES (?, ?, ?, ?)",
                    (entity.id, entity.name, entity.entity_type, json.dumps(entity.properties, ensure_ascii=False)),
                )

            for rel in kg.find_relations():
                conn.execute(
                    "INSERT INTO relations (id, source_id, target_id, relation_type, weight) VALUES (?, ?, ?, ?, ?)",
                    (rel.id, rel.source_id, rel.target_id, rel.relation_type, rel.weight),
                )

            conn.commit()
        finally:
            conn.close()

    def load_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """从 SQLite 加载 KnowledgeGraph。

        Args:
            kg: 目标 KnowledgeGraph
        """
        if not os.path.exists(self._db_path):
            return

        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
            )
            if cursor.fetchone() is None:
                return

            # 加载实体
            cursor = conn.execute("SELECT id, name, entity_type, properties FROM entities")
            for row in cursor:
                entity = Entity(
                    id=row[0],
                    name=row[1],
                    entity_type=row[2],
                    properties=json.loads(row[3]),
                )
                kg.add_entity(entity)

            # 加载关系
            cursor = conn.execute("SELECT id, source_id, target_id, relation_type, weight FROM relations")
            for row in cursor:
                relation = Relation(
                    id=row[0],
                    source_id=row[1],
                    target_id=row[2],
                    relation_type=row[3],
                    weight=row[4],
                )
                kg.add_relation(relation)
        finally:
            conn.close()
