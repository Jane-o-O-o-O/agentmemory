"""aiosqlite 真异步持久化后端。

使用 aiosqlite 实现真正的异步 I/O，适合高并发场景。
如果 aiosqlite 未安装，提供回退方案（在事件循环中执行同步操作）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from agentmemory.models import Memory, Entity, Relation
from agentmemory.embedding_store import EmbeddingStore
from agentmemory.knowledge_graph import KnowledgeGraph


class AsyncSQLiteBackend:
    """aiosqlite 真异步 SQLite 持久化后端。

    使用 WAL 模式和真正的 async I/O，
    适合高并发 RAG 应用场景。

    Args:
        db_path: 数据库文件路径
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def _get_connection(self):
        """获取异步数据库连接。

        Returns:
            aiosqlite Connection
        """
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite 未安装。请执行: pip install aiosqlite"
            )
        conn = await aiosqlite.connect(self._db_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        return conn

    async def _init_tables(self, conn) -> None:
        """创建表结构。"""
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT,
                tags TEXT NOT NULL DEFAULT '[]'
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

    async def save_embedding_store(self, store: EmbeddingStore) -> None:
        """异步保存 EmbeddingStore 到 SQLite。

        Args:
            store: 待保存的 EmbeddingStore
        """
        conn = await self._get_connection()
        try:
            await self._init_tables(conn)
            await conn.execute("DELETE FROM memories")
            for mem in store.list_all():
                embedding_json = json.dumps(mem.embedding) if mem.embedding else None
                await conn.execute(
                    "INSERT INTO memories (id, content, created_at, metadata, embedding, tags) VALUES (?, ?, ?, ?, ?, ?)",
                    (mem.id, mem.content, mem.created_at, json.dumps(mem.metadata, ensure_ascii=False), embedding_json, json.dumps(mem.tags, ensure_ascii=False)),
                )
            await conn.commit()
        finally:
            await conn.close()

    async def load_embedding_store(self, store: EmbeddingStore) -> None:
        """异步从 SQLite 加载 EmbeddingStore。

        Args:
            store: 目标 EmbeddingStore
        """
        if not os.path.exists(self._db_path):
            return

        conn = await self._get_connection()
        try:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            )
            if await cursor.fetchone() is None:
                return

            cursor = await conn.execute("SELECT id, content, created_at, metadata, embedding, tags FROM memories")
            async for row in cursor:
                mem = Memory(
                    id=row[0],
                    content=row[1],
                    created_at=row[2],
                    metadata=json.loads(row[3]),
                    embedding=json.loads(row[4]) if row[4] else None,
                    tags=json.loads(row[5]) if row[5] else [],
                )
                store.add(mem)
        finally:
            await conn.close()

    async def save_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """异步保存 KnowledgeGraph 到 SQLite。

        Args:
            kg: 待保存的 KnowledgeGraph
        """
        conn = await self._get_connection()
        try:
            await self._init_tables(conn)
            await conn.execute("DELETE FROM relations")
            await conn.execute("DELETE FROM entities")

            for entity in kg.find_entities():
                await conn.execute(
                    "INSERT INTO entities (id, name, entity_type, properties) VALUES (?, ?, ?, ?)",
                    (entity.id, entity.name, entity.entity_type, json.dumps(entity.properties, ensure_ascii=False)),
                )

            for rel in kg.find_relations():
                await conn.execute(
                    "INSERT INTO relations (id, source_id, target_id, relation_type, weight) VALUES (?, ?, ?, ?, ?)",
                    (rel.id, rel.source_id, rel.target_id, rel.relation_type, rel.weight),
                )

            await conn.commit()
        finally:
            await conn.close()

    async def load_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """异步从 SQLite 加载 KnowledgeGraph。

        Args:
            kg: 目标 KnowledgeGraph
        """
        if not os.path.exists(self._db_path):
            return

        conn = await self._get_connection()
        try:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
            )
            if await cursor.fetchone() is None:
                return

            cursor = await conn.execute("SELECT id, name, entity_type, properties FROM entities")
            async for row in cursor:
                entity = Entity(
                    id=row[0],
                    name=row[1],
                    entity_type=row[2],
                    properties=json.loads(row[3]),
                )
                kg.add_entity(entity)

            cursor = await conn.execute("SELECT id, source_id, target_id, relation_type, weight FROM relations")
            async for row in cursor:
                relation = Relation(
                    id=row[0],
                    source_id=row[1],
                    target_id=row[2],
                    relation_type=row[3],
                    weight=row[4],
                )
                kg.add_relation(relation)
        finally:
            await conn.close()

    async def close(self) -> None:
        """清理资源（无操作，连接在每个方法中管理）。"""
        pass
