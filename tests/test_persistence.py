"""测试持久化存储：JSON 和 SQLite 后端"""

import json
import os
import pytest
import tempfile

from agentmemory.models import Memory, Entity, Relation
from agentmemory.embedding_store import EmbeddingStore
from agentmemory.knowledge_graph import KnowledgeGraph
from agentmemory.persistence import JSONBackend, SQLiteBackend


class TestJSONBackend:
    """JSON 持久化后端测试"""

    def test_save_and_load_embedding_store(self, tmp_path):
        """EmbeddingStore 保存到 JSON 再加载，数据不丢"""
        path = str(tmp_path / "store.json")
        store = EmbeddingStore(dimension=3)
        m1 = Memory(content="hello", embedding=[1.0, 0.0, 0.0])
        m2 = Memory(content="world", embedding=[0.0, 1.0, 0.0])
        store.add(m1)
        store.add(m2)

        backend = JSONBackend(path)
        backend.save_embedding_store(store)

        # 重新加载到新 store
        new_store = EmbeddingStore(dimension=3)
        backend.load_embedding_store(new_store)
        assert new_store.count() == 2
        loaded_contents = {m.content for m in new_store.list_all()}
        assert loaded_contents == {"hello", "world"}

    def test_save_and_load_knowledge_graph(self, tmp_path):
        """KnowledgeGraph 保存到 JSON 再加载，数据不丢"""
        path = str(tmp_path / "graph.json")
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        r = Relation(source_id=e1.id, target_id=e2.id, relation_type="knows")
        kg.add_relation(r)

        backend = JSONBackend(path)
        backend.save_knowledge_graph(kg)

        new_kg = KnowledgeGraph()
        backend.load_knowledge_graph(new_kg)
        assert new_kg.entity_count() == 2
        assert new_kg.relation_count() == 1
        loaded_names = {e.name for e in new_kg.find_entities()}
        assert loaded_names == {"Alice", "Bob"}

    def test_load_nonexistent_file(self, tmp_path):
        """加载不存在的文件不报错（空数据）"""
        path = str(tmp_path / "nonexistent.json")
        backend = JSONBackend(path)

        store = EmbeddingStore(dimension=3)
        backend.load_embedding_store(store)
        assert store.count() == 0

        kg = KnowledgeGraph()
        backend.load_knowledge_graph(kg)
        assert kg.entity_count() == 0

    def test_save_preserves_metadata(self, tmp_path):
        """保存保留 metadata"""
        path = str(tmp_path / "store.json")
        store = EmbeddingStore(dimension=3)
        m = Memory(content="test", embedding=[1.0, 0.0, 0.0], metadata={"source": "user"})
        store.add(m)

        backend = JSONBackend(path)
        backend.save_embedding_store(store)

        new_store = EmbeddingStore(dimension=3)
        backend.load_embedding_store(new_store)
        loaded = new_store.list_all()[0]
        assert loaded.metadata == {"source": "user"}

    def test_save_preserves_entity_properties(self, tmp_path):
        """保存保留 entity properties"""
        path = str(tmp_path / "graph.json")
        kg = KnowledgeGraph()
        e = Entity(name="Python", entity_type="language", properties={"version": "3.12"})
        kg.add_entity(e)

        backend = JSONBackend(path)
        backend.save_knowledge_graph(kg)

        new_kg = KnowledgeGraph()
        backend.load_knowledge_graph(new_kg)
        loaded = new_kg.find_entities(name="Python")[0]
        assert loaded.properties == {"version": "3.12"}

    def test_overwrite_existing_file(self, tmp_path):
        """多次保存会覆盖"""
        path = str(tmp_path / "store.json")
        backend = JSONBackend(path)

        store1 = EmbeddingStore(dimension=3)
        store1.add(Memory(content="first", embedding=[1.0, 0.0, 0.0]))
        backend.save_embedding_store(store1)

        store2 = EmbeddingStore(dimension=3)
        store2.add(Memory(content="second", embedding=[0.0, 1.0, 0.0]))
        store2.add(Memory(content="third", embedding=[0.0, 0.0, 1.0]))
        backend.save_embedding_store(store2)

        loaded = EmbeddingStore(dimension=3)
        backend.load_embedding_store(loaded)
        assert loaded.count() == 2


class TestSQLiteBackend:
    """SQLite 持久化后端测试"""

    def test_save_and_load_embedding_store(self, tmp_path):
        """EmbeddingStore 保存到 SQLite 再加载"""
        path = str(tmp_path / "store.db")
        store = EmbeddingStore(dimension=3)
        m1 = Memory(content="hello", embedding=[1.0, 0.0, 0.0])
        m2 = Memory(content="world", embedding=[0.0, 1.0, 0.0])
        store.add(m1)
        store.add(m2)

        backend = SQLiteBackend(path)
        backend.save_embedding_store(store)

        new_store = EmbeddingStore(dimension=3)
        backend.load_embedding_store(new_store)
        assert new_store.count() == 2
        loaded_contents = {m.content for m in new_store.list_all()}
        assert loaded_contents == {"hello", "world"}

    def test_save_and_load_knowledge_graph(self, tmp_path):
        """KnowledgeGraph 保存到 SQLite 再加载"""
        path = str(tmp_path / "graph.db")
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        r = Relation(source_id=e1.id, target_id=e2.id, relation_type="knows", weight=0.8)
        kg.add_relation(r)

        backend = SQLiteBackend(path)
        backend.save_knowledge_graph(kg)

        new_kg = KnowledgeGraph()
        backend.load_knowledge_graph(new_kg)
        assert new_kg.entity_count() == 2
        assert new_kg.relation_count() == 1

        # 验证关系权重
        rels = new_kg.find_relations(relation_type="knows")
        assert rels[0].weight == 0.8

    def test_load_nonexistent_db(self, tmp_path):
        """加载不存在的数据库不报错"""
        path = str(tmp_path / "nonexistent.db")
        backend = SQLiteBackend(path)

        store = EmbeddingStore(dimension=3)
        backend.load_embedding_store(store)
        assert store.count() == 0

        kg = KnowledgeGraph()
        backend.load_knowledge_graph(kg)
        assert kg.entity_count() == 0

    def test_save_preserves_metadata(self, tmp_path):
        """SQLite 保存保留 metadata"""
        path = str(tmp_path / "store.db")
        store = EmbeddingStore(dimension=3)
        store.add(Memory(content="test", embedding=[1.0, 0.0, 0.0], metadata={"key": "val"}))

        backend = SQLiteBackend(path)
        backend.save_embedding_store(store)

        new_store = EmbeddingStore(dimension=3)
        backend.load_embedding_store(new_store)
        assert new_store.list_all()[0].metadata == {"key": "val"}

    def test_overwrite_existing_db(self, tmp_path):
        """多次保存会覆盖旧数据"""
        path = str(tmp_path / "store.db")
        backend = SQLiteBackend(path)

        store1 = EmbeddingStore(dimension=3)
        store1.add(Memory(content="old", embedding=[1.0, 0.0, 0.0]))
        backend.save_embedding_store(store1)

        store2 = EmbeddingStore(dimension=3)
        store2.add(Memory(content="new1", embedding=[0.0, 1.0, 0.0]))
        store2.add(Memory(content="new2", embedding=[0.0, 0.0, 1.0]))
        backend.save_embedding_store(store2)

        loaded = EmbeddingStore(dimension=3)
        backend.load_embedding_store(loaded)
        assert loaded.count() == 2
        loaded_contents = {m.content for m in loaded.list_all()}
        assert loaded_contents == {"new1", "new2"}

    def test_large_dataset(self, tmp_path):
        """SQLite 处理较大数据集"""
        path = str(tmp_path / "large.db")
        store = EmbeddingStore(dimension=8)
        for i in range(100):
            embedding = [float(i % 8 == j) for j in range(8)]
            store.add(Memory(content=f"item-{i}", embedding=embedding))

        backend = SQLiteBackend(path)
        backend.save_embedding_store(store)

        new_store = EmbeddingStore(dimension=8)
        backend.load_embedding_store(new_store)
        assert new_store.count() == 100
