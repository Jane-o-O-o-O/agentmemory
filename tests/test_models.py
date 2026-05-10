"""测试数据模型：Memory, Entity, Relation, SearchResult"""

import pytest
import time

from agentmemory.models import Memory, Entity, Relation, SearchResult


class TestMemory:
    """Memory 数据模型测试"""

    def test_create_memory_with_defaults(self):
        """创建 Memory 时，id 和 created_at 自动生成"""
        mem = Memory(content="Hello world")
        assert mem.content == "Hello world"
        assert isinstance(mem.id, str)
        assert len(mem.id) > 0
        assert isinstance(mem.created_at, float)
        assert mem.metadata == {}

    def test_create_memory_with_all_fields(self):
        """支持传入所有字段"""
        mem = Memory(
            id="custom-id",
            content="Test content",
            created_at=1234567890.0,
            metadata={"source": "user"},
            embedding=[0.1, 0.2, 0.3],
        )
        assert mem.id == "custom-id"
        assert mem.content == "Test content"
        assert mem.created_at == 1234567890.0
        assert mem.metadata == {"source": "user"}
        assert mem.embedding == [0.1, 0.2, 0.3]

    def test_memory_to_dict(self):
        """Memory 可以转换为 dict"""
        mem = Memory(id="m1", content="hello")
        d = mem.to_dict()
        assert d["id"] == "m1"
        assert d["content"] == "hello"
        assert "created_at" in d
        assert "metadata" in d

    def test_memory_from_dict(self):
        """可以从 dict 创建 Memory"""
        d = {
            "id": "m2",
            "content": "world",
            "created_at": 100.0,
            "metadata": {"k": "v"},
            "embedding": None,
        }
        mem = Memory.from_dict(d)
        assert mem.id == "m2"
        assert mem.content == "world"
        assert mem.metadata == {"k": "v"}

    def test_memory_unique_ids(self):
        """两个 Memory 自动生成不同的 id"""
        m1 = Memory(content="a")
        m2 = Memory(content="b")
        assert m1.id != m2.id

    def test_memory_content_required(self):
        """content 不能为空"""
        with pytest.raises(ValueError):
            Memory(content="")

    def test_memory_str_repr(self):
        """str 表示只显示前50字符"""
        mem = Memory(content="a" * 100)
        assert "a" * 50 in str(mem)
        assert len(str(mem)) < 100


class TestEntity:
    """Entity 数据模型测试"""

    def test_create_entity(self):
        """创建 Entity"""
        e = Entity(name="Alice", entity_type="person")
        assert e.name == "Alice"
        assert e.entity_type == "person"
        assert e.properties == {}
        assert isinstance(e.id, str)

    def test_create_entity_with_properties(self):
        """创建带属性的 Entity"""
        e = Entity(
            name="Python",
            entity_type="language",
            properties={"paradigm": "multi", "version": "3.12"},
        )
        assert e.properties["paradigm"] == "multi"

    def test_entity_to_dict(self):
        """Entity 序列化"""
        e = Entity(id="e1", name="Bob", entity_type="person")
        d = e.to_dict()
        assert d["name"] == "Bob"
        assert d["entity_type"] == "person"

    def test_entity_from_dict(self):
        """Entity 反序列化"""
        d = {"id": "e1", "name": "Bob", "entity_type": "person", "properties": {}}
        e = Entity.from_dict(d)
        assert e.name == "Bob"

    def test_entity_name_required(self):
        """name 不能为空"""
        with pytest.raises(ValueError):
            Entity(name="", entity_type="test")

    def test_entity_type_required(self):
        """entity_type 不能为空"""
        with pytest.raises(ValueError):
            Entity(name="X", entity_type="")


class TestRelation:
    """Relation 数据模型测试"""

    def test_create_relation(self):
        """创建 Relation"""
        r = Relation(
            source_id="e1",
            target_id="e2",
            relation_type="knows",
        )
        assert r.source_id == "e1"
        assert r.target_id == "e2"
        assert r.relation_type == "knows"
        assert r.weight == 1.0

    def test_relation_with_weight(self):
        """创建带权重的 Relation"""
        r = Relation(
            source_id="e1",
            target_id="e2",
            relation_type="depends_on",
            weight=0.8,
        )
        assert r.weight == 0.8

    def test_relation_to_dict(self):
        """Relation 序列化"""
        r = Relation(id="r1", source_id="e1", target_id="e2", relation_type="likes")
        d = r.to_dict()
        assert d["source_id"] == "e1"
        assert d["relation_type"] == "likes"

    def test_relation_from_dict(self):
        """Relation 反序列化"""
        d = {
            "id": "r1",
            "source_id": "e1",
            "target_id": "e2",
            "relation_type": "likes",
            "weight": 1.0,
        }
        r = Relation.from_dict(d)
        assert r.source_id == "e1"
        assert r.weight == 1.0

    def test_relation_source_required(self):
        """source_id 不能为空"""
        with pytest.raises(ValueError):
            Relation(source_id="", target_id="e2", relation_type="x")

    def test_relation_target_required(self):
        """target_id 不能为空"""
        with pytest.raises(ValueError):
            Relation(source_id="e1", target_id="", relation_type="x")


class TestSearchResult:
    """SearchResult 数据模型测试"""

    def test_create_search_result(self):
        """创建 SearchResult"""
        mem = Memory(content="hello")
        sr = SearchResult(memory=mem, score=0.95)
        assert sr.memory is mem
        assert sr.score == 0.95

    def test_search_result_default_context(self):
        """context 默认为空列表"""
        mem = Memory(content="test")
        sr = SearchResult(memory=mem, score=0.5)
        assert sr.context == []

    def test_search_result_with_context(self):
        """可以附带上下文"""
        mem = Memory(content="test")
        context = [Memory(content="related1"), Memory(content="related2")]
        sr = SearchResult(memory=mem, score=0.5, context=context)
        assert len(sr.context) == 2
