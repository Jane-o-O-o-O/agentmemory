"""测试混合记忆框架：HybridMemory"""

import pytest

from agentmemory.models import Memory, Entity, Relation, SearchResult
from agentmemory.hybrid_memory import HybridMemory


class TestHybridMemory:
    """HybridMemory 混合记忆 API 测试"""

    def test_create_hybrid_memory(self):
        """创建 HybridMemory 实例"""
        hm = HybridMemory(dimension=3)
        assert hm.embedding_store.count() == 0
        assert hm.knowledge_graph.entity_count() == 0

    # --- 记忆管理 ---

    def test_remember_adds_memory(self):
        """remember 添加一条记忆"""
        hm = HybridMemory(dimension=3)
        mem = hm.remember("Hello world", embedding=[1.0, 0.0, 0.0])
        assert mem.content == "Hello world"
        assert mem.embedding == [1.0, 0.0, 0.0]
        assert hm.embedding_store.count() == 1

    def test_remember_with_metadata(self):
        """remember 支持 metadata"""
        hm = HybridMemory(dimension=3)
        mem = hm.remember("test", embedding=[1, 0, 0], metadata={"source": "user"})
        assert mem.metadata["source"] == "user"

    def test_remember_without_embedding_raises(self):
        """remember 不传 embedding 应抛出 ValueError"""
        hm = HybridMemory(dimension=3)
        with pytest.raises(ValueError, match="embedding"):
            hm.remember("hello")

    def test_forget_removes_memory(self):
        """forget 删除记忆"""
        hm = HybridMemory(dimension=3)
        mem = hm.remember("temp", embedding=[1, 0, 0])
        assert hm.embedding_store.count() == 1
        hm.forget(mem.id)
        assert hm.embedding_store.count() == 0

    def test_forget_nonexistent_raises(self):
        """forget 不存在的记忆应抛出 KeyError"""
        hm = HybridMemory(dimension=3)
        with pytest.raises(KeyError):
            hm.forget("nonexistent")

    # --- 向量搜索 ---

    def test_search_by_vector(self):
        """通过向量查询搜索记忆"""
        hm = HybridMemory(dimension=3)
        hm.remember("cat", embedding=[1.0, 0.0, 0.0])
        hm.remember("dog", embedding=[0.0, 1.0, 0.0])
        hm.remember("kitten", embedding=[0.95, 0.05, 0.0])

        results = hm.search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].memory.content == "cat"
        assert results[0].score >= results[1].score

    def test_search_with_threshold(self):
        """搜索支持 threshold 过滤"""
        hm = HybridMemory(dimension=3)
        hm.remember("good", embedding=[1.0, 0.0, 0.0])
        hm.remember("bad", embedding=[0.0, 1.0, 0.0])

        results = hm.search(query_embedding=[1.0, 0.0, 0.0], top_k=10, threshold=0.9)
        assert len(results) == 1
        assert results[0].memory.content == "good"

    # --- 知识图谱管理 ---

    def test_add_entity(self):
        """add_entity 添加实体"""
        hm = HybridMemory(dimension=3)
        e = hm.add_entity("Alice", "person")
        assert e.name == "Alice"
        assert hm.knowledge_graph.entity_count() == 1

    def test_add_relation_between_entities(self):
        """add_relation 添加关系"""
        hm = HybridMemory(dimension=3)
        e1 = hm.add_entity("Alice", "person")
        e2 = hm.add_entity("Bob", "person")
        r = hm.add_relation(e1.id, e2.id, "knows")
        assert r.relation_type == "knows"
        assert hm.knowledge_graph.relation_count() == 1

    def test_get_entity_neighbors(self):
        """get_neighbors 获取实体邻居"""
        hm = HybridMemory(dimension=3)
        e1 = hm.add_entity("Alice", "person")
        e2 = hm.add_entity("Bob", "person")
        e3 = hm.add_entity("Charlie", "person")
        hm.add_relation(e1.id, e2.id, "knows")
        hm.add_relation(e1.id, e3.id, "knows")

        neighbors = hm.get_neighbors(e1.id)
        names = {e.name for e in neighbors}
        assert names == {"Bob", "Charlie"}

    # --- 混合搜索 ---

    def test_hybrid_search_combines_vector_and_graph(self):
        """hybrid_search 同时使用向量搜索和图谱上下文"""
        hm = HybridMemory(dimension=3)

        # 添加一些记忆
        m1 = hm.remember("Alice is a programmer", embedding=[1.0, 0.0, 0.0])
        m2 = hm.remember("Bob is a designer", embedding=[0.0, 1.0, 0.0])
        m3 = hm.remember("Alice uses Python", embedding=[0.9, 0.1, 0.0])

        # 添加实体和关系
        e_alice = hm.add_entity("Alice", "person")
        e_python = hm.add_entity("Python", "language")
        hm.add_relation(e_alice.id, e_python.id, "knows")

        results = hm.hybrid_search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=3,
            graph_depth=1,
        )
        assert len(results) > 0
        # 结果应包含 vector_score 和 graph_context
        for r in results:
            assert hasattr(r, 'score')
            assert r.score >= 0

    def test_hybrid_search_no_graph_context(self):
        """无图谱上下文时 hybrid_search 退化为纯向量搜索"""
        hm = HybridMemory(dimension=3)
        hm.remember("test", embedding=[1.0, 0.0, 0.0])

        results = hm.hybrid_search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=5,
            graph_depth=0,
        )
        assert len(results) == 1
        assert results[0].memory.content == "test"

    def test_hybrid_search_empty(self):
        """空记忆库的 hybrid_search 返回空列表"""
        hm = HybridMemory(dimension=3)
        results = hm.hybrid_search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=5,
        )
        assert results == []

    # --- 统计 ---

    def test_stats(self):
        """stats 返回统计信息"""
        hm = HybridMemory(dimension=3)
        hm.remember("a", embedding=[1, 0, 0])
        hm.remember("b", embedding=[0, 1, 0])
        e = hm.add_entity("X", "thing")
        stats = hm.stats()
        assert stats["memory_count"] == 2
        assert stats["entity_count"] == 1
        assert stats["relation_count"] == 0
