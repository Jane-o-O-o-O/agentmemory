"""测试向量存储与搜索：EmbeddingStore"""

import pytest
import numpy as np

from agentmemory.models import Memory
from agentmemory.embedding_store import EmbeddingStore, cosine_similarity


class TestCosineSimilarity:
    """cosine_similarity 工具函数测试"""

    def test_identical_vectors(self):
        """相同向量的余弦相似度为 1.0"""
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        """正交向量的余弦相似度为 0.0"""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        """反向向量的余弦相似度为 -1.0"""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_zero_vector_raises(self):
        """零向量应抛出 ValueError"""
        with pytest.raises(ValueError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_different_lengths_raises(self):
        """维度不同的向量应抛出 ValueError"""
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_known_value(self):
        """验证已知余弦相似度"""
        a = [1.0, 2.0]
        b = [2.0, 1.0]
        expected = (1 * 2 + 2 * 1) / (np.sqrt(5) * np.sqrt(5))
        assert abs(cosine_similarity(a, b) - expected) < 1e-9


class TestEmbeddingStore:
    """EmbeddingStore 向量存储测试"""

    def test_add_memory_with_embedding(self):
        """添加带 embedding 的 Memory"""
        store = EmbeddingStore(dimension=3)
        mem = Memory(content="hello", embedding=[1.0, 0.0, 0.0])
        store.add(mem)
        assert store.count() == 1

    def test_add_memory_without_embedding_raises(self):
        """添加无 embedding 的 Memory 应抛出 ValueError"""
        store = EmbeddingStore(dimension=3)
        mem = Memory(content="hello")
        with pytest.raises(ValueError, match="embedding"):
            store.add(mem)

    def test_add_wrong_dimension_raises(self):
        """embedding 维度不匹配应抛出 ValueError"""
        store = EmbeddingStore(dimension=3)
        mem = Memory(content="hello", embedding=[1.0, 2.0])
        with pytest.raises(ValueError, match="维度"):
            store.add(mem)

    def test_search_returns_top_k(self):
        """搜索返回按相似度排序的前 k 个结果"""
        store = EmbeddingStore(dimension=3)
        mem1 = Memory(content="aaa", embedding=[1.0, 0.0, 0.0])
        mem2 = Memory(content="bbb", embedding=[0.0, 1.0, 0.0])
        mem3 = Memory(content="ccc", embedding=[0.9, 0.1, 0.0])
        store.add(mem1)
        store.add(mem2)
        store.add(mem3)

        results = store.search(query=[1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # 结果按相似度降序
        assert results[0].score >= results[1].score
        # 最相似的应该是 mem1
        assert results[0].memory.content == "aaa"

    def test_search_empty_store(self):
        """空 store 搜索返回空列表"""
        store = EmbeddingStore(dimension=3)
        results = store.search(query=[1.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_search_top_k_larger_than_count(self):
        """top_k 大于 store 容量时返回全部"""
        store = EmbeddingStore(dimension=3)
        store.add(Memory(content="a", embedding=[1.0, 0.0, 0.0]))
        results = store.search(query=[1.0, 0.0, 0.0], top_k=100)
        assert len(results) == 1

    def test_search_with_threshold(self):
        """带阈值过滤"""
        store = EmbeddingStore(dimension=3)
        store.add(Memory(content="good", embedding=[1.0, 0.0, 0.0]))
        store.add(Memory(content="bad", embedding=[0.0, 1.0, 0.0]))

        results = store.search(query=[1.0, 0.0, 0.0], top_k=10, threshold=0.9)
        assert len(results) == 1
        assert results[0].memory.content == "good"

    def test_remove_memory(self):
        """删除 Memory"""
        store = EmbeddingStore(dimension=3)
        mem = Memory(content="hello", embedding=[1.0, 0.0, 0.0])
        store.add(mem)
        assert store.count() == 1
        store.remove(mem.id)
        assert store.count() == 0

    def test_remove_nonexistent_raises(self):
        """删除不存在的 Memory 应抛出 KeyError"""
        store = EmbeddingStore(dimension=3)
        with pytest.raises(KeyError):
            store.remove("nonexistent")

    def test_get_memory(self):
        """通过 id 获取 Memory"""
        store = EmbeddingStore(dimension=3)
        mem = Memory(content="hello", embedding=[1.0, 0.0, 0.0])
        store.add(mem)
        result = store.get(mem.id)
        assert result is mem

    def test_get_nonexistent_returns_none(self):
        """获取不存在的 Memory 返回 None"""
        store = EmbeddingStore(dimension=3)
        assert store.get("nonexistent") is None

    def test_count(self):
        """count 返回正确的数量"""
        store = EmbeddingStore(dimension=3)
        assert store.count() == 0
        store.add(Memory(content="a", embedding=[1, 0, 0]))
        assert store.count() == 1
        store.add(Memory(content="b", embedding=[0, 1, 0]))
        assert store.count() == 2

    def test_list_all(self):
        """list_all 返回所有 Memory"""
        store = EmbeddingStore(dimension=3)
        m1 = Memory(content="a", embedding=[1, 0, 0])
        m2 = Memory(content="b", embedding=[0, 1, 0])
        store.add(m1)
        store.add(m2)
        all_mems = store.list_all()
        assert len(all_mems) == 2
        assert set(m.content for m in all_mems) == {"a", "b"}
