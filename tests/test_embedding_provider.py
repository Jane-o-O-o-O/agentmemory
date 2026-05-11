"""测试 Embedding Provider：HashEmbeddingProvider"""

import math
import pytest

from agentmemory.embedding_provider import HashEmbeddingProvider


class TestHashEmbeddingProvider:
    """HashEmbeddingProvider 确定性 embedding 测试"""

    def test_default_dimension(self):
        """默认维度为 128"""
        provider = HashEmbeddingProvider()
        assert provider.dimension() == 128

    def test_custom_dimension(self):
        """支持自定义维度"""
        provider = HashEmbeddingProvider(dim=64)
        assert provider.dimension() == 64
        vec = provider.embed("hello")
        assert len(vec) == 64

    def test_embed_returns_normalized_vector(self):
        """输出为 L2 归一化向量"""
        provider = HashEmbeddingProvider(dim=32)
        vec = provider.embed("some text here")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-9

    def test_deterministic(self):
        """相同文本产生相同向量"""
        provider = HashEmbeddingProvider()
        text = "hello world"
        v1 = provider.embed(text)
        v2 = provider.embed(text)
        assert v1 == v2

    def test_different_text_different_vectors(self):
        """不同文本产生不同向量"""
        provider = HashEmbeddingProvider()
        v1 = provider.embed("cat")
        v2 = provider.embed("dog")
        assert v1 != v2

    def test_empty_text(self):
        """空文本产生零向量（归一化后为全零）"""
        provider = HashEmbeddingProvider(dim=8)
        vec = provider.embed("")
        # 空文本没有 token，vec 全为 0
        assert vec == [0.0] * 8

    def test_similar_text_similar_vectors(self):
        """相似文本的向量余弦相似度较高"""
        provider = HashEmbeddingProvider(dim=128)
        v1 = provider.embed("machine learning")
        v2 = provider.embed("machine learning algorithms")
        v3 = provider.embed("cooking pasta recipe")

        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        # 共享 "machine" 的两个向量应该比与 cooking 的更相似
        sim_related = dot(v1, v2)
        sim_unrelated = dot(v1, v3)
        assert sim_related > sim_unrelated

    def test_dimension_must_be_positive(self):
        """维度必须 >= 1"""
        with pytest.raises(ValueError, match="维度"):
            HashEmbeddingProvider(dim=0)

    def test_chinese_text(self):
        """支持中文文本"""
        provider = HashEmbeddingProvider(dim=64)
        vec = provider.embed("你好世界")
        assert len(vec) == 64
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm > 0  # 非空文本的向量不为零


class TestHybridMemoryWithProvider:
    """HybridMemory 使用 embedding_provider 的集成测试"""

    def test_remember_without_embedding_with_provider(self):
        """配置 provider 后，remember 不需要手动传 embedding"""
        from agentmemory.hybrid_memory import HybridMemory
        from agentmemory.embedding_provider import HashEmbeddingProvider

        provider = HashEmbeddingProvider(dim=16)
        hm = HybridMemory(dimension=16, embedding_provider=provider)

        mem = hm.remember("hello world")
        assert mem.content == "hello world"
        assert mem.embedding is not None
        assert len(mem.embedding) == 16

    def test_remember_with_explicit_embedding_overrides_provider(self):
        """手动传入的 embedding 覆盖 provider"""
        from agentmemory.hybrid_memory import HybridMemory
        from agentmemory.embedding_provider import HashEmbeddingProvider

        provider = HashEmbeddingProvider(dim=16)
        hm = HybridMemory(dimension=16, embedding_provider=provider)

        manual_vec = [1.0] + [0.0] * 15
        mem = hm.remember("hello", embedding=manual_vec)
        assert mem.embedding == manual_vec

    def test_search_with_text_query(self):
        """search_text 方法：文本查询"""
        from agentmemory.hybrid_memory import HybridMemory
        from agentmemory.embedding_provider import HashEmbeddingProvider

        provider = HashEmbeddingProvider(dim=16)
        hm = HybridMemory(dimension=16, embedding_provider=provider)

        hm.remember("cat")
        hm.remember("dog")
        hm.remember("kitten")

        results = hm.search_text("cat", top_k=2)
        assert len(results) == 2
        # 最相关的应该是 "cat"
        assert results[0].memory.content == "cat"

    def test_hybrid_search_text(self):
        """hybrid_search_text 方法：文本混合搜索"""
        from agentmemory.hybrid_memory import HybridMemory
        from agentmemory.embedding_provider import HashEmbeddingProvider

        provider = HashEmbeddingProvider(dim=16)
        hm = HybridMemory(dimension=16, embedding_provider=provider)

        hm.remember("Python is a programming language")
        hm.remember("Java is also popular")
        e1 = hm.add_entity("Python", "language")
        e2 = hm.add_entity("Alice", "person")
        hm.add_relation(e2.id, e1.id, "knows")

        results = hm.hybrid_search_text("Python", top_k=2, graph_depth=1)
        assert len(results) > 0

    def test_search_text_without_provider_raises(self):
        """没有配置 provider 时 search_text 应报错"""
        from agentmemory.hybrid_memory import HybridMemory

        hm = HybridMemory(dimension=3)
        with pytest.raises(ValueError, match="embedding_provider"):
            hm.search_text("hello")

    def test_auto_dimension_from_provider(self):
        """配置 provider 时 dimension 可以自动推断"""
        from agentmemory.hybrid_memory import HybridMemory
        from agentmemory.embedding_provider import HashEmbeddingProvider

        provider = HashEmbeddingProvider(dim=32)
        # 不传 dimension，让 provider 推断
        hm = HybridMemory(embedding_provider=provider)
        assert hm.dimension == 32

    def test_both_dimension_and_provider_requires_match(self):
        """同时传 dimension 和 provider 时维度必须匹配"""
        from agentmemory.hybrid_memory import HybridMemory
        from agentmemory.embedding_provider import HashEmbeddingProvider

        provider = HashEmbeddingProvider(dim=32)
        with pytest.raises(ValueError, match="维度不匹配"):
            HybridMemory(dimension=16, embedding_provider=provider)
