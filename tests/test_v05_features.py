"""v0.5.0 测试 — 多探针LSH、加权搜索、aiosqlite异步持久化、图谱推理、插件架构。"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
from typing import Optional

import pytest

from agentmemory.models import Memory, Entity, Relation, SearchResult
from agentmemory.embedding_store import EmbeddingStore, cosine_similarity
from agentmemory.embedding_provider import HashEmbeddingProvider
from agentmemory.knowledge_graph import KnowledgeGraph
from agentmemory.hybrid_memory import HybridMemory
from agentmemory.lsh_index import LSHIndex
from agentmemory.weighted_search import (
    WeightedScorer,
    ScoringWeights,
    weighted_search,
)
from agentmemory.plugins import PluginRegistry, get_registry
from agentmemory.async_persistence import AsyncSQLiteBackend


# =============================================================================
# 1. 多探针 LSH 测试
# =============================================================================

class TestMultiProbeLSH:
    """多探针 LSH 索引测试。"""

    def test_basic_add_query(self):
        """基本添加和查询"""
        lsh = LSHIndex(dimension=16, num_tables=4, num_hyperplanes=8)
        vec = [1.0] * 16
        lsh.add("v1", vec)
        assert lsh.size() == 1
        candidates = lsh.query(vec, max_candidates=10)
        assert "v1" in candidates

    def test_multi_probe_recall(self):
        """多探针应找到候选结果"""
        import random
        rng = random.Random(42)
        dim = 32

        # 创建索引
        lsh = LSHIndex(
            dimension=dim, num_tables=8, num_hyperplanes=10,
            max_probe_bits=3, min_candidates=5,
        )
        # 添加足够多的向量使哈希分布有意义
        vectors = {}
        for i in range(500):
            vec = [rng.gauss(0, 1) for _ in range(dim)]
            vectors[f"v{i}"] = vec
            lsh.add(f"v{i}", vec)

        # 查询应能找到候选
        query = vectors["v250"]
        candidates = lsh.query(query, max_candidates=100)
        assert len(candidates) > 0, "多探针 LSH 应能找到候选结果"
        assert "v250" in candidates, "查询自身应在候选集中"

        # 测试高探针数能工作
        lsh_high = LSHIndex(
            dimension=dim, num_tables=4, num_hyperplanes=10,
            max_probe_bits=5, min_candidates=10,
        )
        for vid, vec in vectors.items():
            lsh_high.add(vid, vec)
        candidates_high = lsh_high.query(query, max_candidates=100)
        assert len(candidates_high) > 0

    def test_stats(self):
        """统计信息"""
        lsh = LSHIndex(dimension=8, num_tables=3, num_hyperplanes=4)
        for i in range(10):
            lsh.add(f"v{i}", [float(i)] * 8)
        stats = lsh.stats()
        assert stats["num_vectors"] == 10
        assert stats["num_tables"] == 3
        assert stats["num_hyperplanes"] == 4
        assert stats["max_probe_bits"] == 3  # default
        assert stats["total_buckets"] > 0

    def test_remove_and_query(self):
        """删除后查询不应包含已删除向量"""
        lsh = LSHIndex(dimension=4, num_tables=2, num_hyperplanes=4)
        lsh.add("v1", [1.0, 0.0, 0.0, 0.0])
        lsh.add("v2", [0.0, 1.0, 0.0, 0.0])
        lsh.remove("v1")
        candidates = lsh.query([1.0, 0.0, 0.0, 0.0])
        assert "v1" not in candidates

    def test_dimension_mismatch(self):
        """维度不匹配应抛出异常"""
        lsh = LSHIndex(dimension=4)
        with pytest.raises(ValueError, match="维度不匹配"):
            lsh.add("v1", [1.0, 2.0])

    def test_max_probe_bits_validation(self):
        """max_probe_bits 参数验证"""
        with pytest.raises(ValueError, match="max_probe_bits"):
            LSHIndex(dimension=4, max_probe_bits=0)

    def test_empty_query(self):
        """空索引查询"""
        lsh = LSHIndex(dimension=4)
        candidates = lsh.query([1.0, 0.0, 0.0, 0.0])
        assert len(candidates) == 0

    def test_flip_bits(self):
        """翻转位操作"""
        lsh = LSHIndex(dimension=4, num_hyperplanes=4)
        result = lsh._flip_bits("1010", (0, 2))
        assert result == "0000"

    def test_probe_patterns_cached(self):
        """探针模式缓存"""
        lsh = LSHIndex(dimension=4, num_hyperplanes=4)
        patterns1 = lsh._get_probe_patterns(2)
        patterns2 = lsh._get_probe_patterns(2)
        assert patterns1 is patterns2  # 同一对象（缓存命中）

    def test_rebuild(self):
        """重建索引"""
        lsh = LSHIndex(dimension=4, num_tables=2, num_hyperplanes=4)
        for i in range(5):
            lsh.add(f"v{i}", [float(i), float(i), float(i), float(i)])
        lsh.rebuild()
        assert lsh.size() == 5

    def test_large_scale_recall(self):
        """大规模数据测试 — 确保多探针不返回空结果"""
        import random
        rng = random.Random(123)
        dim = 64

        lsh = LSHIndex(
            dimension=dim, num_tables=10, num_hyperplanes=12,
            max_probe_bits=3, min_candidates=5,
        )
        # 添加 1000 个向量
        vectors = {}
        for i in range(1000):
            vec = [rng.gauss(0, 1) for _ in range(dim)]
            vectors[f"v{i}"] = vec
            lsh.add(f"v{i}", vec)

        # 查询应该找到候选
        query = vectors["v0"]
        candidates = lsh.query(query, max_candidates=50)
        assert len(candidates) > 0, "多探针 LSH 在大规模数据下不应返回空结果"
        assert "v0" in candidates, "查询自身应在候选集中"


# =============================================================================
# 2. 加权搜索测试
# =============================================================================

class TestWeightedSearch:
    """加权搜索评分器测试。"""

    def test_scoring_weights_normalize(self):
        """权重归一化"""
        w = ScoringWeights(similarity=2, importance=1, recency=1, frequency=1)
        nw = w.normalize()
        total = nw.similarity + nw.importance + nw.recency + nw.frequency
        assert abs(total - 1.0) < 1e-10

    def test_scoring_weights_validation(self):
        """权重验证"""
        w = ScoringWeights(similarity=1.5)
        with pytest.raises(ValueError, match="必须在"):
            w.validate()

    def test_basic_scorer(self):
        """基本评分器功能"""
        scorer = WeightedScorer()
        mem = Memory(content="test memory", tags=["tag1", "tag2"])
        score = scorer.compute_score(mem, similarity_score=0.8)
        assert 0 <= score <= 1

    def test_importance_from_metadata(self):
        """从 metadata 读取重要性"""
        scorer = WeightedScorer()
        mem = Memory(content="important", metadata={"importance": 0.9})
        score = scorer._compute_importance_score(mem)
        assert score == 0.9

    def test_recency_score_decay(self):
        """时间新鲜度衰减"""
        scorer = WeightedScorer(half_life_hours=1.0)
        # 刚创建的记忆
        mem_new = Memory(content="new")
        # 旧记忆（模拟 2 小时前）
        mem_old = Memory(content="old")
        mem_old.created_at = time.time() - 7200

        score_new = scorer._compute_recency_score(mem_new)
        score_old = scorer._compute_recency_score(mem_old)
        assert score_new > score_old

    def test_frequency_score(self):
        """访问频率得分"""
        scorer = WeightedScorer()
        mem = Memory(content="frequently accessed")
        scorer.record_access(mem.id)
        scorer.record_access(mem.id)
        scorer.record_access(mem.id)
        score = scorer._compute_frequency_score(mem)
        assert score > 0

    def test_rerank(self):
        """重排序功能"""
        scorer = WeightedScorer(
            weights=ScoringWeights(similarity=1.0, importance=0.0, recency=0.0, frequency=0.0)
        )
        mem1 = Memory(content="low similarity")
        mem2 = Memory(content="high similarity")
        results = [
            SearchResult(memory=mem1, score=0.3),
            SearchResult(memory=mem2, score=0.9),
        ]
        reranked = scorer.rerank(results)
        assert reranked[0].memory.content == "high similarity"

    def test_weighted_search_convenience(self):
        """便捷函数"""
        mem = Memory(content="test")
        results = [SearchResult(memory=mem, score=0.8)]
        reranked = weighted_search(results)
        assert len(reranked) == 1

    def test_set_weights(self):
        """动态设置权重"""
        scorer = WeightedScorer()
        new_weights = ScoringWeights(similarity=0.3, importance=0.3, recency=0.3, frequency=0.1)
        scorer.set_weights(new_weights)
        assert abs(scorer.weights.similarity - 0.3) < 1e-10

    def test_get_access_count(self):
        """获取访问次数"""
        scorer = WeightedScorer()
        mem = Memory(content="test")
        assert scorer.get_access_count(mem.id) == 0
        scorer.record_access(mem.id)
        scorer.record_access(mem.id)
        assert scorer.get_access_count(mem.id) == 2


# =============================================================================
# 3. 知识图谱推理测试
# =============================================================================

class TestKnowledgeGraphReasoning:
    """知识图谱路径查找和推理测试。"""

    def _build_sample_graph(self) -> tuple[KnowledgeGraph, dict[str, Entity]]:
        """构建测试用图谱：A-B-C-D, A-E-D"""
        kg = KnowledgeGraph()
        a = Entity(name="A", entity_type="person")
        b = Entity(name="B", entity_type="person")
        c = Entity(name="C", entity_type="person")
        d = Entity(name="D", entity_type="person")
        e = Entity(name="E", entity_type="person")

        entities = {}
        for ent in [a, b, c, d, e]:
            kg.add_entity(ent)
            entities[ent.name] = ent

        # A-B, B-C, C-D, A-E, E-D
        kg.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=b.id, target_id=c.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=c.id, target_id=d.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=a.id, target_id=e.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=e.id, target_id=d.id, relation_type="knows"))

        return kg, entities

    def test_shortest_path(self):
        """最短路径"""
        kg, entities = self._build_sample_graph()
        path = kg.shortest_path(entities["A"].id, entities["D"].id)
        assert path is not None
        assert path[0].name == "A"
        assert path[-1].name == "D"
        # 最短路径应为 3 步（A-E-D 或 A-B-C-D 中的较短者）
        assert len(path) <= 4

    def test_shortest_path_same_node(self):
        """同节点最短路径"""
        kg, entities = self._build_sample_graph()
        path = kg.shortest_path(entities["A"].id, entities["A"].id)
        assert path is not None
        assert len(path) == 1

    def test_shortest_path_unreachable(self):
        """不可达节点"""
        kg = KnowledgeGraph()
        a = Entity(name="A", entity_type="person")
        b = Entity(name="B", entity_type="person")
        c = Entity(name="C", entity_type="person")
        kg.add_entity(a)
        kg.add_entity(b)
        kg.add_entity(c)
        kg.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="knows"))
        # C 孤立
        path = kg.shortest_path(a.id, c.id)
        assert path is None

    def test_find_all_paths(self):
        """查找所有路径"""
        kg, entities = self._build_sample_graph()
        paths = kg.find_all_paths(entities["A"].id, entities["D"].id, max_depth=5)
        assert len(paths) >= 2  # A-B-C-D 和 A-E-D

    def test_find_all_paths_max_paths(self):
        """限制路径数量"""
        kg, entities = self._build_sample_graph()
        paths = kg.find_all_paths(entities["A"].id, entities["D"].id, max_depth=5, max_paths=1)
        assert len(paths) == 1

    def test_common_neighbors(self):
        """共同邻居"""
        kg, entities = self._build_sample_graph()
        # A 和 D 的共同邻居应该是 B, C, E 中的某些
        # A 的邻居: B, E; D 的邻居: C, E
        # 共同: E
        common = kg.common_neighbors(entities["A"].id, entities["D"].id)
        common_names = {e.name for e in common}
        assert "E" in common_names

    def test_connected_components(self):
        """连通分量"""
        kg = KnowledgeGraph()
        a = Entity(name="A", entity_type="p")
        b = Entity(name="B", entity_type="p")
        c = Entity(name="C", entity_type="p")
        d = Entity(name="D", entity_type="p")
        for ent in [a, b, c, d]:
            kg.add_entity(ent)
        kg.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="r"))
        kg.add_relation(Relation(source_id=c.id, target_id=d.id, relation_type="r"))
        components = kg.connected_components()
        assert len(components) == 2

    def test_subgraph(self):
        """子图提取"""
        kg, entities = self._build_sample_graph()
        sub = kg.subgraph({entities["A"].id, entities["B"].id, entities["C"].id})
        assert len(sub["entities"]) == 3
        # A-B, B-C 两条关系
        assert len(sub["relations"]) == 2

    def test_shortest_path_nonexistent(self):
        """不存在的实体"""
        kg = KnowledgeGraph()
        with pytest.raises(KeyError):
            kg.shortest_path("nonexistent", "also_nonexistent")

    def test_common_neighbors_nonexistent(self):
        """不存在的实体"""
        kg = KnowledgeGraph()
        with pytest.raises(KeyError):
            kg.common_neighbors("x", "y")

    def test_find_all_paths_with_relation_filter(self):
        """按关系类型过滤路径"""
        kg = KnowledgeGraph()
        a = Entity(name="A", entity_type="person")
        b = Entity(name="B", entity_type="person")
        c = Entity(name="C", entity_type="person")
        for ent in [a, b, c]:
            kg.add_entity(ent)
        kg.add_relation(Relation(source_id=a.id, target_id=b.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=b.id, target_id=c.id, relation_type="works_at"))
        # 按 knows 过滤应只找到 A-B
        paths = kg.find_all_paths(a.id, c.id, relation_type="knows")
        assert len(paths) == 0  # C 不可达通过 knows


# =============================================================================
# 4. 插件架构测试
# =============================================================================

class TestPluginRegistry:
    """插件注册表测试。"""

    def test_register_backend(self):
        """注册后端"""
        registry = PluginRegistry()
        registry.register_backend("test_backend", dict)
        assert registry.get_backend("test_backend") is dict

    def test_register_provider(self):
        """注册 Provider"""
        registry = PluginRegistry()
        registry.register_provider("test_provider", list)
        assert registry.get_provider("test_provider") is list

    def test_register_scorer(self):
        """注册评分策略"""
        registry = PluginRegistry()
        registry.register_scorer("test_scorer", str)
        assert registry.get_scorer("test_scorer") is str

    def test_register_search_strategy(self):
        """注册搜索策略"""
        registry = PluginRegistry()
        def my_strategy(query, **kwargs):
            return []
        registry.register_search_strategy("custom", my_strategy)
        assert registry.get_search_strategy("custom") is my_strategy

    def test_duplicate_registration(self):
        """重复注册应抛出异常"""
        registry = PluginRegistry()
        registry.register_backend("test", dict)
        with pytest.raises(ValueError, match="已注册"):
            registry.register_backend("test", list)

    def test_list_all(self):
        """列出所有插件"""
        registry = PluginRegistry()
        registry.register_backend("json", dict)
        registry.register_provider("hash", list)
        all_plugins = registry.list_all()
        assert "json" in all_plugins["backends"]
        assert "hash" in all_plugins["providers"]

    def test_unregister(self):
        """取消注册"""
        registry = PluginRegistry()
        registry.register_backend("test", dict)
        assert registry.unregister("test") is True
        assert registry.get_backend("test") is None

    def test_unregister_nonexistent(self):
        """取消不存在的插件"""
        registry = PluginRegistry()
        assert registry.unregister("nonexistent") is False

    def test_global_registry(self):
        """全局注册表"""
        registry = get_registry()
        assert isinstance(registry, PluginRegistry)

    def test_list_empty(self):
        """空注册表"""
        registry = PluginRegistry()
        assert registry.list_backends() == []
        assert registry.list_providers() == []
        assert registry.list_scorers() == []
        assert registry.list_search_strategies() == []


# =============================================================================
# 5. aiosqlite 异步持久化测试
# =============================================================================

class TestAsyncSQLiteBackend:
    """aiosqlite 异步持久化后端测试。"""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return str(tmp_path / "test.db")

    @pytest.fixture
    def sample_store(self):
        store = EmbeddingStore(dimension=4)
        store.add(Memory(content="hello", embedding=[1.0, 0.0, 0.0, 0.0]))
        store.add(Memory(content="world", embedding=[0.0, 1.0, 0.0, 0.0]))
        return store

    @pytest.fixture
    def sample_kg(self):
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_relation(Relation(source_id=e1.id, target_id=e2.id, relation_type="knows"))
        return kg

    @pytest.mark.asyncio
    async def test_save_load_memories(self, tmp_db, sample_store):
        """异步保存和加载记忆"""
        backend = AsyncSQLiteBackend(tmp_db)
        await backend.save_embedding_store(sample_store)

        store2 = EmbeddingStore(dimension=4)
        await backend.load_embedding_store(store2)
        assert store2.count() == 2

    @pytest.mark.asyncio
    async def test_save_load_kg(self, tmp_db, sample_kg):
        """异步保存和加载知识图谱"""
        backend = AsyncSQLiteBackend(tmp_db)
        await backend.save_knowledge_graph(sample_kg)

        kg2 = KnowledgeGraph()
        await backend.load_knowledge_graph(kg2)
        assert kg2.entity_count() == 2
        assert kg2.relation_count() == 1

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, tmp_path):
        """加载不存在的数据库"""
        backend = AsyncSQLiteBackend(str(tmp_path / "nonexistent.db"))
        store = EmbeddingStore(dimension=4)
        await backend.load_embedding_store(store)
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, tmp_db, sample_store, sample_kg):
        """完整的保存-加载往返测试"""
        backend = AsyncSQLiteBackend(tmp_db)
        await backend.save_embedding_store(sample_store)
        await backend.save_knowledge_graph(sample_kg)

        store2 = EmbeddingStore(dimension=4)
        kg2 = KnowledgeGraph()
        await backend.load_embedding_store(store2)
        await backend.load_knowledge_graph(kg2)

        assert store2.count() == 2
        assert kg2.entity_count() == 2
        assert kg2.relation_count() == 1


# =============================================================================
# 6. 集成测试
# =============================================================================

class TestIntegration:
    """集成测试 — 各模块协同工作。"""

    def test_hybrid_memory_with_weighted_scoring(self):
        """HybridMemory + 加权搜索集成"""
        hm = HybridMemory(
            dimension=4,
            weighted_scoring=True,
            scoring_weights=ScoringWeights(similarity=0.6, importance=0.2, recency=0.1, frequency=0.1),
        )
        hm.remember("hello world", embedding=[1.0, 0.0, 0.0, 0.0], tags=["greeting"])
        hm.remember("goodbye world", embedding=[0.0, 1.0, 0.0, 0.0], tags=["farewell"])

        results = hm.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_hybrid_memory_scorer_set_get(self):
        """设置和获取评分器"""
        hm = HybridMemory(dimension=4)
        assert hm.get_scorer() is None

        scorer = WeightedScorer()
        hm.set_scorer(scorer)
        assert hm.get_scorer() is scorer

    def test_lsh_with_hybrid_memory(self):
        """LSH + HybridMemory 集成"""
        hm = HybridMemory(dimension=4, use_lsh=True, lsh_tables=4, lsh_hyperplanes=8)
        for i in range(20):
            vec = [float(i % 4), float((i + 1) % 4), float((i + 2) % 4), float((i + 3) % 4)]
            hm.remember(f"memory {i}", embedding=vec)

        results = hm.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) > 0

    def test_knowledge_graph_reasoning_with_hybrid(self):
        """知识图谱推理 + HybridMemory 集成"""
        hm = HybridMemory(dimension=4)
        e1 = hm.add_entity("Python", "language")
        e2 = hm.add_entity("FastAPI", "framework")
        hm.add_relation(e1.id, e2.id, "powers")

        path = hm.knowledge_graph.shortest_path(e1.id, e2.id)
        assert path is not None
        assert len(path) == 2

    def test_plugin_with_hybrid_memory(self):
        """插件系统 + HybridMemory 集成"""
        registry = get_registry()
        # 注册一个自定义评分器
        registry.register_scorer("test_custom", WeightedScorer)
        assert registry.get_scorer("test_custom") is WeightedScorer
        # 清理
        registry.unregister("test_custom")


# =============================================================================
# 7. 边界条件测试
# =============================================================================

class TestEdgeCases:
    """边界条件测试。"""

    def test_lsh_single_vector(self):
        """单向量 LSH"""
        lsh = LSHIndex(dimension=4, num_tables=2, num_hyperplanes=4)
        lsh.add("only", [1.0, 0.0, 0.0, 0.0])
        candidates = lsh.query([1.0, 0.0, 0.0, 0.0])
        assert "only" in candidates

    def test_weighted_scorer_empty_results(self):
        """空结果重排序"""
        scorer = WeightedScorer()
        results = scorer.rerank([])
        assert results == []

    def test_scoring_weights_zero_sum(self):
        """零权重归一化"""
        w = ScoringWeights(similarity=0, importance=0, recency=0, frequency=0)
        nw = w.normalize()
        # 应返回默认权重
        assert nw.similarity > 0

    def test_knowledge_graph_empty_subgraph(self):
        """空子图"""
        kg = KnowledgeGraph()
        sub = kg.subgraph(set())
        assert len(sub["entities"]) == 0
        assert len(sub["relations"]) == 0

    def test_knowledge_graph_single_entity_component(self):
        """单实体连通分量"""
        kg = KnowledgeGraph()
        e = Entity(name="Loner", entity_type="person")
        kg.add_entity(e)
        components = kg.connected_components()
        assert len(components) == 1
        assert len(components[0]) == 1

    def test_plugin_list_scorers_empty(self):
        """空评分策略列表"""
        registry = PluginRegistry()
        assert registry.list_scorers() == []

    def test_plugin_list_search_strategies_empty(self):
        """空搜索策略列表"""
        registry = PluginRegistry()
        assert registry.list_search_strategies() == []
