"""v0.6.0 功能测试 — 搜索缓存、图谱可视化、ChromaDB 后端、CLI 图谱推理"""

import json
import os
import tempfile
import time

import pytest

from agentmemory import (
    HybridMemory,
    HashEmbeddingProvider,
    SearchCache,
    KnowledgeGraph,
    Memory,
    Entity,
    Relation,
    SearchResult,
    export_dot,
    export_html,
    graph_stats_text,
    __version__,
)
from agentmemory.search_cache import SearchCache
from agentmemory.graph_viz import export_dot, export_html, graph_stats_text


# ============================================================
# SearchCache 测试
# ============================================================

class TestSearchCache:
    """搜索缓存核心功能测试"""

    def test_basic_put_get(self):
        """基本存取操作"""
        cache = SearchCache(max_size=10)
        mem = Memory(content="test")
        results = [SearchResult(memory=mem, score=0.9)]

        # 未命中
        assert cache.get("query") is None

        # 存入
        cache.put("query", results)

        # 命中
        cached = cache.get("query")
        assert cached is not None
        assert len(cached) == 1
        assert cached[0].score == 0.9

    def test_lru_eviction(self):
        """LRU 淘汰策略"""
        cache = SearchCache(max_size=3)
        for i in range(5):
            mem = Memory(content=f"text {i}")
            cache.put(f"q{i}", [SearchResult(memory=mem, score=0.5)])

        # 只有最后 3 个应保留
        assert cache.size == 3
        assert cache.get("q0") is None
        assert cache.get("q1") is None
        assert cache.get("q2") is not None
        assert cache.get("q3") is not None
        assert cache.get("q4") is not None

    def test_lru_order_update(self):
        """访问后条目移到末尾"""
        cache = SearchCache(max_size=3)
        for i in range(3):
            mem = Memory(content=f"text {i}")
            cache.put(f"q{i}", [SearchResult(memory=mem, score=0.5)])

        # 访问 q0 使其移到末尾
        cache.get("q0")

        # 添加 q3，应该淘汰 q1（最久未访问）
        mem = Memory(content="text 3")
        cache.put("q3", [SearchResult(memory=mem, score=0.5)])

        assert cache.get("q0") is not None  # 被访问过，保留
        assert cache.get("q1") is None  # 被淘汰
        assert cache.get("q2") is not None
        assert cache.get("q3") is not None

    def test_ttl_expiration(self):
        """TTL 过期机制"""
        cache = SearchCache(max_size=10, ttl_seconds=0.1)
        mem = Memory(content="test")
        cache.put("query", [SearchResult(memory=mem, score=0.8)])

        # 立即访问应命中
        assert cache.get("query") is not None

        # 等待过期
        time.sleep(0.15)
        assert cache.get("query") is None

    def test_different_params_different_keys(self):
        """不同参数生成不同缓存键"""
        cache = SearchCache(max_size=10)
        mem1 = Memory(content="result 1")
        mem2 = Memory(content="result 2")

        cache.put("query", [SearchResult(memory=mem1, score=0.9)], top_k=5)
        cache.put("query", [SearchResult(memory=mem2, score=0.8)], top_k=10)

        r5 = cache.get("query", top_k=5)
        r10 = cache.get("query", top_k=10)
        assert r5 is not None and r5[0].score == 0.9
        assert r10 is not None and r10[0].score == 0.8

    def test_tags_different_keys(self):
        """标签过滤生成不同缓存键"""
        cache = SearchCache(max_size=10)
        mem1 = Memory(content="tagged", tags=["important"])
        mem2 = Memory(content="untagged")

        cache.put("query", [SearchResult(memory=mem1, score=0.9)], tags=["important"])
        cache.put("query", [SearchResult(memory=mem2, score=0.7)])

        r_tagged = cache.get("query", tags=["important"])
        r_all = cache.get("query")
        assert r_tagged is not None and r_tagged[0].memory.tags == ["important"]
        assert r_all is not None and len(r_all[0].memory.tags) == 0

    def test_stats(self):
        """缓存统计"""
        cache = SearchCache(max_size=10)
        mem = Memory(content="test")

        # 2 misses, 1 hit
        cache.get("q1")
        cache.get("q1")
        cache.put("q1", [SearchResult(memory=mem, score=0.5)])
        cache.get("q1")

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert abs(stats["hit_rate"] - 1 / 3) < 0.001
        assert stats["size"] == 1
        assert stats["max_size"] == 10

    def test_clear(self):
        """清空缓存"""
        cache = SearchCache(max_size=10)
        for i in range(5):
            mem = Memory(content=f"text {i}")
            cache.put(f"q{i}", [SearchResult(memory=mem, score=0.5)])

        assert cache.size == 5
        cleared = cache.clear()
        assert cleared == 5
        assert cache.size == 0

    def test_invalidate(self):
        """使缓存条目失效"""
        cache = SearchCache(max_size=10)
        mem = Memory(content="test")
        cache.put("query", [SearchResult(memory=mem, score=0.5)])

        assert cache.get("query") is not None
        assert cache.invalidate("query") is True
        assert cache.get("query") is None
        assert cache.invalidate("nonexistent") is False

    def test_vector_query_cache(self):
        """向量查询缓存"""
        cache = SearchCache(max_size=10)
        mem = Memory(content="test")
        vec = [0.1] * 128

        cache.put(vec, [SearchResult(memory=mem, score=0.9)])
        cached = cache.get(vec)
        assert cached is not None
        assert cached[0].score == 0.9

    def test_max_size_must_be_positive(self):
        """max_size 必须 >= 1"""
        with pytest.raises(ValueError, match="max_size"):
            SearchCache(max_size=0)

    def test_extra_param(self):
        """extra 参数区分缓存键"""
        cache = SearchCache(max_size=10)
        mem1 = Memory(content="r1")
        mem2 = Memory(content="r2")

        cache.put("q", [SearchResult(memory=mem1, score=0.9)], extra="hybrid")
        cache.put("q", [SearchResult(memory=mem2, score=0.7)])

        assert cache.get("q", extra="hybrid")[0].score == 0.9
        assert cache.get("q")[0].score == 0.7


# ============================================================
# HybridMemory 搜索缓存集成测试
# ============================================================

class TestHybridMemoryCache:
    """HybridMemory 搜索缓存集成测试"""

    def test_cache_disabled_by_default(self):
        """默认不启用缓存"""
        hm = HybridMemory(dimension=128, embedding_provider=HashEmbeddingProvider(dim=128))
        assert hm.get_cache_stats() is None
        assert hm.clear_cache() == 0

    def test_cache_enabled(self):
        """启用缓存"""
        hm = HybridMemory(
            dimension=128,
            embedding_provider=HashEmbeddingProvider(dim=128),
            cache_size=64,
        )
        assert hm.get_cache_stats() is not None
        assert hm.get_cache_stats()["max_size"] == 64

    def test_search_text_uses_cache(self):
        """search_text 使用缓存"""
        hm = HybridMemory(
            dimension=128,
            embedding_provider=HashEmbeddingProvider(dim=128),
            cache_size=32,
        )
        hm.remember("Python programming", tags=["tech"])
        hm.remember("JavaScript web dev", tags=["tech"])

        # 第一次搜索
        r1 = hm.search_text("Python", top_k=2)
        stats1 = hm.get_cache_stats()
        assert stats1["misses"] == 1
        assert stats1["hits"] == 0

        # 第二次相同搜索（应命中缓存）
        r2 = hm.search_text("Python", top_k=2)
        stats2 = hm.get_cache_stats()
        assert stats2["hits"] == 1
        assert stats2["misses"] == 1

        # 结果应相同
        assert len(r1) == len(r2)
        assert r1[0].memory.id == r2[0].memory.id

    def test_cache_invalidation_on_remember(self):
        """新记忆不自动清除缓存（查询结果仍然有效）"""
        hm = HybridMemory(
            dimension=128,
            embedding_provider=HashEmbeddingProvider(dim=128),
            cache_size=32,
        )
        hm.remember("Python", tags=["tech"])
        r1 = hm.search_text("Python")
        r2 = hm.search_text("Python")
        assert hm.get_cache_stats()["hits"] == 1

    def test_clear_cache(self):
        """清空缓存"""
        hm = HybridMemory(
            dimension=128,
            embedding_provider=HashEmbeddingProvider(dim=128),
            cache_size=32,
        )
        hm.remember("test")
        hm.search_text("test")
        assert hm.clear_cache() > 0
        assert hm.get_cache_stats()["size"] == 0

    def test_cache_with_ttl(self):
        """带 TTL 的缓存"""
        hm = HybridMemory(
            dimension=128,
            embedding_provider=HashEmbeddingProvider(dim=128),
            cache_size=32,
            cache_ttl=0.1,
        )
        hm.remember("test content")
        hm.search_text("test")
        assert hm.get_cache_stats()["hits"] == 0

        time.sleep(0.15)
        hm.search_text("test")
        # TTL expired, so both are misses
        assert hm.get_cache_stats()["misses"] == 2


# ============================================================
# 图谱可视化测试
# ============================================================

class TestGraphViz:
    """图谱可视化测试"""

    def _build_graph(self) -> KnowledgeGraph:
        """构建测试图谱"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Python", entity_type="language")
        e3 = Entity(name="Bob", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_entity(e3)
        kg.add_relation(Relation(source_id=e1.id, target_id=e2.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=e3.id, target_id=e2.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=e1.id, target_id=e3.id, relation_type="friend", weight=0.8))
        return kg

    def test_export_dot_basic(self):
        """DOT 导出基本功能"""
        kg = self._build_graph()
        dot = export_dot(kg, title="Test Graph")
        assert "digraph" in dot
        assert "Test Graph" in dot
        assert "Alice" in dot
        assert "Python" in dot
        assert "knows" in dot
        assert "friend" in dot

    def test_export_dot_weight(self):
        """DOT 导出包含权重"""
        kg = self._build_graph()
        dot = export_dot(kg)
        assert "0.8" in dot  # weight=0.8 的关系

    def test_export_dot_custom_colors(self):
        """DOT 自定义颜色"""
        kg = self._build_graph()
        dot = export_dot(kg, type_colors={"person": "#FF0000"})
        assert "#FF0000" in dot

    def test_export_dot_no_properties(self):
        """DOT 不显示属性"""
        kg = KnowledgeGraph()
        e = Entity(name="Test", entity_type="concept", properties={"key": "value"})
        kg.add_entity(e)
        dot = export_dot(kg, show_properties=False)
        assert "key" not in dot

    def test_export_dot_empty_graph(self):
        """空图谱 DOT 导出"""
        kg = KnowledgeGraph()
        dot = export_dot(kg)
        assert "digraph" in dot

    def test_export_html_basic(self):
        """HTML 导出基本功能"""
        kg = self._build_graph()
        html = export_html(kg, title="Test Graph")
        assert "vis-network" in html
        assert "Test Graph" in html
        assert "Alice" in html
        assert "Entities: 3" in html
        assert "Relations: 3" in html

    def test_export_html_empty_graph(self):
        """空图谱 HTML 导出"""
        kg = KnowledgeGraph()
        html = export_html(kg)
        assert "vis-network" in html
        assert "Entities: 0" in html

    def test_graph_stats_text(self):
        """图谱统计文本"""
        kg = self._build_graph()
        stats = graph_stats_text(kg)
        assert "Entities: 3" in stats
        assert "Relations: 3" in stats
        assert "person" in stats
        assert "language" in stats
        assert "knows" in stats
        assert "friend" in stats
        assert "Connected Components" in stats

    def test_graph_stats_empty(self):
        """空图谱统计"""
        kg = KnowledgeGraph()
        stats = graph_stats_text(kg)
        assert "Entities: 0" in stats


# ============================================================
# HybridMemory 图谱推理集成测试
# ============================================================

class TestHybridMemoryGraphReasoning:
    """HybridMemory 图谱推理方法测试"""

    def _build_hm(self) -> HybridMemory:
        """构建带图谱的 HybridMemory"""
        hm = HybridMemory(dimension=128, embedding_provider=HashEmbeddingProvider(dim=128))
        e1 = hm.add_entity("Alice", "person")
        e2 = hm.add_entity("Python", "language")
        e3 = hm.add_entity("Bob", "person")
        e4 = hm.add_entity("Rust", "language")
        hm.add_relation(e1.id, e2.id, "knows")
        hm.add_relation(e3.id, e2.id, "knows")
        hm.add_relation(e3.id, e4.id, "knows")
        hm.add_relation(e1.id, e3.id, "friend")
        return hm

    def test_shortest_path(self):
        """最短路径"""
        hm = self._build_hm()
        entities = hm.knowledge_graph.find_entities()
        names = {e.name: e.id for e in entities}

        path = hm.shortest_path(names["Alice"], names["Rust"])
        assert path is not None
        path_names = [e.name for e in path]
        assert path_names[0] == "Alice"
        assert path_names[-1] == "Rust"

    def test_shortest_path_same_entity(self):
        """相同实体的最短路径"""
        hm = self._build_hm()
        entities = hm.knowledge_graph.find_entities()
        alice = [e for e in entities if e.name == "Alice"][0]
        path = hm.shortest_path(alice.id, alice.id)
        assert len(path) == 1
        assert path[0].name == "Alice"

    def test_shortest_path_unreachable(self):
        """不可达实体"""
        hm = HybridMemory(dimension=128, embedding_provider=HashEmbeddingProvider(dim=128))
        e1 = hm.add_entity("A", "type")
        e2 = hm.add_entity("B", "type")
        path = hm.shortest_path(e1.id, e2.id)
        assert path is None

    def test_find_all_paths(self):
        """查找所有路径"""
        hm = self._build_hm()
        entities = hm.knowledge_graph.find_entities()
        names = {e.name: e.id for e in entities}

        paths = hm.find_all_paths(names["Alice"], names["Python"])
        assert len(paths) >= 2  # 直接 + 通过 Bob

    def test_common_neighbors(self):
        """共同邻居"""
        hm = self._build_hm()
        entities = hm.knowledge_graph.find_entities()
        names = {e.name: e.id for e in entities}

        # Alice 和 Bob 的共同邻居应包含 Python
        common = hm.common_neighbors(names["Alice"], names["Bob"])
        common_names = [e.name for e in common]
        assert "Python" in common_names

    def test_connected_components(self):
        """连通分量"""
        hm = self._build_hm()
        components = hm.connected_components()
        assert len(components) == 1  # 全连通

    def test_connected_components_multiple(self):
        """多个连通分量"""
        hm = HybridMemory(dimension=128, embedding_provider=HashEmbeddingProvider(dim=128))
        e1 = hm.add_entity("A", "type")
        e2 = hm.add_entity("B", "type")
        e3 = hm.add_entity("C", "type")
        hm.add_relation(e1.id, e2.id, "related")

        components = hm.connected_components()
        assert len(components) == 2

    def test_subgraph(self):
        """子图提取"""
        hm = self._build_hm()
        entities = hm.knowledge_graph.find_entities()
        names = {e.name: e.id for e in entities}

        sub = hm.subgraph({names["Alice"], names["Python"]})
        assert len(sub["entities"]) == 2
        assert len(sub["relations"]) == 1

    def test_export_dot(self):
        """HybridMemory 导出 DOT"""
        hm = self._build_hm()
        dot = hm.export_dot(title="Test")
        assert "digraph" in dot
        assert "Alice" in dot

    def test_export_html(self):
        """HybridMemory 导出 HTML"""
        hm = self._build_hm()
        html = hm.export_html(title="Test")
        assert "vis-network" in html


# ============================================================
# ChromaDB 后端测试
# ============================================================

class TestChromaDBBackend:
    """ChromaDB 后端测试（需要 chromadb 安装）"""

    def test_import_check(self):
        """检查导入是否正常（不依赖 chromadb 安装）"""
        try:
            from agentmemory.chromadb_backend import ChromaDBBackend, register_chromadb_plugin
            has_chroma = True
        except ImportError:
            has_chroma = False

        if not has_chroma:
            pytest.skip("chromadb not installed")

    def test_register_plugin(self):
        """测试插件注册（不依赖 chromadb 安装）"""
        try:
            from agentmemory.chromadb_backend import register_chromadb_plugin
            from agentmemory.plugins import PluginRegistry
            registry = PluginRegistry()
            # 手动调用应该可以（如果 chromadb 可用）
        except ImportError:
            pytest.skip("chromadb not installed")


# ============================================================
# CLI 新命令测试
# ============================================================

class TestCLIGraphReasoning:
    """CLI 图谱推理命令测试"""

    def test_cli_shortest_path(self):
        """CLI shortest-path 命令"""
        from agentmemory.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "data")
            # 先添加实体和关系
            cli_main(["--store", store_path, "add-entity", "Alice", "person"])
            cli_main(["--store", store_path, "add-entity", "Python", "language"])

            # 获取实体 ID
            from agentmemory import HybridMemory, HashEmbeddingProvider
            hm = HybridMemory(
                dimension=128,
                embedding_provider=HashEmbeddingProvider(dim=128),
                storage_path=store_path,
                auto_load=True,
            )
            entities = hm.knowledge_graph.find_entities()
            names = {e.name: e.id for e in entities}

            cli_main(["--store", store_path, "add-relation", names["Alice"], names["Python"], "knows"])

            # 测试 shortest-path
            cli_main(["--store", store_path, "shortest-path", names["Alice"], names["Python"]])

    def test_cli_common_neighbors(self):
        """CLI common-neighbors 命令"""
        from agentmemory.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "data")
            cli_main(["--store", store_path, "add-entity", "A", "type"])
            cli_main(["--store", store_path, "add-entity", "B", "type"])
            cli_main(["--store", store_path, "add-entity", "C", "type"])

            from agentmemory import HybridMemory, HashEmbeddingProvider
            hm = HybridMemory(
                dimension=128,
                embedding_provider=HashEmbeddingProvider(dim=128),
                storage_path=store_path,
                auto_load=True,
            )
            entities = hm.knowledge_graph.find_entities()
            names = {e.name: e.id for e in entities}

            cli_main(["--store", store_path, "add-relation", names["A"], names["C"], "related"])
            cli_main(["--store", store_path, "add-relation", names["B"], names["C"], "related"])
            cli_main(["--store", store_path, "common-neighbors", names["A"], names["B"]])

    def test_cli_connected_components(self):
        """CLI connected-components 命令"""
        from agentmemory.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "data")
            cli_main(["--store", store_path, "add-entity", "X", "type"])
            cli_main(["--store", store_path, "add-entity", "Y", "type"])
            cli_main(["--store", store_path, "connected-components"])

    def test_cli_graph_export(self):
        """CLI graph-export 命令"""
        from agentmemory.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "data")
            output_file = os.path.join(tmpdir, "graph.html")

            cli_main(["--store", store_path, "add-entity", "Test", "concept"])
            cli_main(["--store", store_path, "graph-export", "--output", output_file])

            assert os.path.exists(output_file)
            with open(output_file) as f:
                content = f.read()
            assert "vis-network" in content

    def test_cli_graph_stats(self):
        """CLI graph-stats 命令"""
        from agentmemory.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "data")
            cli_main(["--store", store_path, "add-entity", "Node", "type"])
            cli_main(["--store", store_path, "graph-stats"])

    def test_cli_cache_stats_disabled(self):
        """CLI cache-stats（缓存未启用）"""
        from agentmemory.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            cli_main(["cache-stats"])


# ============================================================
# 版本和集成测试
# ============================================================

class TestVersionAndIntegration:
    """版本和集成测试"""

    def test_version(self):
        """版本号"""
        assert __version__ == "0.6.0"

    def test_full_workflow(self):
        """完整工作流：记忆+图谱+缓存+可视化"""
        hm = HybridMemory(
            dimension=128,
            embedding_provider=HashEmbeddingProvider(dim=128),
            cache_size=16,
        )

        # 添加记忆
        hm.remember("Alice is a Python developer", tags=["person", "tech"])
        hm.remember("Bob writes Rust", tags=["person", "tech"])
        hm.remember("Python is great for ML", tags=["tech"])

        # 添加实体关系
        alice = hm.add_entity("Alice", "person", {"role": "developer"})
        python = hm.add_entity("Python", "language")
        bob = hm.add_entity("Bob", "person")
        hm.add_relation(alice.id, python.id, "knows")
        hm.add_relation(bob.id, python.id, "knows")
        hm.add_relation(alice.id, bob.id, "colleague")

        # 搜索（带缓存）
        r1 = hm.search_text("Python developer", top_k=3)
        r2 = hm.search_text("Python developer", top_k=3)
        assert hm.get_cache_stats()["hits"] == 1

        # 图谱推理
        path = hm.shortest_path(alice.id, bob.id)
        assert path is not None

        common = hm.common_neighbors(alice.id, bob.id)
        assert len(common) >= 1

        # 可视化导出
        dot = hm.export_dot()
        assert "Alice" in dot
        html_str = hm.export_html()
        assert "vis-network" in html_str

        # 统计
        stats = hm.stats()
        assert stats["memory_count"] == 3
        assert stats["entity_count"] == 3

    def test_persistence_with_new_features(self):
        """带新功能的持久化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "data")

            # 创建并保存
            hm = HybridMemory(
                dimension=128,
                embedding_provider=HashEmbeddingProvider(dim=128),
                storage_path=store_path,
                storage_backend="json",
                auto_save=True,
            )
            hm.remember("test memory", tags=["test"])
            hm.add_entity("TestEntity", "concept")
            hm.save()

            # 加载到新实例
            hm2 = HybridMemory(
                dimension=128,
                embedding_provider=HashEmbeddingProvider(dim=128),
                storage_path=store_path,
                storage_backend="json",
                auto_load=True,
            )
            assert len(hm2.list_all()) == 1
            assert hm2.knowledge_graph.entity_count() == 1

            # 可视化导出应正常
            dot = hm2.export_dot()
            assert "TestEntity" in dot
