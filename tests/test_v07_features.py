"""v0.7.0 测试 — 向量量化、RAG 管道、可观测性指标集成测试。"""

from __future__ import annotations

import json
import math
import time

import pytest

from agentmemory.embedding_provider import HashEmbeddingProvider
from agentmemory.hybrid_memory import HybridMemory
from agentmemory.metrics import (
    Counter,
    Timer,
    Gauge,
    MetricsCollector,
    HealthChecker,
    HealthStatus,
    HealthCheck,
    HealthReport,
    check_memory_health,
    check_lsh_health,
)
from agentmemory.models import Memory, SearchResult
from agentmemory.rag_pipeline import (
    RAGPipeline,
    Reranker,
    RAGContext,
    RAGResult,
    ContextStrategy,
    estimate_tokens,
)
from agentmemory.vector_quantizer import (
    ScalarQuantizer,
    ProductQuantizer,
    CompressedVectorStore,
    QuantizationStats,
)


# ===========================================================================
# 向量量化 (VectorQuantizer) 测试
# ===========================================================================


class TestScalarQuantizer:
    """ScalarQuantizer (SQ8) 测试"""

    def test_init_valid(self):
        sq = ScalarQuantizer(128)
        assert sq.dim == 128
        assert not sq.fitted

    def test_init_invalid_dim(self):
        with pytest.raises(ValueError, match="正整数"):
            ScalarQuantizer(0)
        with pytest.raises(ValueError, match="正整数"):
            ScalarQuantizer(-1)

    def test_fit_basic(self):
        sq = ScalarQuantizer(4)
        vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        result = sq.fit(vectors)
        assert result is sq  # 链式调用
        assert sq.fitted

    def test_fit_empty_raises(self):
        sq = ScalarQuantizer(4)
        with pytest.raises(ValueError, match="不能为空"):
            sq.fit([])

    def test_fit_dim_mismatch_raises(self):
        sq = ScalarQuantizer(4)
        with pytest.raises(ValueError, match="维度不匹配"):
            sq.fit([[1.0, 2.0]])  # dim=2 != 4

    def test_quantize_dequantize_roundtrip(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]])
        original = [5.0, 5.0, 5.0, 5.0]
        compressed = sq.quantize(original)
        assert isinstance(compressed, bytes)
        assert len(compressed) == 4
        restored = sq.dequantize(compressed)
        # 允许一定误差（量化损失）
        for o, r in zip(original, restored):
            assert abs(o - r) < 1.0

    def test_quantize_not_fitted_raises(self):
        sq = ScalarQuantizer(4)
        with pytest.raises(RuntimeError, match="fit"):
            sq.quantize([1.0, 2.0, 3.0, 4.0])

    def test_dequantize_not_fitted_raises(self):
        sq = ScalarQuantizer(4)
        with pytest.raises(RuntimeError, match="fit"):
            sq.dequantize(b"\x00\x00\x00\x00")

    def test_quantize_dim_mismatch_raises(self):
        sq = ScalarQuantizer(4)
        sq.fit([[1.0, 2.0, 3.0, 4.0]])
        with pytest.raises(ValueError, match="维度不匹配"):
            sq.quantize([1.0, 2.0])

    def test_dequantize_length_mismatch_raises(self):
        sq = ScalarQuantizer(4)
        sq.fit([[1.0, 2.0, 3.0, 4.0]])
        with pytest.raises(ValueError, match="长度不匹配"):
            sq.dequantize(b"\x00\x00")

    def test_auto_fit(self):
        sq = ScalarQuantizer(4)
        sq.auto_fit([1.0, 2.0, 3.0, 4.0])
        assert sq.fitted
        sq.auto_fit([5.0, 6.0, 7.0, 8.0])
        compressed = sq.quantize([3.0, 4.0, 5.0, 6.0])
        assert len(compressed) == 4

    def test_auto_fit_dim_mismatch_raises(self):
        sq = ScalarQuantizer(4)
        with pytest.raises(ValueError, match="维度不匹配"):
            sq.auto_fit([1.0, 2.0])

    def test_batch_quantize_dequantize(self):
        sq = ScalarQuantizer(4)
        vectors = [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(10)]
        sq.fit(vectors)
        compressed = sq.quantize_batch(vectors)
        assert len(compressed) == 10
        restored = sq.dequantize_batch(compressed)
        assert len(restored) == 10

    def test_compressed_size(self):
        sq = ScalarQuantizer(128)
        assert sq.compressed_size() == 128  # 1 byte per dim

    def test_stats(self):
        sq = ScalarQuantizer(128)
        sq.fit([[0.0] * 128])
        stats = sq.stats()
        assert stats.original_dim == 128
        assert stats.compressed_size_bytes == 128
        assert stats.compression_ratio == pytest.approx(4.0)  # float32=4bytes, sq8=1byte
        assert stats.method == "SQ8"

    def test_constant_dimension_handling(self):
        """常量维度（所有向量在某维度相同）不崩溃"""
        sq = ScalarQuantizer(3)
        sq.fit([[1.0, 5.0, 1.0], [1.0, 3.0, 1.0]])
        compressed = sq.quantize([1.0, 4.0, 1.0])
        restored = sq.dequantize(compressed)
        assert len(restored) == 3


class TestProductQuantizer:
    """ProductQuantizer (PQ) 测试"""

    def test_init_valid(self):
        pq = ProductQuantizer(dim=16, num_subspaces=4, num_centroids=32)
        assert pq.dim == 16
        assert pq.num_subspaces == 4
        assert pq.num_centroids == 32
        assert not pq.fitted

    def test_init_invalid_dim(self):
        with pytest.raises(ValueError, match="正整数"):
            ProductQuantizer(dim=0)

    def test_init_not_divisible(self):
        with pytest.raises(ValueError, match="整除"):
            ProductQuantizer(dim=15, num_subspaces=4)

    def test_init_invalid_centroids(self):
        with pytest.raises(ValueError, match="1~256"):
            ProductQuantizer(dim=8, num_centroids=0)
        with pytest.raises(ValueError, match="1~256"):
            ProductQuantizer(dim=8, num_centroids=300)

    def test_fit_basic(self):
        rng = __import__("random").Random(42)
        vectors = [[rng.random() for _ in range(16)] for _ in range(50)]
        pq = ProductQuantizer(dim=16, num_subspaces=4, num_centroids=8)
        result = pq.fit(vectors, seed=42)
        assert result is pq
        assert pq.fitted

    def test_fit_empty_raises(self):
        pq = ProductQuantizer(dim=8, num_subspaces=2)
        with pytest.raises(ValueError, match="不能为空"):
            pq.fit([])

    def test_quantize_dequantize_roundtrip(self):
        rng = __import__("random").Random(42)
        vectors = [[rng.random() for _ in range(16)] for _ in range(100)]
        pq = ProductQuantizer(dim=16, num_subspaces=4, num_centroids=16)
        pq.fit(vectors, seed=42)

        original = vectors[0]
        compressed = pq.quantize(original)
        assert isinstance(compressed, bytes)
        assert len(compressed) == 4  # num_subspaces
        restored = pq.dequantize(compressed)
        assert len(restored) == 16

    def test_quantize_not_fitted_raises(self):
        pq = ProductQuantizer(dim=8, num_subspaces=2)
        with pytest.raises(RuntimeError, match="fit"):
            pq.quantize([1.0] * 8)

    def test_batch_operations(self):
        rng = __import__("random").Random(42)
        vectors = [[rng.random() for _ in range(8)] for _ in range(50)]
        pq = ProductQuantizer(dim=8, num_subspaces=2, num_centroids=8)
        pq.fit(vectors, seed=42)
        compressed = pq.quantize_batch(vectors[:10])
        restored = pq.dequantize_batch(compressed)
        assert len(restored) == 10

    def test_compressed_size(self):
        pq = ProductQuantizer(dim=16, num_subspaces=4)
        assert pq.compressed_size() == 4

    def test_stats(self):
        rng = __import__("random").Random(42)
        vectors = [[rng.random() for _ in range(16)] for _ in range(50)]
        pq = ProductQuantizer(dim=16, num_subspaces=4, num_centroids=16)
        pq.fit(vectors, seed=42)
        stats = pq.stats()
        assert stats.original_dim == 16
        assert stats.compressed_size_bytes == 4
        assert stats.compression_ratio == pytest.approx(16.0)  # 16*4/4 = 16x
        assert "PQ" in stats.method


class TestCompressedVectorStore:
    """CompressedVectorStore 测试"""

    def test_add_get(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]])
        store = CompressedVectorStore(sq)
        store.add("v1", [5.0, 5.0, 5.0, 5.0])
        assert store.size == 1
        restored = store.get("v1")
        assert restored is not None
        assert len(restored) == 4

    def test_get_nonexistent(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0] * 4])
        store = CompressedVectorStore(sq)
        assert store.get("missing") is None

    def test_get_compressed(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0] * 4, [10.0] * 4])
        store = CompressedVectorStore(sq)
        store.add("v1", [5.0] * 4)
        raw = store.get_compressed("v1")
        assert isinstance(raw, bytes)
        assert len(raw) == 4

    def test_remove(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0] * 4])
        store = CompressedVectorStore(sq)
        store.add("v1", [1.0] * 4)
        assert store.remove("v1")
        assert store.size == 0
        assert not store.remove("v1")

    def test_contains(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0] * 4])
        store = CompressedVectorStore(sq)
        store.add("v1", [1.0] * 4)
        assert store.contains("v1")
        assert not store.contains("v2")

    def test_list_ids(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0] * 4])
        store = CompressedVectorStore(sq)
        store.add("v1", [1.0] * 4)
        store.add("v2", [2.0] * 4)
        ids = store.list_ids()
        assert set(ids) == {"v1", "v2"}

    def test_memory_usage(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0] * 4])
        store = CompressedVectorStore(sq)
        assert store.memory_usage_bytes() == 0
        store.add("v1", [1.0] * 4)
        assert store.memory_usage_bytes() == 4

    def test_stats(self):
        sq = ScalarQuantizer(4)
        sq.fit([[0.0] * 4, [10.0] * 4])
        store = CompressedVectorStore(sq)
        store.add("v1", [5.0] * 4)
        stats = store.stats()
        assert stats["num_vectors"] == 1
        assert "compression_ratio" in stats

    def test_with_pq(self):
        rng = __import__("random").Random(42)
        vectors = [[rng.random() for _ in range(8)] for _ in range(50)]
        pq = ProductQuantizer(dim=8, num_subspaces=2, num_centroids=8)
        pq.fit(vectors, seed=42)
        store = CompressedVectorStore(pq)
        store.add("v1", vectors[0])
        restored = store.get("v1")
        assert len(restored) == 8


# ===========================================================================
# RAG Pipeline 测试
# ===========================================================================


class TestEstimateTokens:
    """token 估算测试"""

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_english(self):
        tokens = estimate_tokens("hello world test")
        assert tokens > 0

    def test_chinese(self):
        tokens = estimate_tokens("你好世界测试")
        assert tokens > 0

    def test_mixed(self):
        tokens = estimate_tokens("hello 你好 world 世界")
        assert tokens > 0


class TestReranker:
    """Reranker 测试"""

    def _make_result(self, score: float, content: str, age_offset: float = 0) -> SearchResult:
        mem = Memory(content=content, created_at=time.time() - age_offset)
        return SearchResult(memory=mem, score=score)

    def test_empty(self):
        r = Reranker()
        assert r.rerank([]) == []

    def test_basic_rerank(self):
        r = Reranker(freshness_weight=0.0)
        results = [
            self._make_result(0.5, "a"),
            self._make_result(0.9, "b"),
            self._make_result(0.7, "c"),
        ]
        reranked = r.rerank(results)
        assert len(reranked) == 3
        assert reranked[0].memory.content == "b"

    def test_freshness_bias(self):
        r = Reranker(freshness_weight=0.9)
        results = [
            self._make_result(0.9, "old", age_offset=3600),
            self._make_result(0.5, "new", age_offset=0),
        ]
        reranked = r.rerank(results)
        # 新的应该排更高
        assert reranked[0].memory.content == "new"

    def test_min_score_filter(self):
        r = Reranker(min_score=0.5)
        results = [
            self._make_result(0.3, "low"),
            self._make_result(0.8, "high"),
        ]
        reranked = r.rerank(results)
        assert len(reranked) == 1
        assert reranked[0].memory.content == "high"

    def test_max_results(self):
        r = Reranker()
        results = [self._make_result(float(i) / 10, f"m{i}") for i in range(10)]
        reranked = r.rerank(results, max_results=3)
        assert len(reranked) == 3

    def test_invalid_weights(self):
        with pytest.raises(ValueError):
            Reranker(freshness_weight=1.5)
        with pytest.raises(ValueError):
            Reranker(diversity_weight=-0.1)

    def test_rerank_diversified_empty(self):
        r = Reranker()
        assert r.rerank_diversified([]) == []

    def test_rerank_diversified_no_embeddings(self):
        r = Reranker(diversity_weight=0.5)
        results = [
            self._make_result(0.9, "a"),
            self._make_result(0.8, "b"),
        ]
        # 没有向量时降级为普通排序
        reranked = r.rerank_diversified(results)
        assert len(reranked) == 2


class TestRAGPipeline:
    """RAGPipeline 测试"""

    def _make_memory(self):
        dim = 64
        provider = HashEmbeddingProvider(dim=dim)
        mem = HybridMemory(dimension=dim, embedding_provider=provider)
        mem.remember("Python 是一种编程语言", tags=["python"])
        mem.remember("机器学习是 AI 的子领域", tags=["ml"])
        mem.remember("深度学习使用神经网络", tags=["dl"])
        mem.remember("知识图谱存储实体和关系", tags=["kg"])
        mem.remember("RAG 检索增强生成技术", tags=["rag"])
        return mem

    def test_pipeline_basic(self):
        memory = self._make_memory()
        pipeline = RAGPipeline(memory=memory, top_k=3)
        result = pipeline.run("什么是编程语言")
        assert isinstance(result, RAGResult)
        assert result.query == "什么是编程语言"
        assert result.pipeline_time_ms > 0
        assert len(result.search_results) > 0
        assert "编程" in result.prompt or "编程" in result.context.text

    def test_pipeline_with_reranker(self):
        memory = self._make_memory()
        reranker = Reranker(freshness_weight=0.3)
        pipeline = RAGPipeline(memory=memory, top_k=3, reranker=reranker)
        result = pipeline.run("AI 技术")
        assert result.reranked

    def test_pipeline_run_with_sources(self):
        memory = self._make_memory()
        pipeline = RAGPipeline(memory=memory, top_k=2)
        prompt, sources = pipeline.run_with_sources("编程")
        assert isinstance(prompt, str)
        assert isinstance(sources, list)

    def test_pipeline_max_tokens(self):
        memory = self._make_memory()
        pipeline = RAGPipeline(memory=memory, top_k=10, max_context_tokens=50)
        result = pipeline.run("test")
        assert result.context.truncated or result.context.total_tokens <= 50

    def test_pipeline_strategies(self):
        memory = self._make_memory()
        for strategy in ContextStrategy:
            pipeline = RAGPipeline(
                memory=memory,
                top_k=3,
                context_strategy=strategy,
            )
            result = pipeline.run("test")
            assert isinstance(result.context, RAGContext)

    def test_pipeline_custom_template(self):
        memory = self._make_memory()
        template = "Context: {context}\nQuestion: {query}\nAnswer:"
        pipeline = RAGPipeline(memory=memory, prompt_template=template)
        result = pipeline.run("test")
        assert "Context:" in result.prompt
        assert "Question:" in result.prompt

    def test_pipeline_retrieve_only(self):
        memory = self._make_memory()
        pipeline = RAGPipeline(memory=memory, top_k=3)
        results = pipeline.retrieve("编程")
        assert isinstance(results, list)

    def test_pipeline_hybrid_search(self):
        memory = self._make_memory()
        pipeline = RAGPipeline(memory=memory, top_k=3)
        results = pipeline.retrieve("编程", use_hybrid=True)
        assert isinstance(results, list)

    def test_pipeline_min_score_filter(self):
        memory = self._make_memory()
        pipeline = RAGPipeline(memory=memory, min_score=0.99)
        result = pipeline.run("completely unrelated xyz query 12345")
        # 高阈值应该过滤掉大多数结果
        assert len(result.search_results) <= 5


# ===========================================================================
# 可观测性指标 (Metrics) 测试
# ===========================================================================


class TestCounter:
    """Counter 测试"""

    def test_init(self):
        c = Counter("test", "desc")
        assert c.value == 0
        assert c.name == "test"
        assert c.description == "desc"

    def test_init_with_value(self):
        c = Counter("test", "", initial_value=10)
        assert c.value == 10

    def test_increment(self):
        c = Counter("test")
        c.increment()
        assert c.value == 1
        c.increment(5)
        assert c.value == 6

    def test_decrement(self):
        c = Counter("test", "", initial_value=10)
        c.decrement(3)
        assert c.value == 7

    def test_reset(self):
        c = Counter("test", "", initial_value=10)
        c.reset()
        assert c.value == 0
        c.reset(42)
        assert c.value == 42


class TestTimer:
    """Timer 测试"""

    def test_init(self):
        t = Timer("test", "desc")
        assert t.count == 0
        assert t.name == "test"

    def test_start_stop(self):
        t = Timer("test")
        t.start()
        time.sleep(0.01)
        elapsed = t.stop()
        assert elapsed > 0
        assert t.count == 1

    def test_stop_without_start_raises(self):
        t = Timer("test")
        with pytest.raises(RuntimeError, match="start"):
            t.stop()

    def test_record(self):
        t = Timer("test")
        t.record(10.0)
        t.record(20.0)
        assert t.count == 2
        assert t.mean() == 15.0
        assert t.min_duration() == 10.0
        assert t.max_duration() == 20.0
        assert t.total() == 30.0

    def test_percentiles(self):
        t = Timer("test")
        for i in range(100):
            t.record(float(i))
        assert t.p50() == 50.0
        assert t.p95() == 95.0
        assert t.p99() == 99.0

    def test_empty_percentiles(self):
        t = Timer("test")
        assert t.p50() == 0.0
        assert t.p95() == 0.0
        assert t.mean() == 0.0
        assert t.min_duration() == 0.0
        assert t.max_duration() == 0.0

    def test_summary(self):
        t = Timer("test")
        t.record(10.0)
        s = t.summary()
        assert s["name"] == "test"
        assert s["count"] == 1
        assert s["mean_ms"] == 10.0

    def test_reset(self):
        t = Timer("test")
        t.record(10.0)
        t.reset()
        assert t.count == 0


class TestGauge:
    """Gauge 测试"""

    def test_init(self):
        g = Gauge("test", "desc")
        assert g.value == 0.0

    def test_set(self):
        g = Gauge("test")
        g.set(42.5)
        assert g.value == 42.5

    def test_increment_decrement(self):
        g = Gauge("test")
        g.increment(10.0)
        assert g.value == 10.0
        g.decrement(3.0)
        assert g.value == 7.0


class TestMetricsCollector:
    """MetricsCollector 测试"""

    def test_counter(self):
        mc = MetricsCollector()
        c = mc.counter("requests", "请求计数")
        c.increment()
        snap = mc.snapshot()
        assert snap["counters"]["requests"]["value"] == 1

    def test_timer(self):
        mc = MetricsCollector()
        t = mc.timer("latency", "延迟")
        t.record(10.0)
        snap = mc.snapshot()
        assert snap["timers"]["latency"]["count"] == 1

    def test_gauge(self):
        mc = MetricsCollector()
        g = mc.gauge("connections", "连接数")
        g.set(5.0)
        snap = mc.snapshot()
        assert snap["gauges"]["connections"]["value"] == 5.0

    def test_export_json(self):
        mc = MetricsCollector(namespace="test")
        mc.counter("c1").increment()
        j = mc.export_json()
        data = json.loads(j)
        assert data["namespace"] == "test"
        assert "counters" in data

    def test_export_prometheus(self):
        mc = MetricsCollector(namespace="test")
        mc.counter("requests").increment()
        mc.gauge("memory_mb").set(512.0)
        prom = mc.export_prometheus()
        assert "test_requests 1" in prom
        assert "test_memory_mb 512.0" in prom
        assert "counter" in prom
        assert "gauge" in prom

    def test_prometheus_timer(self):
        mc = MetricsCollector(namespace="test")
        t = mc.timer("latency")
        t.record(10.0)
        t.record(20.0)
        prom = mc.export_prometheus()
        assert "test_latency_count 2" in prom
        assert "quantile" in prom

    def test_reset(self):
        mc = MetricsCollector()
        mc.counter("c1").increment()
        mc.gauge("g1").set(42.0)
        mc.reset()
        snap = mc.snapshot()
        assert snap["counters"]["c1"]["value"] == 0
        assert snap["gauges"]["g1"]["value"] == 0.0

    def test_get_existing(self):
        """获取已存在的指标不创建新的"""
        mc = MetricsCollector()
        c1 = mc.counter("test")
        c1.increment()
        c2 = mc.counter("test")
        assert c2.value == 1


class TestHealthChecker:
    """HealthChecker 测试"""

    def test_empty(self):
        hc = HealthChecker()
        report = hc.report()
        assert report.overall_status == HealthStatus.HEALTHY

    def test_healthy(self):
        hc = HealthChecker()
        hc.add_check(HealthCheck("a", HealthStatus.HEALTHY))
        report = hc.report()
        assert report.overall_status == HealthStatus.HEALTHY

    def test_degraded(self):
        hc = HealthChecker()
        hc.add_check(HealthCheck("a", HealthStatus.HEALTHY))
        hc.add_check(HealthCheck("b", HealthStatus.DEGRADED))
        report = hc.report()
        assert report.overall_status == HealthStatus.DEGRADED

    def test_unhealthy_worst(self):
        hc = HealthChecker()
        hc.add_check(HealthCheck("a", HealthStatus.HEALTHY))
        hc.add_check(HealthCheck("b", HealthStatus.UNHEALTHY))
        hc.add_check(HealthCheck("c", HealthStatus.DEGRADED))
        report = hc.report()
        assert report.overall_status == HealthStatus.UNHEALTHY

    def test_clear(self):
        hc = HealthChecker()
        hc.add_check(HealthCheck("a", HealthStatus.HEALTHY))
        hc.clear()
        report = hc.report()
        assert report.overall_status == HealthStatus.HEALTHY
        assert len(report.checks) == 0


class TestCheckFunctions:
    """健康检查函数测试"""

    def test_check_memory_health_empty(self):
        mem = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(64))
        check = check_memory_health(mem)
        assert check.status == HealthStatus.HEALTHY
        assert "空" in check.message

    def test_check_memory_health_with_data(self):
        mem = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(64))
        mem.remember("test memory")
        check = check_memory_health(mem)
        assert check.status == HealthStatus.HEALTHY
        assert "1" in check.message

    def test_check_lsh_health_disabled(self):
        mem = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(64))
        check = check_lsh_health(mem)
        assert check.status == HealthStatus.HEALTHY

    def test_check_lsh_health_enabled(self):
        mem = HybridMemory(
            dimension=64,
            embedding_provider=HashEmbeddingProvider(64),
            use_lsh=True,
        )
        mem.remember("test")
        check = check_lsh_health(mem)
        assert check.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


# ===========================================================================
# HybridMemory 集成测试
# ===========================================================================


class TestHybridMemoryMetricsIntegration:
    """HybridMemory 可观测性集成测试"""

    def _make_mem(self) -> HybridMemory:
        return HybridMemory(
            dimension=64,
            embedding_provider=HashEmbeddingProvider(64),
        )

    def test_metrics_snapshot_after_operations(self):
        mem = self._make_mem()
        mem.remember("test1")
        mem.remember("test2")
        mem.search_text("test")
        snap = mem.metrics_snapshot()
        assert snap["counters"]["remember_count"]["value"] == 2
        assert snap["counters"]["search_count"]["value"] == 1
        assert snap["gauges"]["memory_count"]["value"] == 2

    def test_metrics_json(self):
        mem = self._make_mem()
        mem.remember("test")
        j = mem.metrics_json()
        data = json.loads(j)
        assert "counters" in data
        assert data["counters"]["remember_count"]["value"] == 1

    def test_metrics_prometheus(self):
        mem = self._make_mem()
        mem.remember("test")
        prom = mem.metrics_prometheus()
        assert "agentmemory_" in prom

    def test_search_timer_tracked(self):
        mem = self._make_mem()
        mem.remember("hello world")
        mem.search_text("hello")
        snap = mem.metrics_snapshot()
        assert snap["timers"]["search_latency_ms"]["count"] >= 1

    def test_remember_timer_tracked(self):
        mem = self._make_mem()
        mem.remember("test")
        snap = mem.metrics_snapshot()
        assert snap["timers"]["remember_latency_ms"]["count"] >= 1

    def test_forget_counter(self):
        mem = self._make_mem()
        m = mem.remember("test")
        mem.forget(m.id)
        snap = mem.metrics_snapshot()
        assert snap["counters"]["forget_count"]["value"] == 1
        assert snap["gauges"]["memory_count"]["value"] == 0

    def test_health_check(self):
        mem = self._make_mem()
        mem.remember("test")
        report = mem.health_check()
        assert report["overall_status"] == "healthy"
        assert len(report["checks"]) == 2

    def test_health_check_empty(self):
        mem = self._make_mem()
        report = mem.health_check()
        assert report["overall_status"] == "healthy"


class TestHybridMemoryRAGIntegration:
    """HybridMemory RAG 集成测试"""

    def _make_mem(self) -> HybridMemory:
        mem = HybridMemory(
            dimension=64,
            embedding_provider=HashEmbeddingProvider(64),
        )
        mem.remember("Python 是一种编程语言", tags=["python"])
        mem.remember("机器学习是 AI 的子领域", tags=["ml"])
        mem.remember("深度学习使用神经网络", tags=["dl"])
        return mem

    def test_rag_basic(self):
        mem = self._make_mem()
        result = mem.rag("什么是编程语言")
        assert "prompt" in result
        assert "context_text" in result
        assert "sources" in result
        assert "pipeline_time_ms" in result

    def test_rag_with_tags(self):
        mem = self._make_mem()
        result = mem.rag("编程", tags=["python"])
        assert isinstance(result["sources"], list)

    def test_rag_with_hybrid(self):
        mem = self._make_mem()
        result = mem.rag("AI", use_hybrid=True)
        assert "prompt" in result

    def test_rag_max_tokens(self):
        mem = self._make_mem()
        result = mem.rag("test", max_context_tokens=20)
        assert isinstance(result["total_tokens"], int)

    def test_rag_top_k(self):
        mem = self._make_mem()
        result = mem.rag("test", top_k=1)
        assert len(result["sources"]) <= 1


class TestHybridMemoryCompressionIntegration:
    """HybridMemory 向量压缩集成测试"""

    def _make_mem(self) -> HybridMemory:
        dim = 32
        mem = HybridMemory(
            dimension=dim,
            embedding_provider=HashEmbeddingProvider(dim),
        )
        for i in range(20):
            mem.remember(f"memory item {i}", tags=[f"tag{i % 3}"])
        return mem

    def test_compress_sq8(self):
        mem = self._make_mem()
        stats = mem.compress_vectors(method="sq8")
        assert stats["num_vectors"] == 20
        assert stats["compression_ratio"] == pytest.approx(4.0, abs=0.1)

    def test_compress_pq(self):
        mem = self._make_mem()
        stats = mem.compress_vectors(method="pq", num_subspaces=4)
        assert stats["num_vectors"] == 20
        assert stats["compression_ratio"] > 1.0

    def test_compress_invalid_method(self):
        mem = self._make_mem()
        with pytest.raises(ValueError, match="不支持"):
            mem.compress_vectors(method="unknown")

    def test_compressed_search(self):
        mem = self._make_mem()
        mem.compress_vectors(method="sq8")
        provider = HashEmbeddingProvider(32)
        query_emb = provider.embed("memory item")
        results = mem.compressed_search(query_emb, top_k=5)
        assert len(results) > 0
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_compressed_search_without_compress_raises(self):
        mem = self._make_mem()
        with pytest.raises(RuntimeError, match="compress"):
            mem.compressed_search([0.0] * 32)

    def test_compress_empty(self):
        mem = HybridMemory(
            dimension=32,
            embedding_provider=HashEmbeddingProvider(32),
        )
        stats = mem.compress_vectors()
        assert "error" in stats


# ===========================================================================
# 集成测试：全链路
# ===========================================================================


class TestFullPipelineIntegration:
    """全链路集成测试：添加→搜索→量化→RAG→指标→健康检查"""

    def test_complete_workflow(self):
        dim = 64
        provider = HashEmbeddingProvider(dim)
        mem = HybridMemory(dimension=dim, embedding_provider=provider)

        # 1. 添加记忆
        for i in range(10):
            mem.remember(f"test content {i}", tags=[f"tag{i % 3}"])
        assert mem.embedding_store.count() == 10

        # 2. 搜索
        results = mem.search_text("test content")
        assert len(results) > 0

        # 3. 向量压缩
        comp_stats = mem.compress_vectors(method="sq8")
        assert comp_stats["num_vectors"] == 10

        # 4. 压缩搜索
        query_emb = provider.embed("test content")
        comp_results = mem.compressed_search(query_emb, top_k=3)
        assert len(comp_results) > 0

        # 5. RAG
        rag_result = mem.rag("什么是测试内容", top_k=3)
        assert "prompt" in rag_result

        # 6. 指标快照
        snap = mem.metrics_snapshot()
        assert snap["counters"]["remember_count"]["value"] == 10

        # 7. 健康检查
        health = mem.health_check()
        assert health["overall_status"] == "healthy"

        # 8. Prometheus 导出
        prom = mem.metrics_prometheus()
        assert "agentmemory_" in prom

    def test_metrics_with_cache(self):
        """带缓存的指标追踪"""
        dim = 64
        provider = HashEmbeddingProvider(dim)
        mem = HybridMemory(
            dimension=dim,
            embedding_provider=provider,
            cache_size=10,
        )
        mem.remember("cached test")

        # 两次搜索
        mem.search_text("cached test")
        mem.search_text("cached test")

        snap = mem.metrics_snapshot()
        assert snap["counters"]["search_count"]["value"] == 2

    def test_session_metrics_preserved(self):
        """Session 代理方法可用"""
        dim = 64
        provider = HashEmbeddingProvider(dim)
        mem = HybridMemory(dimension=dim, embedding_provider=provider)
        with mem.session() as s:
            s.remember("session test")
            snap = s.metrics_snapshot()
            assert snap["counters"]["remember_count"]["value"] == 1
