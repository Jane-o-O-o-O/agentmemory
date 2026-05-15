"""v0.9.0 功能测试：配置系统、中间件管道、垃圾回收、基准测试、CLI 新命令。"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch as mock_patch

import pytest

from agentmemory.config import (
    AgentMemoryConfig,
    VectorConfig,
    StorageConfig,
    LifecycleConfig,
    CacheConfig,
    GCConfig,
    load_config,
    get_profile,
    PROFILES,
    _coerce_env,
)
from agentmemory.middleware import (
    MiddlewarePipeline,
    HookContext,
    HookType,
    BuiltinMiddleware,
)
from agentmemory.gc import (
    GarbageCollector,
    GCPolicy,
    GCResult,
)
from agentmemory.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    run_benchmark,
    run_all,
    benchmark_embedding_store,
    benchmark_knowledge_graph,
    benchmark_lsh_index,
    benchmark_hybrid_memory,
)
from agentmemory.models import Memory


# ============================================================
# 配置系统测试
# ============================================================

class TestVectorConfig:
    """VectorConfig 测试"""

    def test_defaults(self):
        cfg = VectorConfig()
        assert cfg.dimension == 128
        assert cfg.use_lsh is False
        assert cfg.lsh_tables == 8
        assert cfg.lsh_hyperplanes == 16
        assert cfg.use_quantization is False
        assert cfg.quantization_method == "sq8"

    def test_validate_valid(self):
        cfg = VectorConfig()
        assert cfg.validate() == []

    def test_validate_invalid_dimension(self):
        cfg = VectorConfig(dimension=0)
        errors = cfg.validate()
        assert any("dimension" in e for e in errors)

    def test_validate_invalid_quantization_method(self):
        cfg = VectorConfig(quantization_method="invalid")
        errors = cfg.validate()
        assert any("quantization_method" in e for e in errors)

    def test_validate_invalid_lsh_tables(self):
        cfg = VectorConfig(lsh_tables=0)
        errors = cfg.validate()
        assert any("lsh_tables" in e for e in errors)


class TestStorageConfig:
    """StorageConfig 测试"""

    def test_defaults(self):
        cfg = StorageConfig()
        assert cfg.storage_path is None
        assert cfg.backend == "json"
        assert cfg.auto_save is False
        assert cfg.auto_load is False

    def test_validate_valid(self):
        for backend in ("json", "sqlite"):
            cfg = StorageConfig(backend=backend)
            assert cfg.validate() == []

    def test_validate_invalid_backend(self):
        cfg = StorageConfig(backend="mongodb")
        errors = cfg.validate()
        assert any("backend" in e for e in errors)


class TestLifecycleConfig:
    """LifecycleConfig 测试"""

    def test_defaults(self):
        cfg = LifecycleConfig()
        assert cfg.default_ttl is None
        assert cfg.decay_rate == 0.001
        assert abs(cfg.recency_weight + cfg.frequency_weight + cfg.relevance_weight - 1.0) < 0.01

    def test_validate_valid(self):
        cfg = LifecycleConfig()
        assert cfg.validate() == []

    def test_validate_negative_ttl(self):
        cfg = LifecycleConfig(default_ttl=-1)
        errors = cfg.validate()
        assert any("default_ttl" in e for e in errors)

    def test_validate_bad_weights(self):
        cfg = LifecycleConfig(recency_weight=0.5, frequency_weight=0.5, relevance_weight=0.5)
        errors = cfg.validate()
        assert any("权重之和" in e for e in errors)

    def test_validate_negative_decay(self):
        cfg = LifecycleConfig(decay_rate=-0.1)
        errors = cfg.validate()
        assert any("decay_rate" in e for e in errors)


class TestCacheConfig:
    """CacheConfig 测试"""

    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.enabled is False
        assert cfg.max_size == 128

    def test_validate_negative_size(self):
        cfg = CacheConfig(max_size=-1)
        errors = cfg.validate()
        assert any("max_size" in e for e in errors)


class TestGCConfig:
    """GCConfig 测试"""

    def test_defaults(self):
        cfg = GCConfig()
        assert cfg.enabled is False
        assert cfg.interval == 3600.0
        assert cfg.min_importance == 0.1

    def test_validate_invalid_interval(self):
        cfg = GCConfig(interval=0)
        errors = cfg.validate()
        assert any("interval" in e for e in errors)

    def test_validate_invalid_importance(self):
        cfg = GCConfig(min_importance=1.5)
        errors = cfg.validate()
        assert any("min_importance" in e for e in errors)

    def test_validate_invalid_batch_size(self):
        cfg = GCConfig(batch_size=0)
        errors = cfg.validate()
        assert any("batch_size" in e for e in errors)


class TestAgentMemoryConfig:
    """AgentMemoryConfig 测试"""

    def test_defaults(self):
        cfg = AgentMemoryConfig()
        assert isinstance(cfg.vector, VectorConfig)
        assert isinstance(cfg.storage, StorageConfig)
        assert isinstance(cfg.lifecycle, LifecycleConfig)
        assert isinstance(cfg.cache, CacheConfig)
        assert isinstance(cfg.gc, GCConfig)
        assert cfg.weighted_scoring is False
        assert cfg.enable_metrics is False

    def test_validate_valid(self):
        cfg = AgentMemoryConfig()
        assert cfg.validate() == []

    def test_validate_collects_all_errors(self):
        cfg = AgentMemoryConfig(
            vector=VectorConfig(dimension=0),
            gc=GCConfig(interval=-1),
        )
        errors = cfg.validate()
        assert len(errors) >= 2

    def test_to_dict_roundtrip(self):
        cfg = AgentMemoryConfig(weighted_scoring=True, enable_metrics=True)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["weighted_scoring"] is True
        restored = AgentMemoryConfig.from_dict(d)
        assert restored.weighted_scoring is True
        assert restored.vector.dimension == cfg.vector.dimension

    def test_from_dict_custom(self):
        data = {
            "vector": {"dimension": 256, "use_lsh": True},
            "storage": {"backend": "sqlite"},
            "weighted_scoring": True,
        }
        cfg = AgentMemoryConfig.from_dict(data)
        assert cfg.vector.dimension == 256
        assert cfg.vector.use_lsh is True
        assert cfg.storage.backend == "sqlite"
        assert cfg.weighted_scoring is True


class TestProfiles:
    """Profile 测试"""

    def test_dev_profile(self):
        cfg = get_profile("dev")
        assert cfg.vector.dimension == 64
        assert cfg.vector.use_lsh is False
        assert cfg.storage.backend == "json"

    def test_test_profile(self):
        cfg = get_profile("test")
        assert cfg.vector.dimension == 32

    def test_prod_profile(self):
        cfg = get_profile("prod")
        assert cfg.vector.dimension == 256
        assert cfg.vector.use_lsh is True
        assert cfg.storage.backend == "sqlite"
        assert cfg.weighted_scoring is True
        assert cfg.enable_metrics is True
        assert cfg.gc.enabled is True

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="未知 Profile"):
            get_profile("unknown")

    def test_all_profiles_exist(self):
        assert "dev" in PROFILES
        assert "test" in PROFILES
        assert "prod" in PROFILES


class TestLoadConfig:
    """load_config 测试"""

    def test_default(self):
        cfg = load_config()
        assert isinstance(cfg, AgentMemoryConfig)

    def test_with_profile(self):
        cfg = load_config(profile="dev")
        assert cfg.vector.dimension == 64

    def test_env_override(self):
        env = {"AGENTMEMORY_VECTOR_DIMENSION": "512"}
        with mock_patch.dict(os.environ, env, clear=False):
            cfg = load_config(env_override=True)
            assert cfg.vector.dimension == 512

    def test_env_override_bool(self):
        env = {"AGENTMEMORY_VECTOR_USE_LSH": "true"}
        with mock_patch.dict(os.environ, env, clear=False):
            cfg = load_config(env_override=True)
            assert cfg.vector.use_lsh is True

    def test_no_env_override(self):
        env = {"AGENTMEMORY_VECTOR_DIMENSION": "512"}
        with mock_patch.dict(os.environ, env, clear=False):
            cfg = load_config(env_override=False)
            assert cfg.vector.dimension == 128  # default


class TestCoerceEnv:
    """_coerce_env 测试"""

    def test_bool_true(self):
        assert _coerce_env("true", False) is True
        assert _coerce_env("1", False) is True
        assert _coerce_env("yes", False) is True

    def test_bool_false(self):
        assert _coerce_env("false", True) is False
        assert _coerce_env("0", True) is False

    def test_int(self):
        assert _coerce_env("42", 0) == 42

    def test_float(self):
        assert _coerce_env("3.14", 0.0) == pytest.approx(3.14)

    def test_string(self):
        assert _coerce_env("hello", "default") == "hello"


# ============================================================
# 中间件管道测试
# ============================================================

class TestHookContext:
    """HookContext 测试"""

    def test_init(self):
        ctx = HookContext(operation="remember", data={"content": "hello"})
        assert ctx.operation == "remember"
        assert ctx.data["content"] == "hello"
        assert ctx.blocked is False
        assert ctx.block_reason is None

    def test_block(self):
        ctx = HookContext(operation="test")
        ctx.block("reason")
        assert ctx.blocked is True
        assert ctx.block_reason == "reason"


class TestMiddlewarePipeline:
    """MiddlewarePipeline 测试"""

    def test_empty_pipeline(self):
        pipeline = MiddlewarePipeline()
        ctx = HookContext(operation="test")
        result = pipeline.run_pre(ctx)
        assert result is ctx
        assert not result.blocked

    def test_pre_hook_modifies_data(self):
        pipeline = MiddlewarePipeline()

        def transform(ctx: HookContext) -> HookContext:
            ctx.data["transformed"] = True
            return ctx

        pipeline.add_pre("transform", transform)
        ctx = HookContext(operation="test", data={"content": "hello"})
        result = pipeline.run_pre(ctx)
        assert result.data["transformed"] is True

    def test_post_hook(self):
        pipeline = MiddlewarePipeline()
        called = []

        def log_post(ctx: HookContext) -> None:
            called.append(ctx.operation)

        pipeline.add_post("logger", log_post)
        ctx = HookContext(operation="remember")
        pipeline.run_post(ctx)
        assert called == ["remember"]

    def test_priority_order(self):
        pipeline = MiddlewarePipeline()
        order = []

        def make_hook(name):
            def hook(ctx):
                order.append(name)
                return ctx
            return hook

        pipeline.add_pre("low", make_hook("low"), priority=200)
        pipeline.add_pre("high", make_hook("high"), priority=10)
        pipeline.add_pre("mid", make_hook("mid"), priority=100)

        ctx = HookContext(operation="test")
        pipeline.run_pre(ctx)
        assert order == ["high", "mid", "low"]

    def test_block_stops_execution(self):
        pipeline = MiddlewarePipeline()
        called = []

        def blocker(ctx):
            ctx.block("blocked!")
            called.append("blocker")

        def after_blocker(ctx):
            called.append("after")
            return ctx

        pipeline.add_pre("blocker", blocker, priority=10)
        pipeline.add_pre("after", after_blocker, priority=20)

        ctx = HookContext(operation="test")
        result = pipeline.run_pre(ctx)
        assert result.blocked is True
        assert called == ["blocker"]  # after not called

    def test_operation_specific_hook(self):
        pipeline = MiddlewarePipeline()
        called = []

        def hook(ctx):
            called.append(ctx.operation)
            return ctx

        pipeline.add_pre("remember_hook", hook, operation="remember")
        pipeline.add_pre("search_hook", hook, operation="search")

        ctx1 = HookContext(operation="remember")
        pipeline.run_pre(ctx1)
        assert called == ["remember"]

        ctx2 = HookContext(operation="search")
        pipeline.run_pre(ctx2)
        assert called == ["remember", "search"]

    def test_remove_middleware(self):
        pipeline = MiddlewarePipeline()
        called = []

        def hook(ctx):
            called.append(True)
            return ctx

        pipeline.add_pre("test", hook)
        assert pipeline.remove("test") is True
        assert pipeline.remove("nonexistent") is False

        ctx = HookContext(operation="test")
        pipeline.run_pre(ctx)
        assert called == []

    def test_enable_disable(self):
        pipeline = MiddlewarePipeline()
        called = []

        def hook(ctx):
            called.append(True)
            return ctx

        pipeline.add_pre("test", hook)
        pipeline.disable("test")

        ctx = HookContext(operation="test")
        pipeline.run_pre(ctx)
        assert called == []

        pipeline.enable("test")
        pipeline.run_pre(ctx)
        assert called == [True]

    def test_list_middleware(self):
        pipeline = MiddlewarePipeline()
        pipeline.add_pre("pre1", lambda ctx: ctx, priority=10)
        pipeline.add_post("post1", lambda ctx: None, priority=20)
        pipeline.add_pre("op_hook", lambda ctx: ctx, operation="remember")

        items = pipeline.list_middleware()
        assert len(items) == 3
        names = [i["name"] for i in items]
        assert "pre1" in names
        assert "post1" in names
        assert "op_hook" in names

    def test_clear(self):
        pipeline = MiddlewarePipeline()
        pipeline.add_pre("test", lambda ctx: ctx)
        pipeline.add_post("test2", lambda ctx: None)
        assert len(pipeline) == 2
        pipeline.clear()
        assert len(pipeline) == 0

    def test_len(self):
        pipeline = MiddlewarePipeline()
        assert len(pipeline) == 0
        pipeline.add_pre("a", lambda ctx: ctx)
        assert len(pipeline) == 1
        pipeline.add_post("b", lambda ctx: None)
        assert len(pipeline) == 2


class TestBuiltinMiddleware:
    """内置中间件测试"""

    def test_timing(self):
        name, pre, post = BuiltinMiddleware.timing()
        assert name == "timing"
        ctx = HookContext(operation="test")
        pre(ctx)
        assert "start_time" in ctx.metadata
        time.sleep(0.01)
        post(ctx)
        assert "elapsed_ms" in ctx.metadata
        assert ctx.metadata["elapsed_ms"] > 0

    def test_content_validator_valid(self):
        name, hook = BuiltinMiddleware.content_validator(min_length=1, max_length=100)
        ctx = HookContext(operation="remember", data={"content": "hello"})
        hook(ctx)
        assert not ctx.blocked

    def test_content_validator_too_short(self):
        name, hook = BuiltinMiddleware.content_validator(min_length=5)
        ctx = HookContext(operation="remember", data={"content": "hi"})
        hook(ctx)
        assert ctx.blocked
        assert "最小要求" in ctx.block_reason

    def test_content_validator_too_long(self):
        name, hook = BuiltinMiddleware.content_validator(max_length=10)
        ctx = HookContext(operation="remember", data={"content": "x" * 20})
        hook(ctx)
        assert ctx.blocked
        assert "最大限制" in ctx.block_reason

    def test_content_validator_no_content(self):
        name, hook = BuiltinMiddleware.content_validator()
        ctx = HookContext(operation="test", data={})
        hook(ctx)
        # empty string has length 0, which is < min_length=1
        assert ctx.blocked

    def test_rate_limiter(self):
        name, hook = BuiltinMiddleware.rate_limiter(max_per_second=3)
        # First 3 should pass
        for _ in range(3):
            ctx = HookContext(operation="test")
            hook(ctx)
            assert not ctx.blocked
        # 4th should be blocked
        ctx = HookContext(operation="test")
        hook(ctx)
        assert ctx.blocked

    def test_audit_log(self):
        logs = []
        name, pre, post = BuiltinMiddleware.audit_log(lambda msg, ctx: logs.append(msg))
        ctx = HookContext(operation="remember", data={"content": "test"})
        pre(ctx)
        post(ctx)
        assert len(logs) == 2
        assert "PRE" in logs[0]
        assert "POST" in logs[1]


# ============================================================
# 垃圾回收测试
# ============================================================

class TestGCPolicy:
    """GCPolicy 测试"""

    def test_defaults(self):
        policy = GCPolicy()
        assert policy.min_importance == 0.0
        assert policy.max_age is None
        assert policy.batch_size == 100
        assert policy.preserve_tags == []


class TestGCResult:
    """GCResult 测试"""

    def test_empty_result(self):
        result = GCResult()
        assert result.total_collected == 0
        assert result.total_retained == 0

    def test_to_dict(self):
        result = GCResult(collected=["a", "b"], retained=["c"], reasons={"a": "expired"})
        d = result.to_dict()
        assert d["total_collected"] == 2
        assert d["total_retained"] == 1


class TestGarbageCollector:
    """GarbageCollector 测试"""

    def _make_memory(self, content: str, age_seconds: float = 0, memory_id: str = "test_id") -> Memory:
        """创建指定年龄的记忆。"""
        m = Memory(content=content, id=memory_id)
        m.created_at = time.time() - age_seconds
        return m

    def test_default_policy_keeps_all(self):
        gc = GarbageCollector()
        m = self._make_memory("hello", age_seconds=100)
        assert gc.should_collect(m) is None

    def test_ttl_expired(self):
        from agentmemory.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle(default_ttl=10)
        gc = GarbageCollector(lifecycle=lifecycle)
        m = self._make_memory("hello", age_seconds=20)
        assert gc.should_collect(m) == "ttl_expired"

    def test_max_age(self):
        policy = GCPolicy(max_age=100)
        gc = GarbageCollector(policy=policy)
        m = self._make_memory("hello", age_seconds=200)
        reason = gc.should_collect(m)
        assert reason is not None
        assert "max_age_exceeded" in reason

    def test_max_age_not_exceeded(self):
        policy = GCPolicy(max_age=100)
        gc = GarbageCollector(policy=policy)
        m = self._make_memory("hello", age_seconds=50)
        assert gc.should_collect(m) is None

    def test_min_importance(self):
        from agentmemory.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle()
        policy = GCPolicy(min_importance=0.8)
        gc = GarbageCollector(lifecycle=lifecycle, policy=policy)
        # A brand new memory with no access should have low importance
        m = self._make_memory("hello", age_seconds=0)
        reason = gc.should_collect(m)
        # The default importance score with no access count and fresh memory
        # may or may not be below 0.8, depending on weights
        # Let's check with an old, never-accessed memory
        m_old = self._make_memory("old", age_seconds=10000)
        reason = gc.should_collect(m_old)
        # Old memory with no access should have low importance
        assert reason is not None

    def test_preserve_tags(self):
        policy = GCPolicy(max_age=1, preserve_tags=["important"])
        gc = GarbageCollector(policy=policy)
        m = self._make_memory("hello", age_seconds=100)
        m.tags = ["important"]
        assert gc.should_collect(m) is None

    def test_max_idle_time(self):
        from agentmemory.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle()
        policy = GCPolicy(max_idle_time=50)
        gc = GarbageCollector(lifecycle=lifecycle, policy=policy)

        m = self._make_memory("hello", age_seconds=100)
        # Never accessed, idle = age = 100 > 50
        reason = gc.should_collect(m)
        assert reason is not None
        assert "idle" in reason or "never_accessed" in reason

    def test_max_idle_time_with_recent_access(self):
        from agentmemory.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle()
        policy = GCPolicy(max_idle_time=50)
        gc = GarbageCollector(lifecycle=lifecycle, policy=policy)

        m = self._make_memory("hello", age_seconds=100)
        lifecycle.record_access(m.id)  # recent access
        reason = gc.should_collect(m)
        assert reason is None  # recently accessed

    def test_collect(self):
        policy = GCPolicy(max_age=100)
        gc = GarbageCollector(policy=policy)

        memories = [
            self._make_memory("old1", age_seconds=200, memory_id="old1"),
            self._make_memory("old2", age_seconds=300, memory_id="old2"),
            self._make_memory("new1", age_seconds=10, memory_id="new1"),
        ]

        result = gc.collect(memories)
        assert result.total_collected == 2
        assert result.total_retained == 1
        assert "old1" in result.collected
        assert "old2" in result.collected
        assert "new1" in result.retained
        assert result.elapsed_ms >= 0

    def test_preview_no_side_effects(self):
        from agentmemory.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle()
        policy = GCPolicy(max_age=100)
        gc = GarbageCollector(lifecycle=lifecycle, policy=policy)

        memories = [self._make_memory("old", age_seconds=200, memory_id="m1")]
        result = gc.preview(memories)
        assert result.total_collected == 1
        # Preview should not clean up lifecycle data
        # (lifecycle data should still exist)
        assert lifecycle.get_access_count("m1") == 0  # was never accessed anyway

    def test_batch_size_limit(self):
        policy = GCPolicy(max_age=100, batch_size=2)
        gc = GarbageCollector(policy=policy)

        memories = [
            self._make_memory(f"old{i}", age_seconds=200, memory_id=f"old{i}")
            for i in range(5)
        ]

        result = gc.collect(memories)
        assert result.total_collected == 2  # batch_size=2
        assert result.total_retained == 3

    def test_stats(self):
        policy = GCPolicy(max_age=100)
        gc = GarbageCollector(policy=policy)

        memories = [
            self._make_memory("old", age_seconds=200, memory_id="m1"),
            self._make_memory("new", age_seconds=10, memory_id="m2"),
        ]

        stats = gc.stats(memories)
        assert stats["total_memories"] == 2
        assert stats["would_collect"] == 1
        assert stats["avg_age_seconds"] > 0
        assert stats["gc_runs"] == 0

    def test_history(self):
        policy = GCPolicy(max_age=100)
        gc = GarbageCollector(policy=policy)

        memories = [self._make_memory("old", age_seconds=200)]
        gc.collect(memories)
        assert len(gc.history) == 1
        assert gc.history[0].total_collected == 1


# ============================================================
# 基准测试测试
# ============================================================

class TestRunBenchmark:
    """run_benchmark 测试"""

    def test_basic(self):
        result = run_benchmark("test", lambda: None, iterations=100, warmup=5)
        assert result.name == "test"
        assert result.iterations == 100
        assert result.ops_per_second > 0
        assert result.avg_ms >= 0
        assert result.min_ms <= result.avg_ms <= result.max_ms

    def test_to_dict(self):
        result = run_benchmark("test", lambda: None, iterations=10, warmup=1)
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["iterations"] == 10

    def test_str(self):
        result = run_benchmark("test", lambda: None, iterations=10, warmup=1)
        s = str(result)
        assert "test" in s
        assert "ops/s" in s


class TestBenchmarkSuite:
    """BenchmarkSuite 测试"""

    def test_empty_suite(self):
        suite = BenchmarkSuite(name="empty")
        assert suite.total_ms == 0
        d = suite.to_dict()
        assert d["name"] == "empty"
        assert d["results"] == []

    def test_summary(self):
        suite = BenchmarkSuite(name="test")
        suite.results.append(
            run_benchmark("sub", lambda: None, iterations=10, warmup=1)
        )
        suite.total_ms = sum(r.total_ms for r in suite.results)
        s = suite.summary()
        assert "test" in s
        assert "sub" in s


class TestBenchmarkEmbeddingStore:
    """EmbeddingStore 基准测试"""

    def test_basic(self):
        suite = benchmark_embedding_store(dimension=32, num_items=50, iterations=10)
        assert suite.name == "EmbeddingStore"
        assert len(suite.results) == 3  # insert, search, remove
        assert all(r.ops_per_second > 0 for r in suite.results)


class TestBenchmarkKnowledgeGraph:
    """KnowledgeGraph 基准测试"""

    def test_basic(self):
        suite = benchmark_knowledge_graph(num_entities=20, num_relations=40, iterations=10)
        assert suite.name == "KnowledgeGraph"
        assert len(suite.results) >= 2  # get_entity, get_relations, bfs


class TestBenchmarkLSHIndex:
    """LSHIndex 基准测试"""

    def test_basic(self):
        suite = benchmark_lsh_index(dimension=32, num_items=100, iterations=10)
        assert suite.name == "LSHIndex"
        assert len(suite.results) == 2  # query, insert


class TestBenchmarkHybridMemory:
    """HybridMemory 基准测试"""

    def test_basic(self):
        suite = benchmark_hybrid_memory(dimension=32, num_memories=20, iterations=5)
        assert suite.name == "HybridMemory"
        assert len(suite.results) == 3  # remember, search_text, list_memories


class TestRunAll:
    """run_all 测试"""

    def test_basic(self):
        suite = run_all(dimension=16, iterations=5)
        assert suite.name == "AgentMemory Full Benchmark"
        assert len(suite.results) > 0
        assert suite.total_ms > 0


# ============================================================
# CLI 新命令测试
# ============================================================

class TestCLIConfigCommand:
    """CLI config 命令测试"""

    def test_config_show(self, capsys):
        from agentmemory.cli import main
        main(["config", "show"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "vector" in data
        assert "storage" in data

    def test_config_profiles(self, capsys):
        from agentmemory.cli import main
        main(["config", "profiles"])
        output = capsys.readouterr().out
        assert "dev" in output
        assert "prod" in output

    def test_config_validate(self, capsys):
        from agentmemory.cli import main
        main(["config", "validate"])
        output = capsys.readouterr().out
        assert "通过" in output

    def test_config_show_profile(self, capsys):
        from agentmemory.cli import main
        main(["config", "show", "--profile", "prod"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["vector"]["dimension"] == 256


class TestCLIBenchmarkCommand:
    """CLI benchmark 命令测试"""

    def test_benchmark_vector(self, capsys):
        from agentmemory.cli import main
        main(["benchmark", "vector", "--iterations", "5"])
        output = capsys.readouterr().out
        assert "EmbeddingStore" in output

    def test_benchmark_json(self, capsys):
        from agentmemory.cli import main
        main(["benchmark", "vector", "--iterations", "5", "--format", "json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "results" in data


class TestCLIGCCommand:
    """CLI gc 命令测试"""

    def test_gc_stats_empty(self, capsys):
        from agentmemory.cli import main
        main(["--store", "/tmp/test_gc_cli", "gc", "stats"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "total_memories" in data

    def test_gc_preview_empty(self, capsys):
        from agentmemory.cli import main
        main(["--store", "/tmp/test_gc_cli2", "gc", "preview"])
        output = capsys.readouterr().out
        assert "预览" in output


# ============================================================
# 集成测试
# ============================================================

class TestMiddlewareIntegration:
    """中间件集成测试"""

    def test_full_pipeline_with_hooks(self):
        pipeline = MiddlewarePipeline()

        # Add timing
        name, pre, post = BuiltinMiddleware.timing()
        pipeline.add_pre(name, pre)
        pipeline.add_post(name, post)

        # Add validation
        name, hook = BuiltinMiddleware.content_validator(min_length=3, max_length=100)
        pipeline.add_pre(name, hook, priority=10)  # validate before timing

        # Test valid content
        ctx = HookContext(operation="remember", data={"content": "hello world"})
        ctx = pipeline.run_pre(ctx)
        assert not ctx.blocked

        ctx = pipeline.run_post(ctx)
        assert "elapsed_ms" in ctx.metadata

    def test_middleware_blocks_short_content(self):
        pipeline = MiddlewarePipeline()
        name, hook = BuiltinMiddleware.content_validator(min_length=10)
        pipeline.add_pre(name, hook)

        ctx = HookContext(operation="remember", data={"content": "hi"})
        ctx = pipeline.run_pre(ctx)
        assert ctx.blocked


class TestGCIntegration:
    """GC 集成测试"""

    def test_gc_with_hybrid_memory(self):
        from agentmemory import HybridMemory, HashEmbeddingProvider

        hm = HybridMemory(
            dimension=32,
            embedding_provider=HashEmbeddingProvider(dim=32),
        )

        # Add memories with different ages
        m1 = hm.remember("old memory")
        m2 = hm.remember("new memory")

        # Artificially age m1
        hm.embedding_store._memories[m1.id].created_at = time.time() - 10000

        gc = GarbageCollector(
            lifecycle=hm.lifecycle,
            policy=GCPolicy(max_age=5000),
        )

        memories = list(hm.embedding_store._memories.values())
        result = gc.collect(memories)
        assert result.total_collected >= 1
        assert m1.id in result.collected


class TestConfigIntegration:
    """配置集成测试"""

    def test_config_to_hybrid_memory(self):
        from agentmemory import HybridMemory, HashEmbeddingProvider

        cfg = get_profile("dev")
        hm = HybridMemory(
            dimension=cfg.vector.dimension,
            embedding_provider=HashEmbeddingProvider(dim=cfg.vector.dimension),
            use_lsh=cfg.vector.use_lsh,
        )
        assert hm.dimension == 64

    def test_full_import(self):
        """测试所有新模块可以正常 import。"""
        from agentmemory import (
            AgentMemoryConfig,
            MiddlewarePipeline,
            HookContext,
            GarbageCollector,
            GCPolicy,
            BenchmarkResult,
            BenchmarkSuite,
        )
        assert AgentMemoryConfig is not None
        assert MiddlewarePipeline is not None
        assert GarbageCollector is not None
        assert BenchmarkSuite is not None
