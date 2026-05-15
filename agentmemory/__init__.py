"""agentmemory - 混合记忆框架：向量搜索 + 知识图谱"""

from agentmemory.models import Memory, Entity, Relation, SearchResult
from agentmemory.embedding_store import EmbeddingStore, cosine_similarity
from agentmemory.embedding_provider import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
)
from agentmemory.embedding_cache import CachedEmbeddingProvider
from agentmemory.knowledge_graph import KnowledgeGraph
from agentmemory.hybrid_memory import HybridMemory, MemorySession
from agentmemory.persistence import JSONBackend, SQLiteBackend
from agentmemory.async_persistence import AsyncSQLiteBackend
from agentmemory.lifecycle import MemoryLifecycle
from agentmemory.lsh_index import LSHIndex
from agentmemory.search_filter import SearchFilter, filter_search_results
from agentmemory.async_api import AsyncHybridMemory
from agentmemory.weighted_search import (
    WeightedScorer,
    ScoringWeights,
    weighted_search,
)
from agentmemory.plugins import PluginRegistry, get_registry
from agentmemory.search_cache import SearchCache
from agentmemory.graph_viz import export_dot, export_html, graph_stats_text
from agentmemory.metrics import (
    MetricsCollector,
    Counter,
    Timer,
    Gauge,
    HealthChecker,
    HealthStatus,
    HealthCheck,
    HealthReport,
)
from agentmemory.rag_pipeline import (
    RAGPipeline,
    Reranker,
    RAGContext,
    RAGResult,
    ContextStrategy,
)
from agentmemory.vector_quantizer import (
    ScalarQuantizer,
    ProductQuantizer,
    CompressedVectorStore,
    QuantizationStats,
)
from agentmemory.consolidator import (
    MemoryConsolidator,
    ConsolidationResult,
)
from agentmemory.snapshot import (
    SnapshotManager,
    SnapshotMetadata,
    SnapshotDiff,
)
from agentmemory.namespace import (
    Namespace,
    NamespaceManager,
)
from agentmemory.events import (
    EventBus,
    EventType,
    EventContext,
    EventHandler,
    get_event_bus,
    reset_event_bus,
)
from agentmemory.analytics import (
    MemoryAnalyzer,
    MemoryReport,
    AccessPattern,
    TemporalDistribution,
    TagCloud,
    ContentAnalysis,
)
from agentmemory.streaming import (
    StreamingSearcher,
    SearchProgress,
    StreamConfig,
    stream_search,
)
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
)
from agentmemory.import_export import (
    MemoryBankFormat,
    JSONLExporter,
    FullExportFormat,
    MarkdownExporter,
    ExportManager,
)
from agentmemory.embedding_providers_ext import (
    CohereEmbeddingProvider,
    VoyageEmbeddingProvider,
    RemoteEmbeddingProvider,
)

__all__ = [
    "Memory", "Entity", "Relation", "SearchResult",
    "EmbeddingStore", "cosine_similarity",
    "EmbeddingProvider", "HashEmbeddingProvider",
    "OpenAIEmbeddingProvider", "HuggingFaceEmbeddingProvider",
    "CachedEmbeddingProvider",
    "KnowledgeGraph",
    "HybridMemory", "MemorySession",
    "AsyncHybridMemory",
    "JSONBackend", "SQLiteBackend", "AsyncSQLiteBackend",
    "MemoryLifecycle",
    "LSHIndex",
    "SearchFilter", "filter_search_results",
    "WeightedScorer", "ScoringWeights", "weighted_search",
    "PluginRegistry", "get_registry",
    "SearchCache",
    "export_dot", "export_html", "graph_stats_text",
    # v0.7.0: 可观测性
    "MetricsCollector", "Counter", "Timer", "Gauge",
    "HealthChecker", "HealthStatus", "HealthCheck", "HealthReport",
    # v0.7.0: RAG 管道
    "RAGPipeline", "Reranker", "RAGContext", "RAGResult", "ContextStrategy",
    # v0.7.0: 向量量化
    "ScalarQuantizer", "ProductQuantizer", "CompressedVectorStore", "QuantizationStats",
    # v0.8.0: 记忆整合
    "MemoryConsolidator", "ConsolidationResult",
    # v0.8.0: 快照系统
    "SnapshotManager", "SnapshotMetadata", "SnapshotDiff",
    # v0.8.0: 命名空间
    "Namespace", "NamespaceManager",
    # v0.8.0: 事件系统
    "EventBus", "EventType", "EventContext", "EventHandler", "get_event_bus", "reset_event_bus",
    # v0.8.0: 分析系统
    "MemoryAnalyzer", "MemoryReport", "AccessPattern", "TemporalDistribution", "TagCloud", "ContentAnalysis",
    # v0.8.0: 流式搜索
    "StreamingSearcher", "SearchProgress", "StreamConfig", "stream_search",
    # v0.9.0: 配置系统
    "AgentMemoryConfig", "VectorConfig", "StorageConfig", "LifecycleConfig",
    "CacheConfig", "GCConfig", "load_config", "get_profile", "PROFILES",
    # v0.9.0: 中间件
    "MiddlewarePipeline", "HookContext", "HookType", "BuiltinMiddleware",
    # v0.9.0: 垃圾回收
    "GarbageCollector", "GCPolicy", "GCResult",
    # v0.9.0: 基准测试
    "BenchmarkResult", "BenchmarkSuite", "run_benchmark", "run_all",
    # v1.0.0: 导入/导出
    "MemoryBankFormat", "JSONLExporter", "FullExportFormat", "MarkdownExporter", "ExportManager",
    # v1.0.0: 扩展 Embedding Provider
    "CohereEmbeddingProvider", "VoyageEmbeddingProvider", "RemoteEmbeddingProvider",
]

__version__ = "1.0.0"
