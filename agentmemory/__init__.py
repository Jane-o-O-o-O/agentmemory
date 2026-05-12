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
from agentmemory.lifecycle import MemoryLifecycle
from agentmemory.lsh_index import LSHIndex
from agentmemory.search_filter import SearchFilter, filter_search_results
from agentmemory.async_api import AsyncHybridMemory

__all__ = [
    "Memory", "Entity", "Relation", "SearchResult",
    "EmbeddingStore", "cosine_similarity",
    "EmbeddingProvider", "HashEmbeddingProvider",
    "OpenAIEmbeddingProvider", "HuggingFaceEmbeddingProvider",
    "CachedEmbeddingProvider",
    "KnowledgeGraph",
    "HybridMemory", "MemorySession",
    "AsyncHybridMemory",
    "JSONBackend", "SQLiteBackend",
    "MemoryLifecycle",
    "LSHIndex",
    "SearchFilter", "filter_search_results",
]

__version__ = "0.4.0"
