# 🧠 AgentMemory

**Hybrid memory framework for AI agents** — combining vector similarity search with knowledge graph relational queries. Pure Python, zero external dependencies beyond numpy.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Vector Search** | Cosine similarity search over embedding vectors |
| **Knowledge Graph** | Entity/relation CRUD with BFS graph traversal |
| **Hybrid Search** | Vector search enriched with graph context |
| **JSON Persistence** | Human-readable storage for small datasets |
| **SQLite Persistence** | Efficient storage for larger datasets |
| **Auto Save/Load** | Transparent persistence on every operation |
| **Type-Safe** | Full type annotations and dataclass models |

---

## 🚀 Quick Start

### Installation

```bash
pip install agentmemory
```

### Basic Usage

```python
from agentmemory import HybridMemory

# Create memory system (3-dimensional embeddings for demo)
hm = HybridMemory(dimension=384)

# Add memories
hm.remember("Alice is a Python developer", embedding=alice_vec)
hm.remember("Bob works on Rust projects", embedding=bob_vec)

# Search by similarity
results = hm.search(query_embedding=query_vec, top_k=3)
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

### With Knowledge Graph

```python
# Add entities and relationships
alice = hm.add_entity("Alice", "person", {"role": "developer"})
python = hm.add_entity("Python", "language")
hm.add_relation(alice.id, python.id, "knows")

# Hybrid search combines vector + graph context
results = hm.hybrid_search(
    query_embedding=query_vec,
    top_k=5,
    graph_depth=2,
)
```

### With Persistence

```python
# JSON backend — human-readable files
hm = HybridMemory(
    dimension=384,
    storage_path="./data",
    storage_backend="json",
    auto_save=True,    # save after every remember/forget
    auto_load=True,    # load existing data on init
)

# SQLite backend — better for larger datasets
hm = HybridMemory(
    dimension=384,
    storage_path="./data/memory.db",
    storage_backend="sqlite",
    auto_save=True,
    auto_load=True,
)

# Manual save/load
hm = HybridMemory(dimension=384, storage_path="./data")
hm.remember("some memory", embedding=vec)
hm.save()  # explicit save

# Later, in a new session:
hm2 = HybridMemory(dimension=384, storage_path="./data")
hm2.load()  # load persisted data
```

---

## 📖 API Reference

### HybridMemory

```python
HybridMemory(
    dimension: int,                              # embedding vector dimension
    storage_path: str = None,                    # persistence directory/file
    storage_backend: str = "json",               # "json" or "sqlite"
    auto_save: bool = False,                     # auto-save after mutations
    auto_load: bool = False,                     # auto-load on init
)
```

#### Memory Operations

| Method | Description |
|---|---|
| `remember(content, embedding, metadata)` | Add a memory |
| `forget(memory_id)` | Remove a memory |
| `get_memory(memory_id)` | Get memory by ID |
| `list_all()` | List all memories |

#### Knowledge Graph Operations

| Method | Description |
|---|---|
| `add_entity(name, type, properties)` | Add entity |
| `add_relation(src, dst, type, weight)` | Add relationship |
| `get_neighbors(entity_id, relation_type)` | Get adjacent entities |

#### Search Operations

| Method | Description |
|---|---|
| `search(query_embedding, top_k, threshold)` | Vector similarity search |
| `hybrid_search(query_embedding, top_k, threshold, graph_depth)` | Vector + graph search |

#### Persistence Operations

| Method | Description |
|---|---|
| `save()` | Save all data to disk |
| `load()` | Load data from disk |
| `stats()` | Get memory/entity/relation counts |

### Low-Level Components

```python
from agentmemory import EmbeddingStore, KnowledgeGraph, JSONBackend, SQLiteBackend

# Use independently
store = EmbeddingStore(dimension=384)
kg = KnowledgeGraph()
backend = JSONBackend("./my_data")
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

No external services required — all tests run with pure Python.

---

## 📁 Project Structure

```
agentmemory/
├── agentmemory/
│   ├── __init__.py          # Public API exports
│   ├── models.py            # Memory, Entity, Relation, SearchResult
│   ├── embedding_store.py   # In-memory vector store + cosine similarity
│   ├── knowledge_graph.py   # Entity/relation graph with BFS traversal
│   ├── hybrid_memory.py     # Unified API combining both stores
│   └── persistence.py       # JSON and SQLite storage backends
├── tests/
│   ├── test_models.py
│   ├── test_embedding_store.py
│   ├── test_knowledge_graph.py
│   ├── test_hybrid_memory.py
│   ├── test_persistence.py
│   └── test_hybrid_persistence.py
├── pyproject.toml
└── README.md
```

---

## 📄 License

MIT

---

<p align="center">
  Built with ❤️ by <a href="https://nousresearch.com">Nous Research</a>
  · <a href="https://github.com/nousresearch/agentmemory">GitHub</a>
</p>
