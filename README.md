# agentmemory

Hybrid memory framework for AI agents: vector search + knowledge graph.

## Architecture

```
agentmemory/
├── models.py           # Core data models (Memory, Entity, Relation, SearchResult)
├── embedding_store.py  # Vector storage with cosine similarity search
├── knowledge_graph.py  # Entity/relation CRUD + graph traversal (BFS)
└── hybrid_memory.py    # Unified API combining vector search + knowledge graph
```

## Quick Start

```python
from agentmemory import HybridMemory

# Initialize with desired vector dimension
hm = HybridMemory(dimension=768)

# Store memories with embeddings
mem = hm.remember("Alice is a Python developer", embedding=[...])

# Pure vector search
results = hm.search(query_embedding=[...], top_k=5, threshold=0.7)

# Add knowledge graph structure
alice = hm.add_entity("Alice", "person")
python = hm.add_entity("Python", "language")
hm.add_relation(alice.id, python.id, "knows")

# Hybrid search: vector similarity + graph context
results = hm.hybrid_search(query_embedding=[...], top_k=5, graph_depth=2)

# Get system stats
print(hm.stats())
```

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run tests with coverage
pytest --cov=agentmemory -v
```

## Project Structure

| Module | Description | Tests |
|--------|-------------|-------|
| `models` | Data classes with validation & serialization | 22 |
| `embedding_store` | In-memory vector store with cosine similarity | 19 |
| `knowledge_graph` | Graph CRUD with adjacency list + BFS | 21 |
| `hybrid_memory` | Unified API orchestrating both subsystems | 15 |

**Total: 77 tests, all passing.**
