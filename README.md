# 🧠 AgentMemory

[![PyPI version](https://img.shields.io/pypi/v/agentmemory?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/agentmemory/)
[![Python](https://img.shields.io/pypi/pyversions/agentmemory?logo=python&logoColor=white)](https://pypi.org/project/agentmemory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/github/actions/workflow/status/nousresearch/agentmemory/tests.yml?branch=main&label=tests)](https://github.com/nousresearch/agentmemory/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Hybrid memory for AI agents** — combining the precision of vector similarity search with the relational power of knowledge graphs. Give your agents persistent, queryable, and interconnected memory that scales from prototyping to production.

> _"Memory is not just retrieval — it's understanding the relationships between what you know."_

---

## 🏗️ Architecture

```
                        ┌──────────────────────────────────────┐
                        │            🤖 AI Agent               │
                        │   (LangChain / AutoGen / Custom)     │
                        └──────────────┬───────────────────────┘
                                       │
                                       ▼
                     ┌─────────────────────────────────────────┐
                     │         📦 AgentMemory Core API         │
                     │                                         │
                     │  remember()  ·  recall()  ·  relate()   │
                     │  forget()    ·  context() ·  query()    │
                     └───────┬──────────────────┬──────────────┘
                             │                  │
                ┌────────────┘                  └────────────┐
                ▼                                            ▼
  ┌──────────────────────┐                  ┌─────────────────────────┐
  │   🔍 Vector Store    │                  │   🕸️  Knowledge Graph    │
  │    (ChromaDB)        │◄────sync────────►│      (Neo4j)            │
  │                      │   & cross-ref    │                         │
  │  • Embedding index   │                  │  • Entity nodes         │
  │  • Semantic search   │                  │  • Relationship edges   │
  │  • Metadata filters  │                  │  • Temporal edges       │
  │  • Similarity scores │                  │  • Graph traversal      │
  └──────────┬───────────┘                  └────────────┬────────────┘
             │                                           │
             ▼                                           ▼
  ┌──────────────────────┐                  ┌─────────────────────────┐
  │  🧮 Sentence-        │                  │  💾 Neo4j Storage       │
  │     Transformers     │                  │     Backend             │
  │  (Embedding Model)   │                  │  (Bolt Protocol)        │
  └──────────────────────┘                  └─────────────────────────┘
```

### How It Works

1. **Store** — Memories are embedded via `sentence-transformers` and indexed in ChromaDB for fast vector search, while entities and their relationships are modeled as nodes and edges in Neo4j.
2. **Retrieve** — `recall()` combines cosine similarity from the vector store with graph-aware context (related entities, temporal decay) to return ranked, enriched results.
3. **Relate** — `relate()` explicitly creates graph edges between memories or entities, enabling multi-hop reasoning and knowledge traversal.
4. **Sync** — Both stores stay in sync automatically; every `remember()` updates the vector index and the knowledge graph in a single atomic operation.

---

## ✨ Features

| Category | Feature |
|---|---|
| **Hybrid Storage** | Vector search + knowledge graph in a unified API |
| **Semantic Recall** | Natural-language queries ranked by embedding similarity |
| **Graph Traversal** | Multi-hop relational queries across connected entities |
| **Temporal Awareness** | Built-in timestamps, decay scoring, and time-windowed queries |
| **Metadata Filtering** | Filter by tags, source, session, agent ID, or custom fields |
| **Auto-Entity Extraction** | Optionally extract entities and relationships from raw text |
| **Pluggable Embeddings** | Swap `sentence-transformers` models via config (local or API) |
| **Lazy Init** | Both backends connect on first use — zero config for prototyping |
| **Async Support** | Full `async`/`await` API alongside synchronous methods |
| **Type-Safe** | Comprehensive type hints and Pydantic models throughout |
| **Extensible** | Custom graph schemas, embedding functions, and storage backends |

---

## 🚀 Quick Start

### Installation

```bash
# Core install (ChromaDB only — no graph features)
pip install agentmemory

# Full install with Neo4j knowledge graph support
pip install agentmemory[graph]

# Everything (graph + dev tools)
pip install agentmemory[all]
```

> **Prerequisites:** Python ≥ 3.9. For graph features, a running Neo4j instance (≥ 5.x) is required. See [Neo4j Setup](#-neo4j-setup) below.

### 30-Second Example

```python
from agentmemory import AgentMemory

# Initialize — ChromaDB for vectors, Neo4j for graph (optional)
mem = AgentMemory(
    collection="my-agent",
    neo4j_uri="bolt://localhost:7687",   # omit to disable graph
    neo4j_user="neo4j",
    neo4j_password="password",
)

# Store a memory
mem.remember(
    "The user prefers dark mode and monospace fonts.",
    metadata={"source": "settings", "session": "s1"},
)

# Recall semantically similar memories
results = mem.rerecall("What are the user's UI preferences?")
print(results)
# [MemoryResult(text="The user prefers dark mode and monospace fonts.",
#               score=0.94, metadata={...})]
```

---

## 📖 API Reference

### `remember()` — Store a Memory

```python
mem.remember(
    text: str,                          # the memory content
    metadata: dict = None,              # arbitrary key-value metadata
    entities: list[str] = None,         # explicit entities to extract/link
    relations: list[tuple] = None,      # (subject, predicate, object) triples
    embedding: list[float] = None,      # pre-computed embedding (optional)
    id: str = None,                     # custom memory ID
) -> str  # returns memory ID
```

**Example with explicit graph relations:**

```python
mem.remember(
    "Alice manages the ML team and reports to Bob.",
    relations=[
        ("Alice", "MANAGES", "ML Team"),
        ("Alice", "REPORTS_TO", "Bob"),
    ],
)
```

### `recall()` — Semantic Search

```python
results = mem.recall(
    query: str,                         # natural language query
    top_k: int = 5,                     # max results
    filters: dict = None,               # metadata filters
    time_window: str = None,            # e.g., "7d", "30d", "1h"
    include_graph: bool = True,         # enrich with graph context
) -> list[MemoryResult]
```

**Example:**

```python
# Find recent memories about "deployment" from a specific session
results = mem.recall(
    "When did we last deploy to production?",
    top_k=3,
    filters={"source": "slack", "session": "ops-channel"},
    time_window="14d",
)

for r in results:
    print(f"[{r.score:.2f}] {r.text}")
    if r.related_entities:
        print(f"   Entities: {r.related_entities}")
```

### `relate()` — Create Graph Edges

```python
mem.relate(
    subject: str,                       # source entity
    predicate: str,                     # relationship type
    object: str,                        # target entity
    properties: dict = None,            # edge properties
)
```

**Example:**

```python
mem.relate("Alice", "WORKS_ON", "Project Alpha", properties={"since": "2025-01"})
mem.relate("Project Alpha", "USES", "Neo4j")
```

### `query()` — Graph Traversal

```python
results = mem.query(
    start_entity: str,                  # starting node
    relation: str = None,               # edge type filter (None = any)
    depth: int = 2,                     # traversal depth
    direction: str = "OUTGOING",        # OUTGOING | INCOMING | BOTH
) -> list[GraphResult]
```

**Example:**

```python
# Find everything connected to Alice within 2 hops
connections = mem.query("Alice", depth=2, direction="BOTH")
for c in connections:
    print(f"  {c.subject} --[{c.predicate}]--> {c.object}")
#   Alice --[MANAGES]--> ML Team
#   Alice --[REPORTS_TO]--> Bob
#   ML Team --[USES]--> Neo4j
```

### `context()` — Build Agent Context Window

```python
ctx = mem.context(
    query: str,
    max_tokens: int = 2000,            # budget for context
    include_graph: bool = True,
) -> str  # formatted context string ready to inject into prompts
```

### `forget()` — Remove Memories

```python
mem.forget(memory_id="abc123")                    # by ID
mem.forget(filters={"session": "old-session"})     # by filter
mem.forget(query="outdated API endpoint", top_k=5) # by similarity
```

---

## ⚙️ Neo4j Setup

**Option A — Docker (recommended for dev):**

```bash
docker run -d \
  --name neo4j-agentmemory \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5-community
```

**Option B — Neo4j Aura (free cloud tier):**

Sign up at [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free) and pass the connection URI to `AgentMemory`.

**Option C — No graph (vectors only):**

Simply omit the `neo4j_*` parameters. AgentMemory falls back to ChromaDB-only mode with no graph features.

---

## 🔧 Configuration

```python
from agentmemory import AgentMemory

mem = AgentMemory(
    # Collection / namespace
    collection="my-agent",

    # ChromaDB (vector store)
    chroma_path="./data/chromadb",           # persist directory
    chroma_host=None,                         # or remote: "localhost:8000"

    # Neo4j (knowledge graph)
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",

    # Embedding model
    embedding_model="all-MiniLM-L6-v2",      # sentence-transformers model
    embedding_device="cpu",                   # "cuda" for GPU

    # Behavior
    auto_extract_entities=True,               # auto NER on remember()
    temporal_decay=True,                      # score decay over time
    max_memories=100_000,                     # collection size cap
)
```

---

## 🤝 Integrations

AgentMemory works with any agent framework:

```python
# LangChain
from langchain.agents import AgentExecutor
from agentmemory.integrations import LangChainMemory

memory = LangChainMemory(mem)
agent = AgentExecutor(memory=memory, ...)

# AutoGen
from agentmemory.integrations import AutoGenMemory
memory = AutoGenMemory(mem, agent_id="assistant-1")

# Raw LLM prompt injection
context = mem.context("What do I know about this user?", max_tokens=1500)
prompt = f"Based on the following context:\n{context}\n\nAnswer the question..."
```

---

## 📁 Project Structure

```
agentmemory/
├── agentmemory/
│   ├── __init__.py
│   ├── core.py              # AgentMemory main class
│   ├── vectorstore.py       # ChromaDB backend
│   ├── graphstore.py        # Neo4j backend
│   ├── embeddings.py        # sentence-transformers wrapper
│   ├── models.py            # Pydantic data models
│   ├── sync.py              # Cross-store synchronization
│   └── integrations/        # LangChain, AutoGen, etc.
├── tests/
├── docs/
├── examples/
├── pyproject.toml
└── README.md
```

---

## 🧪 Running Tests

```bash
# Unit tests (mocked backends — no infra needed)
pytest tests/unit/

# Integration tests (requires Neo4j + ChromaDB)
docker compose up -d
pytest tests/integration/
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ by <a href="https://nousresearch.com">Nous Research</a>
  · <a href="https://github.com/nousresearch/agentmemory">GitHub</a>
  · <a href="https://agentmemory.readthedocs.io">Docs</a>
  · <a href="https://discord.gg/nousresearch">Discord</a>
</p>