# 🧠 AgentMemory

**混合记忆框架** — 结合向量相似度搜索与知识图谱关系查询。纯 Python，零外部依赖（仅 numpy）。

---

## ✨ 功能特性

| 功能 | 说明 |
|---|---|
| **向量搜索** | 余弦相似度搜索 |
| **知识图谱** | 实体/关系 CRUD + BFS 遍历 |
| **混合搜索** | 向量搜索 + 图谱上下文增强 |
| **标签系统** | 记忆标签分类与过滤搜索 |
| **批量操作** | batch_remember / batch_search / batch_forget |
| **Embedding Provider** | 内置 Hash/可选 OpenAI/HuggingFace |
| **JSON/SQLite 持久化** | 人类可读文件或高性能数据库 |
| **CLI 工具** | 命令行管理记忆 |
| **导入/导出** | JSON/CSV 格式数据迁移 |
| **类型安全** | 完整类型注解 + dataclass 模型 |

---

## 🚀 快速开始

### 安装

```bash
pip install agentmemory
```

### Python API

```python
from agentmemory import HybridMemory, HashEmbeddingProvider

# 创建记忆系统（内置 Hash Embedding，零配置）
hm = HybridMemory(
    dimension=128,
    embedding_provider=HashEmbeddingProvider(dim=128),
)

# 添加记忆（自动计算向量）
hm.remember("Alice 是 Python 开发者", tags=["人物", "技术"])
hm.remember("Bob 喜欢 Rust", tags=["人物", "技术"])

# 文本搜索
results = hm.search_text("Python 开发", top_k=3)
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content} {r.memory.tags}")

# 按标签过滤搜索
code_results = hm.search_text("编程", tags=["技术"])

# 批量添加
memories = hm.batch_remember(
    contents=["记忆1", "记忆2", "记忆3"],
    tagss=[["tag1"], ["tag2"], ["tag1", "tag3"]],
)
```

### 知识图谱

```python
# 添加实体和关系
alice = hm.add_entity("Alice", "person", {"role": "developer"})
python = hm.add_entity("Python", "language")
hm.add_relation(alice.id, python.id, "knows")

# 混合搜索：向量 + 图谱上下文
results = hm.hybrid_search_text("Python", graph_depth=2)
```

### 持久化

```python
# JSON 后端 — 人类可读
hm = HybridMemory(
    dimension=128,
    embedding_provider=HashEmbeddingProvider(dim=128),
    storage_path="./data",
    storage_backend="json",
    auto_save=True,
    auto_load=True,
)

# SQLite 后端 — 大规模数据
hm = HybridMemory(
    dimension=128,
    embedding_provider=HashEmbeddingProvider(dim=128),
    storage_path="./data/memory.db",
    storage_backend="sqlite",
    auto_save=True,
    auto_load=True,
)
```

---

## 💻 CLI 工具

```bash
# 添加记忆
agentmemory remember "这是一条记忆" --tags tag1 tag2

# 搜索记忆
agentmemory search "查询内容" --top-k 5 --tags tag1

# 列出记忆
agentmemory list
agentmemory list --tag tag1

# 查看标签
agentmemory tags

# 删除记忆
agentmemory forget <memory_id>

# 统计信息
agentmemory stats

# 导出数据
agentmemory export --format json --output data.json
agentmemory export --format csv --output data.csv

# 导入数据
agentmemory import data.json
agentmemory import data.csv --format csv

# 知识图谱
agentmemory add-entity "Python" "language" --props version=3.11
agentmemory add-relation <src_id> <dst_id> "related_to"
agentmemory graph
agentmemory graph --entity-id <id>

# 使用 SQLite 后端
agentmemory --store ./data --backend sqlite stats
```

---

## 📖 API 参考

### HybridMemory

```python
HybridMemory(
    dimension: int,                              # 向量维度
    embedding_provider: EmbeddingProvider = None, # Embedding 提供者
    storage_path: str = None,                    # 持久化路径
    storage_backend: str = "json",               # "json" 或 "sqlite"
    auto_save: bool = False,                     # 自动保存
    auto_load: bool = False,                     # 自动加载
)
```

#### 记忆操作

| 方法 | 说明 |
|---|---|
| `remember(content, embedding, metadata, tags)` | 添加记忆 |
| `batch_remember(contents, embeddings, metadatas, tagss)` | 批量添加 |
| `forget(memory_id)` | 删除记忆 |
| `batch_forget(memory_ids)` | 批量删除 |
| `get_memory(memory_id)` | 获取记忆 |
| `list_all()` | 列出所有记忆 |
| `add_tag(memory_id, tag)` | 添加标签 |
| `remove_tag(memory_id, tag)` | 移除标签 |
| `get_all_tags()` | 获取所有标签及计数 |

#### 搜索操作

| 方法 | 说明 |
|---|---|
| `search(query_embedding, top_k, threshold, tags)` | 向量搜索 |
| `search_text(query, top_k, threshold, tags)` | 文本搜索 |
| `batch_search(query_embeddings, top_k, threshold, tags)` | 批量搜索 |
| `hybrid_search(query_embedding, top_k, threshold, graph_depth, tags)` | 混合搜索 |
| `hybrid_search_text(query, top_k, threshold, graph_depth, tags)` | 文本混合搜索 |

#### 知识图谱操作

| 方法 | 说明 |
|---|---|
| `add_entity(name, type, properties)` | 添加实体 |
| `add_relation(src, dst, type, weight)` | 添加关系 |
| `get_neighbors(entity_id, relation_type)` | 获取邻居 |

#### 导入/导出

| 方法 | 说明 |
|---|---|
| `export_json(pretty=True)` | 导出为 JSON 字符串 |
| `import_json(json_str, overwrite=False)` | 从 JSON 导入 |
| `export_csv()` | 导出为 CSV 字符串 |
| `import_csv(csv_str)` | 从 CSV 导入 |

#### Embedding Provider

```python
from agentmemory import HashEmbeddingProvider, OpenAIEmbeddingProvider

# 零依赖哈希 Embedding（开发/测试）
provider = HashEmbeddingProvider(dim=128)

# OpenAI Embedding（需要 pip install openai）
provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="sk-...")

# HuggingFace Embedding（需要 pip install sentence-transformers）
provider = HuggingFaceEmbeddingProvider(model="all-MiniLM-L6-v2")
```

---

## 🧪 运行测试

```bash
pytest tests/ -v
```

无需外部服务，全部纯 Python 测试。

---

## 📁 项目结构

```
agentmemory/
├── agentmemory/
│   ├── __init__.py            # 公共 API 导出
│   ├── models.py              # Memory, Entity, Relation, SearchResult
│   ├── embedding_store.py     # 向量存储 + 余弦相似度
│   ├── embedding_provider.py  # Embedding 提供者抽象层
│   ├── knowledge_graph.py     # 知识图谱 + BFS 遍历
│   ├── hybrid_memory.py       # 统一 API（搜索/标签/批量/导入导出）
│   ├── persistence.py         # JSON/SQLite 持久化后端
│   └── cli.py                 # 命令行工具
├── tests/
│   ├── test_models.py
│   ├── test_embedding_store.py
│   ├── test_embedding_provider.py
│   ├── test_knowledge_graph.py
│   ├── test_hybrid_memory.py
│   ├── test_persistence.py
│   ├── test_hybrid_persistence.py
│   ├── test_batch_tags_export.py
│   └── test_cli.py
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
