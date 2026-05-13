# 🧠 AgentMemory

**混合记忆框架** — 结合向量相似度搜索与知识图谱关系查询。纯 Python，零外部依赖（仅 numpy）。

---

## ✨ 功能特性

| 功能 | 说明 |
|---|---|
| **向量搜索** | 余弦相似度搜索 + LSH 近似加速 |
| **多探针 LSH** | 翻转多位扩大搜索，大规模数据召回率优化 |
| **知识图谱** | 实体/关系 CRUD + BFS 遍历 + 图谱推理 |
| **图谱推理** | 最短路径、所有路径、共同邻居、连通分量、子图提取 |
| **图谱可视化** | 导出 Graphviz DOT / 交互式 HTML（vis.js） |
| **混合搜索** | 向量搜索 + 图谱上下文增强 |
| **加权搜索** | 融合相似度 + 重要性 + 时间衰减 + 访问频率排序 |
| **搜索缓存** | LRU 缓存加速频繁查询，支持 TTL 过期 |
| **搜索过滤器** | 按 metadata/时间/重要性/标签多维过滤 |
| **标签系统** | 记忆标签分类与过滤搜索 |
| **生命周期** | TTL 过期、重要性评分、时间衰减 |
| **批量操作** | batch_remember / batch_search / batch_forget |
| **高级 API** | update_memory / merge_memories / forget_where |
| **异步 API** | AsyncHybridMemory，支持 asyncio.gather 并发 |
| **异步持久化** | aiosqlite 真异步 SQLite 后端 |
| **Session 管理** | with hm.session() 自动 save/load |
| **Embedding 缓存** | CachedEmbeddingProvider LRU 缓存避免重复计算 |
| **Embedding Provider** | 内置 Hash/可选 OpenAI/HuggingFace |
| **插件架构** | PluginRegistry 统一管理后端/Provider/评分策略 |
| **ChromaDB 后端** | 可选 ChromaDB 向量数据库后端（需安装 chromadb） |
| **JSON/SQLite 持久化** | 人类可读文件或高性能数据库 |
| **CLI 工具** | 命令行管理记忆（含 interactive/batch-import/visualize/graph） |
| **导入/导出** | JSON/CSV 格式数据迁移 |
| **类型安全** | 完整类型注解 + dataclass 模型 |

---

## 🚀 快速开始

### 安装

```bash
pip install agentmemory

# 可选：ChromaDB 后端
pip install agentmemory chromadb
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

### 搜索缓存

```python
# 启用搜索缓存，加速频繁查询
hm = HybridMemory(
    dimension=128,
    embedding_provider=HashEmbeddingProvider(dim=128),
    cache_size=256,   # LRU 缓存容量
    cache_ttl=300,    # 缓存 TTL（秒），None 表示永不过期
)

hm.remember("Python 开发")
hm.search_text("Python")  # 首次：计算向量
hm.search_text("Python")  # 命中缓存，跳过向量计算

print(hm.get_cache_stats())  # {'hits': 1, 'misses': 1, 'hit_rate': 0.5, ...}
```

### 加权搜索

```python
from agentmemory import HybridMemory, HashEmbeddingProvider, ScoringWeights

# 一行启用加权搜索
hm = HybridMemory(
    dimension=128,
    embedding_provider=HashEmbeddingProvider(dim=128),
    weighted_scoring=True,
)

# 自定义权重
weights = ScoringWeights(
    similarity=0.6,   # 向量相似度
    importance=0.2,   # 记忆重要性
    recency=0.15,     # 时间新鲜度
    frequency=0.05,   # 访问频率
)
hm = HybridMemory(
    dimension=128,
    embedding_provider=HashEmbeddingProvider(dim=128),
    weighted_scoring=True,
    scoring_weights=weights,
)

# 添加高重要性记忆
hm.remember("关键决策", importance=0.9)
```

### Session 会话管理

```python
# 使用 context manager 自动管理持久化
hm = HybridMemory(
    dimension=128,
    embedding_provider=HashEmbeddingProvider(dim=128),
    storage_path="./data",
    auto_save=True,
)

with hm.session() as s:
    s.remember("重要笔记", tags=["笔记"])
    s.remember("会议记录", tags=["会议"])
    # 退出时自动保存
```

### 搜索过滤器

```python
from agentmemory import SearchFilter

# 多维度过滤
f = SearchFilter(
    tags=["重要"],
    content_contains=["报告"],
    created_after=time.time() - 86400,  # 最近 24 小时
    metadata_filters={"source": "api"},
)

# 全局过滤器 — 所有搜索自动应用
hm.set_default_filter(f)
results = hm.search_text("季度报告")  # 自动过滤
```

### 异步 API

```python
import asyncio
from agentmemory import HybridMemory, HashEmbeddingProvider, AsyncHybridMemory

async def main():
    hm = HybridMemory(dimension=128, embedding_provider=HashEmbeddingProvider(dim=128))
    async with AsyncHybridMemory(hm) as am:
        # 并发添加
        await am.abatch_remember(["记忆A", "记忆B", "记忆C"])

        # 并发搜索
        results = await am.abatch_search([query1, query2])

        # 单条操作
        mem = await am.aremember("异步记忆")
        results = await am.asearch_text("搜索")

asyncio.run(main())
```

### 知识图谱 + 推理

```python
# 添加实体和关系
alice = hm.add_entity("Alice", "person", {"role": "developer"})
python = hm.add_entity("Python", "language")
bob = hm.add_entity("Bob", "person")

hm.add_relation(alice.id, python.id, "knows")
hm.add_relation(bob.id, python.id, "knows")
hm.add_relation(alice.id, bob.id, "colleague")

# 图谱推理
path = hm.shortest_path(alice.id, bob.id)       # 最短路径
paths = hm.find_all_paths(alice.id, python.id)   # 所有路径
common = hm.common_neighbors(alice.id, bob.id)   # 共同邻居
components = hm.connected_components()            # 连通分量
sub = hm.subgraph({alice.id, python.id})          # 子图提取

# 混合搜索：向量 + 图谱上下文
results = hm.hybrid_search_text("Python", graph_depth=2)

# 图谱可视化导出
dot_str = hm.export_dot(title="My Graph")      # Graphviz DOT
html_str = hm.export_html(title="My Graph")    # 交互式 HTML
```

### LSH 加速搜索

```python
# 启用 LSH 索引，适合大规模数据（>10k 条记忆）
hm = HybridMemory(
    dimension=128,
    embedding_provider=HashEmbeddingProvider(dim=128),
    use_lsh=True,
    lsh_tables=8,       # 哈希表数量（越多召回率越高）
    lsh_hyperplanes=16,  # 每表超平面数（越细粒度）
)

# 搜索会自动使用 LSH 加速
results = hm.search_text("查询内容", top_k=5)
```

### 记忆生命周期

```python
# 添加带 TTL 的记忆（300 秒后过期）
hm.remember("临时通知", ttl=300)

# 添加高重要性记忆
hm.remember("关键决策", importance=0.9)

# 查看生命周期信息
info = hm.get_lifecycle_info(memory_id)
print(info["decay_factor"])     # 时间衰减因子
print(info["access_count"])     # 访问次数
print(info["time_remaining"])   # 剩余存活时间

# 清理过期记忆
expired_ids = hm.cleanup_expired()

# 按条件删除
deleted = hm.forget_where(lambda m: "临时" in m.content)
```

### 高级操作

```python
# 更新记忆（自动重新计算 embedding）
hm.update_memory(memory_id, content="新内容", tags=["新标签"])

# 合并记忆
merged = hm.merge_memories([id1, id2, id3], new_content="合并结果")
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

# ChromaDB 后端 — 原生向量数据库（需 pip install chromadb）
from agentmemory.chromadb_backend import ChromaDBBackend
backend = ChromaDBBackend("./chroma_data")
```

### 插件架构

```python
from agentmemory import PluginRegistry, get_registry

# 注册自定义后端
registry = get_registry()
registry.register_backend("my_backend", MyBackendClass)

# 列出所有插件
print(registry.list_all())
# {'backends': ['json', 'sqlite', 'my_backend'], 'providers': [...], ...}
```

---

## 💻 CLI 工具

```bash
# 添加记忆
agentmemory remember "这是一条记忆" --tags tag1 tag2

# 搜索记忆
agentmemory search "查询内容" --top-k 5 --tags tag1

# 混合搜索（向量 + 图谱）
agentmemory search "查询内容" --hybrid

# 列出记忆
agentmemory list
agentmemory list --tag tag1

# 查看记忆详情（含生命周期信息）
agentmemory inspect <memory_id>

# 查看标签
agentmemory tags

# 删除记忆
agentmemory forget <memory_id>

# 清理过期记忆
agentmemory cleanup

# 统计信息
agentmemory stats

# 版本信息
agentmemory version

# 导出数据
agentmemory export --format json --output data.json
agentmemory export --format csv --output data.csv

# 导入数据
agentmemory import data.json
agentmemory import data.csv --format csv

# 批量导入（每行一条记忆）
agentmemory batch-import words.txt --tags vocabulary

# 交互式 REPL
agentmemory interactive --store ./data

# 可视化统计
agentmemory visualize

# 知识图谱
agentmemory add-entity "Python" "language" --props version=3.11
agentmemory add-relation <src_id> <dst_id> "related_to"
agentmemory graph
agentmemory graph --entity-id <id>

# 图谱推理
agentmemory shortest-path <source_id> <target_id>
agentmemory common-neighbors <entity1_id> <entity2_id>
agentmemory connected-components

# 图谱可视化导出
agentmemory graph-export --format html --output graph.html
agentmemory graph-export --format dot --output graph.dot
agentmemory graph-stats

# 搜索缓存统计
agentmemory cache-stats

# 使用 SQLite 后端 + LSH 加速
agentmemory --store ./data --backend sqlite --lsh stats
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
    use_lsh: bool = False,                       # 启用 LSH 索引
    lsh_tables: int = 8,                         # LSH 哈希表数量
    lsh_hyperplanes: int = 16,                   # LSH 超平面数量
    default_ttl: float = None,                   # 默认 TTL（秒）
    decay_rate: float = 0.001,                   # 衰减速率
    weighted_scoring: bool = False,               # 启用加权搜索
    scoring_weights: ScoringWeights = None,       # 自定义评分权重
    cache_size: int = 0,                         # 搜索缓存容量（0=禁用）
    cache_ttl: float = None,                     # 缓存 TTL（秒）
)
```

#### 记忆操作

| 方法 | 说明 |
|---|---|
| `remember(content, embedding, metadata, tags, importance, ttl)` | 添加记忆 |
| `update_memory(memory_id, content, metadata, tags)` | 更新记忆 |
| `merge_memories(memory_ids, new_content)` | 合并记忆 |
| `batch_remember(contents, embeddings, metadatas, tagss)` | 批量添加 |
| `forget(memory_id)` | 删除记忆 |
| `forget_where(predicate)` | 按条件删除 |
| `batch_forget(memory_ids)` | 批量删除 |
| `get_memory(memory_id)` | 获取记忆 |
| `list_all()` | 列出所有记忆 |
| `add_tag(memory_id, tag)` | 添加标签 |
| `remove_tag(memory_id, tag)` | 移除标签 |
| `get_all_tags()` | 获取所有标签及计数 |
| `session()` | 创建上下文管理器 |

#### 搜索操作

| 方法 | 说明 |
|---|---|
| `search(query_embedding, top_k, threshold, tags)` | 向量搜索 |
| `search_text(query, top_k, threshold, tags)` | 文本搜索（支持缓存） |
| `batch_search(query_embeddings, top_k, threshold, tags)` | 批量搜索 |
| `hybrid_search(query_embedding, top_k, threshold, graph_depth, tags)` | 混合搜索 |
| `hybrid_search_text(query, top_k, threshold, graph_depth, tags)` | 文本混合搜索 |

#### 知识图谱操作

| 方法 | 说明 |
|---|---|
| `add_entity(name, type, properties)` | 添加实体 |
| `add_relation(src, dst, type, weight)` | 添加关系 |
| `get_neighbors(entity_id, relation_type)` | 获取邻居 |
| `shortest_path(source_id, target_id, max_depth)` | 最短路径 |
| `find_all_paths(source_id, target_id, max_depth, max_paths)` | 所有路径 |
| `common_neighbors(entity_id_1, entity_id_2)` | 共同邻居 |
| `connected_components()` | 连通分量 |
| `subgraph(entity_ids)` | 子图提取 |
| `export_dot(title)` | 导出 DOT 格式 |
| `export_html(title)` | 导出 HTML 格式 |

#### 缓存与评分

| 方法 | 说明 |
|---|---|
| `get_cache_stats()` | 获取缓存统计 |
| `clear_cache()` | 清空缓存 |
| `set_scorer(scorer)` | 设置加权评分器 |
| `get_scorer()` | 获取当前评分器 |

#### 搜索过滤器

| 方法/类 | 说明 |
|---|---|
| `set_default_filter(filter)` | 设置全局搜索过滤器 |
| `SearchFilter(tags, metadata_filters, ...)` | 多维度过滤器 |

#### 生命周期

| 方法 | 说明 |
|---|---|
| `get_lifecycle_info(memory_id)` | 获取生命周期信息 |
| `cleanup_expired()` | 清理过期记忆 |

#### 导入/导出

| 方法 | 说明 |
|---|---|
| `export_json(pretty=True)` | 导出为 JSON 字符串 |
| `import_json(json_str, overwrite=False)` | 从 JSON 导入 |
| `export_csv()` | 导出为 CSV 字符串 |
| `import_csv(csv_str)` | 从 CSV 导入 |

### SearchCache

```python
from agentmemory import SearchCache

cache = SearchCache(max_size=256, ttl_seconds=300)
cache.put("query", results, top_k=5)
cached = cache.get("query", top_k=5)
print(cache.stats)  # {'hits': N, 'misses': N, 'hit_rate': 0.XX, ...}
```

### Embedding Provider

```python
from agentmemory import HashEmbeddingProvider, OpenAIEmbeddingProvider

# 零依赖哈希 Embedding（开发/测试）
provider = HashEmbeddingProvider(dim=128)

# OpenAI Embedding（需要 pip install openai）
provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="sk-...")

# HuggingFace Embedding（需要 pip install sentence-transformers）
provider = HuggingFaceEmbeddingProvider(model="all-MiniLM-L6-v2")
```

### AsyncHybridMemory

```python
from agentmemory import AsyncHybridMemory

am = AsyncHybridMemory(hm, max_workers=4)
# 或
async with AsyncHybridMemory(hm) as am:
    await am.aremember("text")
    await am.asearch_text("query")
    await am.abatch_remember(["a", "b", "c"])
```

### Graph Visualization

```python
from agentmemory import export_dot, export_html, graph_stats_text

# Graphviz DOT 格式
dot = export_dot(kg, title="My Graph", rankdir="LR")
with open("graph.dot", "w") as f:
    f.write(dot)

# 交互式 HTML（vis.js）
html = export_html(kg, title="My Graph")
with open("graph.html", "w") as f:
    f.write(html)

# 文本统计报告
print(graph_stats_text(kg))
```

---

## 🧪 运行测试

```bash
pytest tests/ -v
```

无需外部服务，全部纯 Python 测试（393 个测试）。

---

## 📁 项目结构

```
agentmemory/
├── agentmemory/
│   ├── __init__.py            # 公共 API 导出
│   ├── models.py              # Memory, Entity, Relation, SearchResult
│   ├── embedding_store.py     # 向量存储 + 余弦相似度 + LSH 加速
│   ├── embedding_provider.py  # Embedding 提供者抽象层
│   ├── embedding_cache.py     # LRU 缓存的 EmbeddingProvider
│   ├── knowledge_graph.py     # 知识图谱 + BFS 遍历 + 图谱推理
│   ├── hybrid_memory.py       # 统一 API（搜索/标签/批量/生命周期/缓存/推理）
│   ├── async_api.py           # 异步 API 封装
│   ├── search_filter.py       # 搜索过滤器
│   ├── search_cache.py        # 搜索结果 LRU 缓存
│   ├── persistence.py         # JSON/SQLite 持久化后端
│   ├── async_persistence.py   # aiosqlite 异步持久化
│   ├── chromadb_backend.py    # ChromaDB 向量数据库后端
│   ├── lsh_index.py           # 多探针 LSH 近似最近邻索引
│   ├── lifecycle.py           # 记忆生命周期管理
│   ├── weighted_search.py     # 加权搜索排序
│   ├── plugins.py             # 插件注册架构
│   ├── graph_viz.py           # 知识图谱可视化（DOT/HTML）
│   └── cli.py                 # 命令行工具（含 graph 推理命令）
├── tests/
│   ├── test_*.py              # 测试文件
│   └── benchmark.py           # 性能基准测试
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## 📄 License

MIT

---

<p align="center">
  Built with ❤️ by <a href="https://nousresearch.com">Nous Research</a>
  · <a href="https://github.com/Jane-o-O-o-O/agentmemory">GitHub</a>
</p>
