# Changelog

## v0.7.0 (2026-05-14)

### 新增
- **向量量化集成 (compress_vectors / compressed_search)**：SQ8 标量量化（4x 压缩比）和 PQ 乘积量化集成到 HybridMemory，支持压缩后近似最近邻搜索
- **RAG 管道集成 (rag())**：一行调用完成检索→重排序→上下文组装→Prompt 生成，支持自定义 token 限制、标签过滤、混合检索
- **可观测性集成 (metrics_snapshot / health_check)**：MetricsCollector 自动追踪 remember/search/forget 的计数器、计时器、仪表盘指标；HealthChecker 检查记忆存储和 LSH 索引健康状态
- **指标导出 (metrics_json / metrics_prometheus)**：支持 JSON 和 Prometheus 文本格式导出运行时指标
- **CLI 新命令**：rag（RAG 检索增强生成）、metrics（运行时指标）、health（健康检查）、compress（向量压缩）
- **MemorySession 代理**：Session 上下文管理器新增 metrics_snapshot、health_check、rag、compress_vectors、compressed_search 代理方法

### 修复
- metrics.py check_memory_health/check_lsh_health 使用正确的 stats 字典键（memory_count 替代 total_memories）

### 改进
- __init__.py 导出 MetricsCollector、Counter、Timer、Gauge、HealthChecker、HealthStatus、HealthCheck、HealthReport、RAGPipeline、Reranker、RAGContext、RAGResult、ContextStrategy、ScalarQuantizer、ProductQuantizer、CompressedVectorStore、QuantizationStats
- 测试从 393 个增加到 505 个（+28%，+112 个新测试）
- README 新增 v0.7.0 功能文档和使用示例

## v0.6.0 (2026-05-13)

### 新增
- **搜索结果缓存 (SearchCache)**：LRU 缓存加速频繁查询，支持 TTL 过期、命中率统计、向量/文本双模式缓存
- **知识图谱可视化 (graph_viz)**：导出 Graphviz DOT 格式和交互式 HTML（vis.js Network），支持自定义颜色/布局
- **图谱统计报告 (graph_stats_text)**：实体/关系类型分布、连通分量、度数统计
- **ChromaDB 后端 (chromadb_backend)**：可选的 ChromaDB 向量数据库持久化后端，通过插件架构注册
- **HybridMemory 图谱推理方法**：shortest_path、find_all_paths、common_neighbors、connected_components、subgraph、export_dot、export_html
- **HybridMemory 缓存管理**：get_cache_stats()、clear_cache()，search_text 自动缓存
- **CLI 图谱推理命令**：shortest-path、common-neighbors、connected-components、graph-export、graph-stats、cache-stats
- **SearchCache 集成**：HybridMemory 构造参数 cache_size/cache_ttl，一行代码启用搜索缓存

### 改进
- README 全面更新：涵盖 v0.5.0 和 v0.6.0 所有功能（加权搜索、图谱推理、插件系统、aiosqlite、缓存、可视化、ChromaDB）
- __init__.py 导出 SearchCache、export_dot、export_html、graph_stats_text
- 测试从 345 个增加到 393 个（+14%）

### 内部
- 版本升级到 0.6.0
- 新增 search_cache.py、graph_viz.py、chromadb_backend.py 模块

---

## v0.5.0 (2026-05-12)

### 新增
- **多探针 LSH**：翻转 1~3 位扩大搜索范围，预计算探针模式缓存，大规模数据（1000+）召回率优化
- **加权搜索 (WeightedScorer)**：融合向量相似度(0.5) + 重要性(0.2) + 时间衰减(0.2) + 访问频率(0.1)
- **aiosqlite 异步持久化 (AsyncSQLiteBackend)**：真正的 async I/O，适合高并发 RAG 场景
- **知识图谱推理**：shortest_path(BFS)、find_all_paths(DFS)、common_neighbors、connected_components、subgraph
- **插件架构 (PluginRegistry)**：统一注册表管理 backends/providers/scorers/search_strategies
- **HybridMemory 加权搜索集成**：weighted_scoring 参数、set_scorer()/get_scorer() 方法

### 改进
- 测试从 287 个增加到 345 个（+20%）

### 内部
- 版本升级到 0.5.0
- 新增 weighted_search.py、async_persistence.py、plugins.py 模块

---

## v0.4.0 (2026-05-12)

### 新增
- **异步 API (AsyncHybridMemory)**：所有主要方法的 async 版本，支持 asyncio.gather 并发操作
- **搜索过滤器 (SearchFilter)**：按 metadata、时间范围、重要性范围、标签、内容多维过滤
- **全局搜索过滤器**：HybridMemory.set_default_filter() 设置后自动应用于所有搜索
- **Session 上下文管理器**：`with hm.session() as s:` 自动在退出时保存数据
- **Embedding 缓存 (CachedEmbeddingProvider)**：LRU 缓存避免重复计算，支持命中率统计
- **CLI interactive 命令**：交互式 REPL 模式，直接在命令行管理记忆
- **CLI batch-import 命令**：从文本文件批量导入记忆（每行一条）
- **CLI visualize 命令**：文本可视化统计（柱状图、标签分布）
- **性能基准测试**：LSH vs 暴力搜索在不同数据规模下的性能对比脚本
- **PyPI 发布配置**：LICENSE (MIT)、MANIFEST.in、完整 pyproject.toml 元数据

### 改进
- HybridMemory 新增 session() 方法返回 MemorySession
- search() 和 hybrid_search() 自动应用默认过滤器
- __init__.py 导出所有新模块
- 测试从 235 个增加到 287 个（+22%）

### 内部
- 版本升级到 0.4.0
- 新增 async_api.py、embedding_cache.py、search_filter.py 模块
- 项目结构新增 LICENSE、MANIFEST.in

---

## v0.3.0 (2026-05-12)

### 新增
- **LSH 近似最近邻索引**：纯 Python 实现的局部敏感哈希索引，O(1) 近似查找，适合大规模数据（>10k）
- **记忆生命周期管理**：TTL 过期、重要性评分、时间衰减、访问频率追踪
- **HybridMemory.update_memory()**：更新记忆内容/元数据/标签，自动重新计算 embedding
- **HybridMemory.merge_memories()**：合并多条记忆为一条，标签去重，元数据合并
- **HybridMemory.forget_where()**：按条件批量删除记忆
- **HybridMemory.cleanup_expired()**：自动清理过期记忆
- **HybridMemory.get_lifecycle_info()**：查看记忆生命周期详情
- **remember() 支持 importance 和 ttl 参数**
- **CLI version 子命令**：显示版本信息
- **CLI inspect 子命令**：查看记忆详情和生命周期信息
- **CLI cleanup 子命令**：清理过期记忆
- **CLI search --hybrid 标志**：直接使用混合搜索
- **CLI --lsh 全局标志**：启用 LSH 索引

### 改进
- EmbeddingStore 支持可选 LSH 索引加速搜索
- stats() 输出增加 LSH 状态和维度信息
- MemoryLifecycle 模块独立可用
- LSHIndex 模块独立可用
- 测试从 168 个增加到 235 个

### 内部
- 版本升级到 0.3.0
- 新增 lifecycle.py 和 lsh_index.py 模块

---

## v0.2.0 (2026-05-11)

### 新增
- CLI 工具：命令行管理记忆
- 批量操作：batch_remember / batch_search / batch_forget
- 标签系统：记忆标签分类与过滤搜索
- 导入/导出：JSON/CSV 格式数据迁移

### 改进
- EmbeddingProvider 抽象层：Hash / OpenAI / HuggingFace
- 文本搜索 API：search_text / hybrid_search_text

---

## v0.1.0 (2026-05-10)

### 初始版本
- 向量存储与余弦相似度搜索
- 知识图谱：实体/关系 CRUD + BFS 遍历
- 混合搜索：向量相似度 + 图谱上下文
- JSON/SQLite 持久化后端
- 完整类型注解和 dataclass 模型
