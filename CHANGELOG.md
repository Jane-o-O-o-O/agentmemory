# Changelog

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
