"""内置性能基准测试套件。

提供 AgentMemory 各核心模块的性能基准测试：
- 向量存储插入/搜索
- 知识图谱 CRUD
- LSH 索引搜索
- 混合检索
- 批量操作
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BenchmarkResult:
    """单个基准测试结果。

    Attributes:
        name: 测试名称
        iterations: 迭代次数
        total_ms: 总耗时（毫秒）
        ops_per_second: 每秒操作数
        avg_ms: 平均耗时（毫秒）
        min_ms: 最小耗时（毫秒）
        max_ms: 最大耗时（毫秒）
        p50_ms: P50 耗时（毫秒）
        p95_ms: P95 耗时（毫秒）
        p99_ms: P99 耗时（毫秒）
    """

    name: str
    iterations: int
    total_ms: float
    ops_per_second: float
    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict。"""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_ms": round(self.total_ms, 3),
            "ops_per_second": round(self.ops_per_second, 1),
            "avg_ms": round(self.avg_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
        }

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.ops_per_second:.0f} ops/s "
            f"(avg={self.avg_ms:.3f}ms, p50={self.p50_ms:.3f}ms, "
            f"p95={self.p95_ms:.3f}ms, p99={self.p99_ms:.3f}ms) "
            f"[{self.iterations} iters]"
        )


@dataclass
class BenchmarkSuite:
    """基准测试套件结果。

    Attributes:
        name: 套件名称
        results: 各测试结果列表
        total_ms: 总耗时
    """

    name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict。"""
        return {
            "name": self.name,
            "results": [r.to_dict() for r in self.results],
            "total_ms": round(self.total_ms, 3),
        }

    def summary(self) -> str:
        """生成摘要文本。"""
        lines = [f"=== {self.name} ==="]
        for r in self.results:
            lines.append(f"  {r}")
        lines.append(f"  Total: {self.total_ms:.1f}ms")
        return "\n".join(lines)


def _percentile(sorted_values: list[float], p: float) -> float:
    """计算百分位数。"""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * p / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def run_benchmark(
    name: str,
    fn: Callable[[], Any],
    iterations: int = 1000,
    warmup: int = 10,
) -> BenchmarkResult:
    """运行单个基准测试。

    Args:
        name: 测试名称
        fn: 待测函数
        iterations: 迭代次数
        warmup: 预热次数

    Returns:
        BenchmarkResult 结果
    """
    # 预热
    for _ in range(warmup):
        fn()

    timings: list[float] = []
    start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)  # ms
    total_ms = (time.perf_counter() - start) * 1000

    timings.sort()
    avg_ms = sum(timings) / len(timings)
    ops_per_second = (iterations / total_ms) * 1000 if total_ms > 0 else 0

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_ms=total_ms,
        ops_per_second=ops_per_second,
        avg_ms=avg_ms,
        min_ms=timings[0],
        max_ms=timings[-1],
        p50_ms=_percentile(timings, 50),
        p95_ms=_percentile(timings, 95),
        p99_ms=_percentile(timings, 99),
    )


def benchmark_embedding_store(
    dimension: int = 128,
    num_items: int = 1000,
    search_k: int = 10,
    iterations: int = 100,
) -> BenchmarkSuite:
    """向量存储基准测试。

    Args:
        dimension: 向量维度
        num_items: 预填充数量
        search_k: 搜索返回数量
        iterations: 测试迭代次数

    Returns:
        BenchmarkSuite 结果
    """
    from agentmemory.embedding_store import EmbeddingStore
    from agentmemory.models import Memory

    suite = BenchmarkSuite(name="EmbeddingStore")
    store = EmbeddingStore(dimension=dimension)

    # 预填充
    ids: list[str] = []
    for i in range(num_items):
        vec = [random.random() for _ in range(dimension)]
        m = Memory(content=f"item_{i}", embedding=vec)
        store.add(m)
        ids.append(m.id)

    # 插入测试
    def insert_one():
        vec = [random.random() for _ in range(dimension)]
        m = Memory(content=f"new_{random.randint(0,999999)}", embedding=vec)
        store.add(m)

    suite.results.append(run_benchmark("insert", insert_one, iterations=iterations))

    # 搜索测试
    query = [random.random() for _ in range(dimension)]

    def search():
        store.search(query, top_k=search_k)

    suite.results.append(run_benchmark("search", search, iterations=iterations))

    # 删除测试
    delete_idx = [0]

    def delete_one():
        if delete_idx[0] < len(ids):
            try:
                store.remove(ids[delete_idx[0]])
            except KeyError:
                pass
            delete_idx[0] += 1

    suite.results.append(run_benchmark("remove", delete_one, iterations=min(iterations, len(ids))))

    suite.total_ms = sum(r.total_ms for r in suite.results)
    return suite


def benchmark_knowledge_graph(
    num_entities: int = 500,
    num_relations: int = 1000,
    iterations: int = 100,
) -> BenchmarkSuite:
    """知识图谱基准测试。

    Args:
        num_entities: 预填充实体数量
        num_relations: 预填充关系数量
        iterations: 测试迭代次数

    Returns:
        BenchmarkSuite 结果
    """
    from agentmemory.knowledge_graph import KnowledgeGraph
    from agentmemory.models import Entity, Relation

    suite = BenchmarkSuite(name="KnowledgeGraph")
    kg = KnowledgeGraph()

    # 预填充
    entity_ids: list[str] = []
    for i in range(num_entities):
        e = Entity(name=f"entity_{i}", entity_type="test")
        kg.add_entity(e)
        entity_ids.append(e.id)

    for i in range(num_relations):
        src = random.choice(entity_ids)
        dst = random.choice(entity_ids)
        if src != dst:
            r = Relation(source_id=src, target_id=dst, relation_type=f"rel_{i % 5}")
            kg.add_relation(r)

    # 实体查询
    def query_entity():
        kg.get_entity(random.choice(entity_ids))

    suite.results.append(run_benchmark("get_entity", query_entity, iterations=iterations))

    # 关系查询
    def query_relations():
        kg.find_relations(source_id=random.choice(entity_ids))

    suite.results.append(run_benchmark("find_relations", query_relations, iterations=iterations))

    # BFS 遍历
    def bfs():
        kg.bfs(random.choice(entity_ids), max_depth=3)

    suite.results.append(run_benchmark("bfs_depth3", bfs, iterations=min(iterations, 50)))

    suite.total_ms = sum(r.total_ms for r in suite.results)
    return suite


def benchmark_lsh_index(
    dimension: int = 128,
    num_items: int = 5000,
    iterations: int = 100,
) -> BenchmarkSuite:
    """LSH 索引基准测试。

    Args:
        dimension: 向量维度
        num_items: 预填充数量
        iterations: 测试迭代次数

    Returns:
        BenchmarkSuite 结果
    """
    from agentmemory.lsh_index import LSHIndex

    suite = BenchmarkSuite(name="LSHIndex")
    lsh = LSHIndex(dimension=dimension, num_tables=8, num_hyperplanes=16)

    # 预填充
    ids: list[str] = []
    for i in range(num_items):
        vec = [random.random() for _ in range(dimension)]
        mid = f"item_{i}"
        lsh.add(mid, vec)
        ids.append(mid)

    # 查询测试
    query = [random.random() for _ in range(dimension)]

    def search():
        lsh.query(query, max_candidates=10)

    suite.results.append(run_benchmark("query_k10", search, iterations=iterations))

    # 插入测试
    counter = [num_items]

    def insert():
        vec = [random.random() for _ in range(dimension)]
        mid = f"new_{counter[0]}"
        lsh.add(mid, vec)
        counter[0] += 1

    suite.results.append(run_benchmark("insert", insert, iterations=iterations))

    suite.total_ms = sum(r.total_ms for r in suite.results)
    return suite


def benchmark_hybrid_memory(
    dimension: int = 64,
    num_memories: int = 200,
    iterations: int = 50,
) -> BenchmarkSuite:
    """混合记忆系统基准测试。

    Args:
        dimension: 向量维度
        num_memories: 预填充数量
        iterations: 测试迭代次数

    Returns:
        BenchmarkSuite 结果
    """
    from agentmemory import HybridMemory, HashEmbeddingProvider

    suite = BenchmarkSuite(name="HybridMemory")
    hm = HybridMemory(
        dimension=dimension,
        embedding_provider=HashEmbeddingProvider(dim=dimension),
    )

    # 预填充
    for i in range(num_memories):
        hm.remember(f"Memory content number {i} about topic {i % 20}", tags=[f"tag_{i % 10}"])

    # remember 测试
    counter = [num_memories]

    def remember():
        hm.remember(f"New memory {counter[0]}")
        counter[0] += 1

    suite.results.append(run_benchmark("remember", remember, iterations=iterations))

    # search_text 测试
    def search():
        hm.search_text("topic 5", top_k=5)

    suite.results.append(run_benchmark("search_text", search, iterations=iterations))

    # list_memories 测试
    def list_mem():
        list(hm.list_all())

    suite.results.append(run_benchmark("list_memories", list_mem, iterations=iterations))

    suite.total_ms = sum(r.total_ms for r in suite.results)
    return suite


def run_all(
    dimension: int = 64,
    iterations: int = 50,
) -> BenchmarkSuite:
    """运行所有基准测试。

    Args:
        dimension: 向量维度（较小值加速测试）
        iterations: 每个测试的迭代次数

    Returns:
        BenchmarkSuite 汇总结果
    """
    suite = BenchmarkSuite(name="AgentMemory Full Benchmark")

    suite.results.extend(
        benchmark_embedding_store(
            dimension=dimension, num_items=500, iterations=iterations
        ).results
    )
    suite.results.extend(
        benchmark_knowledge_graph(
            num_entities=200, num_relations=400, iterations=iterations
        ).results
    )
    suite.results.extend(
        benchmark_lsh_index(
            dimension=dimension, num_items=2000, iterations=iterations
        ).results
    )
    suite.results.extend(
        benchmark_hybrid_memory(
            dimension=dimension, num_memories=100, iterations=iterations
        ).results
    )

    suite.total_ms = sum(r.total_ms for r in suite.results)
    return suite
