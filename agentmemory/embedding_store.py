"""向量存储与相似度搜索，支持暴力搜索和 LSH 近似搜索。"""

from __future__ import annotations

import math
from typing import Optional

from agentmemory.models import Memory, SearchResult


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    Args:
        a: 第一个向量
        b: 第二个向量

    Returns:
        余弦相似度值 (-1 ~ 1)

    Raises:
        ValueError: 向量维度不同或为零向量
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不匹配: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量不支持余弦相似度计算")

    return dot / (norm_a * norm_b)


class EmbeddingStore:
    """基于内存的向量存储，支持余弦相似度搜索和可选 LSH 索引加速。

    当启用 use_lsh 时，搜索将使用 LSH 近似最近邻索引，
    大幅提升大规模数据（>10k）的搜索性能。

    Args:
        dimension: 向量维度
        use_lsh: 是否启用 LSH 索引（默认 False）
        lsh_tables: LSH 哈希表数量（仅 use_lsh=True 时生效）
        lsh_hyperplanes: 每个表的超平面数量（仅 use_lsh=True 时生效）
    """

    def __init__(
        self,
        dimension: int,
        use_lsh: bool = False,
        lsh_tables: int = 8,
        lsh_hyperplanes: int = 16,
    ) -> None:
        self._dimension = dimension
        self._memories: dict[str, Memory] = {}
        self._use_lsh = use_lsh
        self._lsh_index = None

        if use_lsh:
            from agentmemory.lsh_index import LSHIndex
            self._lsh_index = LSHIndex(
                dimension=dimension,
                num_tables=lsh_tables,
                num_hyperplanes=lsh_hyperplanes,
            )

    @property
    def dimension(self) -> int:
        """向量维度"""
        return self._dimension

    @property
    def use_lsh(self) -> bool:
        """是否启用 LSH 索引"""
        return self._use_lsh

    def add(self, memory: Memory) -> None:
        """添加 Memory 到存储。

        Args:
            memory: 待添加的 Memory，必须有 embedding

        Raises:
            ValueError: embedding 为空或维度不匹配
        """
        if memory.embedding is None:
            raise ValueError(f"Memory {memory.id} 没有 embedding")
        if len(memory.embedding) != self._dimension:
            raise ValueError(
                f"embedding 维度不匹配: 期望 {self._dimension}, 实际 {len(memory.embedding)}"
            )
        self._memories[memory.id] = memory

        if self._lsh_index is not None:
            self._lsh_index.add(memory.id, memory.embedding)

    def remove(self, memory_id: str) -> None:
        """根据 id 删除 Memory。

        Args:
            memory_id: 待删除的 Memory ID

        Raises:
            KeyError: 该 ID 不存在
        """
        if memory_id not in self._memories:
            raise KeyError(f"Memory {memory_id} 不存在")
        del self._memories[memory_id]

        if self._lsh_index is not None:
            try:
                self._lsh_index.remove(memory_id)
            except KeyError:
                pass

    def get(self, memory_id: str) -> Optional[Memory]:
        """根据 id 获取 Memory。

        Args:
            memory_id: Memory ID

        Returns:
            对应的 Memory，不存在返回 None
        """
        return self._memories.get(memory_id)

    def count(self) -> int:
        """返回存储中的 Memory 数量"""
        return len(self._memories)

    def list_all(self) -> list[Memory]:
        """返回所有 Memory 列表"""
        return list(self._memories.values())

    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[dict] = None, tags: Optional[list[str]] = None) -> Memory:
        """更新记忆内容、元数据或标签。

        Args:
            memory_id: 记忆 ID
            content: 新内容（可选）
            metadata: 新元数据（可选，与现有合并）
            tags: 新标签列表（可选，替换现有）

        Returns:
            更新后的 Memory

        Raises:
            KeyError: 记忆不存在
        """
        mem = self._memories.get(memory_id)
        if mem is None:
            raise KeyError(f"Memory {memory_id} 不存在")
        if content is not None:
            mem.content = content
        if metadata is not None:
            mem.metadata.update(metadata)
        if tags is not None:
            mem.tags = list(dict.fromkeys(tags))
        return mem

    def search(
        self,
        query: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """向量相似度搜索，支持标签过滤。

        当启用 LSH 且数据量较大时，自动使用 LSH 加速。

        Args:
            query: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值，低于此值的结果将被过滤
            tags: 标签过滤列表（AND 逻辑：记忆必须包含所有指定标签）

        Returns:
            按相似度降序排列的 SearchResult 列表
        """
        if len(self._memories) == 0:
            return []

        # 决定搜索范围：LSH 候选集或全量
        if self._lsh_index is not None and self._lsh_index.size() > 100:
            # LSH 加速：先获取候选集
            candidates = self._lsh_index.query(query, max_candidates=top_k * 20)
            search_pool = [
                self._memories[mid] for mid in candidates if mid in self._memories
            ]
            # 补充全量中的标签匹配（LSH 可能遗漏）
            if tags:
                for mem in self._memories.values():
                    if mem.id not in candidates and all(mem.has_tag(t) for t in tags):
                        search_pool.append(mem)
        else:
            search_pool = list(self._memories.values())

        results: list[SearchResult] = []
        for mem in search_pool:
            # 标签过滤
            if tags and not all(mem.has_tag(t) for t in tags):
                continue
            score = cosine_similarity(query, mem.embedding)  # type: ignore[arg-type]
            if score >= threshold:
                results.append(SearchResult(memory=mem, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get_all_tags(self) -> dict[str, int]:
        """获取所有标签及其使用次数。

        Returns:
            标签名称到使用次数的映射
        """
        tag_counts: dict[str, int] = {}
        for mem in self._memories.values():
            for tag in mem.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    def find_by_tag(self, tag: str) -> list[Memory]:
        """查找包含指定标签的所有记忆。

        Args:
            tag: 标签名称（不区分大小写）

        Returns:
            包含该标签的 Memory 列表
        """
        return [m for m in self._memories.values() if m.has_tag(tag)]

    def rebuild_lsh_index(self) -> None:
        """重建 LSH 索引（在大量删除操作后调用以优化性能）。"""
        if self._lsh_index is not None:
            self._lsh_index.clear()
            for mem in self._memories.values():
                if mem.embedding is not None:
                    self._lsh_index.add(mem.id, mem.embedding)
