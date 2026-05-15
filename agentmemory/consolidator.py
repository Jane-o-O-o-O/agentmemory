"""记忆整合器：自动去重、合并相似记忆、压缩旧记忆。

提供三种整合策略：
1. 去重（deduplicate）：找出高度相似的记忆对，合并重复内容
2. 合并（merge_similar）：将相似但不同的记忆合并为摘要
3. 老化压缩（compress_aged）：将旧记忆压缩为摘要形式

整合过程完全在 EmbeddingStore + KnowledgeGraph 层面操作，不影响底层存储。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentmemory.embedding_store import EmbeddingStore, cosine_similarity
from agentmemory.models import Memory, SearchResult


@dataclass
class ConsolidationResult:
    """整合操作结果。

    Attributes:
        merged_count: 被合并的记忆数量
        removed_count: 被移除的记忆数量
        created_count: 新创建的记忆数量（合并后的摘要）
        details: 每次合并/移除的详细信息
        duration_ms: 操作耗时（毫秒）
    """

    merged_count: int = 0
    removed_count: int = 0
    created_count: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "merged_count": self.merged_count,
            "removed_count": self.removed_count,
            "created_count": self.created_count,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2),
        }


class MemoryConsolidator:
    """记忆整合器。

    提供自动去重、相似合并、老化压缩能力。

    Args:
        similarity_threshold: 相似度阈值（0~1），超过此值视为重复，默认 0.92
        min_age_hours: 老化压缩最小年龄（小时），默认 24
        max_content_length: 压缩后最大内容长度（字符），默认 200
        summarizer: 自定义摘要函数 (list[Memory]) -> str，不提供则用简单拼接
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        min_age_hours: float = 24.0,
        max_content_length: int = 200,
        summarizer: Optional[Callable[[list[Memory]], str]] = None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold 必须在 0~1 之间，got {similarity_threshold}")
        if min_age_hours < 0:
            raise ValueError(f"min_age_hours 不能为负，got {min_age_hours}")
        if max_content_length < 10:
            raise ValueError(f"max_content_length 不能小于 10，got {max_content_length}")

        self._similarity_threshold = similarity_threshold
        self._min_age_hours = min_age_hours
        self._max_content_length = max_content_length
        self._summarizer = summarizer

    def find_duplicates(
        self,
        store: EmbeddingStore,
        threshold: Optional[float] = None,
    ) -> list[tuple[Memory, Memory, float]]:
        """查找高度相似的记忆对。

        Args:
            store: 向量存储
            threshold: 覆盖默认相似度阈值

        Returns:
            (memory_a, memory_b, similarity) 元组列表，按相似度降序
        """
        thresh = threshold if threshold is not None else self._similarity_threshold
        all_memories = store.list_all()
        pairs: list[tuple[Memory, Memory, float]] = []

        for i in range(len(all_memories)):
            for j in range(i + 1, len(all_memories)):
                mem_a = all_memories[i]
                mem_b = all_memories[j]
                if mem_a.embedding is None or mem_b.embedding is None:
                    continue
                sim = cosine_similarity(mem_a.embedding, mem_b.embedding)
                if sim >= thresh:
                    pairs.append((mem_a, mem_b, sim))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def deduplicate(
        self,
        store: EmbeddingStore,
        on_merge: Optional[Callable[[Memory, Memory, Memory], None]] = None,
    ) -> ConsolidationResult:
        """自动去重：合并高度相似的记忆对。

        对每对重复记忆，保留较新的那条，将旧记忆的内容追加到新记忆的 metadata 中。

        Args:
            store: 向量存储
            on_merge: 合并回调 (kept, removed, merged_result)

        Returns:
            整合结果
        """
        t0 = time.time()
        result = ConsolidationResult()
        pairs = self.find_duplicates(store)

        removed_ids: set[str] = set()
        for mem_a, mem_b, sim in pairs:
            # 跳过已处理的
            if mem_a.id in removed_ids or mem_b.id in removed_ids:
                continue

            # 保留较新的
            if mem_a.created_at >= mem_b.created_at:
                kept, removed = mem_a, mem_b
            else:
                kept, removed = mem_b, mem_a

            # 合并：将被移除的记忆信息记录到 metadata
            merged_sources = kept.metadata.get("_merged_from", [])
            merged_sources.append({
                "id": removed.id,
                "content": removed.content,
                "similarity": round(sim, 4),
                "created_at": removed.created_at,
            })
            kept.metadata["_merged_from"] = merged_sources
            kept.metadata["_dedup_count"] = len(merged_sources)

            # 合并标签
            seen_tags = set(t.lower() for t in kept.tags)
            for tag in removed.tags:
                if tag.lower() not in seen_tags:
                    kept.tags.append(tag)
                    seen_tags.add(tag.lower())

            # 重新计算向量（保留新记忆的）
            store.remove(removed.id)
            removed_ids.add(removed.id)

            result.merged_count += 1
            result.removed_count += 1
            result.details.append({
                "action": "dedup",
                "kept_id": kept.id,
                "removed_id": removed.id,
                "similarity": round(sim, 4),
            })

            if on_merge:
                on_merge(kept, removed, kept)

        result.duration_ms = (time.time() - t0) * 1000
        return result

    def merge_similar(
        self,
        store: EmbeddingStore,
        threshold: Optional[float] = None,
        max_merge_size: int = 5,
    ) -> ConsolidationResult:
        """合并相似记忆为摘要。

        将相似度超过阈值的记忆聚合为一条摘要记忆。
        使用贪心聚类：以每个未处理的记忆为种子，收集所有相似记忆。

        Args:
            store: 向量存储
            threshold: 相似度阈值
            max_merge_size: 每个聚类最多合并的记忆数

        Returns:
            整合结果
        """
        t0 = time.time()
        thresh = threshold if threshold is not None else self._similarity_threshold
        result = ConsolidationResult()
        all_memories = store.list_all()

        if len(all_memories) < 2:
            result.duration_ms = (time.time() - t0) * 1000
            return result

        processed: set[str] = set()

        for seed in all_memories:
            if seed.id in processed or seed.embedding is None:
                continue

            # 贪心聚类
            cluster: list[Memory] = [seed]
            processed.add(seed.id)

            for candidate in all_memories:
                if candidate.id in processed or candidate.embedding is None:
                    continue
                if len(cluster) >= max_merge_size:
                    break

                # 与聚类中心（种子）比较
                sim = cosine_similarity(seed.embedding, candidate.embedding)
                if sim >= thresh:
                    cluster.append(candidate)
                    processed.add(candidate.id)

            if len(cluster) < 2:
                continue

            # 生成摘要
            summary = self._create_summary(cluster)

            # 创建新记忆（使用种子的向量）
            merged_mem = Memory(
                content=summary,
                embedding=seed.embedding,  # 保留种子向量
                metadata={
                    "_consolidated": True,
                    "_source_count": len(cluster),
                    "_source_ids": [m.id for m in cluster],
                    "_source_contents": [m.content[:100] for m in cluster],
                },
                tags=list({t for m in cluster for t in m.tags}),
            )

            # 删除原始记忆
            for mem in cluster:
                try:
                    store.remove(mem.id)
                except KeyError:
                    pass
                result.removed_count += 1

            # 添加摘要记忆
            store.add(merged_mem)
            result.created_count += 1
            result.merged_count += len(cluster)
            result.details.append({
                "action": "merge",
                "cluster_size": len(cluster),
                "summary_id": merged_mem.id,
                "source_ids": [m.id for m in cluster],
            })

        result.duration_ms = (time.time() - t0) * 1000
        return result

    def compress_aged(
        self,
        store: EmbeddingStore,
        lifecycle: Any,
        min_age_hours: Optional[float] = None,
    ) -> ConsolidationResult:
        """压缩老旧记忆。

        将超过指定年龄且访问频率低的记忆内容截断压缩。

        Args:
            store: 向量存储
            lifecycle: MemoryLifecycle 实例
            min_age_hours: 最小年龄（小时），覆盖默认值

        Returns:
            整合结果
        """
        t0 = time.time()
        age_hours = min_age_hours if min_age_hours is not None else self._min_age_hours
        min_age_secs = age_hours * 3600
        result = ConsolidationResult()
        now = time.time()

        for mem in store.list_all():
            age = now - mem.created_at
            if age < min_age_secs:
                continue

            access_count = lifecycle.get_access_count(mem.id)

            # 只压缩低频访问的记忆
            if access_count > 5:
                continue

            original_len = len(mem.content)
            if original_len <= self._max_content_length:
                continue

            # 截断压缩
            if self._summarizer:
                mem.content = self._summarizer([mem])
            else:
                mem.content = mem.content[: self._max_content_length] + "..."

            mem.metadata["_compressed"] = True
            mem.metadata["_original_length"] = original_len
            mem.metadata["_compressed_at"] = now

            # 重新计算向量（如果有 provider）
            # 注意：这里不重新计算，保持原始向量

            result.merged_count += 1
            result.details.append({
                "action": "compress",
                "memory_id": mem.id,
                "original_length": original_len,
                "compressed_length": len(mem.content),
                "age_hours": round(age / 3600, 1),
                "access_count": access_count,
            })

        result.duration_ms = (time.time() - t0) * 1000
        return result

    def _create_summary(self, memories: list[Memory]) -> str:
        """为一组记忆创建摘要。

        Args:
            memories: 记忆列表

        Returns:
            摘要文本
        """
        if self._summarizer:
            return self._summarizer(memories)

        # 简单策略：拼接去重内容
        contents = list(dict.fromkeys(m.content.strip() for m in memories))
        if len(contents) == 1:
            return contents[0]

        summary_parts: list[str] = []
        for c in contents:
            if len(c) > 100:
                summary_parts.append(c[:100] + "...")
            else:
                summary_parts.append(c)

        combined = " | ".join(summary_parts)
        if len(combined) > self._max_content_length:
            combined = combined[: self._max_content_length] + "..."

        return combined

    def analyze(
        self,
        store: EmbeddingStore,
    ) -> dict[str, Any]:
        """分析存储中的重复和相似情况（不修改数据）。

        Args:
            store: 向量存储

        Returns:
            分析结果字典
        """
        pairs = self.find_duplicates(store)
        all_memories = store.list_all()

        # 统计相似度分布
        sim_buckets = {"0.95-1.0": 0, "0.90-0.95": 0, "0.85-0.90": 0, "<0.85": 0}
        all_sims: list[float] = []

        for i in range(min(len(all_memories), 200)):  # 限制计算量
            for j in range(i + 1, min(len(all_memories), 200)):
                if all_memories[i].embedding and all_memories[j].embedding:
                    sim = cosine_similarity(all_memories[i].embedding, all_memories[j].embedding)
                    all_sims.append(sim)
                    if sim >= 0.95:
                        sim_buckets["0.95-1.0"] += 1
                    elif sim >= 0.90:
                        sim_buckets["0.90-0.95"] += 1
                    elif sim >= 0.85:
                        sim_buckets["0.85-0.90"] += 1
                    else:
                        sim_buckets["<0.85"] += 1

        avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0

        return {
            "total_memories": len(all_memories),
            "duplicate_pairs": len(pairs),
            "similarity_distribution": sim_buckets,
            "average_similarity": round(avg_sim, 4),
            "estimated_mergeable": len(pairs),
            "potential_savings": len(pairs),
        }
