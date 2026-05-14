"""RAG Pipeline — 基于混合记忆的检索增强生成管道。

支持查询→检索→重排序→上下文组装的完整 RAG 流程。
可配置 token 限制、上下文窗口策略、相关性阈值等。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from agentmemory.models import Memory, SearchResult


class ContextStrategy(Enum):
    """上下文组装策略"""

    RELEVANCE = "relevance"  # 按相关性分数排序
    RECENCY = "recency"  # 按时间排序（最新优先）
    HYBRID = "hybrid"  # 综合相关性 + 时间新鲜度
    DIVERSIFIED = "diversified"  # 多样性最大化（去重类似内容）


@dataclass
class RAGContext:
    """RAG 上下文组装结果。

    Attributes:
        text: 组装后的上下文文本
        sources: 引用的 Memory 列表
        total_tokens: 估算的 token 数量
        truncated: 是否发生了截断
        assembly_time_ms: 上下文组装耗时（毫秒）
    """

    text: str
    sources: list[Memory] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False
    assembly_time_ms: float = 0.0


@dataclass
class RAGResult:
    """RAG 管道完整结果。

    Attributes:
        query: 原始查询
        context: 组装的 RAG 上下文
        prompt: 最终生成用 prompt
        search_results: 检索结果
        reranked: 是否执行了重排序
        pipeline_time_ms: 管道总耗时（毫秒）
    """

    query: str
    context: RAGContext
    prompt: str
    search_results: list[SearchResult] = field(default_factory=list)
    reranked: bool = False
    pipeline_time_ms: float = 0.0


def estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数量。

    使用简单的启发式规则：英文约 4 字符/token，中文约 1.5 字符/token。

    Args:
        text: 输入文本

    Returns:
        估算的 token 数量
    """
    if not text:
        return 0
    # 计算中文字符数
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - cjk_count
    return int(cjk_count / 1.5 + other_count / 4)


class Reranker:
    """搜索结果重排序器。

    支持基于多种信号的重排序：相关性分数、时间新鲜度、多样性。

    Args:
        freshness_weight: 时间新鲜度权重 (0~1)
        diversity_weight: 多样性权重 (0~1)
        min_score: 最低相关性阈值
    """

    def __init__(
        self,
        freshness_weight: float = 0.3,
        diversity_weight: float = 0.0,
        min_score: float = 0.0,
    ) -> None:
        if not 0.0 <= freshness_weight <= 1.0:
            raise ValueError(f"freshness_weight 必须在 0~1 之间: {freshness_weight}")
        if not 0.0 <= diversity_weight <= 1.0:
            raise ValueError(f"diversity_weight 必须在 0~1 之间: {diversity_weight}")
        self._freshness_weight = freshness_weight
        self._diversity_weight = diversity_weight
        self._min_score = min_score

    def rerank(
        self, results: list[SearchResult], max_results: Optional[int] = None
    ) -> list[SearchResult]:
        """重排序搜索结果。

        Args:
            results: 原始搜索结果
            max_results: 最大返回数量

        Returns:
            重排序后的搜索结果
        """
        if not results:
            return []

        now = time.time()
        max_age = max(now - r.memory.created_at for r in results) or 1.0

        scored: list[tuple[float, SearchResult]] = []
        for r in results:
            if r.score < self._min_score:
                continue
            # 基础分 = 原始相关性
            combined = r.score * (1.0 - self._freshness_weight - self._diversity_weight)

            # 时间新鲜度
            age = now - r.memory.created_at
            freshness = 1.0 - (age / max_age) if max_age > 0 else 1.0
            combined += freshness * self._freshness_weight

            # 多样性（简易：与已选结果的内容差异度）
            combined += self._diversity_weight * 0.5  # 无历史上下文时给中间值

            scored.append((combined, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = [sr for _, sr in scored]
        if max_results is not None:
            reranked = reranked[:max_results]
        return reranked

    def rerank_diversified(
        self,
        results: list[SearchResult],
        max_results: Optional[int] = None,
        similarity_fn: Optional[Callable[[list[float], list[float]], float]] = None,
    ) -> list[SearchResult]:
        """多样性重排序：MMR (Maximal Marginal Relevance) 算法。

        逐步选择结果，每次选择与已选集合差异最大的高分结果。

        Args:
            results: 原始搜索结果
            max_results: 最大返回数量
            similarity_fn: 向量相似度函数（用于多样性计算）

        Returns:
            多样性排序后的搜索结果
        """
        if not results:
            return []

        max_results = max_results or len(results)
        results_with_emb = [r for r in results if r.memory.embedding is not None]

        if not results_with_emb or similarity_fn is None:
            # 没有向量或没有相似度函数，降级为普通排序
            return sorted(results, key=lambda r: r.score, reverse=True)[:max_results]

        selected: list[SearchResult] = []
        candidates = list(results_with_emb)

        while candidates and len(selected) < max_results:
            best_idx = -1
            best_mmr = float("-inf")

            for i, c in enumerate(candidates):
                # 与已选结果的最大相似度
                max_sim = 0.0
                for s in selected:
                    sim = similarity_fn(c.memory.embedding, s.memory.embedding)
                    max_sim = max(max_sim, sim)

                # MMR = λ * relevance - (1-λ) * max_similarity_to_selected
                lam = 1.0 - self._diversity_weight
                mmr = lam * c.score - (1.0 - lam) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))
            else:
                break

        return selected


class RAGPipeline:
    """检索增强生成 (RAG) 管道。

    整合检索、重排序、上下文组装的完整流程。

    Args:
        memory: HybridMemory 实例
        max_context_tokens: 上下文最大 token 数
        context_strategy: 上下文组装策略
        prompt_template: 提示词模板（支持 {context} 和 {query} 占位符）
        reranker: 重排序器（可选）
        top_k: 检索结果数量
        min_score: 最低相关性阈值
        separator: 上下文片段之间的分隔符
    """

    DEFAULT_TEMPLATE = (
        "基于以下上下文信息回答问题。\n\n"
        "## 上下文\n{context}\n\n"
        "## 问题\n{query}\n\n"
        "请根据上下文给出准确、简洁的回答。"
    )

    def __init__(
        self,
        memory: Any,  # HybridMemory
        max_context_tokens: int = 2000,
        context_strategy: ContextStrategy = ContextStrategy.RELEVANCE,
        prompt_template: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        separator: str = "\n---\n",
    ) -> None:
        self._memory = memory
        self._max_context_tokens = max_context_tokens
        self._context_strategy = context_strategy
        self._prompt_template = prompt_template or self.DEFAULT_TEMPLATE
        self._reranker = reranker
        self._top_k = top_k
        self._min_score = min_score
        self._separator = separator

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        tags: Optional[list[str]] = None,
        use_hybrid: bool = False,
    ) -> list[SearchResult]:
        """从记忆中检索相关结果。

        Args:
            query: 查询文本
            top_k: 返回数量（覆盖默认值）
            tags: 标签过滤
            use_hybrid: 是否使用混合搜索（向量+图谱）

        Returns:
            搜索结果列表
        """
        k = top_k or self._top_k

        if use_hybrid and hasattr(self._memory, "hybrid_search_text"):
            return self._memory.hybrid_search_text(
                query=query, top_k=k, tags=tags
            )
        elif hasattr(self._memory, "search_text"):
            return self._memory.search_text(query=query, top_k=k, tags=tags)
        else:
            raise ValueError("memory 对象必须支持 search_text 或 hybrid_search_text 方法")

    def rerank(
        self, results: list[SearchResult], query: Optional[str] = None
    ) -> list[SearchResult]:
        """重排序搜索结果。

        Args:
            results: 搜索结果
            query: 原始查询（当前未使用，预留给语义重排序）

        Returns:
            重排序后的结果
        """
        if self._reranker is None:
            return results

        if self._context_strategy == ContextStrategy.DIVERSIFIED:
            from agentmemory.embedding_store import cosine_similarity

            return self._reranker.rerank_diversified(
                results,
                max_results=self._top_k,
                similarity_fn=cosine_similarity,
            )
        return self._reranker.rerank(results, max_results=self._top_k)

    def assemble_context(
        self,
        results: list[SearchResult],
    ) -> RAGContext:
        """组装 RAG 上下文。

        根据策略对结果排序，然后在 token 限制内拼接上下文。

        Args:
            results: 搜索结果

        Returns:
            RAGContext 组装结果
        """
        start = time.time()

        # 排序
        if self._context_strategy == ContextStrategy.RECENCY:
            results = sorted(results, key=lambda r: r.memory.created_at, reverse=True)
        elif self._context_strategy == ContextStrategy.RELEVANCE:
            results = sorted(results, key=lambda r: r.score, reverse=True)
        # DIVERSIFIED 和 HYBRID 已在 rerank 阶段处理

        # 拼接上下文
        context_parts: list[str] = []
        sources: list[Memory] = []
        total_tokens = 0
        truncated = False

        for i, result in enumerate(results):
            # 构建片段
            source_tag = f"[来源 {i + 1}]"
            score_tag = f"(相关度: {result.score:.2f})"
            snippet = f"{source_tag} {score_tag}\n{result.memory.content}"

            snippet_tokens = estimate_tokens(snippet)

            if total_tokens + snippet_tokens > self._max_context_tokens:
                # 尝试截断当前片段
                remaining = self._max_context_tokens - total_tokens
                if remaining > 20:
                    # 粗略截断
                    chars = remaining * 3  # 约 3 字符/token
                    snippet = snippet[:int(chars)] + "..."
                    context_parts.append(snippet)
                    sources.append(result.memory)
                    truncated = True
                else:
                    truncated = True
                break

            context_parts.append(snippet)
            sources.append(result.memory)
            total_tokens += snippet_tokens

        elapsed = (time.time() - start) * 1000

        return RAGContext(
            text=self._separator.join(context_parts),
            sources=sources,
            total_tokens=total_tokens,
            truncated=truncated,
            assembly_time_ms=elapsed,
        )

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        tags: Optional[list[str]] = None,
        use_hybrid: bool = False,
    ) -> RAGResult:
        """执行完整的 RAG 管道。

        流程：检索 → 重排序 → 上下文组装 → Prompt 生成

        Args:
            query: 用户查询
            top_k: 检索数量
            tags: 标签过滤
            use_hybrid: 是否使用混合检索

        Returns:
            RAGResult 包含完整管道结果
        """
        start = time.time()

        # 1. 检索
        results = self.retrieve(query=query, top_k=top_k, tags=tags, use_hybrid=use_hybrid)

        # 2. 过滤低分
        results = [r for r in results if r.score >= self._min_score]

        # 3. 重排序
        reranked = False
        if self._reranker is not None:
            results = self.rerank(results, query=query)
            reranked = True

        # 4. 上下文组装
        context = self.assemble_context(results)

        # 5. 生成 Prompt
        prompt = self._prompt_template.format(context=context.text, query=query)

        elapsed = (time.time() - start) * 1000

        return RAGResult(
            query=query,
            context=context,
            prompt=prompt,
            search_results=results,
            reranked=reranked,
            pipeline_time_ms=elapsed,
        )

    def run_with_sources(
        self,
        query: str,
        top_k: Optional[int] = None,
        tags: Optional[list[str]] = None,
        use_hybrid: bool = False,
    ) -> tuple[str, list[Memory]]:
        """执行 RAG 管道，返回 (prompt, sources) 元组。

        Args:
            query: 用户查询
            top_k: 检索数量
            tags: 标签过滤
            use_hybrid: 是否使用混合检索

        Returns:
            (prompt, sources) 元组
        """
        result = self.run(
            query=query, top_k=top_k, tags=tags, use_hybrid=use_hybrid
        )
        return result.prompt, result.context.sources
