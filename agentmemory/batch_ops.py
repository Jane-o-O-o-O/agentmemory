"""批量操作 — 高性能批量 add/search/delete 接口。

提供优化的批量操作，减少单次调用开销，支持并行处理。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentmemory.models import Memory, SearchResult


@dataclass
class BatchResult:
    """批量操作结果。

    Attributes:
        total: 总操作数
        succeeded: 成功数
        failed: 失败数
        errors: 错误信息列表
        duration_ms: 操作耗时（毫秒）
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.succeeded / self.total if self.total > 0 else 0.0


@dataclass
class BatchSearchResult:
    """批量搜索结果。

    Attributes:
        query: 原始查询
        results: 搜索结果列表
        duration_ms: 搜索耗时（毫秒）
    """

    query: str
    results: list[SearchResult] = field(default_factory=list)
    duration_ms: float = 0.0


class BatchOperator:
    """批量操作器，提供高性能批量接口。

    包装 HybridMemory 的单条操作为优化的批量操作。

    Args:
        memory: HybridMemory 实例
        max_workers: 最大并行线程数（用于并行搜索）
    """

    def __init__(self, memory: Any, max_workers: int = 4) -> None:
        self._memory = memory
        self._max_workers = max_workers

    def batch_add(
        self,
        items: list[dict[str, Any]],
    ) -> BatchResult:
        """批量添加记忆。

        Args:
            items: 记忆数据列表，每项为 dict，支持 keys:
                - content (str, required)
                - embedding (list[float], optional)
                - metadata (dict, optional)
                - tags (list[str], optional)
                - importance (float, optional)

        Returns:
            BatchResult 操作结果
        """
        start = time.time()
        result = BatchResult(total=len(items))

        for i, item in enumerate(items):
            try:
                content = item.get("content")
                if not content:
                    result.errors.append({"index": i, "error": "content 不能为空"})
                    result.failed += 1
                    continue

                self._memory.remember(
                    content=content,
                    embedding=item.get("embedding"),
                    metadata=item.get("metadata"),
                    tags=item.get("tags"),
                    importance=item.get("importance"),
                )
                result.succeeded += 1
            except Exception as e:
                result.errors.append({"index": i, "error": str(e)})
                result.failed += 1

        result.duration_ms = (time.time() - start) * 1000
        return result

    def batch_delete(
        self,
        memory_ids: list[str],
        ignore_missing: bool = True,
    ) -> BatchResult:
        """批量删除记忆。

        Args:
            memory_ids: 记忆 ID 列表
            ignore_missing: 是否忽略不存在的 ID

        Returns:
            BatchResult 操作结果
        """
        start = time.time()
        result = BatchResult(total=len(memory_ids))

        for mid in memory_ids:
            try:
                self._memory.forget(mid)
                result.succeeded += 1
            except KeyError:
                if not ignore_missing:
                    result.errors.append({"id": mid, "error": "记忆不存在"})
                    result.failed += 1
                else:
                    result.succeeded += 1  # 忽略视为成功
            except Exception as e:
                result.errors.append({"id": mid, "error": str(e)})
                result.failed += 1

        result.duration_ms = (time.time() - start) * 1000
        return result

    def batch_search(
        self,
        queries: list[str],
        top_k: int = 5,
        tags: Optional[list[str]] = None,
        parallel: bool = False,
    ) -> list[BatchSearchResult]:
        """批量搜索。

        Args:
            queries: 查询列表
            top_k: 每个查询返回的结果数
            tags: 标签过滤
            parallel: 是否并行搜索

        Returns:
            每个查询的搜索结果列表
        """
        if not parallel or len(queries) <= 1:
            return self._batch_search_sequential(queries, top_k, tags)
        return self._batch_search_parallel(queries, top_k, tags)

    def _batch_search_sequential(
        self,
        queries: list[str],
        top_k: int,
        tags: Optional[list[str]],
    ) -> list[BatchSearchResult]:
        """顺序批量搜索"""
        results: list[BatchSearchResult] = []

        for query in queries:
            start = time.time()
            search_results = self._memory.search_text(
                query=query, top_k=top_k, tags=tags
            )
            elapsed = (time.time() - start) * 1000
            results.append(
                BatchSearchResult(
                    query=query,
                    results=search_results,
                    duration_ms=elapsed,
                )
            )

        return results

    def _batch_search_parallel(
        self,
        queries: list[str],
        top_k: int,
        tags: Optional[list[str]],
    ) -> list[BatchSearchResult]:
        """并行批量搜索"""
        results_map: dict[int, BatchSearchResult] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {}
            for i, query in enumerate(queries):
                future = executor.submit(
                    self._memory.search_text,
                    query=query,
                    top_k=top_k,
                    tags=tags,
                )
                future_to_idx[future] = (i, query)

            for future in as_completed(future_to_idx):
                idx, query = future_to_idx[future]
                start = time.time()
                try:
                    search_results = future.result()
                except Exception:
                    search_results = []
                elapsed = (time.time() - start) * 1000
                results_map[idx] = BatchSearchResult(
                    query=query,
                    results=search_results,
                    duration_ms=elapsed,
                )

        return [results_map[i] for i in range(len(queries))]

    def batch_update(
        self,
        updates: list[dict[str, Any]],
    ) -> BatchResult:
        """批量更新记忆。

        Args:
            updates: 更新数据列表，每项为 dict，支持 keys:
                - id (str, required): 记忆 ID
                - content (str, optional): 新内容
                - metadata (dict, optional): 新元数据（合并）
                - tags (list[str], optional): 新标签（替换）

        Returns:
            BatchResult 操作结果
        """
        start = time.time()
        result = BatchResult(total=len(updates))

        for i, update in enumerate(updates):
            try:
                memory_id = update.get("id")
                if not memory_id:
                    result.errors.append({"index": i, "error": "id 不能为空"})
                    result.failed += 1
                    continue

                self._memory.update_memory(
                    memory_id=memory_id,
                    content=update.get("content"),
                    metadata=update.get("metadata"),
                    tags=update.get("tags"),
                )
                result.succeeded += 1
            except KeyError as e:
                result.errors.append({"index": i, "error": str(e)})
                result.failed += 1
            except Exception as e:
                result.errors.append({"index": i, "error": str(e)})
                result.failed += 1

        result.duration_ms = (time.time() - start) * 1000
        return result

    def batch_tag(
        self,
        memory_ids: list[str],
        tag: str,
        remove: bool = False,
    ) -> BatchResult:
        """批量添加/移除标签。

        Args:
            memory_ids: 记忆 ID 列表
            tag: 标签名称
            remove: True 为移除标签，False 为添加

        Returns:
            BatchResult 操作结果
        """
        start = time.time()
        result = BatchResult(total=len(memory_ids))

        for mid in memory_ids:
            try:
                if remove:
                    self._memory.remove_tag(mid, tag)
                else:
                    self._memory.add_tag(mid, tag)
                result.succeeded += 1
            except KeyError as e:
                result.errors.append({"id": mid, "error": str(e)})
                result.failed += 1
            except Exception as e:
                result.errors.append({"id": mid, "error": str(e)})
                result.failed += 1

        result.duration_ms = (time.time() - start) * 1000
        return result

    def batch_export(
        self,
        memory_ids: list[str],
    ) -> list[dict[str, Any]]:
        """批量导出记忆数据。

        Args:
            memory_ids: 记忆 ID 列表

        Returns:
            记忆数据字典列表
        """
        exported: list[dict[str, Any]] = []
        for mid in memory_ids:
            mem = self._memory.get_memory(mid)
            if mem is not None:
                data = mem.to_dict()
                # 附加生命周期信息
                lc = self._memory.get_lifecycle_info(mid)
                if lc:
                    data["_lifecycle"] = lc
                exported.append(data)
        return exported
