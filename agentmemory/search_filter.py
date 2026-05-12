"""搜索过滤器增强 — 按 metadata、时间范围、重要性范围过滤搜索结果。

提供 SearchFilter 数据类和 filter_results 工具函数，
可与 HybridMemory 的搜索结果配合使用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agentmemory.models import Memory, SearchResult


@dataclass
class SearchFilter:
    """搜索结果过滤器。

    支持按 metadata 键值对、时间范围、重要性范围、标签等
    多维度过滤搜索结果。

    Args:
        metadata_filters: metadata 键值对匹配（所有条件必须满足）
        created_after: 创建时间下限（Unix 时间戳）
        created_before: 创建时间上限（Unix 时间戳）
        min_importance: 最小重要性评分
        max_importance: 最大重要性评分
        tags: 必须包含的标签列表（AND 逻辑）
        exclude_tags: 排除包含这些标签的结果
        content_contains: 内容必须包含的子串列表（任一匹配即可）
        content_not_contains: 内容不能包含的子串列表
    """

    metadata_filters: dict[str, Any] = field(default_factory=dict)
    created_after: Optional[float] = None
    created_before: Optional[float] = None
    min_importance: Optional[float] = None
    max_importance: Optional[float] = None
    tags: Optional[list[str]] = None
    exclude_tags: Optional[list[str]] = None
    content_contains: Optional[list[str]] = None
    content_not_contains: Optional[list[str]] = None

    def matches(self, memory: Memory, lifecycle: Any = None) -> bool:
        """判断记忆是否匹配所有过滤条件。

        Args:
            memory: 待检查的记忆
            lifecycle: MemoryLifecycle 实例（用于获取重要性）

        Returns:
            是否匹配所有条件
        """
        # Metadata 匹配
        for key, value in self.metadata_filters.items():
            if memory.metadata.get(key) != value:
                return False

        # 时间范围
        if self.created_after is not None and memory.created_at < self.created_after:
            return False
        if self.created_before is not None and memory.created_at > self.created_before:
            return False

        # 重要性范围
        if lifecycle is not None:
            info = lifecycle.get_lifecycle_info(memory)
            importance = info.get("importance_score", 1.0)
            if self.min_importance is not None and importance < self.min_importance:
                return False
            if self.max_importance is not None and importance > self.max_importance:
                return False

        # 标签匹配（AND）
        if self.tags:
            memory_tags_lower = {t.lower() for t in memory.tags}
            for tag in self.tags:
                if tag.lower() not in memory_tags_lower:
                    return False

        # 排除标签
        if self.exclude_tags:
            memory_tags_lower = {t.lower() for t in memory.tags}
            for tag in self.exclude_tags:
                if tag.lower() in memory_tags_lower:
                    return False

        # 内容包含
        if self.content_contains:
            content_lower = memory.content.lower()
            if not any(sub.lower() in content_lower for sub in self.content_contains):
                return False

        # 内容排除
        if self.content_not_contains:
            content_lower = memory.content.lower()
            for sub in self.content_not_contains:
                if sub.lower() in content_lower:
                    return False

        return True


def filter_search_results(
    results: list[SearchResult],
    search_filter: SearchFilter,
    lifecycle: Any = None,
) -> list[SearchResult]:
    """过滤搜索结果列表。

    Args:
        results: 原始搜索结果
        search_filter: 过滤条件
        lifecycle: MemoryLifecycle 实例

    Returns:
        过滤后的搜索结果，保持原有排序
    """
    return [
        r for r in results
        if search_filter.matches(r.memory, lifecycle)
    ]
