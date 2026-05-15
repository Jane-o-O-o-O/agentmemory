"""记忆分析：统计分析、访问模式、时间分布、标签云。

提供对记忆存储的深度分析能力，帮助理解数据特征和使用模式。
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from agentmemory.models import Memory


@dataclass
class AccessPattern:
    """访问模式分析结果。

    Attributes:
        total_accesses: 总访问次数
        unique_memories_accessed: 被访问过的唯一记忆数
        most_accessed: 访问最多的记忆 (id, count) 列表
        least_accessed: 访问最少的记忆 (id, count) 列表
        avg_access_per_memory: 平均每条记忆的访问次数
        access_distribution: 访问次数分布 {count_range: memory_count}
    """

    total_accesses: int = 0
    unique_memories_accessed: int = 0
    most_accessed: list[tuple[str, int]] = field(default_factory=list)
    least_accessed: list[tuple[str, int]] = field(default_factory=list)
    avg_access_per_memory: float = 0.0
    access_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class TemporalDistribution:
    """时间分布分析结果。

    Attributes:
        oldest_memory_age_hours: 最旧记忆的年龄（小时）
        newest_memory_age_hours: 最新记忆的年龄（小时）
        median_age_hours: 中位年龄（小时）
        hourly_distribution: 每小时创建的记忆数（最近24小时）
        daily_distribution: 每天创建的记忆数（最近30天）
        creation_rate_per_hour: 平均每小时创建速率
    """

    oldest_memory_age_hours: float = 0.0
    newest_memory_age_hours: float = 0.0
    median_age_hours: float = 0.0
    hourly_distribution: dict[int, int] = field(default_factory=dict)
    daily_distribution: dict[str, int] = field(default_factory=dict)
    creation_rate_per_hour: float = 0.0


@dataclass
class TagCloud:
    """标签云分析结果。

    Attributes:
        tags: 标签到使用次数的映射
        top_tags: 使用最多的标签
        orphan_tags: 只使用一次的标签
        tag_co_occurrence: 经常一起出现的标签对
        total_unique_tags: 唯一标签总数
    """

    tags: dict[str, int] = field(default_factory=dict)
    top_tags: list[tuple[str, int]] = field(default_factory=list)
    orphan_tags: list[str] = field(default_factory=list)
    tag_co_occurrence: list[tuple[str, str, int]] = field(default_factory=list)
    total_unique_tags: int = 0


@dataclass
class ContentAnalysis:
    """内容分析结果。

    Attributes:
        avg_content_length: 平均内容长度
        min_content_length: 最短内容长度
        max_content_length: 最长内容长度
        total_characters: 总字符数
        content_length_distribution: 长度分布
        duplicate_candidates: 疑似重复的内容对
    """

    avg_content_length: float = 0.0
    min_content_length: int = 0
    max_content_length: int = 0
    total_characters: int = 0
    content_length_distribution: dict[str, int] = field(default_factory=dict)
    duplicate_candidates: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass
class MemoryReport:
    """综合记忆分析报告。

    Attributes:
        total_memories: 总记忆数
        access_pattern: 访问模式分析
        temporal: 时间分布
        tag_cloud: 标签云
        content: 内容分析
        health_score: 存储健康评分（0~100）
        recommendations: 优化建议列表
        generated_at: 报告生成时间
    """

    total_memories: int = 0
    access_pattern: AccessPattern = field(default_factory=AccessPattern)
    temporal: TemporalDistribution = field(default_factory=TemporalDistribution)
    tag_cloud: TagCloud = field(default_factory=TagCloud)
    content: ContentAnalysis = field(default_factory=ContentAnalysis)
    health_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "total_memories": self.total_memories,
            "access_pattern": {
                "total_accesses": self.access_pattern.total_accesses,
                "unique_memories_accessed": self.access_pattern.unique_memories_accessed,
                "most_accessed": self.access_pattern.most_accessed[:10],
                "avg_access_per_memory": round(self.access_pattern.avg_access_per_memory, 2),
                "access_distribution": self.access_pattern.access_distribution,
            },
            "temporal": {
                "oldest_memory_age_hours": round(self.temporal.oldest_memory_age_hours, 1),
                "newest_memory_age_hours": round(self.temporal.newest_memory_age_hours, 1),
                "median_age_hours": round(self.temporal.median_age_hours, 1),
                "creation_rate_per_hour": round(self.temporal.creation_rate_per_hour, 4),
                "daily_distribution": self.temporal.daily_distribution,
            },
            "tag_cloud": {
                "total_unique_tags": self.tag_cloud.total_unique_tags,
                "top_tags": self.tag_cloud.top_tags[:20],
                "orphan_count": len(self.tag_cloud.orphan_tags),
                "top_co_occurrences": self.tag_cloud.tag_co_occurrence[:10],
            },
            "content": {
                "avg_content_length": round(self.content.avg_content_length, 1),
                "min_content_length": self.content.min_content_length,
                "max_content_length": self.content.max_content_length,
                "total_characters": self.content.total_characters,
                "distribution": self.content.content_length_distribution,
            },
            "health_score": round(self.health_score, 1),
            "recommendations": self.recommendations,
            "generated_at": self.generated_at,
        }


class MemoryAnalyzer:
    """记忆分析器。

    对 HybridMemory 或独立的 EmbeddingStore 进行全面分析。

    Args:
        top_n: 排行榜显示数量，默认 10
    """

    def __init__(self, top_n: int = 10) -> None:
        self._top_n = top_n

    def generate_report(
        self,
        memories: list[Memory],
        lifecycle: Optional[Any] = None,
    ) -> MemoryReport:
        """生成综合分析报告。

        Args:
            memories: 记忆列表
            lifecycle: MemoryLifecycle 实例（用于访问统计）

        Returns:
            MemoryReport 完整报告
        """
        report = MemoryReport(total_memories=len(memories))

        if not memories:
            report.health_score = 100.0
            report.recommendations = ["存储为空，无需优化"]
            return report

        report.access_pattern = self.analyze_access_pattern(memories, lifecycle)
        report.temporal = self.analyze_temporal(memories)
        report.tag_cloud = self.analyze_tags(memories)
        report.content = self.analyze_content(memories)
        report.health_score = self._compute_health_score(report)
        report.recommendations = self._generate_recommendations(report)

        return report

    def analyze_access_pattern(
        self,
        memories: list[Memory],
        lifecycle: Optional[Any] = None,
    ) -> AccessPattern:
        """分析访问模式。

        Args:
            memories: 记忆列表
            lifecycle: MemoryLifecycle 实例

        Returns:
            AccessPattern 分析结果
        """
        pattern = AccessPattern()

        if lifecycle is None:
            return pattern

        access_counts: list[tuple[str, int]] = []
        total = 0

        for mem in memories:
            count = lifecycle.get_access_count(mem.id)
            access_counts.append((mem.id, count))
            total += count

        pattern.total_accesses = total
        pattern.unique_memories_accessed = sum(1 for _, c in access_counts if c > 0)
        pattern.avg_access_per_memory = total / len(memories) if memories else 0.0

        # 排序
        access_counts.sort(key=lambda x: x[1], reverse=True)
        pattern.most_accessed = access_counts[: self._top_n]
        pattern.least_accessed = [
            (mid, c) for mid, c in access_counts[-self._top_n :] if c > 0
        ]
        pattern.least_accessed.reverse()

        # 分布统计
        ranges = {"0": 0, "1-5": 0, "6-20": 0, "21-100": 0, ">100": 0}
        for _, c in access_counts:
            if c == 0:
                ranges["0"] += 1
            elif c <= 5:
                ranges["1-5"] += 1
            elif c <= 20:
                ranges["6-20"] += 1
            elif c <= 100:
                ranges["21-100"] += 1
            else:
                ranges[">100"] += 1
        pattern.access_distribution = ranges

        return pattern

    def analyze_temporal(self, memories: list[Memory]) -> TemporalDistribution:
        """分析时间分布。

        Args:
            memories: 记忆列表

        Returns:
            TemporalDistribution 分析结果
        """
        temporal = TemporalDistribution()
        now = time.time()

        if not memories:
            return temporal

        ages = [(now - m.created_at) for m in memories]
        ages.sort()

        temporal.oldest_memory_age_hours = ages[-1] / 3600
        temporal.newest_memory_age_hours = ages[0] / 3600
        median_idx = len(ages) // 2
        temporal.median_age_hours = ages[median_idx] / 3600

        # 最近24小时的小时分布
        hourly: dict[int, int] = {h: 0 for h in range(24)}
        for m in memories:
            age_hours = (now - m.created_at) / 3600
            if age_hours <= 24:
                import datetime
                dt = datetime.datetime.fromtimestamp(m.created_at)
                hourly[dt.hour] = hourly.get(dt.hour, 0) + 1
        temporal.hourly_distribution = hourly

        # 最近30天的日分布
        daily: dict[str, int] = {}
        for m in memories:
            age_days = (now - m.created_at) / 86400
            if age_days <= 30:
                import datetime
                dt = datetime.datetime.fromtimestamp(m.created_at)
                day_key = dt.strftime("%Y-%m-%d")
                daily[day_key] = daily.get(day_key, 0) + 1
        temporal.daily_distribution = dict(sorted(daily.items()))

        # 创建速率
        time_span_hours = ages[-1] / 3600
        if time_span_hours > 0:
            temporal.creation_rate_per_hour = len(memories) / time_span_hours

        return temporal

    def analyze_tags(self, memories: list[Memory]) -> TagCloud:
        """分析标签云。

        Args:
            memories: 记忆列表

        Returns:
            TagCloud 分析结果
        """
        cloud = TagCloud()
        tag_counter: Counter[str] = Counter()
        co_occurrence: dict[tuple[str, str], int] = defaultdict(int)

        for mem in memories:
            unique_tags = list(set(t.lower() for t in mem.tags))
            for tag in unique_tags:
                tag_counter[tag] += 1

            # 共现统计
            for i in range(len(unique_tags)):
                for j in range(i + 1, len(unique_tags)):
                    pair = tuple(sorted([unique_tags[i], unique_tags[j]]))
                    co_occurrence[pair] += 1

        cloud.tags = dict(tag_counter)
        cloud.total_unique_tags = len(tag_counter)
        cloud.top_tags = tag_counter.most_common(self._top_n)
        cloud.orphan_tags = [tag for tag, count in tag_counter.items() if count == 1]

        # 排序共现对
        sorted_co = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)
        cloud.tag_co_occurrence = [
            (pair[0], pair[1], count) for pair, count in sorted_co[: self._top_n]
        ]

        return cloud

    def analyze_content(self, memories: list[Memory]) -> ContentAnalysis:
        """分析内容特征。

        Args:
            memories: 记忆列表

        Returns:
            ContentAnalysis 分析结果
        """
        analysis = ContentAnalysis()

        if not memories:
            return analysis

        lengths = [len(m.content) for m in memories]
        analysis.avg_content_length = sum(lengths) / len(lengths)
        analysis.min_content_length = min(lengths)
        analysis.max_content_length = max(lengths)
        analysis.total_characters = sum(lengths)

        # 长度分布
        ranges = {"<50": 0, "50-200": 0, "200-500": 0, "500-1000": 0, ">1000": 0}
        for l in lengths:
            if l < 50:
                ranges["<50"] += 1
            elif l < 200:
                ranges["50-200"] += 1
            elif l < 500:
                ranges["200-500"] += 1
            elif l < 1000:
                ranges["500-1000"] += 1
            else:
                ranges[">1000"] += 1
        analysis.content_length_distribution = ranges

        return analysis

    def _compute_health_score(self, report: MemoryReport) -> float:
        """计算存储健康评分。

        Args:
            report: 分析报告

        Returns:
            健康评分（0~100）
        """
        score = 100.0

        # 大量未访问的记忆（扣分）
        if report.total_memories > 0:
            unvisited_ratio = (
                report.access_pattern.access_distribution.get("0", 0)
                / report.total_memories
            )
            score -= unvisited_ratio * 20

        # 标签覆盖率低（扣分）
        if report.total_memories > 0:
            tagged_ratio = 1 - (
                report.tag_cloud.orphan_tags.__len__() / max(report.total_memories, 1)
            )
            if tagged_ratio < 0.3:
                score -= 10

        # 内容过长（扣分）
        if report.content.avg_content_length > 2000:
            score -= 10

        # 过多重复候选（扣分）
        if len(report.content.duplicate_candidates) > report.total_memories * 0.2:
            score -= 15

        return max(0.0, min(100.0, score))

    def _generate_recommendations(self, report: MemoryReport) -> list[str]:
        """生成优化建议。

        Args:
            report: 分析报告

        Returns:
            建议列表
        """
        recommendations: list[str] = []

        # 大量未访问记忆
        unvisited = report.access_pattern.access_distribution.get("0", 0)
        if unvisited > report.total_memories * 0.5:
            recommendations.append(
                f"有 {unvisited} 条记忆从未被访问，建议定期运行 cleanup_expired 或 deduplicate 清理"
            )

        # 标签覆盖
        if report.tag_cloud.total_unique_tags < 3 and report.total_memories > 20:
            recommendations.append(
                "标签使用率较低，建议为记忆添加标签以提高检索效率"
            )

        # 重复候选
        if len(report.content.duplicate_candidates) > 5:
            recommendations.append(
                f"发现 {len(report.content.duplicate_candidates)} 对疑似重复记忆，"
                "建议运行 MemoryConsolidator.deduplicate() 去重"
            )

        # 内容过长
        if report.content.max_content_length > 5000:
            recommendations.append(
                "部分记忆内容过长，建议使用 MemoryConsolidator.compress_aged() 压缩"
            )

        # 创建速率异常
        if report.temporal.creation_rate_per_hour > 100:
            recommendations.append(
                "记忆创建速率过高，建议检查是否有批量写入可以合并"
            )

        if not recommendations:
            recommendations.append("存储状态良好，无需优化")

        return recommendations
