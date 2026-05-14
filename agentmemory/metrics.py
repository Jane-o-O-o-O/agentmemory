"""可观测性模块 — 内存统计、性能指标、健康检查、指标导出。

为 agentmemory 提供运行时观测能力，支持计数器、计时器、健康检查。
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class HealthStatus(Enum):
    """健康状态"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """单个健康检查结果。

    Attributes:
        name: 检查名称
        status: 健康状态
        message: 描述信息
        details: 附加详情
    """

    name: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """综合健康报告。

    Attributes:
        overall_status: 总体健康状态
        checks: 各项检查结果
        timestamp: 报告时间戳
    """

    overall_status: HealthStatus
    checks: list[HealthCheck] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class Counter:
    """计数器，支持递增和递减。

    Args:
        name: 计数器名称
        description: 描述
        initial_value: 初始值
    """

    def __init__(self, name: str, description: str = "", initial_value: int = 0) -> None:
        self._name = name
        self._description = description
        self._value = initial_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def value(self) -> int:
        return self._value

    def increment(self, amount: int = 1) -> None:
        """递增"""
        self._value += amount

    def decrement(self, amount: int = 1) -> None:
        """递减"""
        self._value -= amount

    def reset(self, value: int = 0) -> None:
        """重置"""
        self._value = value


class Timer:
    """计时器，记录多次操作的耗时。

    Args:
        name: 计时器名称
        description: 描述
    """

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._durations: list[float] = []
        self._start: Optional[float] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def count(self) -> int:
        """记录次数"""
        return len(self._durations)

    def start(self) -> Timer:
        """开始计时"""
        self._start = time.time()
        return self

    def stop(self) -> float:
        """停止计时，返回耗时（毫秒）。

        Returns:
            耗时（毫秒）

        Raises:
            RuntimeError: 未调用 start()
        """
        if self._start is None:
            raise RuntimeError("请先调用 start()")
        elapsed = (time.time() - self._start) * 1000
        self._durations.append(elapsed)
        self._start = None
        return elapsed

    def record(self, duration_ms: float) -> None:
        """直接记录一次耗时（毫秒）。

        Args:
            duration_ms: 耗时（毫秒）
        """
        self._durations.append(duration_ms)

    def mean(self) -> float:
        """平均耗时（毫秒）"""
        if not self._durations:
            return 0.0
        return sum(self._durations) / len(self._durations)

    def min_duration(self) -> float:
        """最短耗时（毫秒）"""
        return min(self._durations) if self._durations else 0.0

    def max_duration(self) -> float:
        """最长耗时（毫秒）"""
        return max(self._durations) if self._durations else 0.0

    def total(self) -> float:
        """总耗时（毫秒）"""
        return sum(self._durations)

    def p50(self) -> float:
        """P50 耗时"""
        return self._percentile(50)

    def p95(self) -> float:
        """P95 耗时"""
        return self._percentile(95)

    def p99(self) -> float:
        """P99 耗时"""
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        """计算百分位耗时"""
        if not self._durations:
            return 0.0
        sorted_d = sorted(self._durations)
        idx = int(len(sorted_d) * p / 100)
        idx = min(idx, len(sorted_d) - 1)
        return sorted_d[idx]

    def summary(self) -> dict[str, Any]:
        """获取统计摘要"""
        return {
            "name": self._name,
            "count": self.count,
            "mean_ms": round(self.mean(), 2),
            "min_ms": round(self.min_duration(), 2),
            "max_ms": round(self.max_duration(), 2),
            "total_ms": round(self.total(), 2),
            "p50_ms": round(self.p50(), 2),
            "p95_ms": round(self.p95(), 2),
            "p99_ms": round(self.p99(), 2),
        }

    def reset(self) -> None:
        """重置计时器"""
        self._durations.clear()
        self._start = None


class Gauge:
    """仪表盘指标，记录当前值。

    Args:
        name: 指标名称
        description: 描述
        initial_value: 初始值
    """

    def __init__(self, name: str, description: str = "", initial_value: float = 0.0) -> None:
        self._name = name
        self._description = description
        self._value = initial_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        """设置值"""
        self._value = value

    def increment(self, amount: float = 1.0) -> None:
        """递增"""
        self._value += amount

    def decrement(self, amount: float = 1.0) -> None:
        """递减"""
        self._value -= amount


class MetricsCollector:
    """指标收集器，统一管理所有指标。

    支持计数器、计时器、仪表盘三种指标类型。
    可导出为 JSON 或 Prometheus 格式。

    Args:
        namespace: 指标命名空间前缀
    """

    def __init__(self, namespace: str = "agentmemory") -> None:
        self._namespace = namespace
        self._counters: dict[str, Counter] = {}
        self._timers: dict[str, Timer] = {}
        self._gauges: dict[str, Gauge] = {}

    def counter(self, name: str, description: str = "", initial_value: int = 0) -> Counter:
        """获取或创建计数器。

        Args:
            name: 计数器名称
            description: 描述
            initial_value: 初始值

        Returns:
            Counter 实例
        """
        if name not in self._counters:
            self._counters[name] = Counter(name, description, initial_value)
        return self._counters[name]

    def timer(self, name: str, description: str = "") -> Timer:
        """获取或创建计时器。

        Args:
            name: 计时器名称
            description: 描述

        Returns:
            Timer 实例
        """
        if name not in self._timers:
            self._timers[name] = Timer(name, description)
        return self._timers[name]

    def gauge(self, name: str, description: str = "", initial_value: float = 0.0) -> Gauge:
        """获取或创建仪表盘。

        Args:
            name: 指标名称
            description: 描述
            initial_value: 初始值

        Returns:
            Gauge 实例
        """
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description, initial_value)
        return self._gauges[name]

    def snapshot(self) -> dict[str, Any]:
        """获取所有指标的快照。

        Returns:
            包含所有指标数据的字典
        """
        return {
            "namespace": self._namespace,
            "counters": {
                name: {"value": c.value, "description": c.description}
                for name, c in self._counters.items()
            },
            "timers": {
                name: t.summary()
                for name, t in self._timers.items()
            },
            "gauges": {
                name: {"value": g.value}
                for name, g in self._gauges.items()
            },
        }

    def export_json(self, indent: int = 2) -> str:
        """导出为 JSON 格式。

        Args:
            indent: 缩进空格数

        Returns:
            JSON 字符串
        """
        return json.dumps(self.snapshot(), indent=indent, ensure_ascii=False)

    def export_prometheus(self) -> str:
        """导出为 Prometheus 文本格式。

        Returns:
            Prometheus 文本
        """
        lines: list[str] = []

        for name, c in self._counters.items():
            metric_name = f"{self._namespace}_{name}"
            lines.append(f"# HELP {metric_name} {c.description}")
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {c.value}")

        for name, t in self._timers.items():
            metric_name = f"{self._namespace}_{name}"
            lines.append(f"# HELP {metric_name} {t.description}")
            lines.append(f"# TYPE {metric_name} summary")
            lines.append(f'{metric_name}_count {t.count}')
            lines.append(f'{metric_name}_sum {round(t.total(), 2)}')
            lines.append(f'{metric_name}{{quantile="0.5"}} {round(t.p50(), 2)}')
            lines.append(f'{metric_name}{{quantile="0.95"}} {round(t.p95(), 2)}')
            lines.append(f'{metric_name}{{quantile="0.99"}} {round(t.p99(), 2)}')

        for name, g in self._gauges.items():
            metric_name = f"{self._namespace}_{name}"
            lines.append(f"# HELP {metric_name} gauge")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {g.value}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """重置所有指标"""
        for c in self._counters.values():
            c.reset()
        for t in self._timers.values():
            t.reset()
        for g in self._gauges.values():
            g.set(0.0)


class HealthChecker:
    """健康检查器。

    收集多项健康检查，生成综合健康报告。

    Args:
        name: 检查器名称
    """

    def __init__(self, name: str = "agentmemory") -> None:
        self._name = name
        self._checks: list[HealthCheck] = []

    def add_check(self, check: HealthCheck) -> None:
        """添加健康检查结果"""
        self._checks.append(check)

    def clear(self) -> None:
        """清除所有检查结果"""
        self._checks.clear()

    def report(self) -> HealthReport:
        """生成综合健康报告。

        Returns:
            HealthReport 实例
        """
        if not self._checks:
            return HealthReport(
                overall_status=HealthStatus.HEALTHY,
                checks=[],
            )

        # 总体状态 = 最差的那个
        priority = {
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.HEALTHY: 2,
        }
        overall = min(self._checks, key=lambda c: priority[c.status]).status

        return HealthReport(overall_status=overall, checks=list(self._checks))


def check_memory_health(memory: Any) -> HealthCheck:
    """检查 HybridMemory 实例的健康状态。

    Args:
        memory: HybridMemory 实例

    Returns:
        HealthCheck 结果
    """
    try:
        stats = memory.stats()
        total = stats.get("memory_count", stats.get("total_memories", 0))
        with_embeddings = stats.get("memories_with_embeddings", total)

        if total == 0:
            return HealthCheck(
                name="memory_store",
                status=HealthStatus.HEALTHY,
                message="记忆存储为空（正常初始化状态）",
                details=stats,
            )

        embed_ratio = with_embeddings / total if total > 0 else 0
        if embed_ratio < 0.5:
            return HealthCheck(
                name="memory_store",
                status=HealthStatus.DEGRADED,
                message=f"超过50%的记忆缺少向量表示 ({with_embeddings}/{total})",
                details=stats,
            )

        return HealthCheck(
            name="memory_store",
            status=HealthStatus.HEALTHY,
            message=f"记忆存储正常，共 {total} 条记忆",
            details=stats,
        )
    except Exception as e:
        return HealthCheck(
            name="memory_store",
            status=HealthStatus.UNHEALTHY,
            message=f"健康检查失败: {e}",
        )


def check_lsh_health(memory: Any) -> HealthCheck:
    """检查 LSH 索引的健康状态。

    Args:
        memory: HybridMemory 实例

    Returns:
        HealthCheck 结果
    """
    try:
        stats = memory.stats()
        use_lsh = stats.get("use_lsh", False)

        if not use_lsh:
            return HealthCheck(
                name="lsh_index",
                status=HealthStatus.HEALTHY,
                message="LSH 索引未启用（使用暴力搜索）",
            )

        total = stats.get("memory_count", stats.get("total_memories", 0))
        lsh_indexed = stats.get("lsh_indexed_count", total)

        if total > 0 and lsh_indexed < total:
            return HealthCheck(
                name="lsh_index",
                status=HealthStatus.DEGRADED,
                message=f"LSH 索引不完整: {lsh_indexed}/{total}",
                details={"indexed": lsh_indexed, "total": total},
            )

        return HealthCheck(
            name="lsh_index",
            status=HealthStatus.HEALTHY,
            message=f"LSH 索引正常，已索引 {lsh_indexed} 条",
        )
    except Exception as e:
        return HealthCheck(
            name="lsh_index",
            status=HealthStatus.UNHEALTHY,
            message=f"LSH 健康检查失败: {e}",
        )
