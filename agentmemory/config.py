"""集中化配置系统：类型安全配置、验证、环境变量覆盖、Profile。

提供统一的配置管理，支持：
- 类型安全的 dataclass 配置
- 环境变量覆盖（AGENTMEMORY_ 前缀）
- 预定义 Profile（dev/test/prod）
- 配置验证和序列化
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class VectorConfig:
    """向量存储配置。

    Attributes:
        dimension: 向量维度
        use_lsh: 是否启用 LSH 索引
        lsh_tables: LSH 哈希表数量
        lsh_hyperplanes: LSH 超平面数量
        use_quantization: 是否启用向量量化
        quantization_method: 量化方法 ('sq8' 或 'pq')
        pq_subspaces: PQ 子空间数量
    """

    dimension: int = 128
    use_lsh: bool = False
    lsh_tables: int = 8
    lsh_hyperplanes: int = 16
    use_quantization: bool = False
    quantization_method: str = "sq8"
    pq_subspaces: int = 8

    def validate(self) -> list[str]:
        """验证向量配置，返回错误列表。"""
        errors: list[str] = []
        if self.dimension < 1:
            errors.append(f"dimension 必须 >= 1，got {self.dimension}")
        if self.lsh_tables < 1:
            errors.append(f"lsh_tables 必须 >= 1，got {self.lsh_tables}")
        if self.lsh_hyperplanes < 1:
            errors.append(f"lsh_hyperplanes 必须 >= 1，got {self.lsh_hyperplanes}")
        if self.quantization_method not in ("sq8", "pq"):
            errors.append(f"quantization_method 必须是 'sq8' 或 'pq'，got {self.quantization_method!r}")
        if self.pq_subspaces < 1:
            errors.append(f"pq_subspaces 必须 >= 1，got {self.pq_subspaces}")
        return errors


@dataclass
class StorageConfig:
    """持久化存储配置。

    Attributes:
        storage_path: 存储路径，None 表示不持久化
        backend: 存储后端 ('json' 或 'sqlite')
        auto_save: 操作后自动保存
        auto_load: 初始化时自动加载
    """

    storage_path: Optional[str] = None
    backend: str = "json"
    auto_save: bool = False
    auto_load: bool = False

    def validate(self) -> list[str]:
        """验证存储配置，返回错误列表。"""
        errors: list[str] = []
        if self.backend not in ("json", "sqlite"):
            errors.append(f"backend 必须是 'json' 或 'sqlite'，got {self.backend!r}")
        return errors


@dataclass
class LifecycleConfig:
    """生命周期配置。

    Attributes:
        default_ttl: 默认 TTL（秒），None 表示永不过期
        decay_rate: 时间衰减速率
        recency_weight: 时间新鲜度权重
        frequency_weight: 访问频率权重
        relevance_weight: 原始相关性权重
    """

    default_ttl: Optional[float] = None
    decay_rate: float = 0.001
    recency_weight: float = 0.3
    frequency_weight: float = 0.3
    relevance_weight: float = 0.4

    def validate(self) -> list[str]:
        """验证生命周期配置，返回错误列表。"""
        errors: list[str] = []
        if self.default_ttl is not None and self.default_ttl <= 0:
            errors.append(f"default_ttl 必须 > 0，got {self.default_ttl}")
        if self.decay_rate < 0:
            errors.append(f"decay_rate 必须 >= 0，got {self.decay_rate}")
        total = self.recency_weight + self.frequency_weight + self.relevance_weight
        if abs(total - 1.0) > 0.01:
            errors.append(f"权重之和应为 1.0，got {total:.3f}")
        return errors


@dataclass
class CacheConfig:
    """搜索缓存配置。

    Attributes:
        enabled: 是否启用缓存
        max_size: 最大缓存条目数
        ttl: 缓存条目 TTL（秒），None 表示不过期
    """

    enabled: bool = False
    max_size: int = 128
    ttl: Optional[float] = None

    def validate(self) -> list[str]:
        """验证缓存配置，返回错误列表。"""
        errors: list[str] = []
        if self.max_size < 0:
            errors.append(f"max_size 必须 >= 0，got {self.max_size}")
        return errors


@dataclass
class GCConfig:
    """垃圾回收配置。

    Attributes:
        enabled: 是否启用自动 GC
        interval: GC 运行间隔（秒）
        min_importance: 最低重要性阈值，低于此值的记忆将被清理
        max_age: 最大存活时间（秒），超过此时间的记忆将被清理（即使有访问）
        min_access_count: 最低访问次数，低于此值且超过 max_idle_time 的记忆将被清理
        max_idle_time: 最大空闲时间（秒），超过此时间未访问的记忆将被清理
        batch_size: 每次 GC 清理的最大条目数
    """

    enabled: bool = False
    interval: float = 3600.0
    min_importance: float = 0.1
    max_age: Optional[float] = None
    min_access_count: int = 0
    max_idle_time: Optional[float] = None
    batch_size: int = 100

    def validate(self) -> list[str]:
        """验证 GC 配置，返回错误列表。"""
        errors: list[str] = []
        if self.interval <= 0:
            errors.append(f"interval 必须 > 0，got {self.interval}")
        if not 0.0 <= self.min_importance <= 1.0:
            errors.append(f"min_importance 必须在 0~1 之间，got {self.min_importance}")
        if self.max_age is not None and self.max_age <= 0:
            errors.append(f"max_age 必须 > 0，got {self.max_age}")
        if self.batch_size < 1:
            errors.append(f"batch_size 必须 >= 1，got {self.batch_size}")
        return errors


@dataclass
class AgentMemoryConfig:
    """AgentMemory 完整配置。

    Attributes:
        vector: 向量存储配置
        storage: 持久化存储配置
        lifecycle: 生命周期配置
        cache: 搜索缓存配置
        gc: 垃圾回收配置
        weighted_scoring: 是否启用加权评分
        enable_metrics: 是否启用指标采集
    """

    vector: VectorConfig = field(default_factory=VectorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    gc: GCConfig = field(default_factory=GCConfig)
    weighted_scoring: bool = False
    enable_metrics: bool = False

    def validate(self) -> list[str]:
        """验证完整配置，返回所有错误。

        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors: list[str] = []
        errors.extend(self.vector.validate())
        errors.extend(self.storage.validate())
        errors.extend(self.lifecycle.validate())
        errors.extend(self.cache.validate())
        errors.extend(self.gc.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMemoryConfig:
        """从 dict 反序列化。

        Args:
            data: 配置字典

        Returns:
            AgentMemoryConfig 实例
        """
        vector = VectorConfig(**data.get("vector", {}))
        storage = StorageConfig(**data.get("storage", {}))
        lifecycle = LifecycleConfig(**data.get("lifecycle", {}))
        cache = CacheConfig(**data.get("cache", {}))
        gc = GCConfig(**data.get("gc", {}))
        return cls(
            vector=vector,
            storage=storage,
            lifecycle=lifecycle,
            cache=cache,
            gc=gc,
            weighted_scoring=data.get("weighted_scoring", False),
            enable_metrics=data.get("enable_metrics", False),
        )

    @classmethod
    def from_env(cls, prefix: str = "AGENTMEMORY_") -> AgentMemoryConfig:
        """从环境变量加载配置，覆盖默认值。

        环境变量格式: {PREFIX}{SECTION}_{KEY}，如：
        - AGENTMEMORY_VECTOR_DIMENSION=256
        - AGENTMEMORY_STORAGE_BACKEND=sqlite
        - AGENTMEMORY_WEIGHTED_SCORING=true

        Args:
            prefix: 环境变量前缀

        Returns:
            AgentMemoryConfig 实例
        """
        cfg = cls()
        env_prefix = prefix.upper()

        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue
            parts = key[len(env_prefix):].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section, field_name = parts
            _apply_env_value(cfg, section, field_name, value)

        return cfg


def _apply_env_value(cfg: AgentMemoryConfig, section: str, field_name: str, value: str) -> None:
    """将环境变量值应用到配置对象。"""
    section_map: dict[str, Any] = {
        "vector": cfg.vector,
        "storage": cfg.storage,
        "lifecycle": cfg.lifecycle,
        "cache": cfg.cache,
        "gc": cfg.gc,
    }

    if section in section_map:
        obj = section_map[section]
        if hasattr(obj, field_name):
            current = getattr(obj, field_name)
            setattr(obj, field_name, _coerce_env(value, current))
    elif section == "agentmemory" or section == "":
        # 顶层字段
        if hasattr(cfg, field_name):
            current = getattr(cfg, field_name)
            setattr(cfg, field_name, _coerce_env(value, current))


def _coerce_env(value: str, target: Any) -> Any:
    """将环境变量字符串强制转换为目标类型。"""
    if target is None:
        # 尝试推断类型
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    if isinstance(target, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(target, int):
        return int(value)
    if isinstance(target, float):
        return float(value)
    return value


# 预定义 Profile
PROFILES: dict[str, AgentMemoryConfig] = {
    "dev": AgentMemoryConfig(
        vector=VectorConfig(dimension=64, use_lsh=False),
        storage=StorageConfig(backend="json", auto_save=False),
        cache=CacheConfig(enabled=False),
        gc=GCConfig(enabled=False),
    ),
    "test": AgentMemoryConfig(
        vector=VectorConfig(dimension=32, use_lsh=False),
        storage=StorageConfig(backend="json", auto_save=False),
        cache=CacheConfig(enabled=False),
        gc=GCConfig(enabled=False),
    ),
    "prod": AgentMemoryConfig(
        vector=VectorConfig(dimension=256, use_lsh=True, lsh_tables=16, lsh_hyperplanes=32),
        storage=StorageConfig(backend="sqlite", auto_save=True, auto_load=True),
        lifecycle=LifecycleConfig(default_ttl=86400 * 30),
        cache=CacheConfig(enabled=True, max_size=1024, ttl=300.0),
        gc=GCConfig(enabled=True, interval=3600, min_importance=0.05),
        weighted_scoring=True,
        enable_metrics=True,
    ),
}


def get_profile(name: str) -> AgentMemoryConfig:
    """获取预定义配置 Profile。

    Args:
        name: Profile 名称 ('dev', 'test', 'prod')

    Returns:
        AgentMemoryConfig 实例

    Raises:
        KeyError: Profile 不存在
    """
    if name not in PROFILES:
        raise KeyError(f"未知 Profile: {name!r}，可用: {list(PROFILES.keys())}")
    return PROFILES[name]


def load_config(
    profile: Optional[str] = None,
    env_override: bool = True,
) -> AgentMemoryConfig:
    """加载配置（Profile + 环境变量覆盖）。

    Args:
        profile: Profile 名称，None 表示使用默认配置
        env_override: 是否应用环境变量覆盖

    Returns:
        AgentMemoryConfig 实例

    Raises:
        ValueError: 配置验证失败
    """
    if profile:
        cfg = get_profile(profile)
    else:
        cfg = AgentMemoryConfig()

    if env_override:
        cfg = _merge_env(cfg)

    errors = cfg.validate()
    if errors:
        raise ValueError(f"配置验证失败: {'; '.join(errors)}")

    return cfg


def _merge_env(cfg: AgentMemoryConfig) -> AgentMemoryConfig:
    """将环境变量合并到已有配置。"""
    cfg_dict = cfg.to_dict()
    prefix = "AGENTMEMORY_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field_name = parts
        if section in cfg_dict and isinstance(cfg_dict[section], dict):
            if field_name in cfg_dict[section]:
                cfg_dict[section][field_name] = _coerce_env(value, cfg_dict[section][field_name])
        elif section in cfg_dict:
            cfg_dict[section] = _coerce_env(value, cfg_dict[section])

    return AgentMemoryConfig.from_dict(cfg_dict)
