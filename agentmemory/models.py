"""数据模型：Memory, Entity, Relation, SearchResult"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def _generate_id() -> str:
    """生成唯一 ID"""
    return uuid.uuid4().hex[:16]


@dataclass
class Memory:
    """记忆单元，存储一条文本信息及其向量表示。

    Attributes:
        id: 唯一标识符
        content: 文本内容
        created_at: 创建时间戳
        metadata: 附加元数据
        embedding: 向量表示（可选）
    """

    content: str
    id: str = field(default_factory=_generate_id)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("content 不能为空")

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """从 dict 反序列化"""
        return cls(
            id=data["id"],
            content=data["content"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )

    def __str__(self) -> str:
        preview = self.content[:50]
        if len(self.content) > 50:
            preview += "..."
        return f"Memory(id={self.id[:8]}, content={preview!r})"


@dataclass
class Entity:
    """知识图谱中的实体节点。

    Attributes:
        id: 唯一标识符
        name: 实体名称
        entity_type: 实体类型
        properties: 附加属性
    """

    name: str
    entity_type: str
    id: str = field(default_factory=_generate_id)
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name 不能为空")
        if not self.entity_type:
            raise ValueError("entity_type 不能为空")

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """从 dict 反序列化"""
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data["entity_type"],
            properties=data.get("properties", {}),
        )


@dataclass
class Relation:
    """知识图谱中两个实体之间的关系。

    Attributes:
        id: 唯一标识符
        source_id: 源实体 ID
        target_id: 目标实体 ID
        relation_type: 关系类型
        weight: 关系权重（默认 1.0）
    """

    source_id: str
    target_id: str
    relation_type: str
    id: str = field(default_factory=_generate_id)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id 不能为空")
        if not self.target_id:
            raise ValueError("target_id 不能为空")

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        """从 dict 反序列化"""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            weight=data.get("weight", 1.0),
        )


@dataclass
class SearchResult:
    """搜索结果，包含匹配的记忆及相似度分数。

    Attributes:
        memory: 匹配的 Memory 对象
        score: 相似度分数（0~1）
        context: 相关联的上下文 Memory 列表
    """

    memory: Memory
    score: float
    context: list[Memory] = field(default_factory=list)
