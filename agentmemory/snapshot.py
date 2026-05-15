"""记忆快照系统：创建时间点快照、回滚、差异比较。

支持：
- 创建快照（保存当前状态的时间点副本）
- 列出所有快照
- 回滚到指定快照
- 比较两个快照的差异
- 自动清理旧快照
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SnapshotMetadata:
    """快照元数据。

    Attributes:
        id: 快照唯一 ID
        name: 快照名称（用户指定或自动生成）
        created_at: 创建时间戳
        description: 快照描述
        memory_count: 快照中的记忆数量
        entity_count: 实体数量
        relation_count: 关系数量
    """

    id: str
    name: str
    created_at: float
    description: str = ""
    memory_count: int = 0
    entity_count: int = 0
    relation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "description": self.description,
            "memory_count": self.memory_count,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
        }


@dataclass
class SnapshotDiff:
    """两个快照之间的差异。

    Attributes:
        snapshot_a_id: 快照 A 的 ID
        snapshot_b_id: 快照 B 的 ID
        memories_added: B 中新增的记忆
        memories_removed: B 中删除的记忆
        memories_modified: B 中修改的记忆（内容/标签变化）
        entities_added: B 中新增的实体
        entities_removed: B 中删除的实体
        relations_added: B 中新增的关系
        relations_removed: B 中删除的关系
    """

    snapshot_a_id: str
    snapshot_b_id: str
    memories_added: list[dict[str, Any]] = field(default_factory=list)
    memories_removed: list[dict[str, Any]] = field(default_factory=list)
    memories_modified: list[dict[str, Any]] = field(default_factory=list)
    entities_added: list[dict[str, Any]] = field(default_factory=list)
    entities_removed: list[dict[str, Any]] = field(default_factory=list)
    relations_added: list[dict[str, Any]] = field(default_factory=list)
    relations_removed: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        """差异摘要统计"""
        return {
            "memories_added": len(self.memories_added),
            "memories_removed": len(self.memories_removed),
            "memories_modified": len(self.memories_modified),
            "entities_added": len(self.entities_added),
            "entities_removed": len(self.entities_removed),
            "relations_added": len(self.relations_added),
            "relations_removed": len(self.relations_removed),
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict"""
        return {
            "snapshot_a_id": self.snapshot_a_id,
            "snapshot_b_id": self.snapshot_b_id,
            "summary": self.summary(),
            "details": {
                "memories_added": self.memories_added,
                "memories_removed": self.memories_removed,
                "memories_modified": self.memories_modified,
                "entities_added": self.entities_added,
                "entities_removed": self.entities_removed,
                "relations_added": self.relations_added,
                "relations_removed": self.relations_removed,
            },
        }

    @property
    def has_changes(self) -> bool:
        """是否有任何差异"""
        return any(v > 0 for v in self.summary().values())


class SnapshotManager:
    """快照管理器。

    管理 HybridMemory 的时间点快照，支持创建、回滚、比较和清理。

    Args:
        max_snapshots: 最大快照数量，超出时自动删除最旧的，默认 10
    """

    def __init__(self, max_snapshots: int = 10) -> None:
        if max_snapshots < 1:
            raise ValueError(f"max_snapshots 不能小于 1，got {max_snapshots}")

        self._max_snapshots = max_snapshots
        self._snapshots: dict[str, dict[str, Any]] = {}  # id -> snapshot data
        self._metadata: dict[str, SnapshotMetadata] = {}  # id -> metadata
        self._name_index: dict[str, str] = {}  # name -> id

    def _generate_id(self) -> str:
        """生成快照 ID"""
        import uuid
        return f"snap_{uuid.uuid4().hex[:12]}"

    def create(
        self,
        memory: Any,
        name: Optional[str] = None,
        description: str = "",
    ) -> SnapshotMetadata:
        """创建快照。

        Args:
            memory: HybridMemory 实例
            name: 快照名称（默认自动生成）
            description: 快照描述

        Returns:
            快照元数据

        Raises:
            ValueError: 快照名称已存在
        """
        snapshot_id = self._generate_id()
        if name is None:
            name = f"snapshot_{int(time.time())}"

        if name in self._name_index:
            raise ValueError(f"快照名称 '{name}' 已存在")

        # 深拷贝当前状态
        memories_data = [m.to_dict() for m in memory.embedding_store.list_all()]
        entities_data = [e.to_dict() for e in memory.knowledge_graph.find_entities()]
        relations_data = [r.to_dict() for r in memory.knowledge_graph.find_relations()]

        snapshot_data = {
            "memories": memories_data,
            "entities": entities_data,
            "relations": relations_data,
        }

        # 存储快照
        self._snapshots[snapshot_id] = snapshot_data
        metadata = SnapshotMetadata(
            id=snapshot_id,
            name=name,
            created_at=time.time(),
            description=description,
            memory_count=len(memories_data),
            entity_count=len(entities_data),
            relation_count=len(relations_data),
        )
        self._metadata[snapshot_id] = metadata
        self._name_index[name] = snapshot_id

        # 自动清理旧快照
        self._cleanup_old()

        return metadata

    def restore(
        self,
        memory: Any,
        snapshot_id_or_name: str,
    ) -> SnapshotMetadata:
        """恢复到指定快照。

        Args:
            memory: HybridMemory 实例
            snapshot_id_or_name: 快照 ID 或名称

        Returns:
            恢复的快照元数据

        Raises:
            KeyError: 快照不存在
        """
        snapshot_id = self._resolve_id(snapshot_id_or_name)
        if snapshot_id not in self._snapshots:
            raise KeyError(f"快照不存在: {snapshot_id_or_name}")

        snapshot_data = self._snapshots[snapshot_id]

        # 重建 EmbeddingStore
        from agentmemory.models import Memory, Entity, Relation

        memory.embedding_store._memories.clear()
        if memory.embedding_store._lsh_index is not None:
            memory.embedding_store._lsh_index.clear()

        for mem_data in snapshot_data["memories"]:
            mem = Memory.from_dict(mem_data)
            memory.embedding_store.add(mem)

        # 重建 KnowledgeGraph
        memory.knowledge_graph._entities.clear()
        memory.knowledge_graph._relations.clear()
        memory.knowledge_graph._entity_relations.clear()
        memory.knowledge_graph._name_type_index.clear()

        for entity_data in snapshot_data["entities"]:
            entity = Entity.from_dict(entity_data)
            memory.knowledge_graph.add_entity(entity)

        for rel_data in snapshot_data["relations"]:
            relation = Relation.from_dict(rel_data)
            memory.knowledge_graph.add_relation(relation)

        return self._metadata[snapshot_id]

    def list_snapshots(self) -> list[SnapshotMetadata]:
        """列出所有快照。

        Returns:
            快照元数据列表，按创建时间降序
        """
        snapshots = list(self._metadata.values())
        snapshots.sort(key=lambda s: s.created_at, reverse=True)
        return snapshots

    def get_metadata(self, snapshot_id_or_name: str) -> Optional[SnapshotMetadata]:
        """获取快照元数据。

        Args:
            snapshot_id_or_name: 快照 ID 或名称

        Returns:
            快照元数据，不存在返回 None
        """
        snapshot_id = self._resolve_id(snapshot_id_or_name)
        return self._metadata.get(snapshot_id)

    def delete(self, snapshot_id_or_name: str) -> bool:
        """删除快照。

        Args:
            snapshot_id_or_name: 快照 ID 或名称

        Returns:
            是否成功删除

        Raises:
            KeyError: 快照不存在
        """
        snapshot_id = self._resolve_id(snapshot_id_or_name)
        if snapshot_id not in self._snapshots:
            raise KeyError(f"快照不存在: {snapshot_id_or_name}")

        metadata = self._metadata.pop(snapshot_id)
        self._name_index.pop(metadata.name, None)
        del self._snapshots[snapshot_id]
        return True

    def diff(
        self,
        snapshot_a_id_or_name: str,
        snapshot_b_id_or_name: str,
    ) -> SnapshotDiff:
        """比较两个快照的差异。

        Args:
            snapshot_a_id_or_name: 快照 A 的 ID 或名称
            snapshot_b_id_or_name: 快照 B 的 ID 或名称

        Returns:
            差异对象

        Raises:
            KeyError: 任何快照不存在
        """
        id_a = self._resolve_id(snapshot_a_id_or_name)
        id_b = self._resolve_id(snapshot_b_id_or_name)

        if id_a not in self._snapshots:
            raise KeyError(f"快照不存在: {snapshot_a_id_or_name}")
        if id_b not in self._snapshots:
            raise KeyError(f"快照不存在: {snapshot_b_id_or_name}")

        data_a = self._snapshots[id_a]
        data_b = self._snapshots[id_b]

        # 构建 ID -> data 映射
        mem_a = {m["id"]: m for m in data_a["memories"]}
        mem_b = {m["id"]: m for m in data_b["memories"]}
        ent_a = {e["id"]: e for e in data_a["entities"]}
        ent_b = {e["id"]: e for e in data_b["entities"]}
        rel_a = {r["id"]: r for r in data_a["relations"]}
        rel_b = {r["id"]: r for r in data_b["relations"]}

        diff = SnapshotDiff(snapshot_a_id=id_a, snapshot_b_id=id_b)

        # 记忆差异
        for mid in mem_b:
            if mid not in mem_a:
                diff.memories_added.append(mem_b[mid])
            elif mem_a[mid] != mem_b[mid]:
                diff.memories_modified.append({
                    "id": mid,
                    "before": mem_a[mid],
                    "after": mem_b[mid],
                })

        for mid in mem_a:
            if mid not in mem_b:
                diff.memories_removed.append(mem_a[mid])

        # 实体差异
        for eid in ent_b:
            if eid not in ent_a:
                diff.entities_added.append(ent_b[eid])
        for eid in ent_a:
            if eid not in ent_b:
                diff.entities_removed.append(ent_a[eid])

        # 关系差异
        for rid in rel_b:
            if rid not in rel_a:
                diff.relations_added.append(rel_b[rid])
        for rid in rel_a:
            if rid not in rel_b:
                diff.relations_removed.append(rel_a[rid])

        return diff

    def export_snapshot(self, snapshot_id_or_name: str) -> str:
        """导出快照为 JSON 字符串。

        Args:
            snapshot_id_or_name: 快照 ID 或名称

        Returns:
            JSON 字符串

        Raises:
            KeyError: 快照不存在
        """
        snapshot_id = self._resolve_id(snapshot_id_or_name)
        if snapshot_id not in self._snapshots:
            raise KeyError(f"快照不存在: {snapshot_id_or_name}")

        export_data = {
            "metadata": self._metadata[snapshot_id].to_dict(),
            "data": self._snapshots[snapshot_id],
        }
        return json.dumps(export_data, ensure_ascii=False, indent=2)

    def import_snapshot(self, json_str: str) -> SnapshotMetadata:
        """从 JSON 字符串导入快照。

        Args:
            json_str: JSON 字符串

        Returns:
            导入的快照元数据
        """
        import_data = json.loads(json_str)
        metadata_dict = import_data["metadata"]
        data = import_data["data"]

        snapshot_id = metadata_dict["id"]

        # 如果 ID 冲突，生成新 ID
        if snapshot_id in self._snapshots:
            snapshot_id = self._generate_id()
            metadata_dict["id"] = snapshot_id

        self._snapshots[snapshot_id] = data
        metadata = SnapshotMetadata(**{
            k: v for k, v in metadata_dict.items()
            if k in SnapshotMetadata.__dataclass_fields__
        })
        self._metadata[snapshot_id] = metadata
        self._name_index[metadata.name] = snapshot_id

        return metadata

    def cleanup(self, keep_latest: int = 3) -> int:
        """清理旧快照，保留最近的 N 个。

        Args:
            keep_latest: 保留最新快照数量

        Returns:
            删除的快照数量
        """
        all_snapshots = self.list_snapshots()
        if len(all_snapshots) <= keep_latest:
            return 0

        to_delete = all_snapshots[keep_latest:]
        count = 0
        for snap in to_delete:
            try:
                self.delete(snap.id)
                count += 1
            except KeyError:
                pass
        return count

    @property
    def count(self) -> int:
        """当前快照数量"""
        return len(self._snapshots)

    def _resolve_id(self, snapshot_id_or_name: str) -> str:
        """将名称解析为 ID，如果是 ID 则直接返回"""
        if snapshot_id_or_name in self._snapshots:
            return snapshot_id_or_name
        if snapshot_id_or_name in self._name_index:
            return self._name_index[snapshot_id_or_name]
        return snapshot_id_or_name  # 返回原始值，让调用方处理 KeyError

    def _cleanup_old(self) -> None:
        """自动清理超出限制的旧快照"""
        while len(self._snapshots) > self._max_snapshots:
            oldest = min(self._metadata.values(), key=lambda m: m.created_at)
            self.delete(oldest.id)
