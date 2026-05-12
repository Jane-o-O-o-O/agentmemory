"""知识图谱：实体/关系的 CRUD 与图遍历"""

from __future__ import annotations

from collections import deque
from typing import Optional

from agentmemory.models import Entity, Relation


class KnowledgeGraph:
    """基于内存的知识图谱，支持实体/关系 CRUD 和 BFS 遍历。"""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        # 辅助索引：entity_id -> list[relation_id]
        self._entity_relations: dict[str, list[str]] = {}
        # 辅助索引：(name, type) -> entity_id
        self._name_type_index: dict[tuple[str, str], str] = {}

    # --- Entity CRUD ---

    def add_entity(self, entity: Entity) -> None:
        """添加实体到图谱。

        Args:
            entity: 待添加的实体

        Raises:
            ValueError: 同名同类型的实体已存在
        """
        key = (entity.name, entity.entity_type)
        if key in self._name_type_index:
            raise ValueError(
                f"实体已存在: name={entity.name}, type={entity.entity_type}"
            )
        self._entities[entity.id] = entity
        self._name_type_index[key] = entity.id
        self._entity_relations[entity.id] = []

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """通过 ID 获取实体。

        Args:
            entity_id: 实体 ID

        Returns:
            实体对象，不存在返回 None
        """
        return self._entities.get(entity_id)

    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> list[Entity]:
        """按名称和/或类型查找实体。

        Args:
            name: 过滤名称
            entity_type: 过滤类型

        Returns:
            匹配的实体列表
        """
        results: list[Entity] = []
        for e in self._entities.values():
            if name is not None and e.name != name:
                continue
            if entity_type is not None and e.entity_type != entity_type:
                continue
            results.append(e)
        return results

    def remove_entity(self, entity_id: str) -> None:
        """删除实体及其关联的所有关系。

        Args:
            entity_id: 实体 ID

        Raises:
            KeyError: 实体不存在
        """
        if entity_id not in self._entities:
            raise KeyError(f"实体 {entity_id} 不存在")

        # 删除关联的所有关系
        related_ids = list(self._entity_relations.get(entity_id, []))
        for rid in related_ids:
            self._remove_relation_internal(rid)

        # 删除实体
        entity = self._entities.pop(entity_id)
        del self._name_type_index[(entity.name, entity.entity_type)]
        self._entity_relations.pop(entity_id, None)

    def entity_count(self) -> int:
        """返回实体数量"""
        return len(self._entities)

    # --- Relation CRUD ---

    def add_relation(self, relation: Relation) -> None:
        """添加关系到图谱。

        Args:
            relation: 待添加的关系

        Raises:
            ValueError: 源或目标实体不存在
        """
        if relation.source_id not in self._entities:
            raise ValueError(f"源实体 {relation.source_id} 不存在")
        if relation.target_id not in self._entities:
            raise ValueError(f"目标实体 {relation.target_id} 不存在")

        self._relations[relation.id] = relation
        self._entity_relations[relation.source_id].append(relation.id)
        self._entity_relations[relation.target_id].append(relation.id)

    def find_relations(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> list[Relation]:
        """按条件查找关系。

        Args:
            source_id: 过滤源实体
            target_id: 过滤目标实体
            relation_type: 过滤关系类型

        Returns:
            匹配的关系列表
        """
        results: list[Relation] = []
        for r in self._relations.values():
            if source_id is not None and r.source_id != source_id:
                continue
            if target_id is not None and r.target_id != target_id:
                continue
            if relation_type is not None and r.relation_type != relation_type:
                continue
            results.append(r)
        return results

    def remove_relation(self, relation_id: str) -> None:
        """删除关系。

        Args:
            relation_id: 关系 ID

        Raises:
            KeyError: 关系不存在
        """
        if relation_id not in self._relations:
            raise KeyError(f"关系 {relation_id} 不存在")
        self._remove_relation_internal(relation_id)

    def _remove_relation_internal(self, relation_id: str) -> None:
        """内部方法：删除关系并更新索引（不检查存在性）。"""
        if relation_id not in self._relations:
            return
        relation = self._relations.pop(relation_id)
        # 从两端实体的索引中移除
        for eid in (relation.source_id, relation.target_id):
            if eid in self._entity_relations:
                try:
                    self._entity_relations[eid].remove(relation_id)
                except ValueError:
                    pass

    def relation_count(self) -> int:
        """返回关系数量"""
        return len(self._relations)

    # --- Graph Traversal ---

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
    ) -> list[Entity]:
        """获取指定实体的邻居节点。

        Args:
            entity_id: 实体 ID
            relation_type: 按关系类型过滤（可选）

        Returns:
            邻居实体列表

        Raises:
            KeyError: 实体不存在
        """
        if entity_id not in self._entities:
            raise KeyError(f"实体 {entity_id} 不存在")

        neighbors: list[Entity] = []
        for rid in self._entity_relations.get(entity_id, []):
            r = self._relations.get(rid)
            if r is None:
                continue
            if relation_type is not None and r.relation_type != relation_type:
                continue

            neighbor_id = r.target_id if r.source_id == entity_id else r.source_id
            neighbor = self._entities.get(neighbor_id)
            if neighbor is not None:
                neighbors.append(neighbor)

        return neighbors

    def bfs(
        self,
        start_id: str,
        max_depth: int = 2,
        relation_type: Optional[str] = None,
    ) -> list[Entity]:
        """从起始实体开始 BFS 广度优先遍历（不包含起始节点）。

        Args:
            start_id: 起始实体 ID
            max_depth: 最大遍历深度
            relation_type: 按关系类型过滤（可选）

        Returns:
            访问到的实体列表（按 BFS 顺序）

        Raises:
            KeyError: 起始实体不存在
        """
        if start_id not in self._entities:
            raise KeyError(f"实体 {start_id} 不存在")

        visited: set[str] = {start_id}
        result: list[Entity] = []
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for neighbor in self.get_neighbors(current_id, relation_type=relation_type):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    result.append(neighbor)
                    queue.append((neighbor.id, depth + 1))

        return result

    # --- 图谱推理 ---

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
        relation_type: Optional[str] = None,
    ) -> Optional[list[Entity]]:
        """查找两个实体之间的最短路径（BFS）。

        Args:
            source_id: 起始实体 ID
            target_id: 目标实体 ID
            max_depth: 最大搜索深度
            relation_type: 按关系类型过滤

        Returns:
            路径上的实体列表（包含起始和目标），不可达返回 None

        Raises:
            KeyError: 实体不存在
        """
        if source_id not in self._entities:
            raise KeyError(f"实体 {source_id} 不存在")
        if target_id not in self._entities:
            raise KeyError(f"实体 {target_id} 不存在")
        if source_id == target_id:
            return [self._entities[source_id]]

        visited: set[str] = {source_id}
        queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])

        while queue:
            current_id, path = queue.popleft()
            if len(path) - 1 >= max_depth:
                continue

            for neighbor in self.get_neighbors(current_id, relation_type=relation_type):
                if neighbor.id == target_id:
                    return [self._entities[pid] for pid in path] + [neighbor]
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))

        return None

    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        max_paths: int = 10,
        relation_type: Optional[str] = None,
    ) -> list[list[Entity]]:
        """查找两个实体之间的所有路径（DFS，限制深度和数量）。

        Args:
            source_id: 起始实体 ID
            target_id: 目标实体 ID
            max_depth: 最大路径深度
            max_paths: 最大路径数
            relation_type: 按关系类型过滤

        Returns:
            所有路径的列表，每条路径是实体列表

        Raises:
            KeyError: 实体不存在
        """
        if source_id not in self._entities:
            raise KeyError(f"实体 {source_id} 不存在")
        if target_id not in self._entities:
            raise KeyError(f"实体 {target_id} 不存在")

        paths: list[list[Entity]] = []
        visited: set[str] = set()

        def _dfs(current_id: str, path: list[str]) -> None:
            if len(paths) >= max_paths:
                return
            if current_id == target_id:
                paths.append([self._entities[pid] for pid in path])
                return
            if len(path) - 1 >= max_depth:
                return

            for neighbor in self.get_neighbors(current_id, relation_type=relation_type):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    path.append(neighbor.id)
                    _dfs(neighbor.id, path)
                    path.pop()
                    visited.discard(neighbor.id)

        visited.add(source_id)
        _dfs(source_id, [source_id])
        return paths

    def common_neighbors(
        self,
        entity_id_1: str,
        entity_id_2: str,
    ) -> list[Entity]:
        """查找两个实体的共同邻居。

        Args:
            entity_id_1: 第一个实体 ID
            entity_id_2: 第二个实体 ID

        Returns:
            共同邻居实体列表

        Raises:
            KeyError: 实体不存在
        """
        neighbors_1 = {e.id for e in self.get_neighbors(entity_id_1)}
        neighbors_2 = {e.id for e in self.get_neighbors(entity_id_2)}
        common_ids = neighbors_1 & neighbors_2
        return [self._entities[eid] for eid in common_ids]

    def connected_components(self) -> list[list[Entity]]:
        """查找图谱中的所有连通分量。

        Returns:
            每个连通分量包含的实体列表
        """
        visited: set[str] = set()
        components: list[list[Entity]] = []

        for entity_id in self._entities:
            if entity_id in visited:
                continue
            component: list[Entity] = []
            queue: deque[str] = deque([entity_id])
            visited.add(entity_id)
            while queue:
                current = queue.popleft()
                component.append(self._entities[current])
                for neighbor in self.get_neighbors(current):
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        queue.append(neighbor.id)
            components.append(component)

        return components

    def subgraph(
        self,
        entity_ids: set[str],
        include_relations: bool = True,
    ) -> dict[str, list]:
        """提取子图（指定实体及其之间的关系）。

        Args:
            entity_ids: 子图包含的实体 ID 集合
            include_relations: 是否包含关系

        Returns:
            包含 entities 和 relations 的字典
        """
        entities = [
            self._entities[eid] for eid in entity_ids if eid in self._entities
        ]

        relations: list[Relation] = []
        if include_relations:
            for r in self._relations.values():
                if r.source_id in entity_ids and r.target_id in entity_ids:
                    relations.append(r)

        return {"entities": entities, "relations": relations}
