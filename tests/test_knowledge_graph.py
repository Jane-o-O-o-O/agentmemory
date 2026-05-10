"""测试知识图谱：KnowledgeGraph"""

import pytest

from agentmemory.models import Entity, Relation
from agentmemory.knowledge_graph import KnowledgeGraph


class TestKnowledgeGraph:
    """KnowledgeGraph 知识图谱测试"""

    # --- Entity CRUD ---

    def test_add_entity(self):
        """添加实体"""
        kg = KnowledgeGraph()
        e = Entity(name="Alice", entity_type="person")
        kg.add_entity(e)
        assert kg.entity_count() == 1

    def test_add_duplicate_entity_raises(self):
        """添加重复实体（同名同类型）应抛出 ValueError"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Alice", entity_type="person")
        kg.add_entity(e1)
        with pytest.raises(ValueError, match="已存在"):
            kg.add_entity(e2)

    def test_add_entity_same_name_different_type(self):
        """同名不同类型不算重复"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type="language")
        e2 = Entity(name="Python", entity_type="library")
        kg.add_entity(e1)
        kg.add_entity(e2)
        assert kg.entity_count() == 2

    def test_get_entity_by_id(self):
        """通过 ID 获取实体"""
        kg = KnowledgeGraph()
        e = Entity(name="Bob", entity_type="person")
        kg.add_entity(e)
        result = kg.get_entity(e.id)
        assert result is e

    def test_get_entity_nonexistent_returns_none(self):
        """获取不存在的实体返回 None"""
        kg = KnowledgeGraph()
        assert kg.get_entity("nonexistent") is None

    def test_find_entities_by_name(self):
        """按名称查找实体"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        results = kg.find_entities(name="Alice")
        assert len(results) == 1
        assert results[0].name == "Alice"

    def test_find_entities_by_type(self):
        """按类型查找实体"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Python", entity_type="language")
        kg.add_entity(e1)
        kg.add_entity(e2)
        results = kg.find_entities(entity_type="person")
        assert len(results) == 1
        assert results[0].name == "Alice"

    def test_find_entities_by_name_and_type(self):
        """按名称+类型联合查找"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type="language")
        e2 = Entity(name="Python", entity_type="library")
        e3 = Entity(name="Rust", entity_type="language")
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_entity(e3)
        results = kg.find_entities(name="Python", entity_type="language")
        assert len(results) == 1
        assert results[0].entity_type == "language"

    def test_remove_entity(self):
        """删除实体，同时删除关联的边"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        r = Relation(source_id=e1.id, target_id=e2.id, relation_type="knows")
        kg.add_relation(r)
        assert kg.relation_count() == 1

        kg.remove_entity(e1.id)
        assert kg.entity_count() == 1
        assert kg.relation_count() == 0  # 关联边也被删除

    def test_remove_nonexistent_entity_raises(self):
        """删除不存在的实体应抛出 KeyError"""
        kg = KnowledgeGraph()
        with pytest.raises(KeyError):
            kg.remove_entity("nonexistent")

    # --- Relation CRUD ---

    def test_add_relation(self):
        """添加关系"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        r = Relation(source_id=e1.id, target_id=e2.id, relation_type="knows")
        kg.add_relation(r)
        assert kg.relation_count() == 1

    def test_add_relation_with_nonexistent_entity_raises(self):
        """引用不存在的实体应抛出 ValueError"""
        kg = KnowledgeGraph()
        r = Relation(source_id="fake1", target_id="fake2", relation_type="knows")
        with pytest.raises(ValueError, match="不存在"):
            kg.add_relation(r)

    def test_find_relations(self):
        """查找关系（按 source、target、type 过滤）"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        e3 = Entity(name="Charlie", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_entity(e3)
        r1 = Relation(source_id=e1.id, target_id=e2.id, relation_type="knows")
        r2 = Relation(source_id=e1.id, target_id=e3.id, relation_type="knows")
        r3 = Relation(source_id=e2.id, target_id=e3.id, relation_type="manages")
        kg.add_relation(r1)
        kg.add_relation(r2)
        kg.add_relation(r3)

        # 按 source 过滤
        results = kg.find_relations(source_id=e1.id)
        assert len(results) == 2

        # 按 relation_type 过滤
        results = kg.find_relations(relation_type="manages")
        assert len(results) == 1
        assert results[0].relation_type == "manages"

        # 按 target 过滤
        results = kg.find_relations(target_id=e3.id)
        assert len(results) == 2

    def test_remove_relation(self):
        """删除关系"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        r = Relation(source_id=e1.id, target_id=e2.id, relation_type="knows")
        kg.add_relation(r)
        kg.remove_relation(r.id)
        assert kg.relation_count() == 0

    # --- Graph Traversal ---

    def test_get_neighbors(self):
        """获取邻居节点"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        e3 = Entity(name="Charlie", entity_type="person")
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_entity(e3)
        kg.add_relation(Relation(source_id=e1.id, target_id=e2.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=e1.id, target_id=e3.id, relation_type="knows"))

        neighbors = kg.get_neighbors(e1.id)
        neighbor_names = {e.name for e in neighbors}
        assert neighbor_names == {"Bob", "Charlie"}

    def test_get_neighbors_with_relation_type(self):
        """按关系类型过滤邻居"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Alice", entity_type="person")
        e2 = Entity(name="Bob", entity_type="person")
        e3 = Entity(name="Acme", entity_type="company")
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_entity(e3)
        kg.add_relation(Relation(source_id=e1.id, target_id=e2.id, relation_type="knows"))
        kg.add_relation(Relation(source_id=e1.id, target_id=e3.id, relation_type="works_at"))

        neighbors = kg.get_neighbors(e1.id, relation_type="knows")
        assert len(neighbors) == 1
        assert neighbors[0].name == "Bob"

    def test_get_neighbors_empty(self):
        """无邻居时返回空列表"""
        kg = KnowledgeGraph()
        e = Entity(name="Alice", entity_type="person")
        kg.add_entity(e)
        assert kg.get_neighbors(e.id) == []

    def test_get_neighbors_nonexistent_raises(self):
        """查询不存在的实体邻居应抛出 KeyError"""
        kg = KnowledgeGraph()
        with pytest.raises(KeyError):
            kg.get_neighbors("nonexistent")

    def test_bfs_traversal(self):
        """BFS 广度优先遍历"""
        kg = KnowledgeGraph()
        e1 = Entity(name="A", entity_type="node")
        e2 = Entity(name="B", entity_type="node")
        e3 = Entity(name="C", entity_type="node")
        e4 = Entity(name="D", entity_type="node")
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_entity(e3)
        kg.add_entity(e4)
        kg.add_relation(Relation(source_id=e1.id, target_id=e2.id, relation_type="link"))
        kg.add_relation(Relation(source_id=e1.id, target_id=e3.id, relation_type="link"))
        kg.add_relation(Relation(source_id=e2.id, target_id=e4.id, relation_type="link"))

        visited = kg.bfs(e1.id, max_depth=2)
        visited_names = {e.name for e in visited}
        assert visited_names == {"B", "C", "D"}

    def test_bfs_max_depth(self):
        """BFS max_depth 限制"""
        kg = KnowledgeGraph()
        e1 = Entity(name="A", entity_type="node")
        e2 = Entity(name="B", entity_type="node")
        e3 = Entity(name="C", entity_type="node")
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_entity(e3)
        kg.add_relation(Relation(source_id=e1.id, target_id=e2.id, relation_type="link"))
        kg.add_relation(Relation(source_id=e2.id, target_id=e3.id, relation_type="link"))

        visited = kg.bfs(e1.id, max_depth=1)
        assert len(visited) == 1
        assert visited[0].name == "B"

    def test_entity_count_and_relation_count(self):
        """计数方法"""
        kg = KnowledgeGraph()
        assert kg.entity_count() == 0
        assert kg.relation_count() == 0
