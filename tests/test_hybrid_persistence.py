"""测试 HybridMemory 持久化集成"""

import pytest

from agentmemory.models import Memory, Entity, Relation
from agentmemory.hybrid_memory import HybridMemory


class TestHybridMemoryPersistence:
    """HybridMemory 持久化集成测试"""

    def test_save_and_load_json(self, tmp_path):
        """通过 JSON 后端保存和加载 HybridMemory"""
        path = str(tmp_path / "data")
        hm = HybridMemory(dimension=3, storage_path=path, storage_backend="json")

        # 添加数据
        hm.remember("hello world", embedding=[1.0, 0.0, 0.0], metadata={"source": "test"})
        hm.remember("foo bar", embedding=[0.0, 1.0, 0.0])
        e1 = hm.add_entity("Alice", "person")
        e2 = hm.add_entity("Bob", "person")
        hm.add_relation(e1.id, e2.id, "knows")

        # 保存
        hm.save()

        # 创建新实例，自动加载
        hm2 = HybridMemory(dimension=3, storage_path=path, storage_backend="json")
        hm2.load()

        assert hm2.embedding_store.count() == 2
        assert hm2.knowledge_graph.entity_count() == 2
        assert hm2.knowledge_graph.relation_count() == 1

    def test_save_and_load_sqlite(self, tmp_path):
        """通过 SQLite 后端保存和加载 HybridMemory"""
        path = str(tmp_path / "data")
        hm = HybridMemory(dimension=3, storage_path=path, storage_backend="sqlite")

        hm.remember("test memory", embedding=[1.0, 0.0, 0.0])
        hm.save()

        hm2 = HybridMemory(dimension=3, storage_path=path, storage_backend="sqlite")
        hm2.load()
        assert hm2.embedding_store.count() == 1
        assert hm2.list_all()[0].content == "test memory"

    def test_load_on_init(self, tmp_path):
        """auto_load=True 时初始化自动加载"""
        path = str(tmp_path / "data")

        # 先创建并保存
        hm = HybridMemory(dimension=3, storage_path=path, storage_backend="json")
        hm.remember("persistent", embedding=[1.0, 0.0, 0.0])
        hm.save()

        # 新实例自动加载
        hm2 = HybridMemory(dimension=3, storage_path=path, storage_backend="json", auto_load=True)
        assert hm2.embedding_store.count() == 1

    def test_auto_save_on_remember(self, tmp_path):
        """auto_save=True 时 remember 自动保存"""
        path = str(tmp_path / "data")
        hm = HybridMemory(dimension=3, storage_path=path, storage_backend="json", auto_save=True)

        hm.remember("auto saved", embedding=[1.0, 0.0, 0.0])

        # 直接从文件验证
        hm2 = HybridMemory(dimension=3, storage_path=path, storage_backend="json")
        hm2.load()
        assert hm2.embedding_store.count() == 1

    def test_no_storage_path_no_persistence(self):
        """不指定 storage_path 时 save/load 不可用"""
        hm = HybridMemory(dimension=3)
        with pytest.raises(ValueError, match="storage_path"):
            hm.save()
        with pytest.raises(ValueError, match="storage_path"):
            hm.load()

    def test_list_all_method(self, tmp_path):
        """list_all 返回所有记忆"""
        hm = HybridMemory(dimension=3)
        hm.remember("a", embedding=[1, 0, 0])
        hm.remember("b", embedding=[0, 1, 0])
        all_mems = hm.list_all()
        assert len(all_mems) == 2

    def test_get_memory(self, tmp_path):
        """get_memory 按 ID 获取"""
        hm = HybridMemory(dimension=3)
        mem = hm.remember("test", embedding=[1, 0, 0])
        loaded = hm.get_memory(mem.id)
        assert loaded is mem

    def test_search_returns_sorted_results(self, tmp_path):
        """搜索结果按相似度降序"""
        hm = HybridMemory(dimension=3)
        hm.remember("very similar", embedding=[1.0, 0.0, 0.0])
        hm.remember("somewhat similar", embedding=[0.8, 0.2, 0.0])
        hm.remember("not similar", embedding=[0.0, 1.0, 0.0])

        results = hm.search(query_embedding=[1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        assert results[0].score >= results[1].score >= results[2].score
