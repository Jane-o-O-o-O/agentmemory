"""批量操作、标签过滤、导出/导入测试"""

import json
import pytest

from agentmemory.models import Memory
from agentmemory.embedding_provider import HashEmbeddingProvider
from agentmemory.hybrid_memory import HybridMemory


@pytest.fixture
def provider():
    return HashEmbeddingProvider(dim=64)


@pytest.fixture
def memory(provider):
    return HybridMemory(dimension=64, embedding_provider=provider)


# --- 标签基础 ---

class TestMemoryTags:
    """Memory 模型标签功能测试"""

    def test_memory_with_tags(self):
        mem = Memory(content="test", tags=["ai", "ml"])
        assert mem.tags == ["ai", "ml"]

    def test_tags_dedup(self):
        mem = Memory(content="test", tags=["ai", "ai", "ml"])
        assert mem.tags == ["ai", "ml"]

    def test_has_tag_case_insensitive(self):
        mem = Memory(content="test", tags=["AI", "ML"])
        assert mem.has_tag("ai")
        assert mem.has_tag("AI")
        assert not mem.has_tag("deep_learning")

    def test_empty_tags(self):
        mem = Memory(content="test")
        assert mem.tags == []
        assert not mem.has_tag("anything")

    def test_to_dict_with_tags(self):
        mem = Memory(content="test", tags=["a", "b"])
        d = mem.to_dict()
        assert d["tags"] == ["a", "b"]

    def test_from_dict_with_tags(self):
        d = {
            "id": "abc123",
            "content": "test",
            "created_at": 1.0,
            "metadata": {},
            "embedding": None,
            "tags": ["x", "y"],
        }
        mem = Memory.from_dict(d)
        assert mem.tags == ["x", "y"]

    def test_from_dict_missing_tags(self):
        """旧版本数据（没有 tags 字段）兼容"""
        d = {
            "id": "abc123",
            "content": "test",
            "created_at": 1.0,
            "metadata": {},
            "embedding": None,
        }
        mem = Memory.from_dict(d)
        assert mem.tags == []

    def test_str_with_tags(self):
        mem = Memory(content="hello", tags=["ai"])
        s = str(mem)
        assert "tags=" in s


# --- 批量操作 ---

class TestBatchOperations:
    """批量操作测试"""

    def test_batch_remember(self, memory):
        memories = memory.batch_remember(
            contents=["hello", "world", "test"],
            tagss=[["greeting"], ["noun"], ["test"]],
        )
        assert len(memories) == 3
        assert memory.stats()["memory_count"] == 3

    def test_batch_remember_with_embeddings(self, memory):
        emb1 = [1.0] * 64
        emb2 = [0.5] * 64
        memories = memory.batch_remember(
            contents=["a", "b"],
            embeddings=[emb1, emb2],
        )
        assert len(memories) == 2

    def test_batch_remember_length_mismatch(self, memory):
        with pytest.raises(ValueError, match="不一致"):
            memory.batch_remember(
                contents=["a", "b"],
                embeddings=[[1.0] * 64],
            )

    def test_batch_forget(self, memory):
        memories = memory.batch_remember(contents=["a", "b", "c"])
        ids = [m.id for m in memories]
        deleted = memory.batch_forget([ids[0], ids[1], "nonexistent"])
        assert len(deleted) == 2
        assert memory.stats()["memory_count"] == 1

    def test_batch_search(self, memory):
        memory.batch_remember(contents=["cat", "dog", "fish"])
        provider = HashEmbeddingProvider(dim=64)
        queries = [provider.embed("cat"), provider.embed("dog")]
        results = memory.batch_search(queries, top_k=2)
        assert len(results) == 2
        assert len(results[0]) <= 2
        assert len(results[1]) <= 2


# --- 标签过滤搜索 ---

class TestTagFiltering:
    """标签过滤搜索测试"""

    def test_search_with_tags(self, memory):
        memory.remember("python programming", tags=["code", "python"])
        memory.remember("javascript web", tags=["code", "js"])
        memory.remember("machine learning", tags=["ai", "python"])

        provider = HashEmbeddingProvider(dim=64)
        q = provider.embed("python")

        # 搜索所有（threshold=-1 确保负分数也返回）
        all_results = memory.search(q, top_k=10, threshold=-1.0)
        assert len(all_results) == 3

        # 只搜 code 标签
        code_results = memory.search(q, top_k=10, tags=["code"], threshold=-1.0)
        assert len(code_results) == 2

        # 搜 code AND python
        both = memory.search(q, top_k=10, tags=["code", "python"], threshold=-1.0)
        assert len(both) == 1

    def test_search_text_with_tags(self, memory):
        memory.remember("hello world", tags=["greeting"])
        memory.remember("goodbye world", tags=["farewell"])

        results = memory.search_text("world", tags=["greeting"])
        assert len(results) == 1
        assert results[0].memory.has_tag("greeting")

    def test_hybrid_search_with_tags(self, memory):
        memory.remember("test data", tags=["test"])
        results = memory.hybrid_search_text("test", tags=["test"])
        assert len(results) >= 1

    def test_add_remove_tag(self, memory):
        mem = memory.remember("content")
        assert not mem.tags

        memory.add_tag(mem.id, "new_tag")
        assert mem.has_tag("new_tag")

        memory.add_tag(mem.id, "new_tag")  # 重复添加
        assert mem.tags.count("new_tag") == 1

        memory.remove_tag(mem.id, "NEW_TAG")  # 大小写
        assert not mem.has_tag("new_tag")

    def test_add_tag_nonexistent(self, memory):
        with pytest.raises(KeyError):
            memory.add_tag("nonexistent", "tag")

    def test_remove_tag_nonexistent(self, memory):
        with pytest.raises(KeyError):
            memory.remove_tag("nonexistent", "tag")

    def test_get_all_tags(self, memory):
        memory.remember("a", tags=["x", "y"])
        memory.remember("b", tags=["x", "z"])
        tags = memory.get_all_tags()
        assert tags["x"] == 2
        assert tags["y"] == 1
        assert tags["z"] == 1


# --- 导出/导入 ---

class TestExportImport:
    """导出/导入测试"""

    def test_export_json(self, memory):
        memory.remember("test", tags=["tag1"])
        memory.add_entity("Python", "language")

        json_str = memory.export_json()
        data = json.loads(json_str)
        assert "memories" in data
        assert "entities" in data
        assert "relations" in data
        assert len(data["memories"]) == 1
        assert data["memories"][0]["tags"] == ["tag1"]

    def test_import_json(self, memory):
        export_data = {
            "version": "1.0",
            "stats": {"memory_count": 1, "entity_count": 1, "relation_count": 0},
            "memories": [
                {
                    "id": "mem1",
                    "content": "hello",
                    "created_at": 1.0,
                    "metadata": {},
                    "embedding": [1.0] * 64,
                    "tags": ["imported"],
                }
            ],
            "entities": [
                {"id": "ent1", "name": "Test", "entity_type": "type1", "properties": {}}
            ],
            "relations": [],
        }
        counts = memory.import_json(json.dumps(export_data))
        assert counts["memories"] == 1
        assert counts["entities"] == 1
        assert memory.stats()["memory_count"] == 1

    def test_import_json_overwrite(self, memory):
        memory.remember("existing")
        assert memory.stats()["memory_count"] == 1

        export_data = {
            "memories": [
                {"id": "new1", "content": "new", "created_at": 2.0, "metadata": {}, "embedding": [0.5] * 64, "tags": []}
            ],
            "entities": [],
            "relations": [],
        }
        memory.import_json(json.dumps(export_data), overwrite=True)
        assert memory.stats()["memory_count"] == 1
        assert memory.list_all()[0].content == "new"

    def test_export_csv(self, memory):
        memory.remember("hello", tags=["a", "b"])
        csv_str = memory.export_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "id" in lines[0]
        assert "tags" in lines[0]

    def test_import_csv(self, memory):
        # 使用 csv.writer 生成正确转义的 CSV
        from io import StringIO
        import csv
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "content", "created_at", "metadata", "tags"])
        writer.writerow(["csv1", "test content", "1.0", "{}", json.dumps(["tag1"])])
        csv_data = output.getvalue()

        count = memory.import_csv(csv_data)
        assert count == 1
        assert memory.stats()["memory_count"] == 1
        mem = memory.get_memory("csv1")
        assert mem is not None
        assert mem.tags == ["tag1"]

    def test_roundtrip_json(self, memory):
        """导出再导入，数据一致"""
        memory.remember("test1", tags=["a"])
        memory.remember("test2", tags=["b"])
        memory.add_entity("E1", "type1")

        json_str = memory.export_json()

        mem2 = HybridMemory(dimension=64, embedding_provider=HashEmbeddingProvider(dim=64))
        mem2.import_json(json_str)

        assert mem2.stats()["memory_count"] == 2
        assert mem2.stats()["entity_count"] == 1
