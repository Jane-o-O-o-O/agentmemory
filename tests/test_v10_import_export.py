"""v1.0 导入/导出模块测试

覆盖 MemoryBankFormat、JSONLExporter、FullExportFormat、MarkdownExporter、ExportManager
的导出、导入、往返一致性和错误处理。
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from agentmemory.models import Memory, Entity, Relation
from agentmemory.import_export import (
    MemoryBankFormat,
    JSONLExporter,
    FullExportFormat,
    MarkdownExporter,
    ExportManager,
)


# ============================================================
# 辅助工具
# ============================================================

def _make_memory(**overrides) -> Memory:
    """创建测试用 Memory 实例。"""
    defaults = {
        "id": "abc123def456",
        "content": "这是一条测试记忆",
        "created_at": 1700000000.0,
        "metadata": {"source": "test"},
        "embedding": [0.1, 0.2, 0.3],
        "tags": ["test", "ai"],
    }
    defaults.update(overrides)
    return Memory(**defaults)


def _make_entity(**overrides) -> Entity:
    """创建测试用 Entity 实例。"""
    defaults = {
        "id": "ent001",
        "name": "Python",
        "entity_type": "language",
        "properties": {"version": "3.12"},
    }
    defaults.update(overrides)
    return Entity(**defaults)


def _make_relation(**overrides) -> Relation:
    """创建测试用 Relation 实例。"""
    defaults = {
        "id": "rel001",
        "source_id": "ent001",
        "target_id": "ent002",
        "relation_type": "uses",
        "weight": 0.9,
    }
    defaults.update(overrides)
    return Relation(**defaults)


def _mock_hybrid_memory(memories=None, entities=None, relations=None):
    """创建 mock HybridMemory 实例。"""
    mock = MagicMock()
    mock.list_all.return_value = memories or []
    kg = MagicMock()
    kg.find_entities.return_value = entities or []
    kg.find_relations.return_value = relations or []
    mock.knowledge_graph = kg
    return mock


# ============================================================
# MemoryBankFormat 测试
# ============================================================

class TestMemoryBankFormat:
    """MemoryBank 格式导出/导入测试"""

    def test_export_single_memory(self):
        """导出单条记忆"""
        mem = _make_memory()
        result = MemoryBankFormat.export([mem])
        lines = result.strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["id"] == mem.id
        assert data["text"] == mem.content
        assert data["tags"] == mem.tags

    def test_export_multiple_memories(self):
        """导出多条记忆"""
        memories = [_make_memory(id=f"mem{i}") for i in range(3)]
        result = MemoryBankFormat.export(memories)
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_export_without_embedding(self):
        """导出不含 embedding 的记忆时不应包含 embedding 字段"""
        mem = _make_memory(embedding=None)
        result = MemoryBankFormat.export([mem])
        data = json.loads(result)
        assert "embedding" not in data

    def test_export_with_embedding(self):
        """导出含 embedding 的记忆时应包含 embedding 字段"""
        mem = _make_memory(embedding=[0.1, 0.2])
        result = MemoryBankFormat.export([mem])
        data = json.loads(result)
        assert data["embedding"] == [0.1, 0.2]

    def test_roundtrip(self):
        """MemoryBank 格式导出再导入应保持一致"""
        memories = [
            _make_memory(id="m1", content="第一条"),
            _make_memory(id="m2", content="第二条", embedding=None),
        ]
        exported = MemoryBankFormat.export(memories)
        imported = MemoryBankFormat.import_data(exported)
        assert len(imported) == 2
        assert imported[0].id == "m1"
        assert imported[0].content == "第一条"
        assert imported[1].embedding is None

    def test_import_empty_string(self):
        """导入空字符串应返回空列表"""
        assert MemoryBankFormat.import_data("") == []
        assert MemoryBankFormat.import_data("   ") == []

    def test_import_invalid_json(self):
        """导入非法 JSON 应抛出 ValueError"""
        with pytest.raises(ValueError, match="JSON 解析失败"):
            MemoryBankFormat.import_data("{invalid json}")

    def test_import_missing_text_field(self):
        """缺少 text 和 content 字段时应抛出 ValueError"""
        data = json.dumps({"id": "x", "metadata": {}})
        with pytest.raises(ValueError, match="缺少 'text' 或 'content' 字段"):
            MemoryBankFormat.import_data(data)


# ============================================================
# JSONLExporter 测试
# ============================================================

class TestJSONLExporter:
    """JSONL 格式导出/导入测试"""

    def test_export_single_memory(self):
        """导出单条记忆为 JSONL"""
        mem = _make_memory()
        result = JSONLExporter.export([mem])
        data = json.loads(result)
        assert data["id"] == mem.id
        assert data["content"] == mem.content

    def test_export_multiple_lines(self):
        """导出多条记忆，每行一条"""
        memories = [_make_memory(id=f"m{i}") for i in range(3)]
        result = JSONLExporter.export(memories)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            assert json.loads(line)["id"] == f"m{i}"

    def test_roundtrip(self):
        """JSONL 格式往返一致性"""
        memories = [
            _make_memory(id="a", content="hello", tags=["x"]),
            _make_memory(id="b", content="world", embedding=None),
        ]
        exported = JSONLExporter.export(memories)
        imported = JSONLExporter.import_data(exported)
        assert len(imported) == 2
        assert imported[0].content == "hello"
        assert imported[0].tags == ["x"]
        assert imported[1].embedding is None

    def test_import_invalid_json(self):
        """导入非法 JSONL 应抛出 ValueError"""
        with pytest.raises(ValueError, match="JSON 解析失败"):
            JSONLExporter.import_data("{bad json}")

    def test_import_missing_required_fields(self):
        """缺少必要字段时应抛出 ValueError"""
        bad_data = json.dumps({"foo": "bar"})
        with pytest.raises(ValueError, match="缺少必要字段"):
            JSONLExporter.import_data(bad_data)

    def test_roundtrip_preserves_metadata(self):
        """往返后 metadata 应保持不变"""
        mem = _make_memory(metadata={"key": "value", "nested": {"a": 1}})
        exported = JSONLExporter.export([mem])
        imported = JSONLExporter.import_data(exported)
        assert imported[0].metadata == {"key": "value", "nested": {"a": 1}}


# ============================================================
# FullExportFormat 测试
# ============================================================

class TestFullExportFormat:
    """完整导出格式测试（记忆 + 实体 + 关系）"""

    def test_export_all_three_types(self):
        """导出包含记忆、实体、关系的数据"""
        memories = [_make_memory()]
        entities = [_make_entity()]
        relations = [_make_relation()]
        result = FullExportFormat.export(memories, entities, relations)
        data = json.loads(result)
        assert len(data["memories"]) == 1
        assert len(data["entities"]) == 1
        assert len(data["relations"]) == 1

    def test_export_empty_lists(self):
        """导出空列表"""
        result = FullExportFormat.export([], [], [])
        data = json.loads(result)
        assert data == {"memories": [], "entities": [], "relations": []}

    def test_roundtrip(self):
        """完整格式往返一致性"""
        memories = [_make_memory(id="m1")]
        entities = [_make_entity(id="e1")]
        relations = [_make_relation(id="r1")]
        exported = FullExportFormat.export(memories, entities, relations)
        result = FullExportFormat.import_data(exported)
        assert len(result["memories"]) == 1
        assert result["memories"][0].id == "m1"
        assert len(result["entities"]) == 1
        assert result["entities"][0].name == "Python"
        assert len(result["relations"]) == 1
        assert result["relations"][0].relation_type == "uses"

    def test_roundtrip_with_empty_entities_and_relations(self):
        """只有记忆，无实体和关系的往返"""
        memories = [_make_memory()]
        exported = FullExportFormat.export(memories, [], [])
        result = FullExportFormat.import_data(exported)
        assert len(result["memories"]) == 1
        assert result["entities"] == []
        assert result["relations"] == []

    def test_import_invalid_json(self):
        """导入非法 JSON 应抛出 ValueError"""
        with pytest.raises(ValueError, match="JSON 解析失败"):
            FullExportFormat.import_data("{not valid}")

    def test_import_non_object_top_level(self):
        """顶层不是对象时应抛出 ValueError"""
        with pytest.raises(ValueError, match="顶层结构必须是 JSON 对象"):
            FullExportFormat.import_data("[1, 2, 3]")


# ============================================================
# MarkdownExporter 测试
# ============================================================

class TestMarkdownExporter:
    """Markdown 格式导出测试"""

    def test_export_contains_table_header(self):
        """导出应包含表头"""
        mem = _make_memory()
        result = MarkdownExporter.export([mem])
        assert "| ID | Content | Tags | Created |" in result
        assert "| --- | --- | --- | --- |" in result

    def test_export_single_memory_row(self):
        """导出单条记忆应有一行数据"""
        mem = _make_memory(id="abcdef123456", content="测试内容")
        result = MarkdownExporter.export([mem])
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + separator + 1 row
        assert "abcdef12" in lines[2]  # 前8位ID
        assert "测试内容" in lines[2]

    def test_export_empty_list(self):
        """导出空列表应返回提示文本"""
        result = MarkdownExporter.export([])
        assert "无记忆数据" in result

    def test_export_truncates_long_content(self):
        """超长内容应被截断到80字符"""
        long_content = "x" * 100
        mem = _make_memory(content=long_content)
        result = MarkdownExporter.export([mem])
        lines = result.strip().split("\n")
        row = lines[2]
        assert "..." in row

    def test_export_escapes_pipe_in_content(self):
        """内容中的管道符应被转义"""
        mem = _make_memory(content="a | b")
        result = MarkdownExporter.export([mem])
        assert "a \\| b" in result

    def test_export_replaces_newline_in_content(self):
        """内容中的换行应被替换为空格"""
        mem = _make_memory(content="line1\nline2")
        result = MarkdownExporter.export([mem])
        assert "line1 line2" in result

    def test_export_multiple_memories(self):
        """导出多条记忆"""
        memories = [_make_memory(id=f"m{i:012d}") for i in range(3)]
        result = MarkdownExporter.export(memories)
        lines = result.strip().split("\n")
        assert len(lines) == 5  # header + separator + 3 rows

    def test_export_shows_tags(self):
        """导出应显示标签"""
        mem = _make_memory(tags=["ai", "ml"])
        result = MarkdownExporter.export([mem])
        assert "ai, ml" in result

    def test_export_shows_created_timestamp(self):
        """导出应显示格式化的时间戳"""
        from datetime import datetime
        mem = _make_memory(created_at=1700000000.0)
        result = MarkdownExporter.export([mem])
        expected = datetime.fromtimestamp(1700000000.0).strftime("%Y-%m-%d")
        assert expected in result


# ============================================================
# ExportManager 测试
# ============================================================

class TestExportManagerExport:
    """ExportManager.export_memories() 测试"""

    def test_export_json_format(self):
        """以 JSON 格式导出"""
        mem = _make_memory()
        mock_mem = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem)
        result = manager.export_memories("json")
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["id"] == mem.id

    def test_export_jsonl_format(self):
        """以 JSONL 格式导出"""
        mem = _make_memory()
        mock_mem = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem)
        result = manager.export_memories("jsonl")
        data = json.loads(result)
        assert data["content"] == mem.content

    def test_export_memorybank_format(self):
        """以 MemoryBank 格式导出"""
        mem = _make_memory()
        mock_mem = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem)
        result = manager.export_memories("memorybank")
        data = json.loads(result)
        assert data["text"] == mem.content

    def test_export_markdown_format(self):
        """以 Markdown 格式导出"""
        mem = _make_memory()
        mock_mem = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem)
        result = manager.export_memories("markdown")
        assert "| ID | Content |" in result

    def test_export_full_format(self):
        """以完整格式导出（含实体和关系）"""
        mem = _make_memory()
        entity = _make_entity()
        relation = _make_relation()
        mock_mem = _mock_hybrid_memory([mem], [entity], [relation])
        manager = ExportManager(mock_mem)
        result = manager.export_memories("full")
        data = json.loads(result)
        assert "memories" in data
        assert "entities" in data
        assert "relations" in data

    def test_export_to_file(self):
        """导出到文件"""
        mem = _make_memory()
        mock_mem = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            result = manager.export_memories("json", output_path=path)
            assert os.path.isfile(path)
            with open(path, "r", encoding="utf-8") as f:
                file_content = f.read()
            assert file_content == result
        finally:
            os.unlink(path)

    def test_export_unsupported_format(self):
        """不支持的导出格式应抛出 ValueError"""
        mock_mem = _mock_hybrid_memory()
        manager = ExportManager(mock_mem)
        with pytest.raises(ValueError, match="不支持的导出格式"):
            manager.export_memories("yaml")

    def test_export_case_insensitive_format(self):
        """格式名称应不区分大小写"""
        mem = _make_memory()
        mock_mem = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem)
        result = manager.export_memories("JSON")
        data = json.loads(result)
        assert isinstance(data, list)


class TestExportManagerImport:
    """ExportManager.import_memories() 测试"""

    def test_import_json_array(self):
        """从 JSON 数组导入"""
        mem = _make_memory()
        content = json.dumps([mem.to_dict()], ensure_ascii=False)
        mock_mem = _mock_hybrid_memory()
        manager = ExportManager(mock_mem)
        result = manager.import_memories(content, "json")
        assert result["memories"] == 1
        mock_mem.remember.assert_called_once()

    def test_import_jsonl(self):
        """从 JSONL 格式导入"""
        mem = _make_memory()
        content = json.dumps(mem.to_dict(), ensure_ascii=False)
        mock_mem = _mock_hybrid_memory()
        manager = ExportManager(mock_mem)
        result = manager.import_memories(content, "jsonl")
        assert result["memories"] == 1

    def test_import_memorybank(self):
        """从 MemoryBank 格式导入"""
        record = {"id": "x", "text": "hello", "metadata": {}, "tags": []}
        content = json.dumps(record)
        mock_mem = _mock_hybrid_memory()
        manager = ExportManager(mock_mem)
        result = manager.import_memories(content, "memorybank")
        assert result["memories"] == 1

    def test_import_full_format(self):
        """从完整格式导入记忆、实体和关系"""
        mem = _make_memory()
        entity = _make_entity()
        relation = _make_relation()
        payload = {
            "memories": [mem.to_dict()],
            "entities": [entity.to_dict()],
            "relations": [relation.to_dict()],
        }
        content = json.dumps(payload, ensure_ascii=False)
        mock_mem = _mock_hybrid_memory()
        manager = ExportManager(mock_mem)
        result = manager.import_memories(content, "full")
        assert result["memories"] == 1
        assert result["entities"] == 1
        assert result["relations"] == 1
        mock_mem.knowledge_graph.add_entity.assert_called_once()
        mock_mem.knowledge_graph.add_relation.assert_called_once()

    def test_import_markdown_raises_error(self):
        """导入 Markdown 格式应抛出 ValueError"""
        mock_mem = _mock_hybrid_memory()
        manager = ExportManager(mock_mem)
        with pytest.raises(ValueError, match="Markdown 格式仅支持导出"):
            manager.import_memories("some content", "markdown")

    def test_import_unsupported_format(self):
        """不支持的导入格式应抛出 ValueError"""
        mock_mem = _mock_hybrid_memory()
        manager = ExportManager(mock_mem)
        with pytest.raises(ValueError, match="不支持的导入格式"):
            manager.import_memories("data", "yaml")

    def test_import_overwrite_clears_existing(self):
        """overwrite=True 时应先清空现有数据"""
        existing = _make_memory(id="old_mem")
        mock_mem = _mock_hybrid_memory([existing])
        mem = _make_memory(id="new_mem", content="新记忆")
        content = json.dumps([mem.to_dict()], ensure_ascii=False)
        manager = ExportManager(mock_mem)
        manager.import_memories(content, "json", overwrite=True)
        mock_mem.forget.assert_called_once_with("old_mem")

    def test_roundtrip_json(self):
        """JSON 格式完整往返"""
        mem = _make_memory()
        mock_mem_export = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem_export)
        exported = manager.export_memories("json")
        mock_mem_import = _mock_hybrid_memory()
        manager2 = ExportManager(mock_mem_import)
        result = manager2.import_memories(exported, "json")
        assert result["memories"] == 1

    def test_roundtrip_jsonl(self):
        """JSONL 格式完整往返"""
        mem = _make_memory()
        mock_mem_export = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem_export)
        exported = manager.export_memories("jsonl")
        mock_mem_import = _mock_hybrid_memory()
        manager2 = ExportManager(mock_mem_import)
        result = manager2.import_memories(exported, "jsonl")
        assert result["memories"] == 1

    def test_roundtrip_memorybank(self):
        """MemoryBank 格式完整往返"""
        mem = _make_memory()
        mock_mem_export = _mock_hybrid_memory([mem])
        manager = ExportManager(mock_mem_export)
        exported = manager.export_memories("memorybank")
        mock_mem_import = _mock_hybrid_memory()
        manager2 = ExportManager(mock_mem_import)
        result = manager2.import_memories(exported, "memorybank")
        assert result["memories"] == 1


class TestExportManagerImportFile:
    """ExportManager.import_file() 测试"""

    def test_import_file_json(self):
        """从 .json 文件导入"""
        mem = _make_memory()
        content = json.dumps([mem.to_dict()], ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            mock_mem = _mock_hybrid_memory()
            manager = ExportManager(mock_mem)
            result = manager.import_file(path)
            assert result["memories"] == 1
        finally:
            os.unlink(path)

    def test_import_file_jsonl(self):
        """从 .jsonl 文件导入"""
        mem = _make_memory()
        content = json.dumps(mem.to_dict(), ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            mock_mem = _mock_hybrid_memory()
            manager = ExportManager(mock_mem)
            result = manager.import_file(path)
            assert result["memories"] == 1
        finally:
            os.unlink(path)

    def test_import_file_with_explicit_format(self):
        """通过 fmt 参数显式指定格式"""
        mem = _make_memory()
        content = json.dumps(mem.to_dict(), ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            suffix=".dat", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            mock_mem = _mock_hybrid_memory()
            manager = ExportManager(mock_mem)
            result = manager.import_file(path, fmt="jsonl")
            assert result["memories"] == 1
        finally:
            os.unlink(path)

    def test_import_file_not_found(self):
        """文件不存在时应抛出 FileNotFoundError"""
        mock_mem = _mock_hybrid_memory()
        manager = ExportManager(mock_mem)
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            manager.import_file("/nonexistent/path/data.json")

    def test_import_file_unknown_extension(self):
        """未知扩展名且未指定 fmt 时应抛出 ValueError"""
        with tempfile.NamedTemporaryFile(
            suffix=".xyz", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("some content")
            path = f.name
        try:
            mock_mem = _mock_hybrid_memory()
            manager = ExportManager(mock_mem)
            with pytest.raises(ValueError, match="无法从扩展名"):
                manager.import_file(path)
        finally:
            os.unlink(path)

    def test_import_file_markdown_raises_error(self):
        """从 .md 文件导入应因不支持 Markdown 导入而失败"""
        with tempfile.NamedTemporaryFile(
            suffix=".md", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("| ID | Content |\n| --- | --- |")
            path = f.name
        try:
            mock_mem = _mock_hybrid_memory()
            manager = ExportManager(mock_mem)
            with pytest.raises(ValueError, match="Markdown 格式仅支持导出"):
                manager.import_file(path)
        finally:
            os.unlink(path)
