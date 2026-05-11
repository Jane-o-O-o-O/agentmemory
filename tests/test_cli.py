"""CLI 工具测试"""

import json
import os
import tempfile

import pytest

from agentmemory.cli import build_parser, main


@pytest.fixture
def store_dir(tmp_path):
    """创建临时存储目录"""
    return str(tmp_path / "test_store")


class TestCLIParser:
    """命令行参数解析测试"""

    def test_parser_remember(self):
        parser = build_parser()
        args = parser.parse_args(["remember", "hello world"])
        assert args.command == "remember"
        assert args.text == "hello world"
        assert args.tags is None

    def test_parser_remember_with_tags(self):
        parser = build_parser()
        args = parser.parse_args(["remember", "hello", "--tags", "tag1", "tag2"])
        assert args.tags == ["tag1", "tag2"]

    def test_parser_search(self):
        parser = build_parser()
        args = parser.parse_args(["search", "query", "--top-k", "10"])
        assert args.command == "search"
        assert args.query == "query"
        assert args.top_k == 10

    def test_parser_search_with_tags(self):
        parser = build_parser()
        args = parser.parse_args(["search", "q", "--tags", "code"])
        assert args.tags == ["code"]

    def test_parser_forget(self):
        parser = build_parser()
        args = parser.parse_args(["forget", "abc123"])
        assert args.command == "forget"
        assert args.id == "abc123"

    def test_parser_list(self):
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_parser_list_with_tag(self):
        parser = build_parser()
        args = parser.parse_args(["list", "--tag", "ai"])
        assert args.tag == "ai"

    def test_parser_tags(self):
        parser = build_parser()
        args = parser.parse_args(["tags"])
        assert args.command == "tags"

    def test_parser_stats(self):
        parser = build_parser()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_parser_export(self):
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "csv", "--output", "out.csv"])
        assert args.command == "export"
        assert args.format == "csv"
        assert args.output == "out.csv"

    def test_parser_import(self):
        parser = build_parser()
        args = parser.parse_args(["import", "data.json", "--overwrite"])
        assert args.command == "import"
        assert args.file == "data.json"
        assert args.overwrite is True

    def test_parser_add_entity(self):
        parser = build_parser()
        args = parser.parse_args(["add-entity", "Python", "language", "--props", "version=3.11"])
        assert args.command == "add-entity"
        assert args.name == "Python"
        assert args.type == "language"
        assert args.props == ["version=3.11"]

    def test_parser_add_relation(self):
        parser = build_parser()
        args = parser.parse_args(["add-relation", "src1", "tgt1", "related_to"])
        assert args.command == "add-relation"

    def test_parser_graph(self):
        parser = build_parser()
        args = parser.parse_args(["graph", "--entity-id", "abc"])
        assert args.entity_id == "abc"

    def test_parser_global_options(self):
        parser = build_parser()
        args = parser.parse_args(["--store", "/tmp/test", "--backend", "sqlite", "--dimension", "256", "stats"])
        assert args.store == "/tmp/test"
        assert args.backend == "sqlite"
        assert args.dimension == 256


class TestCLICommands:
    """CLI 命令执行测试"""

    def test_remember_and_search(self, store_dir, capsys):
        main(["--store", store_dir, "remember", "hello world", "--tags", "test"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["content"] == "hello world"
        assert data["tags"] == ["test"]

        main(["--store", store_dir, "search", "hello"])
        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert len(results) >= 1

    def test_stats(self, store_dir, capsys):
        main(["--store", store_dir, "stats"])
        captured = capsys.readouterr()
        assert "记忆数: 0" in captured.out

    def test_list_empty(self, store_dir, capsys):
        main(["--store", store_dir, "list"])
        captured = capsys.readouterr()
        assert "暂无记忆" in captured.out

    def test_list_with_memories(self, store_dir, capsys):
        main(["--store", store_dir, "remember", "test memory", "--tags", "tag1"])
        capsys.readouterr()

        main(["--store", store_dir, "list"])
        captured = capsys.readouterr()
        assert "test memory" in captured.out

    def test_list_with_tag_filter(self, store_dir, capsys):
        main(["--store", store_dir, "remember", "python code", "--tags", "code"])
        capsys.readouterr()
        main(["--store", store_dir, "remember", "random thought"])
        capsys.readouterr()

        main(["--store", store_dir, "list", "--tag", "code"])
        captured = capsys.readouterr()
        assert "python code" in captured.out
        assert "random thought" not in captured.out

    def test_tags_command(self, store_dir, capsys):
        main(["--store", store_dir, "remember", "a", "--tags", "ai", "ml"])
        capsys.readouterr()
        main(["--store", store_dir, "remember", "b", "--tags", "ai"])
        capsys.readouterr()

        main(["--store", store_dir, "tags"])
        captured = capsys.readouterr()
        assert "ai: 2" in captured.out
        assert "ml: 1" in captured.out

    def test_forget(self, store_dir, capsys):
        main(["--store", store_dir, "remember", "to delete"])
        captured = capsys.readouterr()
        mem_id = json.loads(captured.out)["id"]

        main(["--store", store_dir, "forget", mem_id])
        captured = capsys.readouterr()
        assert "已删除" in captured.out

    def test_forget_nonexistent(self, store_dir):
        with pytest.raises(SystemExit):
            main(["--store", store_dir, "forget", "nonexistent"])

    def test_export_json(self, store_dir, capsys):
        main(["--store", store_dir, "remember", "test data"])
        capsys.readouterr()

        main(["--store", store_dir, "export"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "memories" in data

    def test_export_csv(self, store_dir, capsys):
        main(["--store", store_dir, "remember", "test data"])
        capsys.readouterr()

        main(["--store", store_dir, "export", "--format", "csv"])
        captured = capsys.readouterr()
        assert "id,content" in captured.out

    def test_export_to_file(self, store_dir, capsys):
        main(["--store", store_dir, "remember", "test data"])
        capsys.readouterr()

        out_file = os.path.join(store_dir, "export.json")
        main(["--store", store_dir, "export", "--output", out_file])
        captured = capsys.readouterr()
        assert "已导出" in captured.out
        assert os.path.exists(out_file)

    def test_add_entity_and_graph(self, store_dir, capsys):
        main(["--store", store_dir, "add-entity", "Python", "language", "--props", "version=3.11"])
        captured = capsys.readouterr()
        entity = json.loads(captured.out)
        assert entity["name"] == "Python"

        main(["--store", store_dir, "graph"])
        captured = capsys.readouterr()
        assert "Python" in captured.out

    def test_graph_empty(self, store_dir, capsys):
        main(["--store", store_dir, "graph"])
        captured = capsys.readouterr()
        assert "知识图谱为空" in captured.out

    def test_no_command(self, store_dir, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--store", store_dir])
        assert exc_info.value.code == 0
