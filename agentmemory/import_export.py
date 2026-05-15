"""记忆数据的导入/导出模块。

支持多种格式：JSON、JSONL、MemoryBank、Markdown、完整导出。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

from agentmemory.models import Memory, Entity, Relation


# ---------------------------------------------------------------------------
# MemoryBank 格式 (.jsonl)
# ---------------------------------------------------------------------------


class MemoryBankFormat:
    """MemoryBank 格式：每行一条 JSON，包含 id/text/metadata/embedding/tags。"""

    @staticmethod
    def export(memories: list[Memory]) -> str:
        """导出为 MemoryBank JSONL 格式。"""
        lines: list[str] = []
        for mem in memories:
            record: dict[str, Any] = {
                "id": mem.id,
                "text": mem.content,
                "metadata": mem.metadata,
                "tags": mem.tags,
            }
            if mem.embedding is not None:
                record["embedding"] = mem.embedding
            lines.append(json.dumps(record, ensure_ascii=False))
        return "\n".join(lines)

    @staticmethod
    def import_data(content: str) -> list[Memory]:
        """从 MemoryBank JSONL 格式导入记忆。

        Raises:
            ValueError: 数据格式不合法
        """
        memories: list[Memory] = []
        for line_no, line in enumerate(content.strip().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc
            if "text" not in record and "content" not in record:
                raise ValueError(f"第 {line_no} 行缺少 'text' 或 'content' 字段")
            mem = Memory(
                id=record.get("id", Memory(content="tmp").id),
                content=record.get("text") or record.get("content", ""),
                metadata=record.get("metadata", {}),
                embedding=record.get("embedding"),
                tags=record.get("tags", []),
                created_at=record.get("created_at", 0.0),
            )
            memories.append(mem)
        return memories


# ---------------------------------------------------------------------------
# JSONL 格式
# ---------------------------------------------------------------------------


class JSONLExporter:
    """JSONL 格式：每行一条完整 Memory.to_dict() 的 JSON。"""

    @staticmethod
    def export(memories: list[Memory]) -> str:
        """导出为 JSONL 格式。"""
        lines = [json.dumps(m.to_dict(), ensure_ascii=False) for m in memories]
        return "\n".join(lines)

    @staticmethod
    def import_data(content: str) -> list[Memory]:
        """从 JSONL 格式导入记忆。

        Raises:
            ValueError: 数据格式不合法
        """
        memories: list[Memory] = []
        for line_no, line in enumerate(content.strip().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc
            try:
                memories.append(Memory.from_dict(data))
            except KeyError as exc:
                raise ValueError(f"第 {line_no} 行缺少必要字段: {exc}") from exc
        return memories


# ---------------------------------------------------------------------------
# 完整导出格式（记忆 + 实体 + 关系）
# ---------------------------------------------------------------------------


class FullExportFormat:
    """完整导出格式：JSON 包含 memories/entities/relations 三个数组。"""

    @staticmethod
    def export(
        memories: list[Memory],
        entities: list[Entity],
        relations: list[Relation],
    ) -> str:
        """导出记忆、实体和关系为 JSON 字符串。"""
        payload: dict[str, Any] = {
            "memories": [m.to_dict() for m in memories],
            "entities": [e.to_dict() for e in entities],
            "relations": [r.to_dict() for r in relations],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def import_data(content: str) -> dict[str, list]:
        """从完整导出 JSON 导入，返回含 memories/entities/relations 的字典。

        Raises:
            ValueError: 数据格式不合法
        """
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("顶层结构必须是 JSON 对象")

        result: dict[str, list] = {"memories": [], "entities": [], "relations": []}
        for d in payload.get("memories", []):
            result["memories"].append(Memory.from_dict(d))
        for d in payload.get("entities", []):
            result["entities"].append(Entity.from_dict(d))
        for d in payload.get("relations", []):
            result["relations"].append(Relation.from_dict(d))
        return result


# ---------------------------------------------------------------------------
# Markdown 格式
# ---------------------------------------------------------------------------


class MarkdownExporter:
    """Markdown 表格格式导出记忆（ID / Content / Tags / Created）。"""

    @staticmethod
    def export(memories: list[Memory]) -> str:
        """导出为 Markdown 表格。"""
        if not memories:
            return "*（无记忆数据）*"

        lines = ["| ID | Content | Tags | Created |", "| --- | --- | --- | --- |"]
        for mem in memories:
            mid = mem.id[:8]
            content = mem.content.replace("|", "\\|").replace("\n", " ")
            if len(content) > 80:
                content = content[:77] + "..."
            tags = ", ".join(mem.tags) if mem.tags else ""
            created = datetime.fromtimestamp(mem.created_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            lines.append(f"| {mid} | {content} | {tags} | {created} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 统一导出管理器
# ---------------------------------------------------------------------------

_FORMAT_MAP = {"json", "jsonl", "memorybank", "markdown", "full"}

_EXTENSION_FORMAT_MAP = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".mbk": "memorybank",
    ".md": "markdown",
}


class ExportManager:
    """统一的导入/导出管理器，封装 HybridMemory 操作。"""

    def __init__(self, memory: Any) -> None:
        """初始化管理器。

        Args:
            memory: HybridMemory 实例引用
        """
        self._memory = memory

    def export_memories(
        self,
        fmt: str,
        output_path: Optional[str] = None,
    ) -> str:
        """导出记忆数据。

        Args:
            fmt: 格式 'json'|'jsonl'|'memorybank'|'markdown'|'full'
            output_path: 输出文件路径（可选）

        Returns:
            导出内容字符串

        Raises:
            ValueError: 不支持的格式
        """
        fmt = fmt.lower().strip()
        if fmt not in _FORMAT_MAP:
            raise ValueError(f"不支持的导出格式: {fmt!r}，可选: {', '.join(sorted(_FORMAT_MAP))}")

        memories = self._memory.list_all()

        if fmt == "json":
            content = json.dumps(
                [m.to_dict() for m in memories], ensure_ascii=False, indent=2
            )
        elif fmt == "jsonl":
            content = JSONLExporter.export(memories)
        elif fmt == "memorybank":
            content = MemoryBankFormat.export(memories)
        elif fmt == "markdown":
            content = MarkdownExporter.export(memories)
        elif fmt == "full":
            kg = self._memory.knowledge_graph
            content = FullExportFormat.export(
                memories, kg.find_entities(), kg.find_relations()
            )
        else:
            raise ValueError(f"不支持的导出格式: {fmt!r}")

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    def import_memories(
        self,
        content: str,
        fmt: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """从字符串导入记忆数据。

        Args:
            content: 数据内容
            fmt: 格式 'json'|'jsonl'|'memorybank'|'full'
            overwrite: 导入前是否清空现有数据

        Returns:
            导入统计 {'memories': int, 'entities': int, 'relations': int}

        Raises:
            ValueError: 不支持的格式或数据不合法
        """
        fmt = fmt.lower().strip()
        if fmt not in _FORMAT_MAP:
            raise ValueError(f"不支持的导入格式: {fmt!r}")
        if fmt == "markdown":
            raise ValueError("Markdown 格式仅支持导出，不支持导入")

        if overwrite:
            for mem in self._memory.list_all():
                self._memory.forget(mem.id)

        result: dict[str, Any] = {"memories": 0, "entities": 0, "relations": 0}

        if fmt == "json":
            memories = self._parse_json(content)
            self._batch_add_memories(memories)
            result["memories"] = len(memories)

        elif fmt == "jsonl":
            memories = JSONLExporter.import_data(content)
            self._batch_add_memories(memories)
            result["memories"] = len(memories)

        elif fmt == "memorybank":
            memories = MemoryBankFormat.import_data(content)
            self._batch_add_memories(memories)
            result["memories"] = len(memories)

        elif fmt == "full":
            data = FullExportFormat.import_data(content)
            for mem in data["memories"]:
                self._add_single_memory(mem)
            for entity in data["entities"]:
                self._memory.knowledge_graph.add_entity(entity)
            for relation in data["relations"]:
                self._memory.knowledge_graph.add_relation(relation)
            result["memories"] = len(data["memories"])
            result["entities"] = len(data["entities"])
            result["relations"] = len(data["relations"])

        return result

    def import_file(
        self,
        path: str,
        fmt: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """从文件导入，自动根据扩展名推断格式。

        Args:
            path: 文件路径
            fmt: 格式（可选，不指定则从扩展名推断）
            overwrite: 是否清空现有数据后再导入

        Returns:
            导入统计字典

        Raises:
            ValueError: 无法推断格式
            FileNotFoundError: 文件不存在
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        if fmt is None:
            _, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext not in _EXTENSION_FORMAT_MAP:
                raise ValueError(
                    f"无法从扩展名 {ext!r} 推断格式，请通过 fmt 参数指定"
                )
            fmt = _EXTENSION_FORMAT_MAP[ext]

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.import_memories(content, fmt, overwrite=overwrite)

    def _add_single_memory(self, mem: Memory) -> None:
        """将单条 Memory 对象添加到 HybridMemory。"""
        self._memory.remember(
            content=mem.content,
            embedding=mem.embedding,
            metadata=mem.metadata,
            tags=mem.tags,
        )

    def _batch_add_memories(self, memories: list[Memory]) -> None:
        """批量添加 Memory 对象到 HybridMemory。"""
        for mem in memories:
            self._add_single_memory(mem)

    @staticmethod
    def _parse_json(content: str) -> list[Memory]:
        """解析 JSON 格式（数组或含 memories 键的对象）的记忆数据。

        Raises:
            ValueError: 数据格式不合法
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败: {exc}") from exc

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("memories", [data])
        else:
            raise ValueError("JSON 顶层必须是数组或对象")

        memories: list[Memory] = []
        for i, item in enumerate(items):
            try:
                memories.append(Memory.from_dict(item))
            except (KeyError, TypeError) as exc:
                raise ValueError(f"第 {i + 1} 条记忆数据不合法: {exc}") from exc
        return memories
