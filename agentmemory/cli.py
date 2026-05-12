"""agentmemory CLI — 记忆管理命令行接口

用法:
    agentmemory remember "text" [--tags tag1 tag2]
    agentmemory search "query" [--top-k 5] [--tags tag1] [--hybrid]
    agentmemory forget <id>
    agentmemory list [--tag tag]
    agentmemory tags
    agentmemory stats
    agentmemory inspect <id>
    agentmemory cleanup
    agentmemory version
    agentmemory export [--format json|csv] [--output file]
    agentmemory import <file> [--format json|csv]
    agentmemory add-entity "name" "type" [--props key=value]
    agentmemory add-relation <source_id> <target_id> "type"
    agentmemory graph [--entity-id id]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from agentmemory.embedding_provider import HashEmbeddingProvider
from agentmemory.hybrid_memory import HybridMemory


def _get_memory(args: argparse.Namespace) -> HybridMemory:
    """根据 CLI 参数创建 HybridMemory 实例。

    Args:
        args: 解析后的命令行参数

    Returns:
        配置好的 HybridMemory 实例
    """
    return HybridMemory(
        dimension=args.dimension,
        embedding_provider=HashEmbeddingProvider(dim=args.dimension),
        storage_path=args.store,
        storage_backend=args.backend,
        auto_save=True,
        auto_load=True if args.store else False,
        use_lsh=getattr(args, 'lsh', False),
    )


def cmd_remember(args: argparse.Namespace) -> None:
    """添加记忆"""
    mem = _get_memory(args).remember(
        content=args.text,
        tags=args.tags or [],
    )
    print(json.dumps(mem.to_dict(), ensure_ascii=False, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    """搜索记忆"""
    mem = _get_memory(args)
    if getattr(args, 'hybrid', False):
        results = mem.hybrid_search_text(
            query=args.query,
            top_k=args.top_k,
            tags=args.tags or None,
        )
    else:
        results = mem.search_text(
            query=args.query,
            top_k=args.top_k,
            tags=args.tags or None,
        )
    if not results:
        print("未找到匹配的记忆")
        return
    output = []
    for r in results:
        entry = {
            "id": r.memory.id,
            "content": r.memory.content,
            "score": round(r.score, 4),
            "tags": r.memory.tags,
        }
        if r.context:
            entry["context"] = [{"id": c.id, "content": c.content[:80]} for c in r.context]
        output.append(entry)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def cmd_forget(args: argparse.Namespace) -> None:
    """删除记忆"""
    try:
        _get_memory(args).forget(args.id)
        print(f"已删除记忆: {args.id}")
    except KeyError:
        print(f"记忆不存在: {args.id}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    """列出所有记忆"""
    mem = _get_memory(args)
    if args.tag:
        memories = mem.embedding_store.find_by_tag(args.tag)
    else:
        memories = mem.list_all()

    if not memories:
        print("暂无记忆")
        return

    for m in memories:
        tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
        print(f"  {m.id[:12]}  {m.content[:60]}{tag_str}")


def cmd_tags(args: argparse.Namespace) -> None:
    """列出所有标签"""
    tags = _get_memory(args).get_all_tags()
    if not tags:
        print("暂无标签")
        return
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")


def cmd_stats(args: argparse.Namespace) -> None:
    """显示统计信息"""
    s = _get_memory(args).stats()
    print(f"记忆数: {s['memory_count']}")
    print(f"实体数: {s['entity_count']}")
    print(f"关系数: {s['relation_count']}")
    print(f"向量维度: {s['dimension']}")
    print(f"LSH 索引: {'启用' if s['use_lsh'] else '禁用'}")


def cmd_inspect(args: argparse.Namespace) -> None:
    """查看记忆详情"""
    mem = _get_memory(args)
    memory = mem.get_memory(args.id)
    if memory is None:
        print(f"记忆不存在: {args.id}", file=sys.stderr)
        sys.exit(1)

    info = memory.to_dict()
    # 添加生命周期信息
    lifecycle_info = mem.get_lifecycle_info(args.id)
    if lifecycle_info:
        info["lifecycle"] = {
            "age_seconds": round(lifecycle_info["age_seconds"], 2),
            "is_expired": lifecycle_info["is_expired"],
            "decay_factor": round(lifecycle_info["decay_factor"], 4),
            "access_count": lifecycle_info["access_count"],
        }
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_cleanup(args: argparse.Namespace) -> None:
    """清理过期记忆"""
    expired = _get_memory(args).cleanup_expired()
    if expired:
        print(f"已清理 {len(expired)} 条过期记忆:")
        for mid in expired:
            print(f"  {mid}")
    else:
        print("没有过期记忆需要清理")


def cmd_version(args: argparse.Namespace) -> None:
    """显示版本信息"""
    from agentmemory import __version__
    print(f"agentmemory v{__version__}")
    print("混合记忆框架：向量搜索 + 知识图谱")


def cmd_export(args: argparse.Namespace) -> None:
    """导出数据"""
    mem = _get_memory(args)
    if args.format == "csv":
        data = mem.export_csv()
    else:
        data = mem.export_json()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"已导出到: {args.output}")
    else:
        print(data)


def cmd_import(args: argparse.Namespace) -> None:
    """导入数据"""
    with open(args.file, "r", encoding="utf-8") as f:
        data = f.read()

    mem = _get_memory(args)
    if args.format == "csv":
        count = mem.import_csv(data)
        print(f"已导入 {count} 条记忆")
    else:
        counts = mem.import_json(data, overwrite=args.overwrite)
        print(f"已导入: {counts['memories']} 条记忆, {counts['entities']} 个实体, {counts['relations']} 条关系")


def cmd_add_entity(args: argparse.Namespace) -> None:
    """添加实体"""
    props = {}
    if args.props:
        for p in args.props:
            if "=" in p:
                k, v = p.split("=", 1)
                props[k] = v
    entity = _get_memory(args).add_entity(
        name=args.name,
        entity_type=args.type,
        properties=props,
    )
    print(json.dumps(entity.to_dict(), ensure_ascii=False, indent=2))


def cmd_add_relation(args: argparse.Namespace) -> None:
    """添加关系"""
    relation = _get_memory(args).add_relation(
        source_id=args.source_id,
        target_id=args.target_id,
        relation_type=args.relation_type,
    )
    print(json.dumps(relation.to_dict(), ensure_ascii=False, indent=2))


def cmd_graph(args: argparse.Namespace) -> None:
    """查看知识图谱"""
    mem = _get_memory(args)
    if args.entity_id:
        # 显示特定实体的邻居
        try:
            neighbors = mem.get_neighbors(args.entity_id)
            entity = mem.knowledge_graph.get_entity(args.entity_id)
            if entity:
                print(f"实体: {entity.name} ({entity.entity_type})")
                print(f"邻居: {len(neighbors)} 个")
                for n in neighbors:
                    print(f"  -> {n.name} ({n.entity_type})")
            else:
                print(f"实体不存在: {args.entity_id}", file=sys.stderr)
                sys.exit(1)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    else:
        # 显示图谱概览
        entities = mem.knowledge_graph.find_entities()
        relations = mem.knowledge_graph.find_relations()
        if not entities:
            print("知识图谱为空")
            return
        print(f"实体 ({len(entities)}):")
        for e in entities:
            props = f" {e.properties}" if e.properties else ""
            print(f"  {e.id[:12]}  {e.name} ({e.entity_type}){props}")
        print(f"\n关系 ({len(relations)}):")
        for r in relations:
            src = mem.knowledge_graph.get_entity(r.source_id)
            tgt = mem.knowledge_graph.get_entity(r.target_id)
            src_name = src.name[:20] if src else r.source_id[:12]
            tgt_name = tgt.name[:20] if tgt else r.target_id[:12]
            print(f"  {src_name} --[{r.relation_type}]--> {tgt_name}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        配置好的 ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="agentmemory",
        description="agentmemory — 混合记忆框架 CLI",
    )
    parser.add_argument(
        "--store", "-s", default=None,
        help="持久化存储路径",
    )
    parser.add_argument(
        "--backend", "-b", default="json", choices=["json", "sqlite"],
        help="存储后端 (默认: json)",
    )
    parser.add_argument(
        "--dimension", "-d", type=int, default=128,
        help="向量维度 (默认: 128)",
    )
    parser.add_argument(
        "--lsh", action="store_true",
        help="启用 LSH 近似搜索索引",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # remember
    p = subparsers.add_parser("remember", help="添加记忆")
    p.add_argument("text", help="记忆内容")
    p.add_argument("--tags", nargs="*", help="标签列表")
    p.set_defaults(func=cmd_remember)

    # search
    p = subparsers.add_parser("search", help="搜索记忆")
    p.add_argument("query", help="搜索查询")
    p.add_argument("--top-k", type=int, default=5, help="返回结果数")
    p.add_argument("--tags", nargs="*", help="标签过滤")
    p.add_argument("--hybrid", action="store_true", help="使用混合搜索（向量+图谱）")
    p.set_defaults(func=cmd_search)

    # forget
    p = subparsers.add_parser("forget", help="删除记忆")
    p.add_argument("id", help="记忆 ID")
    p.set_defaults(func=cmd_forget)

    # list
    p = subparsers.add_parser("list", help="列出记忆")
    p.add_argument("--tag", help="按标签过滤")
    p.set_defaults(func=cmd_list)

    # tags
    p = subparsers.add_parser("tags", help="列出标签")
    p.set_defaults(func=cmd_tags)

    # stats
    p = subparsers.add_parser("stats", help="统计信息")
    p.set_defaults(func=cmd_stats)

    # inspect
    p = subparsers.add_parser("inspect", help="查看记忆详情")
    p.add_argument("id", help="记忆 ID")
    p.set_defaults(func=cmd_inspect)

    # cleanup
    p = subparsers.add_parser("cleanup", help="清理过期记忆")
    p.set_defaults(func=cmd_cleanup)

    # version
    p = subparsers.add_parser("version", help="版本信息")
    p.set_defaults(func=cmd_version)

    # export
    p = subparsers.add_parser("export", help="导出数据")
    p.add_argument("--format", choices=["json", "csv"], default="json", help="导出格式")
    p.add_argument("--output", "-o", help="输出文件路径")
    p.set_defaults(func=cmd_export)

    # import
    p = subparsers.add_parser("import", help="导入数据")
    p.add_argument("file", help="导入文件路径")
    p.add_argument("--format", choices=["json", "csv"], default="json", help="导入格式")
    p.add_argument("--overwrite", action="store_true", help="清空现有数据后导入")
    p.set_defaults(func=cmd_import)

    # add-entity
    p = subparsers.add_parser("add-entity", help="添加实体")
    p.add_argument("name", help="实体名称")
    p.add_argument("type", help="实体类型")
    p.add_argument("--props", nargs="*", help="属性 (key=value)")
    p.set_defaults(func=cmd_add_entity)

    # add-relation
    p = subparsers.add_parser("add-relation", help="添加关系")
    p.add_argument("source_id", help="源实体 ID")
    p.add_argument("target_id", help="目标实体 ID")
    p.add_argument("relation_type", help="关系类型")
    p.set_defaults(func=cmd_add_relation)

    # graph
    p = subparsers.add_parser("graph", help="查看知识图谱")
    p.add_argument("--entity-id", help="查看特定实体的邻居")
    p.set_defaults(func=cmd_graph)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI 入口点。

    Args:
        argv: 命令行参数列表（默认使用 sys.argv）
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
