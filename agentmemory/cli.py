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


def cmd_graph_export(args: argparse.Namespace) -> None:
    """导出知识图谱可视化"""
    from agentmemory.graph_viz import export_dot, export_html
    mem = _get_memory(args)
    if args.format == "dot":
        data = export_dot(mem.knowledge_graph, title=args.title)
    else:
        data = export_html(mem.knowledge_graph, title=args.title)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"已导出到: {args.output}")
    else:
        print(data)


def cmd_graph_stats(args: argparse.Namespace) -> None:
    """知识图谱统计报告"""
    from agentmemory.graph_viz import graph_stats_text
    mem = _get_memory(args)
    print(graph_stats_text(mem.knowledge_graph))


def cmd_cache_stats(args: argparse.Namespace) -> None:
    """搜索缓存统计"""
    mem = _get_memory(args)
    stats = mem.get_cache_stats()
    if stats is None:
        print("搜索缓存未启用（使用 --cache-size > 0 启用）")
        return
    print(f"缓存命中: {stats['hits']}")
    print(f"缓存未命中: {stats['misses']}")
    print(f"命中率: {stats['hit_rate']:.2%}")
    print(f"当前大小: {stats['size']}/{stats['max_size']}")
    print(f"TTL: {stats['ttl_seconds']}秒" if stats['ttl_seconds'] else "TTL: 无限制")


def cmd_shortest_path(args: argparse.Namespace) -> None:
    """查找两个实体之间的最短路径"""
    mem = _get_memory(args)
    try:
        path = mem.shortest_path(args.source_id, args.target_id, max_depth=args.max_depth)
        if path is None:
            print("不可达：两个实体之间没有路径")
            return
        print(f"路径（{len(path)} 步）:")
        for i, entity in enumerate(path):
            arrow = "  -> " if i > 0 else "     "
            print(f"{arrow}{entity.name} ({entity.entity_type}) [{entity.id[:12]}]")
    except KeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def cmd_common_neighbors(args: argparse.Namespace) -> None:
    """查找两个实体的共同邻居"""
    mem = _get_memory(args)
    try:
        neighbors = mem.common_neighbors(args.entity1, args.entity2)
        if not neighbors:
            print("没有共同邻居")
            return
        print(f"共同邻居 ({len(neighbors)}):")
        for n in neighbors:
            print(f"  {n.name} ({n.entity_type}) [{n.id[:12]}]")
    except KeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def cmd_connected_components(args: argparse.Namespace) -> None:
    """列出图谱的连通分量"""
    mem = _get_memory(args)
    components = mem.connected_components()
    if not components:
        print("知识图谱为空")
        return
    print(f"连通分量 ({len(components)}):")
    for i, comp in enumerate(components):
        names = [e.name for e in comp[:5]]
        extra = f" +{len(comp) - 5} more" if len(comp) > 5 else ""
        print(f"  #{i + 1} ({len(comp)} entities): {', '.join(names)}{extra}")


def _get_version() -> str:
    """获取版本号"""
    from agentmemory import __version__
    return __version__


def cmd_rag(args: argparse.Namespace) -> None:
    """执行 RAG 检索增强生成"""
    mem = _get_memory(args)
    result = mem.rag(
        query=args.query,
        top_k=args.top_k,
        max_context_tokens=args.max_tokens,
        tags=args.tags or None,
        use_hybrid=getattr(args, 'hybrid', False),
    )
    if getattr(args, 'prompt_only', False):
        print(result["prompt"])
    else:
        output = {
            "prompt": result["prompt"],
            "sources": result["sources"],
            "total_tokens": result["total_tokens"],
            "truncated": result["truncated"],
            "reranked": result["reranked"],
            "pipeline_time_ms": result["pipeline_time_ms"],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))


def cmd_metrics(args: argparse.Namespace) -> None:
    """显示运行时指标"""
    mem = _get_memory(args)
    fmt = getattr(args, 'format', 'text')
    if fmt == 'json':
        print(mem.metrics_json())
    elif fmt == 'prometheus':
        print(mem.metrics_prometheus())
    else:
        snap = mem.metrics_snapshot()
        print("=" * 40)
        print("  agentmemory 运行时指标")
        print("=" * 40)
        print(f"\n  计数器:")
        for name, info in snap.get("counters", {}).items():
            print(f"    {name}: {info['value']}")
        print(f"\n  计时器:")
        for name, info in snap.get("timers", {}).items():
            print(f"    {name}: count={info['count']} mean={info['mean_ms']:.2f}ms p50={info['p50_ms']:.2f}ms p95={info['p95_ms']:.2f}ms")
        print(f"\n  仪表盘:")
        for name, info in snap.get("gauges", {}).items():
            print(f"    {name}: {info['value']}")
        print()


def cmd_health(args: argparse.Namespace) -> None:
    """执行健康检查"""
    mem = _get_memory(args)
    report = mem.health_check()
    status_icons = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}
    overall = report["overall_status"]
    print(f"总体状态: {status_icons.get(overall, '?')} {overall}")
    print()
    for check in report["checks"]:
        icon = status_icons.get(check["status"], "?")
        print(f"  {icon} {check['name']}: {check['message']}")
        if check.get("details"):
            for k, v in check["details"].items():
                print(f"     {k}: {v}")


def cmd_compress(args: argparse.Namespace) -> None:
    """压缩向量存储"""
    mem = _get_memory(args)
    method = args.method
    stats = mem.compress_vectors(method=method, num_subspaces=args.subspaces)
    if "error" in stats:
        print(f"错误: {stats['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"压缩方法: {stats['method']}")
    print(f"向量数量: {stats['num_vectors']}")
    print(f"压缩比: {stats['compression_ratio']}x")
    print(f"每向量字节: {stats['compressed_bytes_per_vector']}")
    print(f"总压缩字节: {stats['total_compressed_bytes']}")


def cmd_interactive(args: argparse.Namespace) -> None:
    """交互式 REPL 模式"""
    mem = _get_memory(args)
    print(f"agentmemory v{_get_version()} — 交互式模式")
    print("输入 remember/search/list/stats/forget/cleanup/quit")
    print()
    while True:
        try:
            line = input("agentmemory> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        try:
            if cmd in ("quit", "exit", "q"):
                print("退出")
                break
            elif cmd == "remember" and arg:
                tags = []
                if "--tags" in arg:
                    text_part, tags_part = arg.split("--tags", 1)
                    text = text_part.strip().strip('"').strip("'")
                    tags = tags_part.strip().split()
                else:
                    text = arg.strip('"').strip("'")
                m = mem.remember(text, tags=tags)
                print(f"  已添加: {m.id[:12]}")
            elif cmd == "search" and arg:
                results = mem.search_text(arg.strip('"').strip("'"))
                if not results:
                    print("  未找到匹配的记忆")
                for r in results:
                    tag_str = f" [{', '.join(r.memory.tags)}]" if r.memory.tags else ""
                    print(f"  {r.memory.id[:12]}  score={r.score:.4f}  {r.memory.content[:50]}{tag_str}")
            elif cmd == "list":
                all_mems = mem.list_all()
                if not all_mems:
                    print("  暂无记忆")
                for m in all_mems:
                    tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                    print(f"  {m.id[:12]}  {m.content[:50]}{tag_str}")
            elif cmd == "stats":
                s = mem.stats()
                print(f"  记忆: {s['memory_count']}  实体: {s['entity_count']}  关系: {s['relation_count']}  LSH: {'✓' if s['use_lsh'] else '✗'}")
            elif cmd == "forget" and arg:
                mem.forget(arg.strip())
                print(f"  已删除: {arg.strip()}")
            elif cmd == "cleanup":
                expired = mem.cleanup_expired()
                print(f"  清理 {len(expired)} 条过期记忆")
            elif cmd == "help":
                print("  remember <text> [--tags t1 t2]  添加记忆")
                print("  search <query>                  搜索记忆")
                print("  list                            列出所有")
                print("  stats                           统计信息")
                print("  forget <id>                     删除记忆")
                print("  cleanup                         清理过期")
                print("  quit                            退出")
            else:
                print(f"  未知命令: {cmd}，输入 help 查看帮助")
        except Exception as e:
            print(f"  错误: {e}")


def cmd_batch_import(args: argparse.Namespace) -> None:
    """从文本文件批量导入记忆（每行一条）"""
    mem = _get_memory(args)
    tags = args.tags or []
    count = 0
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and not text.startswith("#"):
                mem.remember(text, tags=tags)
                count += 1
    print(f"已批量导入 {count} 条记忆" + (f"（标签: {', '.join(tags)}）" if tags else ""))


def cmd_visualize(args: argparse.Namespace) -> None:
    """文本可视化统计"""
    mem = _get_memory(args)
    s = mem.stats()

    print("=" * 50)
    print("  agentmemory 统计可视化")
    print("=" * 50)

    mc = s["memory_count"]
    bar_len = min(mc, 40)
    print(f"\n  记忆数: {mc}")
    print(f"  {'█' * bar_len}{'░' * (40 - bar_len)}")

    ec = s["entity_count"]
    rc = s["relation_count"]
    e_bar = min(ec, 40)
    r_bar = min(rc, 40)
    print(f"\n  实体数: {ec}")
    print(f"  {'█' * e_bar}{'░' * (40 - e_bar)}")
    print(f"\n  关系数: {rc}")
    print(f"  {'█' * r_bar}{'░' * (40 - r_bar)}")

    tags = mem.get_all_tags()
    if tags:
        print(f"\n  标签分布 (共 {len(tags)} 个):")
        max_count = max(tags.values())
        for tag, cnt in sorted(tags.items(), key=lambda x: -x[1])[:10]:
            bar_len = int(cnt / max_count * 30) if max_count > 0 else 0
            print(f"    {tag:15s} {'█' * bar_len} {cnt}")

    print(f"\n  配置:")
    print(f"    向量维度: {s['dimension']}")
    print(f"    LSH 索引: {'启用' if s['use_lsh'] else '禁用'}")
    print("=" * 50)


def cmd_config(args: argparse.Namespace) -> None:
    """查看/管理配置"""
    from agentmemory.config import AgentMemoryConfig, PROFILES, load_config

    if args.config_action == "show":
        if args.profile:
            cfg = load_config(profile=args.profile, env_override=False)
        else:
            cfg = AgentMemoryConfig()
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
    elif args.config_action == "profiles":
        for name in PROFILES:
            print(f"  {name}")
    elif args.config_action == "validate":
        if args.profile:
            cfg = load_config(profile=args.profile, env_override=True)
        else:
            cfg = AgentMemoryConfig()
        errors = cfg.validate()
        if errors:
            print(f"验证失败 ({len(errors)} 个错误):")
            for e in errors:
                print(f"  ✗ {e}")
            sys.exit(1)
        else:
            print("✓ 配置验证通过")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """运行性能基准测试"""
    from agentmemory.benchmarks import (
        benchmark_embedding_store,
        benchmark_knowledge_graph,
        benchmark_lsh_index,
        benchmark_hybrid_memory,
        run_all,
    )

    dim = args.dimension
    iters = args.iterations

    if args.bench_target == "all":
        suite = run_all(dimension=dim, iterations=iters)
    elif args.bench_target == "vector":
        suite = benchmark_embedding_store(dimension=dim, num_items=500, iterations=iters)
    elif args.bench_target == "graph":
        suite = benchmark_knowledge_graph(num_entities=200, num_relations=400, iterations=iters)
    elif args.bench_target == "lsh":
        suite = benchmark_lsh_index(dimension=dim, num_items=2000, iterations=iters)
    elif args.bench_target == "hybrid":
        suite = benchmark_hybrid_memory(dimension=dim, num_memories=100, iterations=iters)
    else:
        suite = run_all(dimension=dim, iterations=iters)

    if args.format == "json":
        print(json.dumps(suite.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(suite.summary())


def cmd_gc(args: argparse.Namespace) -> None:
    """运行垃圾回收"""
    from agentmemory.gc import GarbageCollector, GCPolicy
    from agentmemory.config import GCConfig

    mem = _get_memory(args)
    memories = mem.list_all()

    policy = GCPolicy(
        min_importance=args.min_importance,
        max_age=args.max_age,
        batch_size=args.batch_size,
    )

    gc = GarbageCollector(lifecycle=mem.lifecycle, policy=policy)

    if args.gc_action == "preview":
        result = gc.preview(memories)
        print(f"预览 GC 结果:")
        print(f"  将回收: {result.total_collected} 条记忆")
        print(f"  将保留: {result.total_retained} 条记忆")
        if result.reasons:
            print(f"  回收原因:")
            for mid, reason in list(result.reasons.items())[:20]:
                print(f"    {mid[:12]}...: {reason}")
    elif args.gc_action == "run":
        ids_to_forget = []
        result = gc.collect(memories)
        for mid in result.collected:
            try:
                mem.forget(mid)
                ids_to_forget.append(mid)
            except KeyError:
                pass
        print(f"GC 完成:")
        print(f"  已回收: {len(ids_to_forget)} 条记忆")
        print(f"  耗时: {result.elapsed_ms:.1f}ms")
    elif args.gc_action == "stats":
        stats = gc.stats(memories)
        print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    """启动 Web API 服务器。

    Args:
        args: 解析后的命令行参数
    """
    try:
        import uvicorn
    except ImportError:
        print("请安装 uvicorn: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    from agentmemory.api import create_app
    from agentmemory.embedding_provider import HashEmbeddingProvider

    mem = _get_memory(args)
    api_keys = args.api_keys if args.api_keys else None
    cors = args.cors if args.cors else None

    app = create_app(memory=mem, api_keys=api_keys, cors_origins=cors)
    print(f"启动 agentmemory API 服务...")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  文档: http://{args.host}:{args.port}/docs")
    if api_keys:
        print(f"  API Key 认证: 已启用")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


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

    # shortest-path
    p = subparsers.add_parser("shortest-path", help="查找两个实体之间的最短路径")
    p.add_argument("source_id", help="起始实体 ID")
    p.add_argument("target_id", help="目标实体 ID")
    p.add_argument("--max-depth", type=int, default=10, help="最大搜索深度")
    p.set_defaults(func=cmd_shortest_path)

    # common-neighbors
    p = subparsers.add_parser("common-neighbors", help="查找两个实体的共同邻居")
    p.add_argument("entity1", help="第一个实体 ID")
    p.add_argument("entity2", help="第二个实体 ID")
    p.set_defaults(func=cmd_common_neighbors)

    # connected-components
    p = subparsers.add_parser("connected-components", help="列出图谱的连通分量")
    p.set_defaults(func=cmd_connected_components)

    # graph-export
    p = subparsers.add_parser("graph-export", help="导出知识图谱可视化")
    p.add_argument("--format", choices=["dot", "html"], default="html", help="导出格式")
    p.add_argument("--output", "-o", help="输出文件路径")
    p.add_argument("--title", default="Knowledge Graph", help="图表标题")
    p.set_defaults(func=cmd_graph_export)

    # graph-stats
    p = subparsers.add_parser("graph-stats", help="知识图谱统计报告")
    p.set_defaults(func=cmd_graph_stats)

    # cache-stats
    p = subparsers.add_parser("cache-stats", help="搜索缓存统计")
    p.set_defaults(func=cmd_cache_stats)

    # interactive
    p = subparsers.add_parser("interactive", help="交互式 REPL 模式")
    p.set_defaults(func=cmd_interactive)

    # batch-import
    p = subparsers.add_parser("batch-import", help="从文本文件批量导入（每行一条）")
    p.add_argument("file", help="文本文件路径")
    p.add_argument("--tags", nargs="*", help="标签列表")
    p.set_defaults(func=cmd_batch_import)

    # visualize
    p = subparsers.add_parser("visualize", help="文本可视化统计")
    p.set_defaults(func=cmd_visualize)

    # rag
    p = subparsers.add_parser("rag", help="RAG 检索增强生成")
    p.add_argument("query", help="查询文本")
    p.add_argument("--top-k", type=int, default=5, help="检索数量 (默认: 5)")
    p.add_argument("--max-tokens", type=int, default=2000, help="上下文最大 token 数")
    p.add_argument("--tags", nargs="*", help="标签过滤")
    p.add_argument("--hybrid", action="store_true", help="使用混合检索")
    p.add_argument("--prompt-only", action="store_true", help="只输出 prompt")
    p.set_defaults(func=cmd_rag)

    # metrics
    p = subparsers.add_parser("metrics", help="运行时指标")
    p.add_argument("--format", choices=["text", "json", "prometheus"], default="text", help="输出格式")
    p.set_defaults(func=cmd_metrics)

    # health
    p = subparsers.add_parser("health", help="健康检查")
    p.set_defaults(func=cmd_health)

    # compress
    p = subparsers.add_parser("compress", help="压缩向量存储")
    p.add_argument("--method", choices=["sq8", "pq"], default="sq8", help="量化方法")
    p.add_argument("--subspaces", type=int, default=8, help="PQ 子空间数量")
    p.set_defaults(func=cmd_compress)

    # config
    p = subparsers.add_parser("config", help="配置管理")
    p.add_argument("config_action", choices=["show", "profiles", "validate"], help="操作类型")
    p.add_argument("--profile", "-p", help="Profile 名称 (dev/test/prod)")
    p.set_defaults(func=cmd_config)

    # benchmark
    p = subparsers.add_parser("benchmark", help="运行性能基准测试")
    p.add_argument("bench_target", nargs="?", default="all", choices=["all", "vector", "graph", "lsh", "hybrid"], help="测试目标")
    p.add_argument("--iterations", "-n", type=int, default=50, help="迭代次数")
    p.add_argument("--format", choices=["text", "json"], default="text", help="输出格式")
    p.set_defaults(func=cmd_benchmark)

    # gc
    p = subparsers.add_parser("gc", help="垃圾回收")
    p.add_argument("gc_action", choices=["preview", "run", "stats"], help="操作类型")
    p.add_argument("--min-importance", type=float, default=0.1, help="最低重要性阈值")
    p.add_argument("--max-age", type=float, default=None, help="最大存活时间（秒）")
    p.add_argument("--batch-size", type=int, default=100, help="每批清理数量")
    p.set_defaults(func=cmd_gc)

    # serve
    p = subparsers.add_parser("serve", help="启动 Web API 服务器")
    p.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    p.add_argument("--port", "-p", type=int, default=8000, help="监听端口 (默认: 8000)")
    p.add_argument("--api-keys", nargs="*", help="启用 API Key 认证")
    p.add_argument("--cors", nargs="*", help="CORS 允许的源")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="日志级别")
    p.set_defaults(func=cmd_serve)

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
