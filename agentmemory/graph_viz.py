"""知识图谱可视化 — 导出 DOT 和 HTML 格式。

提供将 KnowledgeGraph 导出为 Graphviz DOT 和交互式 HTML 可视化的能力。
"""

from __future__ import annotations

import html
from typing import Optional

from agentmemory.knowledge_graph import KnowledgeGraph
from agentmemory.models import Entity, Relation


# 默认节点颜色映射
DEFAULT_TYPE_COLORS: dict[str, str] = {
    "person": "#4A90D9",
    "organization": "#E85D75",
    "location": "#50C878",
    "concept": "#FFB347",
    "language": "#9B59B6",
    "technology": "#1ABC9C",
    "event": "#E74C3C",
    "document": "#3498DB",
}

DEFAULT_FALLBACK_COLOR = "#95A5A6"


def _sanitize_dot_id(s: str) -> str:
    """将字符串转为 DOT 安全的标识符。"""
    return s.replace('"', '\\"').replace('\n', '\\n')


def export_dot(
    kg: KnowledgeGraph,
    title: str = "Knowledge Graph",
    show_properties: bool = True,
    type_colors: Optional[dict[str, str]] = None,
    rankdir: str = "TB",
) -> str:
    """导出知识图为 Graphviz DOT 格式。

    Args:
        kg: 知识图谱实例
        title: 图表标题
        show_properties: 是否在节点标签中显示属性
        type_colors: 实体类型到颜色的映射（覆盖默认）
        rankdir: 布局方向 "TB"(上到下) / "LR"(左到右)

    Returns:
        DOT 格式字符串

    Example:
        >>> dot = export_dot(kg)
        >>> with open("graph.dot", "w") as f:
        ...     f.write(dot)
    """
    colors = {**DEFAULT_TYPE_COLORS, **(type_colors or {})}
    lines: list[str] = []
    lines.append(f'digraph "{_sanitize_dot_id(title)}" {{')
    lines.append(f"    rankdir={rankdir};")
    lines.append('    node [shape=box, style="rounded,filled", fontname="Arial"];')
    lines.append('    edge [fontname="Arial", fontsize=10];')
    lines.append("")

    # 节点
    entities = kg.find_entities()
    for entity in entities:
        color = colors.get(entity.entity_type.lower(), DEFAULT_FALLBACK_COLOR)
        label_parts = [f"{entity.name}\\n({entity.entity_type})"]
        if show_properties and entity.properties:
            for k, v in list(entity.properties.items())[:3]:
                label_parts.append(f"{k}: {v}")
        label = "\\n".join(label_parts)
        eid = _sanitize_dot_id(entity.id)
        lines.append(f'    "{eid}" [label="{_sanitize_dot_id(label)}", fillcolor="{color}"];')

    lines.append("")

    # 边
    relations = kg.find_relations()
    for rel in relations:
        src = _sanitize_dot_id(rel.source_id)
        tgt = _sanitize_dot_id(rel.target_id)
        rtype = _sanitize_dot_id(rel.relation_type)
        weight_label = f" [{rel.weight:.1f}]" if rel.weight != 1.0 else ""
        lines.append(f'    "{src}" -> "{tgt}" [label="{rtype}{weight_label}"];')

    lines.append("}")
    return "\n".join(lines)


def export_html(
    kg: KnowledgeGraph,
    title: str = "Knowledge Graph",
    type_colors: Optional[dict[str, str]] = None,
    width: str = "100%",
    height: str = "600px",
) -> str:
    """导出知识图为交互式 HTML（基于 vis.js）。

    生成自包含的 HTML 文件，使用 vis.js Network 实现交互式图谱浏览。

    Args:
        kg: 知识图谱实例
        title: 页面标题
        type_colors: 实体类型到颜色的映射
        width: 画布宽度
        height: 画布高度

    Returns:
        完整的 HTML 字符串

    Example:
        >>> html = export_html(kg)
        >>> with open("graph.html", "w") as f:
        ...     f.write(html)
    """
    colors = {**DEFAULT_TYPE_COLORS, **(type_colors or {})}

    # 构建 nodes JSON
    entities = kg.find_entities()
    nodes: list[str] = []
    for entity in entities:
        color = colors.get(entity.entity_type.lower(), DEFAULT_FALLBACK_COLOR)
        tooltip_parts = [f"Type: {entity.entity_type}"]
        if entity.properties:
            for k, v in entity.properties.items():
                tooltip_parts.append(f"{k}: {v}")
        tooltip = html.escape("\\n".join(tooltip_parts))
        label = html.escape(f"{entity.name}\\n({entity.entity_type})")
        nodes.append(
            f'{{id: "{entity.id}", label: "{label}", '
            f'color: "{color}", title: "{tooltip}"}}'
        )

    # 构建 edges JSON
    relations = kg.find_relations()
    edges: list[str] = []
    for rel in relations:
        label = html.escape(rel.relation_type)
        edges.append(
            f'{{from: "{rel.source_id}", to: "{rel.target_id}", '
            f'label: "{label}", arrows: "to"}}'
        )

    nodes_js = ",\n            ".join(nodes)
    edges_js = ",\n            ".join(edges)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{html.escape(title)}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #header {{ padding: 10px 20px; background: #2c3e50; color: white; }}
        #graph {{ width: {width}; height: {height}; border: 1px solid #ddd; }}
        #info {{ padding: 10px 20px; font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <div id="header"><h2>{html.escape(title)}</h2></div>
    <div id="graph"></div>
    <div id="info">
        Entities: {len(entities)} | Relations: {len(relations)}
        — Drag nodes, scroll to zoom, click for details
    </div>
    <script>
        var nodes = new vis.DataSet([
            {nodes_js}
        ]);
        var edges = new vis.DataSet([
            {edges_js}
        ]);
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            physics: {{ solver: 'forceAtlas2Based', stabilization: {{ iterations: 100 }} }},
            interaction: {{ hover: true, tooltipDelay: 200 }},
            nodes: {{ font: {{ size: 12 }} }},
            edges: {{ font: {{ size: 10, align: 'middle' }}, smooth: {{ type: 'curvedCW', roundness: 0.2 }} }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""


def graph_stats_text(kg: KnowledgeGraph) -> str:
    """生成知识图谱的文本统计报告。

    Args:
        kg: 知识图谱实例

    Returns:
        格式化的统计文本
    """
    entities = kg.find_entities()
    relations = kg.find_relations()

    # 实体类型分布
    type_counts: dict[str, int] = {}
    for e in entities:
        type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

    # 关系类型分布
    rel_type_counts: dict[str, int] = {}
    for r in relations:
        rel_type_counts[r.relation_type] = rel_type_counts.get(r.relation_type, 0) + 1

    # 连通分量
    components = kg.connected_components()

    lines = [
        "=" * 50,
        "  Knowledge Graph Statistics",
        "=" * 50,
        f"  Entities: {len(entities)}",
        f"  Relations: {len(relations)}",
        f"  Connected Components: {len(components)}",
        "",
        "  Entity Types:",
    ]
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(c, 30)
        lines.append(f"    {t:20s} {bar} {c}")

    lines.append("")
    lines.append("  Relation Types:")
    for t, c in sorted(rel_type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(c, 30)
        lines.append(f"    {t:20s} {bar} {c}")

    # 度数统计
    if entities:
        degrees: list[int] = []
        for e in entities:
            neighbors = kg.get_neighbors(e.id)
            degrees.append(len(neighbors))
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        lines.append("")
        lines.append(f"  Degree Statistics:")
        lines.append(f"    Average: {avg_degree:.1f}")
        lines.append(f"    Max: {max_degree}")

    lines.append("=" * 50)
    return "\n".join(lines)
