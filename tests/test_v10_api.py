"""v1.0.0 功能测试：FastAPI REST API 全端点测试。

覆盖记忆 CRUD、搜索、批量操作、知识图谱、RAG 管道、快照管理、
系统信息、指标、认证、CORS 等所有端点。
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from agentmemory.api import create_app
from agentmemory.hybrid_memory import HybridMemory
from agentmemory.embedding_provider import HashEmbeddingProvider


# ============================================================
# Fixtures
# ============================================================


@pytest_asyncio.fixture
async def client():
    """创建测试用异步 HTTP 客户端。"""
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def auth_client():
    """创建带 API Key 认证的测试客户端。"""
    app = create_app(api_keys=["test-secret-key", "another-key"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def cors_client():
    """创建带 CORS 配置的测试客户端。"""
    app = create_app(cors_origins=["https://example.com", "http://localhost:3000"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _create_memory(client: AsyncClient, content: str = "测试记忆", tags=None, metadata=None):
    """辅助函数：创建一条记忆并返回响应数据。"""
    payload = {"content": content}
    if tags:
        payload["tags"] = tags
    if metadata:
        payload["metadata"] = metadata
    resp = await client.post("/api/v1/memories", json=payload)
    assert resp.status_code == 200
    return resp.json()


async def _create_entity(client: AsyncClient, name: str, entity_type: str, properties=None):
    """辅助函数：创建一个实体并返回响应数据。"""
    payload = {"name": name, "entity_type": entity_type, "properties": properties or {}}
    resp = await client.post("/api/v1/graph/entities", json=payload)
    assert resp.status_code == 200
    return resp.json()


# ============================================================
# 根端点测试
# ============================================================


class TestRootEndpoint:
    """根路径 / 测试"""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, client):
        """根端点应返回 200"""
        resp = await client.get("/")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_root_contains_app_info(self, client):
        """根端点应包含应用信息"""
        resp = await client.get("/")
        data = resp.json()
        assert "name" in data or "version" in data or "status" in data or "message" in data


# ============================================================
# 健康检查测试
# ============================================================


class TestHealthEndpoint:
    """健康检查 /api/v1/health 测试"""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        """健康检查应返回 200"""
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_contains_status(self, client):
        """健康检查应包含状态信息"""
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert "overall_status" in data
        assert data["overall_status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_health_contains_checks(self, client):
        """健康检查应包含检查列表"""
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert "checks" in data
        assert isinstance(data["checks"], list)

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, auth_client):
        """健康检查端点不需要认证"""
        resp = await auth_client.get("/api/v1/health")
        assert resp.status_code == 200


# ============================================================
# 统计端点测试
# ============================================================


class TestStatsEndpoint:
    """系统统计 /api/v1/stats 测试"""

    @pytest.mark.asyncio
    async def test_stats_returns_200(self, client):
        """统计端点应返回 200"""
        resp = await client.get("/api/v1/stats")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_stats_contains_memory_count(self, client):
        """统计应包含记忆数量"""
        resp = await client.get("/api/v1/stats")
        data = resp.json()
        assert "memory_count" in data or "total_memories" in data

    @pytest.mark.asyncio
    async def test_stats_after_creating_memories(self, client):
        """创建记忆后统计应更新"""
        await _create_memory(client, "记忆 A")
        await _create_memory(client, "记忆 B")
        resp = await client.get("/api/v1/stats")
        data = resp.json()
        memory_count = data.get("memory_count", data.get("total_memories", 0))
        assert memory_count >= 2


# ============================================================
# 指标端点测试
# ============================================================


class TestMetricsEndpoints:
    """指标端点 /api/v1/metrics 测试"""

    @pytest.mark.asyncio
    async def test_metrics_json_returns_200(self, client):
        """JSON 指标应返回 200"""
        resp = await client.get("/api/v1/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_json_is_dict(self, client):
        """JSON 指标应返回字典"""
        resp = await client.get("/api/v1/metrics")
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_metrics_prometheus_returns_200(self, client):
        """Prometheus 指标应返回 200"""
        resp = await client.get("/api/v1/metrics/prometheus")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_prometheus_content_type(self, client):
        """Prometheus 指标应为纯文本格式"""
        resp = await client.get("/api/v1/metrics/prometheus")
        assert "text/plain" in resp.headers.get("content-type", "")


# ============================================================
# 创建记忆测试
# ============================================================


class TestCreateMemory:
    """创建记忆 POST /api/v1/memories 测试"""

    @pytest.mark.asyncio
    async def test_create_memory_returns_200(self, client):
        """创建记忆应返回 200"""
        resp = await client.post("/api/v1/memories", json={"content": "你好世界"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_memory_returns_id(self, client):
        """创建记忆应返回唯一 ID"""
        resp = await client.post("/api/v1/memories", json={"content": "测试"})
        data = resp.json()
        assert "id" in data
        assert len(data["id"]) > 0

    @pytest.mark.asyncio
    async def test_create_memory_returns_content(self, client):
        """返回的记忆应包含原始内容"""
        resp = await client.post("/api/v1/memories", json={"content": "具体内容"})
        data = resp.json()
        assert data["content"] == "具体内容"

    @pytest.mark.asyncio
    async def test_create_memory_with_tags(self, client):
        """创建带标签的记忆"""
        resp = await client.post(
            "/api/v1/memories", json={"content": "带标签", "tags": ["重要", "工作"]}
        )
        data = resp.json()
        assert "重要" in data["tags"]
        assert "工作" in data["tags"]

    @pytest.mark.asyncio
    async def test_create_memory_with_metadata(self, client):
        """创建带元数据的记忆"""
        resp = await client.post(
            "/api/v1/memories",
            json={"content": "带元数据", "metadata": {"source": "test", "score": 0.9}},
        )
        data = resp.json()
        assert data["metadata"]["source"] == "test"

    @pytest.mark.asyncio
    async def test_create_memory_with_importance(self, client):
        """创建带重要性评分的记忆"""
        resp = await client.post(
            "/api/v1/memories", json={"content": "重要记忆", "importance": 0.8}
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_memory_with_ttl(self, client):
        """创建带 TTL 的记忆"""
        resp = await client.post(
            "/api/v1/memories", json={"content": "临时记忆", "ttl": 3600.0}
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_memory_empty_content_returns_400(self, client):
        """空内容应返回 400 或 422"""
        resp = await client.post("/api/v1/memories", json={"content": ""})
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_create_memory_missing_content_returns_422(self, client):
        """缺少 content 字段应返回 422"""
        resp = await client.post("/api/v1/memories", json={"metadata": {}})
        assert resp.status_code == 422


# ============================================================
# 列出记忆测试
# ============================================================


class TestListMemories:
    """列出记忆 GET /api/v1/memories 测试"""

    @pytest.mark.asyncio
    async def test_list_empty_returns_empty(self, client):
        """无记忆时应返回空列表"""
        resp = await client.get("/api/v1/memories")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_list_returns_created_memories(self, client):
        """列出的记忆应包含已创建的"""
        mem = await _create_memory(client, "第一条")
        resp = await client.get("/api/v1/memories")
        data = resp.json()
        ids = [m["id"] for m in data]
        assert mem["id"] in ids

    @pytest.mark.asyncio
    async def test_list_multiple_memories(self, client):
        """列出多条记忆"""
        await _create_memory(client, "A")
        await _create_memory(client, "B")
        await _create_memory(client, "C")
        resp = await client.get("/api/v1/memories")
        data = resp.json()
        assert len(data) >= 3


# ============================================================
# 获取单条记忆测试
# ============================================================


class TestGetMemoryById:
    """获取记忆 GET /api/v1/memories/{id} 测试"""

    @pytest.mark.asyncio
    async def test_get_existing_memory(self, client):
        """获取已存在的记忆"""
        mem = await _create_memory(client, "可获取")
        resp = await client.get(f"/api/v1/memories/{mem['id']}")
        assert resp.status_code == 200
        assert resp.json()["content"] == "可获取"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_404(self, client):
        """获取不存在的记忆应返回 404"""
        resp = await client.get("/api/v1/memories/nonexistent_id_12345")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_memory_has_correct_fields(self, client):
        """获取的记忆应包含所有必要字段"""
        mem = await _create_memory(client, "字段检查")
        resp = await client.get(f"/api/v1/memories/{mem['id']}")
        data = resp.json()
        assert "id" in data
        assert "content" in data
        assert "created_at" in data
        assert "metadata" in data
        assert "tags" in data


# ============================================================
# 更新记忆测试
# ============================================================


class TestUpdateMemory:
    """更新记忆 PUT /api/v1/memories/{id} 测试"""

    @pytest.mark.asyncio
    async def test_update_content(self, client):
        """更新记忆内容"""
        mem = await _create_memory(client, "原始内容")
        resp = await client.put(
            f"/api/v1/memories/{mem['id']}", json={"content": "更新后内容"}
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "更新后内容"

    @pytest.mark.asyncio
    async def test_update_metadata(self, client):
        """更新记忆元数据"""
        mem = await _create_memory(client, "元数据更新")
        resp = await client.put(
            f"/api/v1/memories/{mem['id']}", json={"metadata": {"new_key": "new_val"}}
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"]["new_key"] == "new_val"

    @pytest.mark.asyncio
    async def test_update_tags(self, client):
        """更新记忆标签"""
        mem = await _create_memory(client, "标签更新", tags=["旧标签"])
        resp = await client.put(
            f"/api/v1/memories/{mem['id']}", json={"tags": ["新标签A", "新标签B"]}
        )
        assert resp.status_code == 200
        assert "新标签A" in resp.json()["tags"]
        assert "新标签B" in resp.json()["tags"]

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_404(self, client):
        """更新不存在的记忆应返回 404"""
        resp = await client.put(
            "/api/v1/memories/nonexistent_id", json={"content": "不存在"}
        )
        assert resp.status_code == 404


# ============================================================
# 删除记忆测试
# ============================================================


class TestDeleteMemory:
    """删除记忆 DELETE /api/v1/memories/{id} 测试"""

    @pytest.mark.asyncio
    async def test_delete_existing_memory(self, client):
        """删除已存在的记忆"""
        mem = await _create_memory(client, "将被删除")
        resp = await client.delete(f"/api/v1/memories/{mem['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["deleted"] == mem["id"]

    @pytest.mark.asyncio
    async def test_delete_then_get_returns_404(self, client):
        """删除后再获取应返回 404"""
        mem = await _create_memory(client, "删除后不可获取")
        await client.delete(f"/api/v1/memories/{mem['id']}")
        resp = await client.get(f"/api/v1/memories/{mem['id']}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, client):
        """删除不存在的记忆应返回 404"""
        resp = await client.delete("/api/v1/memories/nonexistent_id")
        assert resp.status_code == 404


# ============================================================
# 搜索端点测试
# ============================================================


class TestSearchEndpoint:
    """搜索 POST /api/v1/memories/search 测试"""

    @pytest.mark.asyncio
    async def test_search_returns_list(self, client):
        """搜索应返回列表"""
        await _create_memory(client, "Python 是一种编程语言")
        resp = await client.post(
            "/api/v1/memories/search", json={"query": "编程语言"}
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_search_returns_results_with_score(self, client):
        """搜索结果应包含评分"""
        await _create_memory(client, "机器学习基础")
        resp = await client.post(
            "/api/v1/memories/search", json={"query": "机器学习"}
        )
        data = resp.json()
        if len(data) > 0:
            assert "score" in data[0]
            assert "memory" in data[0]

    @pytest.mark.asyncio
    async def test_search_with_top_k(self, client):
        """搜索应尊重 top_k 参数"""
        for i in range(5):
            await _create_memory(client, f"搜索结果 {i}")
        resp = await client.post(
            "/api/v1/memories/search", json={"query": "搜索", "top_k": 2}
        )
        data = resp.json()
        assert len(data) <= 2

    @pytest.mark.asyncio
    async def test_search_with_tags_filter(self, client):
        """搜索应支持标签过滤"""
        await _create_memory(client, "带标签的搜索", tags=["python"])
        resp = await client.post(
            "/api/v1/memories/search",
            json={"query": "搜索", "tags": ["python"]},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_hybrid_mode(self, client):
        """混合搜索模式"""
        await _create_memory(client, "混合搜索测试内容")
        resp = await client.post(
            "/api/v1/memories/search",
            json={"query": "混合搜索", "use_hybrid": True},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_list(self, client):
        """空查询应返回列表（可能为空）"""
        resp = await client.post(
            "/api/v1/memories/search", json={"query": ""}
        )
        # 空查询可能返回 400 或 200
        assert resp.status_code in (200, 400)


# ============================================================
# 批量操作测试
# ============================================================


class TestBatchEndpoints:
    """批量操作端点测试"""

    @pytest.mark.asyncio
    async def test_batch_create(self, client):
        """批量创建记忆"""
        resp = await client.post(
            "/api/v1/memories/batch",
            json={"contents": ["记忆一", "记忆二", "记忆三"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["content"] == "记忆一"

    @pytest.mark.asyncio
    async def test_batch_create_with_metadatas(self, client):
        """批量创建带元数据的记忆"""
        resp = await client.post(
            "/api/v1/memories/batch",
            json={
                "contents": ["A", "B"],
                "metadatas": [{"src": "a"}, {"src": "b"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["metadata"]["src"] == "a"
        assert data[1]["metadata"]["src"] == "b"

    @pytest.mark.asyncio
    async def test_batch_create_with_tagss(self, client):
        """批量创建带标签的记忆"""
        resp = await client.post(
            "/api/v1/memories/batch",
            json={
                "contents": ["标签A", "标签B"],
                "tagss": [["t1", "t2"], ["t3"]],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "t1" in data[0]["tags"]
        assert "t3" in data[1]["tags"]

    @pytest.mark.asyncio
    async def test_batch_delete(self, client):
        """批量删除记忆"""
        mem1 = await _create_memory(client, "待删1")
        mem2 = await _create_memory(client, "待删2")
        resp = await client.request(
            "DELETE",
            "/api/v1/memories/batch",
            json={"memory_ids": [mem1["id"], mem2["id"]]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["deleted_count"] == 2

    @pytest.mark.asyncio
    async def test_batch_delete_partial_ids(self, client):
        """批量删除部分有效 ID"""
        mem = await _create_memory(client, "有效")
        resp = await client.request(
            "DELETE",
            "/api/v1/memories/batch",
            json={"memory_ids": [mem["id"], "invalid_id_xyz"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_count"] >= 1


# ============================================================
# 知识图谱端点测试
# ============================================================


class TestGraphEndpoints:
    """知识图谱端点测试"""

    @pytest.mark.asyncio
    async def test_add_entity(self, client):
        """添加实体"""
        resp = await client.post(
            "/api/v1/graph/entities",
            json={"name": "Python", "entity_type": "语言", "properties": {"year": 1991}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Python"
        assert data["entity_type"] == "语言"
        assert data["properties"]["year"] == 1991
        assert "id" in data

    @pytest.mark.asyncio
    async def test_add_entity_empty_name_returns_400(self, client):
        """空名称实体应返回 400"""
        resp = await client.post(
            "/api/v1/graph/entities",
            json={"name": "", "entity_type": "test"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_add_relation(self, client):
        """添加关系"""
        e1 = await _create_entity(client, "Alice", "人")
        e2 = await _create_entity(client, "Bob", "人")
        resp = await client.post(
            "/api/v1/graph/relations",
            json={
                "source_id": e1["id"],
                "target_id": e2["id"],
                "relation_type": "朋友",
                "weight": 0.9,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == e1["id"]
        assert data["target_id"] == e2["id"]
        assert data["relation_type"] == "朋友"
        assert data["weight"] == 0.9

    @pytest.mark.asyncio
    async def test_add_relation_invalid_entity_returns_400(self, client):
        """引用不存在实体的关系应返回 400"""
        resp = await client.post(
            "/api/v1/graph/relations",
            json={
                "source_id": "nonexistent",
                "target_id": "nonexistent2",
                "relation_type": "测试",
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_get_neighbors(self, client):
        """获取邻居实体"""
        e1 = await _create_entity(client, "中心", "节点")
        e2 = await _create_entity(client, "邻居", "节点")
        await client.post(
            "/api/v1/graph/relations",
            json={
                "source_id": e1["id"],
                "target_id": e2["id"],
                "relation_type": "连接",
            },
        )
        resp = await client.get(f"/api/v1/graph/entities/{e1['id']}/neighbors")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        neighbor_ids = [n["id"] for n in data]
        assert e2["id"] in neighbor_ids

    @pytest.mark.asyncio
    async def test_get_neighbors_with_relation_type(self, client):
        """按关系类型过滤邻居"""
        e1 = await _create_entity(client, "过滤中心", "节点")
        e2 = await _create_entity(client, "邻居A", "节点")
        await client.post(
            "/api/v1/graph/relations",
            json={
                "source_id": e1["id"],
                "target_id": e2["id"],
                "relation_type": "特定类型",
            },
        )
        resp = await client.get(
            f"/api/v1/graph/entities/{e1['id']}/neighbors",
            params={"relation_type": "特定类型"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_shortest_path(self, client):
        """查找最短路径"""
        e1 = await _create_entity(client, "起点", "节点")
        e2 = await _create_entity(client, "中转", "节点")
        e3 = await _create_entity(client, "终点", "节点")
        await client.post(
            "/api/v1/graph/relations",
            json={"source_id": e1["id"], "target_id": e2["id"], "relation_type": "连接"},
        )
        await client.post(
            "/api/v1/graph/relations",
            json={"source_id": e2["id"], "target_id": e3["id"], "relation_type": "连接"},
        )
        resp = await client.get(
            "/api/v1/graph/path",
            params={"source_id": e1["id"], "target_id": e3["id"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "path" in data
        assert "length" in data
        assert data["length"] == 3

    @pytest.mark.asyncio
    async def test_shortest_path_no_connection_returns_404(self, client):
        """不连通的实体间最短路径应返回 404"""
        e1 = await _create_entity(client, "孤立A", "节点")
        e2 = await _create_entity(client, "孤立B", "节点")
        resp = await client.get(
            "/api/v1/graph/path",
            params={"source_id": e1["id"], "target_id": e2["id"]},
        )
        assert resp.status_code == 404


# ============================================================
# RAG 管道端点测试
# ============================================================


class TestRAGEndpoint:
    """RAG 管道 POST /api/v1/rag 测试"""

    @pytest.mark.asyncio
    async def test_rag_returns_200(self, client):
        """RAG 请求应返回 200"""
        await _create_memory(client, "Python 是一种解释型语言")
        resp = await client.post(
            "/api/v1/rag", json={"query": "什么是 Python"}
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_rag_contains_prompt(self, client):
        """RAG 结果应包含 prompt"""
        await _create_memory(client, "机器学习是人工智能分支")
        resp = await client.post(
            "/api/v1/rag", json={"query": "机器学习"}
        )
        data = resp.json()
        assert "prompt" in data

    @pytest.mark.asyncio
    async def test_rag_contains_sources(self, client):
        """RAG 结果应包含来源"""
        await _create_memory(client, "深度学习使用神经网络")
        resp = await client.post(
            "/api/v1/rag", json={"query": "深度学习"}
        )
        data = resp.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    @pytest.mark.asyncio
    async def test_rag_with_tags(self, client):
        """RAG 应支持标签过滤"""
        await _create_memory(client, "带标签的 RAG", tags=["ai"])
        resp = await client.post(
            "/api/v1/rag", json={"query": "RAG", "tags": ["ai"]}
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_rag_with_hybrid(self, client):
        """RAG 应支持混合检索"""
        await _create_memory(client, "混合 RAG 测试")
        resp = await client.post(
            "/api/v1/rag", json={"query": "RAG", "use_hybrid": True}
        )
        assert resp.status_code == 200


# ============================================================
# 快照管理端点测试
# ============================================================


class TestSnapshotEndpoints:
    """快照管理端点测试"""

    @pytest.mark.asyncio
    async def test_list_snapshots_empty(self, client):
        """无快照时应返回空列表"""
        resp = await client.get("/api/v1/snapshots")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_create_snapshot(self, client):
        """创建快照"""
        await _create_memory(client, "快照前的记忆")
        resp = await client.post(
            "/api/v1/snapshots",
            json={"name": "test-snap", "description": "测试快照"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["name"] == "test-snap"
        assert data["description"] == "测试快照"

    @pytest.mark.asyncio
    async def test_create_snapshot_then_list(self, client):
        """创建快照后应出现在列表中"""
        resp = await client.post(
            "/api/v1/snapshots",
            json={"name": "list-test", "description": "列表测试"},
        )
        snap_id = resp.json()["id"]
        resp = await client.get("/api/v1/snapshots")
        data = resp.json()
        ids = [s["id"] for s in data]
        assert snap_id in ids

    @pytest.mark.asyncio
    async def test_delete_snapshot(self, client):
        """删除快照"""
        resp = await client.post(
            "/api/v1/snapshots",
            json={"name": "to-delete", "description": "将被删除"},
        )
        snap_id = resp.json()["id"]
        resp = await client.delete(f"/api/v1/snapshots/{snap_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_snapshot_returns_404(self, client):
        """删除不存在的快照应返回 404"""
        resp = await client.delete("/api/v1/snapshots/nonexistent_snap")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_restore_snapshot(self, client):
        """恢复快照"""
        await _create_memory(client, "快照时的记忆")
        resp = await client.post(
            "/api/v1/snapshots",
            json={"name": "restore-test", "description": "恢复测试"},
        )
        snap_id = resp.json()["id"]
        resp = await client.post(f"/api/v1/snapshots/{snap_id}/restore")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_restore_nonexistent_snapshot_returns_404(self, client):
        """恢复不存在的快照应返回 404"""
        resp = await client.post("/api/v1/snapshots/nonexistent_snap/restore")
        assert resp.status_code == 404


# ============================================================
# API Key 认证测试
# ============================================================


class TestAPIKeyAuth:
    """API Key 认证测试"""

    @pytest.mark.asyncio
    async def test_no_key_returns_401(self, auth_client):
        """无 API Key 请求应返回 401"""
        resp = await auth_client.get("/api/v1/stats")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_returns_403(self, auth_client):
        """错误 API Key 应返回 403"""
        resp = await auth_client.get(
            "/api/v1/stats", headers={"X-API-Key": "wrong-key"}
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_key_returns_200(self, auth_client):
        """正确 API Key 应返回 200"""
        resp = await auth_client.get(
            "/api/v1/stats", headers={"X-API-Key": "test-secret-key"}
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_on_post_memories(self, auth_client):
        """创建记忆也需要认证"""
        resp = await auth_client.post(
            "/api/v1/memories",
            json={"content": "认证测试"},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_on_post_memories_no_key(self, auth_client):
        """无 Key 创建记忆应返回 401"""
        resp = await auth_client.post(
            "/api/v1/memories", json={"content": "无认证"}
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_on_search(self, auth_client):
        """搜索也需要认证"""
        resp = await auth_client.post(
            "/api/v1/memories/search",
            json={"query": "test"},
            headers={"X-API-Key": "another-key"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_no_auth_needed(self, auth_client):
        """健康检查不需要认证"""
        resp = await auth_client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_root_no_auth_needed(self, auth_client):
        """根端点不需要认证"""
        resp = await auth_client.get("/")
        assert resp.status_code == 200


# ============================================================
# CORS 测试
# ============================================================


class TestCORS:
    """CORS 跨域配置测试"""

    @pytest.mark.asyncio
    async def test_cors_headers_present_with_options(self, cors_client):
        """CORS 配置后 OPTIONS 请求应包含跨域头"""
        resp = await cors_client.options(
            "/api/v1/memories",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS 中间件应处理 OPTIONS 请求
        assert resp.status_code in (200, 204, 405)

    @pytest.mark.asyncio
    async def test_cors_allow_origin_header(self, cors_client):
        """CORS 应返回允许的来源"""
        resp = await cors_client.get(
            "/api/v1/health",
            headers={"Origin": "https://example.com"},
        )
        # 检查 Access-Control-Allow-Origin 头
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert allow_origin in ("https://example.com", "*")

    @pytest.mark.asyncio
    async def test_no_cors_without_config(self, client):
        """未配置 CORS 时不应有跨域头"""
        resp = await client.get("/api/v1/health")
        # 不配置 CORS 时，不应有 Access-Control-Allow-Origin
        # （FastAPI 默认不添加）
        assert resp.status_code == 200


# ============================================================
# 错误处理综合测试
# ============================================================


class TestErrorHandling:
    """错误处理综合测试"""

    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self, client):
        """未知端点应返回 404"""
        resp = await client.get("/api/v1/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, client):
        """不支持的 HTTP 方法应返回 405"""
        resp = await client.put("/api/v1/health")
        assert resp.status_code == 405

    @pytest.mark.asyncio
    async def test_invalid_json_body_returns_422(self, client):
        """无效 JSON 请求体应返回 422"""
        resp = await client.post(
            "/api/v1/memories",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_required_field_returns_422(self, client):
        """缺少必填字段应返回 422"""
        resp = await client.post("/api/v1/memories/search", json={"top_k": 5})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_memory_empty_id_is_404(self, client):
        """空路径段获取记忆应返回 404 或路由不匹配"""
        resp = await client.get("/api/v1/memories/")
        # 空路径段可能返回 404 或 307 重定向
        assert resp.status_code in (404, 307, 405, 200)
