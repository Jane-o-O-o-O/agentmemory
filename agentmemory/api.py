"""FastAPI REST API：为 agentmemory 混合记忆框架提供 HTTP 接口。

提供记忆 CRUD、文本搜索、知识图谱操作、RAG 管道、快照管理等端点。
支持 API Key 认证和 CORS 跨域。
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# Pydantic 请求/响应模型
# ============================================================


class RememberRequest(BaseModel):
    """创建记忆的请求体。"""

    content: str = Field(..., description="文本内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    importance: Optional[float] = Field(None, description="重要性评分（0~1）")
    ttl: Optional[float] = Field(None, description="自定义 TTL（秒）")


class UpdateMemoryRequest(BaseModel):
    """更新记忆的请求体。"""

    content: Optional[str] = Field(None, description="新内容")
    metadata: Optional[dict[str, Any]] = Field(None, description="新元数据（合并）")
    tags: Optional[list[str]] = Field(None, description="新标签列表（替换）")


class SearchRequest(BaseModel):
    """搜索请求体。"""

    query: str = Field(..., description="查询文本")
    top_k: int = Field(5, ge=1, le=1000, description="返回结果数量")
    tags: Optional[list[str]] = Field(None, description="标签过滤")
    use_hybrid: bool = Field(False, description="是否使用混合搜索（向量+图谱）")


class BatchRememberRequest(BaseModel):
    """批量创建记忆的请求体。"""

    contents: list[str] = Field(..., description="文本内容列表")
    metadatas: Optional[list[dict[str, Any]]] = Field(None, description="元数据列表")
    tagss: Optional[list[list[str]]] = Field(None, description="标签列表的列表")


class BatchDeleteRequest(BaseModel):
    """批量删除记忆的请求体。"""

    memory_ids: list[str] = Field(..., description="记忆 ID 列表")


class AddEntityRequest(BaseModel):
    """添加实体的请求体。"""

    name: str = Field(..., description="实体名称")
    entity_type: str = Field(..., description="实体类型")
    properties: dict[str, Any] = Field(default_factory=dict, description="附加属性")


class AddRelationRequest(BaseModel):
    """添加关系的请求体。"""

    source_id: str = Field(..., description="源实体 ID")
    target_id: str = Field(..., description="目标实体 ID")
    relation_type: str = Field(..., description="关系类型")
    weight: float = Field(1.0, description="关系权重")


class RAGRequest(BaseModel):
    """RAG 管道请求体。"""

    query: str = Field(..., description="用户查询")
    top_k: int = Field(5, ge=1, description="检索结果数量")
    max_context_tokens: int = Field(2000, ge=1, description="上下文最大 token 数")
    tags: Optional[list[str]] = Field(None, description="标签过滤")
    use_hybrid: bool = Field(False, description="是否使用混合检索")


class SnapshotRequest(BaseModel):
    """创建快照的请求体。"""

    name: Optional[str] = Field(None, description="快照名称")
    description: str = Field("", description="快照描述")


class MemoryResponse(BaseModel):
    """记忆响应模型。"""

    id: str
    content: str
    created_at: float
    metadata: dict[str, Any]
    tags: list[str]


class SearchResultItem(BaseModel):
    """搜索结果项。"""

    memory: MemoryResponse
    score: float
    context: list[MemoryResponse] = Field(default_factory=list)


class EntityResponse(BaseModel):
    """实体响应模型。"""

    id: str
    name: str
    entity_type: str
    properties: dict[str, Any]


class RelationResponse(BaseModel):
    """关系响应模型。"""

    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float


# ============================================================
# 内部辅助函数
# ============================================================


def _memory_to_response(mem: Any) -> dict[str, Any]:
    """将 Memory 对象转换为响应字典。

    Args:
        mem: Memory 数据类实例

    Returns:
        符合 MemoryResponse 的字典
    """
    return {
        "id": mem.id,
        "content": mem.content,
        "created_at": mem.created_at,
        "metadata": mem.metadata,
        "tags": mem.tags,
    }


def _search_result_to_response(result: Any) -> dict[str, Any]:
    """将 SearchResult 转换为响应字典。

    Args:
        result: SearchResult 数据类实例

    Returns:
        符合 SearchResultItem 的字典
    """
    return {
        "memory": _memory_to_response(result.memory),
        "score": result.score,
        "context": [_memory_to_response(m) for m in result.context],
    }


def _entity_to_response(entity: Any) -> dict[str, Any]:
    """将 Entity 对象转换为响应字典。

    Args:
        entity: Entity 数据类实例

    Returns:
        符合 EntityResponse 的字典
    """
    return {
        "id": entity.id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "properties": entity.properties,
    }


def _relation_to_response(relation: Any) -> dict[str, Any]:
    """将 Relation 对象转换为响应字典。

    Args:
        relation: Relation 数据类实例

    Returns:
        符合 RelationResponse 的字典
    """
    return {
        "id": relation.id,
        "source_id": relation.source_id,
        "target_id": relation.target_id,
        "relation_type": relation.relation_type,
        "weight": relation.weight,
    }


# ============================================================
# 应用工厂
# ============================================================


def create_app(
    memory: Optional[Any] = None,
    api_keys: Optional[list[str]] = None,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """创建 FastAPI 应用实例。

    如果未提供 memory 实例，将使用 HashEmbeddingProvider 创建默认的 HybridMemory。

    Args:
        memory: HybridMemory 实例（可选）
        api_keys: 允许的 API Key 列表（可选，启用 X-API-Key 认证）
        cors_origins: 允许的 CORS 来源列表（可选）

    Returns:
        配置好的 FastAPI 应用
    """
    # 延迟导入以避免循环依赖
    from agentmemory.hybrid_memory import HybridMemory
    from agentmemory.embedding_provider import HashEmbeddingProvider

    if memory is None:
        memory = HybridMemory(embedding_provider=HashEmbeddingProvider())

    app = FastAPI(
        title="AgentMemory API",
        description="混合记忆框架 REST API：向量搜索 + 知识图谱",
        version="1.0.0",
    )

    # --- 根端点 ---

    @app.get("/", summary="API 信息")
    async def root() -> dict[str, str]:
        """返回 API 基本信息。"""
        return {
            "name": "agentmemory",
            "version": "1.0.0",
            "description": "混合记忆框架 REST API",
            "docs": "/docs",
        }

    # --- 中间件 ---

    if cors_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # --- 依赖注入 ---

    def get_memory() -> HybridMemory:
        """获取 memory 实例的依赖注入函数。"""
        return memory

    # --- API Key 认证 ---

    async def verify_api_key(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ) -> None:
        """验证 API Key 中间件。

        Args:
            x_api_key: 请求头中的 API Key

        Raises:
            HTTPException: API Key 无效或缺失
        """
        if api_keys is None:
            return
        if x_api_key is None:
            raise HTTPException(status_code=401, detail="缺少 X-API-Key 请求头")
        if x_api_key not in api_keys:
            raise HTTPException(status_code=403, detail="API Key 无效")

    auth_dep = Depends(verify_api_key) if api_keys else None

    # 构建依赖列表的辅助
    def _deps() -> list:
        """返回带认证的依赖列表。"""
        if api_keys:
            return [Depends(verify_api_key)]
        return []

    # ============================================================
    # 记忆管理端点
    # ============================================================

    @app.post("/api/v1/memories", summary="创建记忆", dependencies=_deps())
    async def remember_memory(
        req: RememberRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """添加一条新记忆到存储。

        如果配置了 embedding_provider，将自动计算向量。
        """
        try:
            result = mem.remember(
                content=req.content,
                metadata=req.metadata,
                tags=req.tags,
                importance=req.importance,
                ttl=req.ttl,
            )
            return _memory_to_response(result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/v1/memories", summary="列出所有记忆", dependencies=_deps())
    async def list_memories(
        mem: HybridMemory = Depends(get_memory),
    ) -> list[dict[str, Any]]:
        """返回所有已存储的记忆列表。"""
        return [_memory_to_response(m) for m in mem.list_all()]

    @app.post("/api/v1/memories/search", summary="文本搜索", dependencies=_deps())
    async def search_memories(
        req: SearchRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> list[dict[str, Any]]:
        """文本相似度搜索，支持混合搜索模式。

        use_hybrid=True 时结合向量搜索和知识图谱上下文。
        """
        try:
            if req.use_hybrid:
                results = mem.hybrid_search_text(
                    query=req.query,
                    top_k=req.top_k,
                    tags=req.tags,
                )
            else:
                results = mem.search_text(
                    query=req.query,
                    top_k=req.top_k,
                    tags=req.tags,
                )
            return [_search_result_to_response(r) for r in results]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post(
        "/api/v1/memories/batch", summary="批量创建记忆", dependencies=_deps()
    )
    async def batch_remember(
        req: BatchRememberRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> list[dict[str, Any]]:
        """批量添加多条记忆。"""
        try:
            results = mem.batch_remember(
                contents=req.contents,
                metadatas=req.metadatas,
                tagss=req.tagss,
            )
            return [_memory_to_response(m) for m in results]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete(
        "/api/v1/memories/batch", summary="批量删除记忆", dependencies=_deps()
    )
    async def batch_delete(
        req: BatchDeleteRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """批量删除多条记忆。"""
        deleted = mem.batch_forget(req.memory_ids)
        return {"status": "ok", "deleted_count": len(deleted), "deleted_ids": deleted}

    @app.get("/api/v1/memories/{memory_id}", summary="获取记忆", dependencies=_deps())
    async def get_memory_by_id(
        memory_id: str,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """根据 ID 获取单条记忆。"""
        result = mem.get_memory(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"记忆 {memory_id} 不存在")
        return _memory_to_response(result)

    @app.put("/api/v1/memories/{memory_id}", summary="更新记忆", dependencies=_deps())
    async def update_memory(
        memory_id: str,
        req: UpdateMemoryRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """更新已有记忆的内容、元数据或标签。"""
        try:
            result = mem.update_memory(
                memory_id=memory_id,
                content=req.content,
                metadata=req.metadata,
                tags=req.tags,
            )
            return _memory_to_response(result)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete(
        "/api/v1/memories/{memory_id}", summary="删除记忆", dependencies=_deps()
    )
    async def forget_memory(
        memory_id: str,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, str]:
        """删除指定 ID 的记忆。"""
        try:
            mem.forget(memory_id)
            return {"status": "ok", "deleted": memory_id}
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ============================================================
    # 知识图谱端点
    # ============================================================

    @app.post(
        "/api/v1/graph/entities", summary="添加实体", dependencies=_deps()
    )
    async def add_entity(
        req: AddEntityRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """添加实体到知识图谱。"""
        try:
            entity = mem.add_entity(
                name=req.name,
                entity_type=req.entity_type,
                properties=req.properties,
            )
            return _entity_to_response(entity)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post(
        "/api/v1/graph/relations", summary="添加关系", dependencies=_deps()
    )
    async def add_relation(
        req: AddRelationRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """添加关系到知识图谱。"""
        try:
            relation = mem.add_relation(
                source_id=req.source_id,
                target_id=req.target_id,
                relation_type=req.relation_type,
                weight=req.weight,
            )
            return _relation_to_response(relation)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get(
        "/api/v1/graph/entities/{entity_id}/neighbors",
        summary="获取邻居实体",
        dependencies=_deps(),
    )
    async def get_neighbors(
        entity_id: str,
        relation_type: Optional[str] = Query(None, description="按关系类型过滤"),
        mem: HybridMemory = Depends(get_memory),
    ) -> list[dict[str, Any]]:
        """获取实体的邻居节点列表。"""
        neighbors = mem.get_neighbors(entity_id, relation_type=relation_type)
        return [_entity_to_response(e) for e in neighbors]

    @app.get(
        "/api/v1/graph/path", summary="最短路径", dependencies=_deps()
    )
    async def shortest_path(
        source_id: str = Query(..., description="起始实体 ID"),
        target_id: str = Query(..., description="目标实体 ID"),
        max_depth: int = Query(10, ge=1, description="最大搜索深度"),
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """查找两个实体之间的最短路径。"""
        path = mem.shortest_path(source_id, target_id, max_depth=max_depth)
        if path is None:
            raise HTTPException(
                status_code=404,
                detail=f"实体 {source_id} 和 {target_id} 之间不可达",
            )
        return {
            "path": [_entity_to_response(e) for e in path],
            "length": len(path),
        }

    # ============================================================
    # 系统信息端点
    # ============================================================

    @app.get("/api/v1/stats", summary="系统统计", dependencies=_deps())
    async def get_stats(
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """返回系统统计信息，包括记忆数、实体数、关系数等。"""
        return mem.stats()

    @app.get("/api/v1/health", summary="健康检查")
    async def health_check(
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """执行综合健康检查，检查存储和索引状态。"""
        return mem.health_check()

    @app.get("/api/v1/metrics", summary="指标（JSON）", dependencies=_deps())
    async def get_metrics_json(
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """返回运行时指标的 JSON 格式。"""
        return json.loads(mem.metrics_json())

    @app.get(
        "/api/v1/metrics/prometheus",
        summary="指标（Prometheus）",
        dependencies=_deps(),
    )
    async def get_metrics_prometheus(
        mem: HybridMemory = Depends(get_memory),
    ) -> Any:
        """返回运行时指标的 Prometheus 文本格式。"""
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(mem.metrics_prometheus())

    # ============================================================
    # RAG 管道端点
    # ============================================================

    @app.post("/api/v1/rag", summary="RAG 管道", dependencies=_deps())
    async def run_rag(
        req: RAGRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """执行 RAG（检索增强生成）管道。

        检索相关记忆 → 重排序 → 上下文组装 → Prompt 生成。
        """
        try:
            return mem.rag(
                query=req.query,
                top_k=req.top_k,
                max_context_tokens=req.max_context_tokens,
                tags=req.tags,
                use_hybrid=req.use_hybrid,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ============================================================
    # 快照管理端点
    # ============================================================

    @app.get("/api/v1/snapshots", summary="列出快照", dependencies=_deps())
    async def list_snapshots(
        mem: HybridMemory = Depends(get_memory),
    ) -> list[dict[str, Any]]:
        """列出所有已创建的快照。"""
        snapshots = mem.list_snapshots()
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "created_at": s.created_at,
                "memory_count": getattr(s, "memory_count", None),
            }
            for s in snapshots
        ]

    @app.post("/api/v1/snapshots", summary="创建快照", dependencies=_deps())
    async def create_snapshot(
        req: SnapshotRequest,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """创建当前记忆状态的快照。"""
        try:
            snapshot = mem.create_snapshot(name=req.name, description=req.description)
            return {
                "id": snapshot.id,
                "name": snapshot.name,
                "description": snapshot.description,
                "created_at": snapshot.created_at,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/api/v1/snapshots/{snapshot_id}",
        summary="删除快照",
        dependencies=_deps(),
    )
    async def delete_snapshot(
        snapshot_id: str,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, str]:
        """删除指定快照。"""
        try:
            success = mem.delete_snapshot(snapshot_id)
            if not success:
                raise HTTPException(
                    status_code=404, detail=f"快照 {snapshot_id} 不存在"
                )
            return {"status": "ok", "deleted": snapshot_id}
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post(
        "/api/v1/snapshots/{snapshot_id}/restore",
        summary="恢复快照",
        dependencies=_deps(),
    )
    async def restore_snapshot(
        snapshot_id: str,
        mem: HybridMemory = Depends(get_memory),
    ) -> dict[str, Any]:
        """恢复到指定快照的状态。"""
        try:
            snapshot = mem.restore_snapshot(snapshot_id)
            return {
                "status": "ok",
                "restored_snapshot": {
                    "id": snapshot.id,
                    "name": snapshot.name,
                    "description": snapshot.description,
                    "created_at": snapshot.created_at,
                },
            }
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return app
