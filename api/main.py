"""
api/main.py
-----------
FastAPI入口：定义REST API接口，管理Agent生命周期。
支持：普通响应 / 流式SSE响应 / 健康检查 / 监控指标
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.factory import AgentFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# 全局Agent注册表（服务启动时预热，避免首请求冷启动）
# ─────────────────────────────────────────────────────
_agents: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/关闭生命周期管理。"""
    logger.info("Initializing agents for all use cases...")
    try:
        _agents.update(AgentFactory.create_all())
        logger.info(f"Agents ready: {list(_agents.keys())}")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        # 生产环境：启动失败应该告警，此处不阻断服务启动

    yield  # 服务运行中

    # 关闭时清理资源
    from agent.factory import get_shared_cache
    try:
        await get_shared_cache().close()
    except Exception:
        pass
    logger.info("Service shutdown complete.")


# ─────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG Agent API",
    description="生产级多场景LLM Agent，支持kb_qa / helpdesk / compliance",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# Request / Response 模型
# ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询")
    use_case: str = Field(..., description="业务场景: kb_qa | helpdesk | compliance")
    tenant_id: str = Field(..., description="企业租户ID，用于数据隔离")
    stream: bool = Field(False, description="是否流式输出")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "员工年假政策是什么？",
                "use_case": "kb_qa",
                "tenant_id": "acme_corp",
                "stream": False,
            }
        }


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: int
    use_case: str
    from_cache: bool


class HealthResponse(BaseModel):
    status: str
    agents_ready: list[str]
    uptime_seconds: float


# ─────────────────────────────────────────────────────
# 服务启动时间（用于uptime计算）
# ─────────────────────────────────────────────────────
_start_time = time.time()


# ─────────────────────────────────────────────────────
# API 接口
# ─────────────────────────────────────────────────────

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主聊天接口（非流式）。
    - 自动路由到对应use_case的Agent
    - 集成缓存、检索、生成全流程
    - 目标P50延迟 ≤ 2000ms
    """
    if req.use_case not in _agents:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的use_case: {req.use_case}。"
                   f"可用场景: {list(_agents.keys())}"
        )

    agent = _agents[req.use_case]
    try:
        result = await agent.run(
            query=req.query,
            tenant_id=req.tenant_id,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Agent error | use_case={req.use_case} | error={e}")
        raise HTTPException(status_code=500, detail=f"Agent执行失败: {str(e)}")


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式聊天接口（Server-Sent Events）。
    首token延迟目标 < 500ms，适合实时交互场景（如帮助台）。

    前端消费示例：
        const es = new EventSource('/v1/chat/stream');
        es.onmessage = (e) => appendToUI(e.data);
    """
    if req.use_case not in _agents:
        raise HTTPException(status_code=400, detail=f"不支持的use_case: {req.use_case}")

    agent = _agents[req.use_case]

    async def event_generator():
        try:
            async for token in agent.run_stream(req.query, req.tenant_id):
                # SSE格式：data: <token>\n\n
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx不缓冲SSE
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口（供K8s liveness probe使用）。"""
    return HealthResponse(
        status="healthy",
        agents_ready=list(_agents.keys()),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics")
async def metrics():
    """
    延迟指标接口（供Prometheus/Grafana抓取）。
    生产环境建议改用 prometheus_fastapi_instrumentator。
    """
    from agent.factory import get_shared_llm_client

    llm_stats = get_shared_llm_client().get_latency_stats()

    pipeline_stats = {}
    for use_case, agent in _agents.items():
        pipeline_stats[use_case] = agent.retriever.get_latency_stats()

    from agent.factory import get_shared_cache
    cache_stats = get_shared_cache().get_stats()

    return {
        "llm_latency": llm_stats,
        "retrieval_latency": pipeline_stats,
        "cache": cache_stats,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.delete("/cache/{use_case}/{tenant_id}")
async def invalidate_cache(use_case: str, tenant_id: str):
    """
    使某个租户的缓存失效（文档更新后调用）。
    例：PUT /documents 成功后自动触发此接口。
    """
    from agent.factory import get_shared_cache
    deleted = await get_shared_cache().invalidate(use_case, tenant_id)
    return {"deleted_keys": deleted}


# ─────────────────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,             # 多进程（CPU密集型任务）
        loop="uvloop",         # 更快的事件循环
        log_level="info",
    )
