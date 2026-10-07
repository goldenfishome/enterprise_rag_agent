"""
api/main.py
-----------
FastAPI entry point: defines REST API endpoints and manages the agent lifecycle.
Supports: standard responses / streaming SSE responses / health checks / monitoring metrics
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
# Global agent registry (warmed up at service startup to avoid a cold start on the first request)
# ─────────────────────────────────────────────────────
_agents: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the service startup/shutdown lifecycle."""
    logger.info("Initializing agents for all use cases...")
    try:
        _agents.update(AgentFactory.create_all())
        logger.info(f"Agents ready: {list(_agents.keys())}")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        # In production, startup failures should trigger alerts; service startup is not blocked here

    yield  # Service is running

    # Clean up resources on shutdown
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
    description="Production-grade LLM agent for multiple use cases, supporting kb_qa / helpdesk / compliance",
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
# Request / Response models
# ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    use_case: str = Field(..., description="Business use case: kb_qa | helpdesk | compliance")
    tenant_id: str = Field(..., description="Enterprise tenant ID for data isolation")
    stream: bool = Field(False, description="Whether to stream output")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the employee annual leave policy?",
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
# Service start time (used to calculate uptime)
# ─────────────────────────────────────────────────────
_start_time = time.time()


# ─────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint (non-streaming).
    - Automatically routes to the agent for the corresponding use_case
    - Integrates the full caching, retrieval, and generation workflow
    - Target P50 latency ≤ 2000ms
    """
    if req.use_case not in _agents:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported use_case: {req.use_case}. "
                   f"Available use cases: {list(_agents.keys())}"
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
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint (Server-Sent Events).
    Target first-token latency < 500ms, suitable for real-time interaction (such as a helpdesk).

    Frontend consumption example:
        const es = new EventSource('/v1/chat/stream');
        es.onmessage = (e) => appendToUI(e.data);
    """
    if req.use_case not in _agents:
        raise HTTPException(status_code=400, detail=f"Unsupported use_case: {req.use_case}")

    agent = _agents[req.use_case]

    async def event_generator():
        try:
            async for token in agent.run_stream(req.query, req.tenant_id):
                # SSE format: data: <token>\n\n
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
            "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint (for the K8s liveness probe)."""
    return HealthResponse(
        status="healthy",
        agents_ready=list(_agents.keys()),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics")
async def metrics():
    """
    Latency metrics endpoint (for Prometheus/Grafana scraping).
    In production, switching to prometheus_fastapi_instrumentator is recommended.
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
    Invalidate the cache for a tenant (called after document updates).
    Example: automatically trigger this endpoint after a successful PUT /documents.
    """
    from agent.factory import get_shared_cache
    deleted = await get_shared_cache().invalidate(use_case, tenant_id)
    return {"deleted_keys": deleted}


# ─────────────────────────────────────────────────────
# Startup entry point
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,             # Multiple processes (CPU-intensive tasks)
        loop="uvloop",         # Faster event loop
        log_level="info",
    )
