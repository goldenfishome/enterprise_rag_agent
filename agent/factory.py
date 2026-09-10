"""
agent/factory.py
----------------
AgentFactory：依赖注入工厂，根据use_case组装Agent实例。
生产环境中使用单例模式复用连接池，避免重复创建客户端。
"""

import os
from functools import lru_cache
from config.use_cases import AgentConfig, get_config
from agent.core import EnterpriseRAGAgent
from retrieval.pipeline import OptimizedRAGPipeline
from retrieval.reranker import TwoStageRetriever
from ingestion.embedder import DomainEmbedder
from cache.semantic_cache import SemanticCache
from llm.client import LLMClient


# ─────────────────────────────────────────────
# 单例组件（全局共享，避免重复建连接池）
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_shared_embedder() -> DomainEmbedder:
    return DomainEmbedder(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    )


@lru_cache(maxsize=1)
def get_shared_cache() -> SemanticCache:
    return SemanticCache(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    )


@lru_cache(maxsize=1)
def get_shared_llm_client() -> LLMClient:
    return LLMClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        max_connections=100,
        max_keepalive=20,
    )


@lru_cache(maxsize=1)
def get_shared_reranker() -> TwoStageRetriever:
    return TwoStageRetriever(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )


# ─────────────────────────────────────────────
# Agent工厂
# ─────────────────────────────────────────────

class AgentFactory:
    """
    根据use_case字符串组装完整的EnterpriseRAGAgent。
    所有底层客户端使用单例，只有AgentConfig和Pipeline按场景实例化。
    """

    @staticmethod
    def create(use_case: str, qdrant_url: str = None) -> EnterpriseRAGAgent:
        """
        use_case: "kb_qa" | "helpdesk" | "compliance"
        """
        config: AgentConfig = get_config(use_case)

        # Qdrant向量库连接（每个use_case对应独立collection）
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        collection_name = f"enterprise_{use_case}"

        # 检索流水线（包含embedder + reranker）
        pipeline = OptimizedRAGPipeline(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            embedder=get_shared_embedder(),
            reranker=get_shared_reranker(),
        )

        return EnterpriseRAGAgent(
            config=config,
            retriever=pipeline,
            llm_client=get_shared_llm_client(),
            cache=get_shared_cache(),
        )

    @staticmethod
    def create_all() -> dict[str, EnterpriseRAGAgent]:
        """预热所有use_case的Agent（服务启动时调用）。"""
        return {
            use_case: AgentFactory.create(use_case)
            for use_case in ["kb_qa", "helpdesk", "compliance"]
        }
