"""
agent/factory.py
----------------
AgentFactory: Dependency injection factory that assembles agent instances by use_case.
Uses singletons in production to reuse connection pools and avoid recreating clients.
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
# Singleton components (shared globally to avoid recreating connection pools)
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
# Agent factory
# ─────────────────────────────────────────────

class AgentFactory:
    """
    Assemble a complete EnterpriseRAGAgent based on the use_case string.
    All underlying clients are singletons; only AgentConfig and Pipeline are instantiated per use case.
    """

    @staticmethod
    def create(use_case: str, qdrant_url: str = None) -> EnterpriseRAGAgent:
        """
        use_case: "kb_qa" | "helpdesk" | "compliance"
        """
        config: AgentConfig = get_config(use_case)

        # Qdrant vector store connection (a separate collection for each use_case)
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        collection_name = f"enterprise_{use_case}"

        # Retrieval pipeline (includes embedder + reranker)
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
        """Warm up agents for all use cases (called at service startup)."""
        return {
            use_case: AgentFactory.create(use_case)
            for use_case in ["kb_qa", "helpdesk", "compliance"]
        }
