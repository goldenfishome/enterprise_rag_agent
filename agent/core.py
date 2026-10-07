"""
agent/core.py
-------------
EnterpriseRAGAgent: Core agent class connecting the full retrieval → generation workflow.
Resume reference: production-ready LLM-based agent with RAG retrieval pipelines
"""

import time
import logging
from typing import AsyncIterator, Optional

from config.use_cases import AgentConfig

logger = logging.getLogger(__name__)


class Document:
    """Simplified document data class (from LangChain or custom-built in real projects)."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class EnterpriseRAGAgent:
    """
    Production-grade RAG agent.
    - Supports multi-tenant isolation (tenant_id filtering)
    - Supports streaming output (stream=True)
    - Integrates semantic caching, retrieval pipelines, and LLM generation
    - Implements configurable business logic through AgentConfig
    """

    def __init__(
        self,
        config: AgentConfig,
        retriever,       # OptimizedRAGPipeline instance
        llm_client,      # LLMClient instance
        cache,           # SemanticCache instance
    ):
        self.config = config
        self.retriever = retriever
        self.llm = llm_client
        self.cache = cache

    # ─────────────────────────────────────────
    # Main entry point: non-streaming
    # ─────────────────────────────────────────
    async def run(self, query: str, tenant_id: str) -> dict:
        """
        Full RAG workflow: cache lookup → retrieval → generation → cache write.
        Returns: {answer, sources, latency_ms, from_cache}
        """
        t0 = time.perf_counter()

        # Step 1: Check for a cache hit (return immediately for popular queries)
        if self.config.cache_enabled:
            cached = await self.cache.get(query, self.config.use_case, tenant_id)
            if cached:
                logger.info(f"[{self.config.use_case}] Cache hit | query={query[:50]}")
                return {**cached, "from_cache": True, "latency_ms": 0}

        # Step 2: RAG retrieval
        docs = await self.retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            rerank=self.config.rerank,
            score_threshold=self.config.score_threshold,
            use_hyde=self.config.use_hyde,
        )
        logger.info(f"[{self.config.use_case}] Retrieved {len(docs)} docs")

        # Step 3: Build the prompt context
        context = self._build_context(docs)
        system_prompt = self.config.system_prompt.format(context=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Step 4: LLM generation (non-streaming)
        answer = await self.llm.generate(
            messages=messages,
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"[{self.config.use_case}] Done | latency={latency_ms}ms")

        result = {
            "answer": answer,
            "sources": self._format_sources(docs),
            "latency_ms": latency_ms,
            "use_case": self.config.use_case,
            "from_cache": False,
        }

        # Step 5: Write to cache
        if self.config.cache_enabled:
            await self.cache.set(
                query, self.config.use_case, tenant_id,
                result, ttl=self.config.cache_ttl
            )

        return result

    # ─────────────────────────────────────────
    # Streaming entry point (Server-Sent Events)
    # ─────────────────────────────────────────
    async def run_stream(
        self, query: str, tenant_id: str
    ) -> AsyncIterator[str]:
        """
        Streaming RAG: start streaming generation after retrieval, with first-token latency <500ms.
        Usage: async for token in agent.run_stream(query, tenant_id)
        """
        # Retrieval (same as above, non-streaming)
        docs = await self.retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            rerank=self.config.rerank,
            score_threshold=self.config.score_threshold,
            use_hyde=self.config.use_hyde,
        )

        context = self._build_context(docs)
        system_prompt = self.config.system_prompt.format(context=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Streaming LLM generation
        async for token in self.llm.stream_generate(
            messages=messages,
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ):
            yield token

    # ─────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────
    def _build_context(self, docs: list[Document]) -> str:
        """
        Join retrieved documents into a structured context string.
        With hierarchical chunking, this returns parent chunk content (more complete).
        """
        if not docs:
            return "(No relevant documents retrieved)"
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "")
            header = f"[Source{i}] {source}" + (f" | Page {page}" if page else "")
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _format_sources(self, docs: list[Document]) -> list[dict]:
        """Return a structured source list for rendering citation markers in the frontend."""
        return [
            {
                "index": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", ""),
                "score": doc.metadata.get("relevance_score", None),
            }
            for i, doc in enumerate(docs)
        ]
