"""
agent/core.py
-------------
EnterpriseRAGAgent：核心Agent类，串联检索→生成全流程。
对应简历：production-ready LLM-based agent with RAG retrieval pipelines
"""

import time
import logging
from typing import AsyncIterator, Optional

from config.use_cases import AgentConfig

logger = logging.getLogger(__name__)


class Document:
    """简化的文档数据类（实际项目中来自LangChain或自定义）"""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class EnterpriseRAGAgent:
    """
    生产级RAG Agent。
    - 支持多租户隔离（tenant_id过滤）
    - 支持流式输出（stream=True）
    - 集成语义缓存、检索流水线、LLM生成
    - 通过AgentConfig实现configurable business logic
    """

    def __init__(
        self,
        config: AgentConfig,
        retriever,       # OptimizedRAGPipeline实例
        llm_client,      # LLMClient实例
        cache,           # SemanticCache实例
    ):
        self.config = config
        self.retriever = retriever
        self.llm = llm_client
        self.cache = cache

    # ─────────────────────────────────────────
    # 主入口：非流式
    # ─────────────────────────────────────────
    async def run(self, query: str, tenant_id: str) -> dict:
        """
        完整RAG流程：缓存检查 → 检索 → 生成 → 缓存写入。
        返回: {answer, sources, latency_ms, from_cache}
        """
        t0 = time.perf_counter()

        # Step 1: 缓存命中检查（热点查询直接返回）
        if self.config.cache_enabled:
            cached = await self.cache.get(query, self.config.use_case, tenant_id)
            if cached:
                logger.info(f"[{self.config.use_case}] Cache hit | query={query[:50]}")
                return {**cached, "from_cache": True, "latency_ms": 0}

        # Step 2: RAG检索
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

        # Step 3: 构造Prompt上下文
        context = self._build_context(docs)
        system_prompt = self.config.system_prompt.format(context=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Step 4: LLM生成（非流式）
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

        # Step 5: 写入缓存
        if self.config.cache_enabled:
            await self.cache.set(
                query, self.config.use_case, tenant_id,
                result, ttl=self.config.cache_ttl
            )

        return result

    # ─────────────────────────────────────────
    # 流式入口（Server-Sent Events）
    # ─────────────────────────────────────────
    async def run_stream(
        self, query: str, tenant_id: str
    ) -> AsyncIterator[str]:
        """
        流式RAG：检索完成后开始流式生成，首token延迟<500ms。
        使用方式：async for token in agent.run_stream(query, tenant_id)
        """
        # 检索（同上，非流式）
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

        # 流式LLM生成
        async for token in self.llm.stream_generate(
            messages=messages,
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ):
            yield token

    # ─────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────
    def _build_context(self, docs: list[Document]) -> str:
        """
        将检索到的文档列表拼接成结构化上下文字符串。
        Hierarchical chunking场景下，这里返回的是父chunk内容（更完整）。
        """
        if not docs:
            return "（未检索到相关文档）"
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            header = f"[来源{i}] {source}" + (f" | 第{page}页" if page else "")
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _format_sources(self, docs: list[Document]) -> list[dict]:
        """返回结构化的来源列表，供前端渲染引用角标。"""
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
