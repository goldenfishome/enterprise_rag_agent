"""
config/use_cases.py
-------------------
Distinct AgentConfig configurations for 3 business use cases.
Resume reference: configurable business logic, serving enterprise clients across 3 distinct use cases
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """
    Configurable agent business logic parameters.
    Each enterprise use case has an AgentConfig instance,
    sharing the same RAG + LLM framework with configuration-driven differences in behavior.
    """
    # Use case identifier
    use_case: str

    # Retrieval parameters
    top_k: int = 5                     # Number of documents returned in the final result
    candidate_k: int = 20              # Initial ranking candidate count (Stage 1)
    rerank: bool = True                # Whether to enable Cohere reranking
    score_threshold: float = 0.5       # Reranking relevance threshold (key parameter for filtering hallucinations)
    hybrid_search: bool = True         # Whether to enable dense + sparse hybrid retrieval

    # Chunking parameters (affect document processing during ingestion)
    chunk_size: int = 512
    chunk_overlap: int = 64
    use_hierarchical_chunking: bool = True

    # Embedding parameters
    embedding_model: str = "text-embedding-3-large"
    use_hyde: bool = True              # HyDE query expansion (improves semantic vector matching)
    hyde_max_tokens: int = 150

    # LLM parameters
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.1           # Lower temperature for enterprise use cases to reduce randomness
    stream: bool = True

    # Cache parameters
    cache_ttl: int = 300               # Redis cache TTL (seconds)
    cache_enabled: bool = True

    # System prompt (core distinction between business use cases)
    system_prompt: str = ""

    # Metadata filter field (multi-tenant isolation)
    tenant_filter_field: str = "tenant_id"


# ─────────────────────────────────────────────
# 3 enterprise use case configurations
# ─────────────────────────────────────────────

KB_QA_CONFIG = AgentConfig(
    use_case="kb_qa",
    top_k=5,
    candidate_k=20,
    rerank=True,
    score_threshold=0.5,
    hybrid_search=True,
    chunk_size=512,
    chunk_overlap=64,
    use_hierarchical_chunking=True,
    use_hyde=True,
    cache_ttl=600,          # Knowledge base content changes slowly, so cache longer
    system_prompt=(
        "You are an intelligent assistant for the internal enterprise knowledge base. "
        "Answer questions strictly based on the retrieved documents below. "
        "Every key conclusion must cite its source (e.g., [Source1]). "
        "If the documents contain no relevant information, explicitly state 'This information is not currently available in the knowledge base'; do not guess."
        "\n\nRetrieved documents:\n{context}"
    ),
)

HELPDESK_CONFIG = AgentConfig(
    use_case="helpdesk",
    top_k=3,
    candidate_k=15,
    rerank=True,
    score_threshold=0.55,   # Customer service requires higher precision
    hybrid_search=True,
    chunk_size=256,          # Shorter chunks for support tickets improve retrieval precision
    chunk_overlap=32,
    use_hierarchical_chunking=False,
    use_hyde=False,          # Customer service queries have clear intent, so HyDE offers little benefit
    cache_ttl=180,           # Support tickets are time-sensitive, so use a short cache TTL
    max_tokens=512,          # Customer service answers should be concise
    system_prompt=(
        "You are an intelligent customer service support system that helps support staff quickly find solutions. "
        "Provide concise, actionable steps, prioritizing references to successful past cases. "
        "Response format: [Issue category] → [Recommended solution] → [Reference ticket number]"
        "\n\nReference material:\n{context}"
    ),
)

COMPLIANCE_CONFIG = AgentConfig(
    use_case="compliance",
    top_k=8,
    candidate_k=25,
    rerank=False,            # Compliance requires broad coverage without reranking truncation
    score_threshold=0.4,     # A low threshold ensures risky clauses are not missed
    hybrid_search=True,
    chunk_size=1024,         # Compliance text has long context; large chunks preserve full meaning
    chunk_overlap=128,
    use_hierarchical_chunking=True,
    use_hyde=True,
    cache_ttl=3600,          # Regulatory documents are stable, so use a long cache TTL
    max_tokens=2048,
    temperature=0.0,         # Compliance requires deterministic output
    system_prompt=(
        "You are an enterprise compliance review assistant. "
        "Analyze potential compliance risks in the following documents clause by clause. "
        "Label each risk with: [Risk level: High/Medium/Low][Relevant clauses][Recommended action]. "
        "Base your judgments strictly on the original regulatory text retrieved; do not infer."
        "\n\nReference regulatory documents:\n{context}"
    ),
)

# Registry: use_case -> config
USE_CASE_REGISTRY: dict[str, AgentConfig] = {
    "kb_qa":       KB_QA_CONFIG,
    "helpdesk":    HELPDESK_CONFIG,
    "compliance":  COMPLIANCE_CONFIG,
}


def get_config(use_case: str) -> AgentConfig:
    """Get the configuration by use_case name, or raise ValueError if it does not exist."""
    if use_case not in USE_CASE_REGISTRY:
        raise ValueError(
            f"Unknown use_case: '{use_case}'. "
            f"Supported use cases: {list(USE_CASE_REGISTRY.keys())}"
        )
    return USE_CASE_REGISTRY[use_case]
