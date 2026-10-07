# Enterprise RAG Agent Platform

Production-grade LLM agent platform for enterprise clients, supporting three business use cases.

## Project Structure

```
enterprise_rag_agent/
├── agent/
│   ├── core.py          # Core agent class and main workflow
│   └── factory.py       # Agent factory
├── retrieval/
│   ├── pipeline.py      # Asynchronous parallel RAG retrieval pipeline
│   └── reranker.py      # Two-stage reranking
├── ingestion/
│   ├── chunker.py       # Hierarchical chunking strategy
│   └── embedder.py      # HyDE embedding optimization
├── cache/
│   └── semantic_cache.py # Redis query cache
├── llm/
│   └── client.py        # LLM client with connection pooling and streaming
├── config/
│   └── use_cases.py     # Configuration for three business use cases
├── api/
│   └── main.py          # FastAPI entry point
├── evaluation/
│   └── evaluator.py     # Retrieval evaluation: hallucination rate and relevance
├── tests/
│   └── load_test.py     # Concurrent load testing with Locust
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Mapping to Three Resume Highlights

| Resume highlight | Code files |
|----------|----------|
| 3 distinct use cases + configurable logic | config/use_cases.py, agent/core.py |
| 20% latency reduction + 2s P50 | retrieval/pipeline.py, cache/semantic_cache.py, llm/client.py |
| 25% relevance + 31% hallucination reduction | ingestion/chunker.py, ingestion/embedder.py, retrieval/reranker.py |
