# Enterprise RAG Agent Platform

生产级多场景LLM Agent，面向企业客户，支持3个业务场景。

## 项目结构

```
enterprise_rag_agent/
├── agent/
│   ├── core.py          # Agent基类 + 主流程
│   └── factory.py       # Agent工厂
├── retrieval/
│   ├── pipeline.py      # 异步并行RAG检索流水线
│   └── reranker.py      # 两阶段Rerank
├── ingestion/
│   ├── chunker.py       # Hierarchical分块策略
│   └── embedder.py      # HyDE embedding优化
├── cache/
│   └── semantic_cache.py # Redis语义缓存
├── llm/
│   └── client.py        # LLM客户端（连接池+流式）
├── config/
│   └── use_cases.py     # 3个业务场景配置
├── api/
│   └── main.py          # FastAPI入口
├── evaluation/
│   └── evaluator.py     # 检索效果评估（幻觉率/相关性）
├── tests/
│   └── load_test.py     # Locust并发压测
└── requirements.txt
```

## 快速启动

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 三条简历对应关系

| 简历内容 | 代码文件 |
|----------|----------|
| 3 distinct use cases + configurable logic | config/use_cases.py, agent/core.py |
| 20% latency reduction + 2s P50 | retrieval/pipeline.py, cache/semantic_cache.py, llm/client.py |
| 25% relevance + 31% hallucination reduction | ingestion/chunker.py, ingestion/embedder.py, retrieval/reranker.py |
