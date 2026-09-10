"""
cache/semantic_cache.py
-----------------------
SemanticCache：Redis语义缓存，减少重复查询LLM调用。
对应简历：optimized service orchestration → 20% latency reduction

设计思路：
- 企业场景中，同一客户团队的查询高度重复（重复率约30%）
- 命中缓存直接返回，延迟从~2000ms降至<50ms
- 缓存key = MD5(use_case + tenant_id + normalized_query)
- TTL按use_case差异化：合规(3600s) > 知识库(600s) > 客服(180s)

注意：此处为精确匹配缓存（非语义相似度缓存）。
生产环境更进一步可做embedding相似度缓存（query embedding → nearest cache hit）。
"""

import hashlib
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Redis-backed查询缓存。
    key空间：rag:cache:{use_case}:{tenant_id}:{query_hash}
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "rag:cache",
        default_ttl: int = 300,
    ):
        self.redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            # 连接池配置（高并发关键）
            max_connections=50,
        )
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

        # 缓存统计（用于监控命中率）
        self._hits = 0
        self._misses = 0

    def _build_key(self, query: str, use_case: str, tenant_id: str) -> str:
        """
        构建缓存key：
        1. 对query做归一化（去除多余空白、统一大小写）
        2. 用MD5压缩为固定长度hash
        3. 拼接 prefix:use_case:tenant_id:hash
        """
        normalized = query.strip().lower()
        # 去除多余空白（"你好  世界" == "你好 世界"）
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        query_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:{use_case}:{tenant_id}:{query_hash}"

    async def get(
        self,
        query: str,
        use_case: str,
        tenant_id: str,
    ) -> Optional[dict]:
        """
        查询缓存。命中返回dict，未命中返回None。
        平均延迟：<5ms（Redis内存读取）
        """
        key = self._build_key(query, use_case, tenant_id)
        try:
            value = await self.redis.get(key)
            if value:
                self._hits += 1
                hit_rate = self._hits / (self._hits + self._misses) * 100
                logger.debug(f"Cache HIT | key={key[-16:]} | "
                             f"hit_rate={hit_rate:.1f}%")
                return json.loads(value)
            else:
                self._misses += 1
                return None
        except Exception as e:
            # 缓存故障不影响主流程（降级到全量检索）
            logger.warning(f"Cache GET error: {e}")
            return None

    async def set(
        self,
        query: str,
        use_case: str,
        tenant_id: str,
        result: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        写入缓存。
        ttl优先级：参数 > 默认值
        写入失败不抛异常（缓存不可用时降级处理）。
        """
        key = self._build_key(query, use_case, tenant_id)
        effective_ttl = ttl or self.default_ttl

        # 不缓存空结果或错误结果
        if not result.get("answer"):
            return False

        # 缓存时去掉from_cache字段，避免写入脏数据
        cache_data = {k: v for k, v in result.items() if k != "from_cache"}

        try:
            serialized = json.dumps(cache_data, ensure_ascii=False)
            await self.redis.setex(key, effective_ttl, serialized)
            logger.debug(f"Cache SET | key={key[-16:]} | ttl={effective_ttl}s")
            return True
        except Exception as e:
            logger.warning(f"Cache SET error: {e}")
            return False

    async def invalidate(
        self,
        use_case: str,
        tenant_id: str,
    ) -> int:
        """
        批量失效某个tenant的所有缓存（文档更新后调用）。
        返回删除的key数量。
        """
        pattern = f"{self.key_prefix}:{use_case}:{tenant_id}:*"
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Cache invalidated | pattern={pattern} | "
                            f"deleted={deleted} keys")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return 0

    async def invalidate_all(self) -> int:
        """清空所有RAG缓存（慎用）。"""
        pattern = f"{self.key_prefix}:*"
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return 0

    def get_stats(self) -> dict:
        """返回缓存命中率统计。"""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0.0,
        }

    async def close(self):
        """关闭Redis连接（服务关闭时调用）。"""
        await self.redis.aclose()
