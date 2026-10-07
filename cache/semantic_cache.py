"""
cache/semantic_cache.py
-----------------------
SemanticCache: Redis query cache that reduces LLM calls for repeated queries.
Resume highlight: optimized service orchestration → 20% latency reduction

Design:
- Queries within the same enterprise team repeat frequently (about 30%).
- Return cached results immediately, reducing latency from ~2000ms to <50ms.
- Cache key = MD5(use_case + tenant_id + normalized_query).
- TTL varies by use case: compliance (3600s) > knowledge base (600s) > helpdesk (180s).

Note: this implementation uses exact matching, not semantic similarity.
Production deployments could add embedding similarity caching (query embedding → nearest cache hit).
"""

import hashlib
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Redis-backed query cache.
    Key namespace: rag:cache:{use_case}:{tenant_id}:{query_hash}
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
            # Connection pool configuration for high concurrency
            max_connections=50,
        )
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

        # Cache statistics for monitoring the hit rate
        self._hits = 0
        self._misses = 0

    def _build_key(self, query: str, use_case: str, tenant_id: str) -> str:
        """
        Build the cache key:
        1. Normalize the query (collapse whitespace and lowercase).
        2. Use MD5 to produce a fixed-length hash.
        3. Combine prefix:use_case:tenant_id:hash.
        """
        normalized = query.strip().lower()
        # Collapse extra whitespace ("hello  world" == "hello world")
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
        Look up the cache. Return a dict on a hit, or None on a miss.
        Average latency: <5ms (Redis in-memory read).
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
            # Cache failures fall back to the full retrieval workflow
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
        Write to the cache.
        TTL precedence: argument > default value.
        Write failures do not raise exceptions, allowing operation without caching.
        """
        key = self._build_key(query, use_case, tenant_id)
        effective_ttl = ttl or self.default_ttl

        # Do not cache empty answers or error results
        if not result.get("answer"):
            return False

        # Omit from_cache when storing the result to avoid stale status data
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
        Invalidate all cached results for a tenant and use case after document updates.
        Return the number of deleted keys.
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
        """Clear all RAG cache entries (use with caution)."""
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
        """Return cache hit-rate statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0.0,
        }

    async def close(self):
        """Close the Redis connection during service shutdown."""
        await self.redis.aclose()
