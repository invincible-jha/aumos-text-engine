"""Redis-backed prompt cache manager.

Provides semantic-aware caching for LLM prompt/response pairs with TTL-based
expiration, hit/miss metrics, version-based invalidation, and optional
approximate matching via embedding similarity.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from aumos_common.logging import get_logger

logger: structlog.BoundLogger = get_logger(__name__)

# Default cache settings
_DEFAULT_TTL_SECONDS = 3600           # 1 hour
_DEFAULT_SEMANTIC_THRESHOLD = 0.92    # Similarity threshold for approximate hits
_DEFAULT_MAX_MEMORY_CACHE_SIZE = 512  # In-memory LRU fallback entries
_CACHE_KEY_PREFIX = "aumos:text:cache"
_METRICS_KEY_PREFIX = "aumos:text:cache:metrics"


@dataclass
class CacheEntry:
    """A cached prompt-response pair.

    Attributes:
        cache_key: Hash-derived cache key.
        prompt_hash: SHA-256 hash of the prompt.
        config_hash: SHA-256 hash of the generation config.
        response: Cached LLM response.
        template_version: Version of the prompt template used.
        created_at: Unix timestamp of cache entry creation.
        ttl: Time-to-live in seconds.
    """

    cache_key: str
    prompt_hash: str
    config_hash: str
    response: str
    template_version: str
    created_at: float
    ttl: int


@dataclass
class CacheMetrics:
    """Aggregate cache performance metrics.

    Attributes:
        total_hits: Total cache hits across all keys.
        total_misses: Total cache misses.
        total_semantic_hits: Approximate hits via embedding similarity.
        total_invalidations: Keys invalidated by version bumps.
        memory_usage_bytes: Estimated memory usage of cached data.
    """

    total_hits: int = 0
    total_misses: int = 0
    total_semantic_hits: int = 0
    total_invalidations: int = 0
    memory_usage_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0.0, 1.0].

        Returns:
            Hit rate (hits / total_lookups).
        """
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0


class PromptCacheManager:
    """Redis-backed prompt/response cache with semantic matching.

    Features:
    - Exact-match caching via SHA-256 key derivation
    - Semantic approximate matching via embedding similarity
    - TTL-based expiration
    - Template version-based batch invalidation
    - In-memory LRU fallback when Redis is unavailable
    - Hit/miss/eviction metrics

    Attributes:
        _redis_url: Redis connection URL.
        _default_ttl: Default TTL for cache entries in seconds.
        _semantic_threshold: Cosine similarity threshold for approximate hits.
        _redis_client: Async Redis client (lazy initialized).
        _memory_cache: In-memory fallback cache (key -> CacheEntry).
        _memory_cache_max_size: Maximum in-memory cache entries.
        _embedding_model: sentence-transformers model for semantic matching.
        _metrics: Running cache metrics.
        _log: Structured logger.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = _DEFAULT_TTL_SECONDS,
        semantic_threshold: float = _DEFAULT_SEMANTIC_THRESHOLD,
        enable_semantic_cache: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        memory_cache_max_size: int = _DEFAULT_MAX_MEMORY_CACHE_SIZE,
    ) -> None:
        """Initialize the PromptCacheManager.

        Args:
            redis_url: Redis server URL.
            default_ttl: Default time-to-live for cache entries in seconds.
            semantic_threshold: Minimum similarity for approximate cache hits.
            enable_semantic_cache: Whether to use embedding-based matching.
            embedding_model_name: sentence-transformers model for semantic cache.
            memory_cache_max_size: Maximum in-memory LRU cache entries.
        """
        self._redis_url = redis_url
        self._default_ttl = default_ttl
        self._semantic_threshold = semantic_threshold
        self._enable_semantic_cache = enable_semantic_cache
        self._embedding_model_name = embedding_model_name
        self._memory_cache_max_size = memory_cache_max_size
        self._redis_client: Any = None
        self._memory_cache: dict[str, CacheEntry] = {}
        self._memory_cache_order: list[str] = []  # LRU tracking
        self._embedding_model: Any = None
        self._semantic_index: list[tuple[str, Any]] = []  # (cache_key, embedding)
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def initialize(self) -> None:
        """Connect to Redis and load the embedding model.

        Gracefully degrades to in-memory cache if Redis is unavailable.

        Args:
            None

        Returns:
            None
        """
        # Try to connect to Redis
        try:
            import redis.asyncio as aioredis

            self._redis_client = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis_client.ping()
            self._log.info("connected to Redis cache", url=self._redis_url)
        except Exception as exc:
            self._log.warning(
                "Redis unavailable, falling back to in-memory cache",
                error=str(exc),
            )
            self._redis_client = None

        # Load embedding model for semantic cache
        if self._enable_semantic_cache:
            try:
                from sentence_transformers import SentenceTransformer

                loop = asyncio.get_running_loop()
                self._embedding_model = await loop.run_in_executor(
                    None,
                    SentenceTransformer,
                    self._embedding_model_name,
                )
                self._log.info("semantic cache embedding model loaded")
            except ImportError:
                self._log.warning("sentence-transformers not installed — semantic cache disabled")
                self._enable_semantic_cache = False

    def _make_cache_key(
        self,
        prompt: str,
        config_dict: dict[str, Any],
    ) -> tuple[str, str]:
        """Derive cache key from prompt and config hashes.

        Args:
            prompt: Generation prompt.
            config_dict: Generation configuration as a dict.

        Returns:
            Tuple of (cache_key, prompt_hash).
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        cache_key = f"{_CACHE_KEY_PREFIX}:{prompt_hash[:32]}:{config_hash}"
        return cache_key, prompt_hash

    async def get(
        self,
        prompt: str,
        config_dict: dict[str, Any],
    ) -> str | None:
        """Look up a cached response for the given prompt and config.

        Checks exact match first, then semantic approximate match if enabled.

        Args:
            prompt: Generation prompt to look up.
            config_dict: Generation configuration dict.

        Returns:
            Cached response string, or None if not found.
        """
        cache_key, prompt_hash = self._make_cache_key(prompt, config_dict)

        # Try exact match
        exact_result = await self._get_exact(cache_key)
        if exact_result is not None:
            self._metrics.total_hits += 1
            self._log.debug("cache hit (exact)", cache_key=cache_key[:24])
            return exact_result

        # Try semantic approximate match
        if self._enable_semantic_cache and self._embedding_model is not None:
            semantic_result = await self._get_semantic(prompt)
            if semantic_result is not None:
                self._metrics.total_hits += 1
                self._metrics.total_semantic_hits += 1
                self._log.debug("cache hit (semantic)")
                return semantic_result

        self._metrics.total_misses += 1
        return None

    async def _get_exact(self, cache_key: str) -> str | None:
        """Retrieve an exact cache entry by key.

        Args:
            cache_key: Fully derived cache key.

        Returns:
            Cached response or None.
        """
        if self._redis_client is not None:
            try:
                raw = await self._redis_client.get(cache_key)
                if raw:
                    entry = json.loads(raw)
                    return str(entry["response"])
            except Exception as exc:
                self._log.warning("Redis get failed", error=str(exc))

        # Fallback to memory cache
        entry = self._memory_cache.get(cache_key)
        if entry:
            now = time.time()
            if now - entry.created_at < entry.ttl:
                self._update_lru(cache_key)
                return entry.response
            else:
                self._evict_memory(cache_key)

        return None

    async def _get_semantic(self, prompt: str) -> str | None:
        """Find a semantically similar cached prompt.

        Args:
            prompt: Query prompt to match against cached prompts.

        Returns:
            Cached response for the most similar prompt, or None.
        """
        if not self._semantic_index or self._embedding_model is None:
            return None

        loop = asyncio.get_running_loop()

        def _find_match() -> tuple[str, float] | None:
            import numpy as np

            query_embedding = self._embedding_model.encode(
                [prompt],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

            best_key: str | None = None
            best_score = 0.0

            for cache_key, stored_embedding in self._semantic_index:
                score = float(np.dot(query_embedding, stored_embedding))
                if score > best_score:
                    best_score = score
                    best_key = cache_key

            if best_key and best_score >= self._semantic_threshold:
                return best_key, best_score
            return None

        match = await loop.run_in_executor(None, _find_match)
        if match:
            matched_key, score = match
            self._log.debug("semantic match found", score=round(score, 4))
            return await self._get_exact(matched_key)

        return None

    async def set(
        self,
        prompt: str,
        config_dict: dict[str, Any],
        response: str,
        template_version: str = "1.0.0",
        ttl: int | None = None,
    ) -> None:
        """Store a prompt/response pair in the cache.

        Args:
            prompt: The generation prompt.
            config_dict: Generation configuration dict.
            response: LLM response to cache.
            template_version: Version of the template used (for invalidation).
            ttl: Cache TTL in seconds. Uses default if not specified.

        Returns:
            None
        """
        cache_key, prompt_hash = self._make_cache_key(prompt, config_dict)
        effective_ttl = ttl or self._default_ttl
        config_hash = cache_key.split(":")[-1]

        entry = CacheEntry(
            cache_key=cache_key,
            prompt_hash=prompt_hash,
            config_hash=config_hash,
            response=response,
            template_version=template_version,
            created_at=time.time(),
            ttl=effective_ttl,
        )

        entry_dict = {
            "response": response,
            "template_version": template_version,
            "created_at": entry.created_at,
            "prompt_hash": prompt_hash,
        }

        # Write to Redis
        if self._redis_client is not None:
            try:
                await self._redis_client.setex(
                    cache_key,
                    effective_ttl,
                    json.dumps(entry_dict),
                )
                # Store template version tag for batch invalidation
                version_set_key = f"{_METRICS_KEY_PREFIX}:version:{template_version}"
                await self._redis_client.sadd(version_set_key, cache_key)
            except Exception as exc:
                self._log.warning("Redis set failed", error=str(exc))

        # Write to memory cache (with LRU eviction)
        async with self._lock:
            if len(self._memory_cache) >= self._memory_cache_max_size:
                self._evict_lru()
            self._memory_cache[cache_key] = entry
            self._memory_cache_order.append(cache_key)

        # Update semantic index
        if self._enable_semantic_cache and self._embedding_model is not None:
            await self._update_semantic_index(cache_key, prompt)

        self._metrics.memory_usage_bytes = sum(
            len(e.response) for e in self._memory_cache.values()
        )

        self._log.debug("cache entry stored", cache_key=cache_key[:24], ttl=effective_ttl)

    async def _update_semantic_index(self, cache_key: str, prompt: str) -> None:
        """Add a prompt embedding to the semantic similarity index.

        Args:
            cache_key: Cache key to associate with the embedding.
            prompt: Prompt text to embed.

        Returns:
            None
        """
        loop = asyncio.get_running_loop()

        def _embed() -> Any:
            return self._embedding_model.encode(
                [prompt],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

        embedding = await loop.run_in_executor(None, _embed)
        async with self._lock:
            # Remove old entry for this key if it exists
            self._semantic_index = [(k, e) for k, e in self._semantic_index if k != cache_key]
            self._semantic_index.append((cache_key, embedding))

    async def invalidate_by_version(self, template_version: str) -> int:
        """Invalidate all cache entries for a specific template version.

        Args:
            template_version: Template version whose entries should be invalidated.

        Returns:
            Number of entries invalidated.
        """
        count = 0

        # Invalidate in Redis
        if self._redis_client is not None:
            try:
                version_set_key = f"{_METRICS_KEY_PREFIX}:version:{template_version}"
                keys = await self._redis_client.smembers(version_set_key)
                if keys:
                    await self._redis_client.delete(*keys, version_set_key)
                    count = len(keys)
            except Exception as exc:
                self._log.warning("Redis invalidation failed", error=str(exc))

        # Invalidate in memory cache
        async with self._lock:
            keys_to_remove = [
                k for k, e in self._memory_cache.items()
                if e.template_version == template_version
            ]
            for key in keys_to_remove:
                self._evict_memory(key)
                count += 1

        self._metrics.total_invalidations += count
        self._log.info("cache invalidated by version", template_version=template_version, count=count)
        return count

    async def invalidate_key(self, prompt: str, config_dict: dict[str, Any]) -> bool:
        """Invalidate a specific cache entry by prompt and config.

        Args:
            prompt: Generation prompt of the entry to invalidate.
            config_dict: Config dict of the entry to invalidate.

        Returns:
            True if the key was found and invalidated.
        """
        cache_key, _ = self._make_cache_key(prompt, config_dict)
        found = False

        if self._redis_client is not None:
            try:
                deleted = await self._redis_client.delete(cache_key)
                found = deleted > 0
            except Exception as exc:
                self._log.warning("Redis delete failed", error=str(exc))

        async with self._lock:
            if cache_key in self._memory_cache:
                self._evict_memory(cache_key)
                found = True

        return found

    def get_metrics(self) -> dict[str, Any]:
        """Return current cache metrics.

        Args:
            None

        Returns:
            Dict of metric names to values.
        """
        return {
            "total_hits": self._metrics.total_hits,
            "total_misses": self._metrics.total_misses,
            "total_semantic_hits": self._metrics.total_semantic_hits,
            "total_invalidations": self._metrics.total_invalidations,
            "hit_rate": round(self._metrics.hit_rate, 4),
            "memory_cache_entries": len(self._memory_cache),
            "memory_usage_bytes": self._metrics.memory_usage_bytes,
            "semantic_index_size": len(self._semantic_index),
            "redis_connected": self._redis_client is not None,
        }

    def _update_lru(self, cache_key: str) -> None:
        """Move a key to the end of the LRU order list (most recently used).

        Args:
            cache_key: Key to promote.

        Returns:
            None
        """
        try:
            self._memory_cache_order.remove(cache_key)
        except ValueError:
            pass
        self._memory_cache_order.append(cache_key)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry from the memory cache.

        Args:
            None

        Returns:
            None
        """
        if self._memory_cache_order:
            lru_key = self._memory_cache_order[0]
            self._evict_memory(lru_key)

    def _evict_memory(self, cache_key: str) -> None:
        """Remove an entry from the memory cache and LRU tracker.

        Args:
            cache_key: Key to evict.

        Returns:
            None
        """
        self._memory_cache.pop(cache_key, None)
        try:
            self._memory_cache_order.remove(cache_key)
        except ValueError:
            pass

    async def close(self) -> None:
        """Close the Redis connection.

        Args:
            None

        Returns:
            None
        """
        if self._redis_client is not None:
            await self._redis_client.aclose()
            self._log.info("cache manager Redis connection closed")
