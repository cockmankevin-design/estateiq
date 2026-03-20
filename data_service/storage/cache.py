"""
Redis-based caching layer.

Provides transparent caching for expensive data fetches and
computed results with configurable TTL and serialisation.
"""

import json
import logging
import pickle
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class Cache:
    """
    Redis cache with pandas / object serialisation support.

    Usage
    -----
    >>> cache = Cache("redis://localhost:6379/0")
    >>> cache.set("my_key", df, ttl=300)
    >>> df2 = cache.get_df("my_key")
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        default_ttl: int = 300,
        prefix: str = "estateiq:",
    ):
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._r = None
        self._url = url

    def _client(self):
        if self._r is None:
            try:
                import redis
                self._r = redis.from_url(self._url, decode_responses=False)
                self._r.ping()
                logger.info("Redis connected: %s", self._url)
            except Exception as exc:
                logger.warning("Redis unavailable — caching disabled: %s", exc)
                self._r = _NoopRedis()
        return self._r

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a pickled value from cache."""
        raw = self._client().get(self._key(key))
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception as exc:
            logger.warning("Cache deserialise error for '%s': %s", key, exc)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a pickled value in cache."""
        try:
            raw = pickle.dumps(value)
            self._client().setex(self._key(key), ttl or self.default_ttl, raw)
            return True
        except Exception as exc:
            logger.warning("Cache set error for '%s': %s", key, exc)
            return False

    def get_df(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame stored as Parquet bytes."""
        raw = self._client().get(self._key(key))
        if raw is None:
            return None
        try:
            import io
            return pd.read_parquet(io.BytesIO(raw))
        except Exception as exc:
            logger.warning("Cache DataFrame deserialise error for '%s': %s", key, exc)
            return None

    def set_df(self, key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> bool:
        """Store a DataFrame as Parquet bytes."""
        try:
            import io
            buf = io.BytesIO()
            df.to_parquet(buf, engine="pyarrow")
            self._client().setex(self._key(key), ttl or self.default_ttl, buf.getvalue())
            return True
        except Exception as exc:
            logger.warning("Cache DataFrame set error for '%s': %s", key, exc)
            return False

    def delete(self, key: str) -> None:
        self._client().delete(self._key(key))

    def exists(self, key: str) -> bool:
        return bool(self._client().exists(self._key(key)))

    def flush_prefix(self) -> int:
        """Delete all keys matching this cache's prefix."""
        pattern = f"{self.prefix}*"
        keys = self._client().keys(pattern)
        if keys:
            self._client().delete(*keys)
        return len(keys)

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def cached(
        self,
        ttl: Optional[int] = None,
        key_fn: Optional[Callable] = None,
    ) -> Callable:
        """
        Decorator to cache a function's return value.

        >>> @cache.cached(ttl=600)
        ... def expensive(symbol, start, end):
        ...     ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if key_fn is not None:
                    cache_key = key_fn(*args, **kwargs)
                else:
                    arg_str = "_".join(str(a) for a in args)
                    kwarg_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = f"{func.__name__}:{arg_str}:{kwarg_str}"

                cached_val = self.get(cache_key)
                if cached_val is not None:
                    logger.debug("Cache hit: %s", cache_key)
                    return cached_val

                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result
            return wrapper
        return decorator


# ---------------------------------------------------------------------------
# No-op fallback when Redis is unavailable
# ---------------------------------------------------------------------------

class _NoopRedis:
    """Silent no-op Redis client used when Redis is not available."""

    def get(self, key): return None
    def setex(self, key, ttl, value): pass
    def delete(self, *keys): pass
    def exists(self, key): return 0
    def keys(self, pattern): return []
    def ping(self): return True
