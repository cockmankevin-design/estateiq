"""Abstract base class for all data fetchers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Uniform return type from every fetcher."""

    data: pd.DataFrame
    source: str
    symbol: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.data.empty


class BaseFetcher(ABC):
    """
    Contract every fetcher must satisfy.

    Subclasses implement `_fetch_impl` with their provider-specific
    logic; the public `fetch` method wraps it with retry / logging.
    """

    def __init__(self, api_key: str = "", timeout: int = 30, max_retries: int = 3):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    def _fetch_impl(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Provider-specific data retrieval. Must return a DataFrame."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        **kwargs: Any,
    ) -> FetchResult:
        """Public fetch with automatic retry and structured result."""
        self._logger.info("Fetching %s [%s] %s→%s", symbol, self.source_name, start.date(), end.date())
        errors: List[str] = []
        df = pd.DataFrame()
        try:
            df = self._fetch_impl(symbol, start, end, **kwargs)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self._logger.error("Fetch failed for %s: %s", symbol, msg)
            errors.append(msg)

        return FetchResult(
            data=df,
            source=self.source_name,
            symbol=symbol,
            start=start,
            end=end,
            errors=errors,
        )

    def fetch_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        **kwargs: Any,
    ) -> Dict[str, FetchResult]:
        """Fetch multiple symbols sequentially, returning a dict keyed by symbol."""
        return {s: self.fetch(s, start, end, **kwargs) for s in symbols}

    def validate_dates(self, start: datetime, end: datetime) -> None:
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        if end > datetime.utcnow():
            raise ValueError("end date cannot be in the future")
