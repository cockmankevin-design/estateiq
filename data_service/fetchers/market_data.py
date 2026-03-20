"""
Market data fetcher.

Supports multiple providers (yfinance, Alpha Vantage, Polygon.io)
with automatic fallback.
"""

import logging
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from .base import BaseFetcher

logger = logging.getLogger(__name__)


class MarketDataFetcher(BaseFetcher):
    """
    Fetches OHLCV price and volume data from financial data providers.

    Provider priority: yfinance → Alpha Vantage → Polygon.io
    Falls back to the next provider on any error.
    """

    def __init__(
        self,
        alpha_vantage_key: str = "",
        polygon_key: str = "",
        preferred_provider: str = "yfinance",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        super().__init__(api_key=alpha_vantage_key, timeout=timeout, max_retries=max_retries)
        self.polygon_key = polygon_key
        self.preferred_provider = preferred_provider
        self._av_key = alpha_vantage_key

    @property
    def source_name(self) -> str:
        return "MarketData"

    # ------------------------------------------------------------------
    # Core implementation
    # ------------------------------------------------------------------

    def _fetch_impl(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        providers = self._provider_order()
        last_exc: Optional[Exception] = None
        for provider in providers:
            try:
                df = self._fetch_from(provider, symbol, start, end, interval)
                if not df.empty:
                    return self._normalise(df)
            except Exception as exc:
                logger.warning("Provider %s failed for %s: %s", provider, symbol, exc)
                last_exc = exc

        if last_exc:
            raise last_exc
        return pd.DataFrame()

    def _provider_order(self):
        providers = ["yfinance", "alpha_vantage", "polygon"]
        if self.preferred_provider in providers:
            providers.remove(self.preferred_provider)
            providers.insert(0, self.preferred_provider)
        return providers

    def _fetch_from(
        self,
        provider: str,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> pd.DataFrame:
        if provider == "yfinance":
            return self._fetch_yfinance(symbol, start, end, interval)
        if provider == "alpha_vantage":
            return self._fetch_alpha_vantage(symbol, start, end)
        if provider == "polygon":
            return self._fetch_polygon(symbol, start, end, interval)
        raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    # Provider-specific helpers
    # ------------------------------------------------------------------

    def _fetch_yfinance(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance not installed") from exc

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df.index.name = "date"
        return df

    def _fetch_alpha_vantage(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        if not self._av_key:
            raise RuntimeError("Alpha Vantage API key not configured")

        import requests

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self._av_key,
            "outputsize": "full",
            "datatype": "json",
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json().get("Time Series (Daily)", {})
        df = pd.DataFrame.from_dict(data, orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()
        df = df.loc[start:end]  # type: ignore[misc]
        df.columns = [c.split(". ")[1] for c in df.columns]
        return df

    def _fetch_polygon(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        if not self.polygon_key:
            raise RuntimeError("Polygon API key not configured")

        import requests

        multiplier, timespan = self._parse_interval(interval)
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range"
            f"/{multiplier}/{timespan}"
            f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.polygon_key}
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("date").rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _parse_interval(interval: str):
        mapping = {
            "1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"),
            "1h": (1, "hour"), "1d": (1, "day"), "1w": (1, "week"),
        }
        if interval in mapping:
            return mapping[interval]
        return 1, "day"

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names to lower-case OHLCV."""
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        present = required & set(df.columns)
        return df[list(present)].copy()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def fetch_latest_price(self, symbol: str) -> float:
        """Return the most recent closing price for *symbol*."""
        from datetime import timedelta

        end = datetime.utcnow()
        start = end - timedelta(days=5)
        result = self.fetch(symbol, start, end)
        if result.ok:
            return float(result.data["close"].iloc[-1])
        raise RuntimeError(f"Could not fetch price for {symbol}: {result.errors}")

    def fetch_bulk_close(
        self, symbols: list, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return a DataFrame of closing prices with symbols as columns."""
        frames = {}
        for sym in symbols:
            res = self.fetch(sym, start, end)
            if res.ok and "close" in res.data.columns:
                frames[sym] = res.data["close"]
        return pd.DataFrame(frames) if frames else pd.DataFrame()
