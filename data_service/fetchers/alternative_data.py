"""
Alternative data fetcher.

Covers: sentiment (news/social), property-specific (Zillow/ATTOM),
web traffic, and job posting signals.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseFetcher

logger = logging.getLogger(__name__)


class AlternativeDataFetcher(BaseFetcher):
    """
    Fetches non-traditional alpha signals:
      - News & social sentiment via NewsAPI / GDELT
      - Real estate data from Zillow / ATTOM
      - Google Trends (interest over time)
    """

    def __init__(
        self,
        news_api_key: str = "",
        zillow_key: str = "",
        attom_key: str = "",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        super().__init__(api_key=news_api_key, timeout=timeout, max_retries=max_retries)
        self._zillow_key = zillow_key
        self._attom_key = attom_key

    @property
    def source_name(self) -> str:
        return "AlternativeData"

    def _fetch_impl(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: str = "sentiment",
        **kwargs: Any,
    ) -> pd.DataFrame:
        if data_type == "sentiment":
            return self._fetch_news_sentiment(symbol, start, end)
        if data_type == "real_estate":
            return self._fetch_real_estate(symbol, **kwargs)
        if data_type == "trends":
            return self._fetch_google_trends(symbol, start, end)
        raise ValueError(f"Unknown alternative data type: {data_type}")

    # ------------------------------------------------------------------
    # News Sentiment
    # ------------------------------------------------------------------

    def _fetch_news_sentiment(
        self, query: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError("NewsAPI key not configured")

        import requests

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.api_key,
            "pageSize": 100,
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        if not articles:
            return pd.DataFrame()

        rows = []
        for art in articles:
            rows.append(
                {
                    "date": pd.to_datetime(art["publishedAt"]).normalize(),
                    "title": art.get("title", ""),
                    "description": art.get("description", ""),
                    "source": art.get("source", {}).get("name", ""),
                    "sentiment": self._score_sentiment(
                        f"{art.get('title', '')} {art.get('description', '')}"
                    ),
                }
            )

        df = pd.DataFrame(rows).set_index("date").sort_index()
        # Aggregate daily sentiment
        daily = df["sentiment"].resample("D").mean().rename("sentiment_score")
        return daily.to_frame()

    @staticmethod
    def _score_sentiment(text: str) -> float:
        """
        Lightweight rule-based sentiment (-1 bearish … +1 bullish).
        Replace with a proper NLP model (e.g. finBERT) in production.
        """
        positive = {"rise", "gain", "surge", "beat", "strong", "growth", "profit", "up"}
        negative = {"fall", "drop", "miss", "weak", "loss", "decline", "down", "crash"}
        words = set(text.lower().split())
        score = len(words & positive) - len(words & negative)
        return max(-1.0, min(1.0, score / max(len(words), 1) * 10))

    # ------------------------------------------------------------------
    # Real Estate Data (ATTOM)
    # ------------------------------------------------------------------

    def _fetch_real_estate(
        self, address_or_zip: str, **kwargs: Any
    ) -> pd.DataFrame:
        if not self._attom_key:
            raise RuntimeError("ATTOM API key not configured")

        import requests

        url = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/property/basicprofile"
        headers = {"apikey": self._attom_key, "accept": "application/json"}
        params = {"address": address_or_zip}
        resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        resp.raise_for_status()
        properties = resp.json().get("property", [])
        if not properties:
            return pd.DataFrame()
        return pd.json_normalize(properties)

    # ------------------------------------------------------------------
    # Google Trends
    # ------------------------------------------------------------------

    def _fetch_google_trends(
        self, keyword: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        try:
            from pytrends.request import TrendReq
        except ImportError as exc:
            raise RuntimeError("pytrends not installed") from exc

        pytrends = TrendReq(hl="en-US", tz=360)
        timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
        pytrends.build_payload([keyword], timeframe=timeframe)
        df = pytrends.interest_over_time()
        if df.empty:
            return pd.DataFrame()
        df.index.name = "date"
        return df[[keyword]].rename(columns={keyword: "interest"})

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def fetch_property_comps(
        self, zip_code: str, radius_miles: float = 0.5
    ) -> pd.DataFrame:
        """Return comparable property sales for a given ZIP code."""
        return self._fetch_real_estate(zip_code, radius=radius_miles)

    def fetch_sentiment_series(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Bulk-fetch daily sentiment scores for multiple tickers/topics."""
        frames: Dict[str, pd.Series] = {}
        for sym in symbols:
            res = self.fetch(sym, start, end, data_type="sentiment")
            if res.ok:
                frames[sym] = res.data["sentiment_score"]
        return pd.DataFrame(frames) if frames else pd.DataFrame()
