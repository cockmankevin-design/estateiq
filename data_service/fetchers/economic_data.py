"""
Economic data fetcher.

Sources: FRED (Federal Reserve), World Bank, BLS.
Provides macro indicators: interest rates, inflation, GDP, unemployment, etc.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseFetcher

logger = logging.getLogger(__name__)

# Commonly used FRED series identifiers
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "gdp": "GDP",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "yield_curve": "T10Y2Y",
    "housing_starts": "HOUST",
    "case_shiller": "CSUSHPISA",
    "mortgage_30y": "MORTGAGE30US",
    "sp500": "SP500",
    "vix": "VIXCLS",
}


class EconomicDataFetcher(BaseFetcher):
    """
    Fetches macroeconomic time-series from FRED and other public sources.

    Requires a FRED API key (free at https://fred.stlouisfed.org/docs/api/).
    Falls back to World Bank / BLS for certain indicators.
    """

    def __init__(
        self,
        fred_key: str = "",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        super().__init__(api_key=fred_key, timeout=timeout, max_retries=max_retries)
        self._fred_key = fred_key

    @property
    def source_name(self) -> str:
        return "EconomicData"

    def _fetch_impl(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        # symbol may be a friendly alias or a raw FRED series id
        series_id = FRED_SERIES.get(symbol.lower(), symbol.upper())
        return self._fetch_fred(series_id, start, end)

    # ------------------------------------------------------------------
    # FRED
    # ------------------------------------------------------------------

    def _fetch_fred(
        self, series_id: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        if not self._fred_key:
            raise RuntimeError("FRED API key not configured (set FRED_KEY env var)")

        import requests

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._fred_key,
            "file_type": "json",
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.columns = [series_id.lower()]
        return df.dropna()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def fetch_yield_curve(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return 2y, 10y treasury yields and the spread (10y - 2y)."""
        y2 = self.fetch("treasury_2y", start, end)
        y10 = self.fetch("treasury_10y", start, end)
        if y2.ok and y10.ok:
            df = pd.concat([y2.data, y10.data], axis=1)
            df.columns = ["treasury_2y", "treasury_10y"]
            df["spread"] = df["treasury_10y"] - df["treasury_2y"]
            return df
        return pd.DataFrame()

    def fetch_macro_dashboard(
        self,
        start: datetime,
        end: datetime,
        series: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch a set of macro indicators and return them aligned on a
        monthly frequency.
        """
        if series is None:
            series = ["fed_funds_rate", "cpi", "unemployment", "mortgage_30y"]

        frames: Dict[str, pd.Series] = {}
        for name in series:
            result = self.fetch(name, start, end)
            if result.ok:
                frames[name] = result.data.iloc[:, 0]

        if not frames:
            return pd.DataFrame()

        combined = pd.DataFrame(frames)
        return combined.resample("ME").last().ffill()
