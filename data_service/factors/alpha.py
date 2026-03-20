"""
Alpha factor library.

Classic Fama-French style factors plus momentum, quality,
low-volatility, and real-estate specific factors.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlphaFactors:
    """
    Compute cross-sectional alpha factors from a universe of assets.

    All factor methods accept a `prices` DataFrame (date × symbol, close prices)
    and return a DataFrame of the same shape (factor scores, higher is better).
    """

    # ------------------------------------------------------------------
    # Classic factors
    # ------------------------------------------------------------------

    @staticmethod
    def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
        """12-1 month momentum (Jegadeesh & Titman, 1993)."""
        past = prices.shift(skip)
        return prices / past.shift(lookback - skip) - 1

    @staticmethod
    def value(prices: pd.DataFrame, book_value: pd.DataFrame) -> pd.DataFrame:
        """Book-to-market ratio (Fama & French, 1992)."""
        return book_value / prices

    @staticmethod
    def size(market_cap: pd.DataFrame) -> pd.DataFrame:
        """Log market cap (negative — small-cap premium)."""
        return -np.log(market_cap.clip(lower=1))

    @staticmethod
    def low_volatility(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """Inverse trailing volatility (Blitz & van Vliet, 2007)."""
        daily_ret = prices.pct_change()
        vol = daily_ret.rolling(window).std() * np.sqrt(252)
        return -vol  # lower vol → higher score

    @staticmethod
    def quality(
        roe: pd.DataFrame,
        debt_to_equity: pd.DataFrame,
        gross_margin: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Composite quality score: ROE + gross margin − leverage.
        All inputs are cross-sectional DataFrames (date × symbol).
        """
        def zscore(df):
            mu = df.mean(axis=1)
            sigma = df.std(axis=1)
            return df.sub(mu, axis=0).div(sigma, axis=0)

        q = zscore(roe) + zscore(gross_margin) - zscore(debt_to_equity)
        return q

    @staticmethod
    def mean_reversion(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Short-term reversal (Jegadeesh, 1990) — negative 1-week return."""
        return -(prices / prices.shift(window) - 1)

    # ------------------------------------------------------------------
    # Real-estate specific alpha factors
    # ------------------------------------------------------------------

    @staticmethod
    def cap_rate_momentum(
        noi: pd.DataFrame, price: pd.DataFrame, window: int = 12
    ) -> pd.DataFrame:
        """Change in cap rate (NOI/price) over `window` months."""
        cap_rate = noi / price
        return cap_rate.pct_change(window)

    @staticmethod
    def rental_yield_spread(
        rental_yield: pd.DataFrame, risk_free_rate: pd.Series
    ) -> pd.DataFrame:
        """
        Rental yield minus risk-free rate.
        Higher spread → more attractive income vs. alternatives.
        """
        return rental_yield.sub(risk_free_rate, axis=0)

    @staticmethod
    def appreciation_factor(
        price: pd.DataFrame,
        windows: tuple = (3, 12, 36),
    ) -> pd.DataFrame:
        """Composite of 3m / 12m / 36m price appreciation."""
        factors = [price.pct_change(w) for w in windows]
        return pd.concat(factors, axis=1).mean(axis=1).unstack()

    # ------------------------------------------------------------------
    # Factor normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def cross_section_zscore(factor: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
        """Standardise each row across the cross-section and clip outliers."""
        mu = factor.mean(axis=1)
        sigma = factor.std(axis=1).replace(0, np.nan)
        z = factor.sub(mu, axis=0).div(sigma, axis=0)
        return z.clip(-clip, clip)

    @staticmethod
    def rank_transform(factor: pd.DataFrame) -> pd.DataFrame:
        """Convert factor values to uniform ranks in [0, 1]."""
        return factor.rank(axis=1, pct=True)

    @classmethod
    def neutralise(
        cls,
        factor: pd.DataFrame,
        sector: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Sector-neutralise a factor by demeaning within each sector group.
        `sector` must be a date × symbol DataFrame of sector labels.
        """
        result = factor.copy()
        for date in factor.index:
            row = factor.loc[date].dropna()
            sec = sector.loc[date, row.index] if date in sector.index else None
            if sec is None or sec.isnull().all():
                continue
            for _, group in row.groupby(sec):
                result.loc[date, group.index] = group - group.mean()
        return result
