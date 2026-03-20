"""
Momentum strategy.

Ranks assets by recent return, goes long top-N, short bottom-N.
Implements classic 12-1 cross-sectional momentum with optional
time-series momentum overlay.
"""

import logging
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Cross-sectional momentum (Jegadeesh & Titman, 1993).

    Parameters
    ----------
    lookback : int
        Formation period in trading days (default 252 − 21 = ~11 months).
    skip : int
        Days to skip at the end of the formation period (default 21 = 1 month).
    n_long : int
        Number of top-momentum assets to hold long.
    n_short : int
        Number of bottom-momentum assets to hold short (0 = long-only).
    rebalance_freq : str
        Pandas offset alias for rebalance dates (e.g. 'ME', 'W', 'Q').
    vol_scale : bool
        If True, scale weights by inverse volatility.
    vol_lookback : int
        Lookback window for volatility scaling.
    """

    def __init__(
        self,
        lookback: int = 231,
        skip: int = 21,
        n_long: int = 5,
        n_short: int = 0,
        rebalance_freq: str = "ME",
        vol_scale: bool = True,
        vol_lookback: int = 63,
    ):
        super().__init__(
            name="momentum",
            params={
                "lookback": lookback,
                "skip": skip,
                "n_long": n_long,
                "n_short": n_short,
                "rebalance_freq": rebalance_freq,
                "vol_scale": vol_scale,
                "vol_lookback": vol_lookback,
            },
        )

    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> List[Signal]:
        close = prices["close"] if "close" in prices.columns else prices
        if isinstance(close, pd.Series):
            close = close.to_frame()

        lookback = self._get_param("lookback", 231)
        skip = self._get_param("skip", 21)
        n_long = self._get_param("n_long", 5)
        n_short = self._get_param("n_short", 0)
        rebalance_freq = self._get_param("rebalance_freq", "ME")
        vol_scale = self._get_param("vol_scale", True)
        vol_lookback = self._get_param("vol_lookback", 63)

        returns = close.pct_change()
        rebalance_dates = close.resample(rebalance_freq).last().index

        signals: List[Signal] = []

        for date in rebalance_dates:
            if date not in close.index:
                continue
            loc = close.index.get_loc(date)
            if loc < lookback:
                continue

            # Formation window: [loc-lookback, loc-skip]
            form_start = loc - lookback
            form_end = loc - skip
            if form_end <= form_start:
                continue

            form_returns = close.iloc[form_end] / close.iloc[form_start] - 1
            ranked = form_returns.dropna().sort_values(ascending=False)

            # Volatility scaling weights
            if vol_scale and loc >= vol_lookback:
                vol = returns.iloc[loc - vol_lookback: loc].std()
                vol = vol.replace(0, np.nan).fillna(1.0)
                inv_vol = 1.0 / vol
                inv_vol /= inv_vol.sum()
            else:
                n = len(ranked)
                inv_vol = pd.Series(1.0 / n, index=ranked.index)

            # Long signals
            for sym in ranked.index[:n_long]:
                signals.append(
                    Signal(
                        timestamp=date.to_pydatetime(),
                        symbol=sym,
                        signal_type=SignalType.LONG,
                        confidence=float(np.clip(form_returns[sym], 0, 1)),
                        target_weight=float(inv_vol.get(sym, 1.0 / n_long)),
                        price=float(close.loc[date, sym]),
                        metadata={"momentum_return": float(form_returns[sym])},
                    )
                )

            # Short signals (optional)
            for sym in ranked.index[-n_short:] if n_short > 0 else []:
                signals.append(
                    Signal(
                        timestamp=date.to_pydatetime(),
                        symbol=sym,
                        signal_type=SignalType.SHORT,
                        confidence=float(np.clip(-form_returns[sym], 0, 1)),
                        target_weight=float(inv_vol.get(sym, 1.0 / n_short)),
                        price=float(close.loc[date, sym]),
                        metadata={"momentum_return": float(form_returns[sym])},
                    )
                )

        return signals

    # ------------------------------------------------------------------
    # Time-series momentum overlay
    # ------------------------------------------------------------------

    def time_series_signal(
        self, prices: pd.Series, short_window: int = 20, long_window: int = 200
    ) -> pd.Series:
        """
        Simple dual-moving-average crossover.
        Returns a series of +1 (long), -1 (short), 0 (flat).
        """
        sma_short = prices.rolling(short_window).mean()
        sma_long = prices.rolling(long_window).mean()
        signal = np.sign(sma_short - sma_long)
        return pd.Series(signal, index=prices.index, name="ts_momentum")
