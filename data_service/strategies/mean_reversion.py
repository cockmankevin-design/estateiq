"""
Mean-reversion strategy.

Implements z-score based entry/exit on rolling mean ± k·σ bands.
Suitable for pairs, ETFs, or single-asset oscillation trades.
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Z-score mean-reversion.

    Entry: z < -entry_z  → LONG  (price below lower band)
           z >  entry_z  → SHORT (price above upper band)
    Exit : |z| < exit_z  → EXIT  (price returns to mean)

    Parameters
    ----------
    window : int
        Rolling window for mean / std computation.
    entry_z : float
        Z-score threshold to enter a position.
    exit_z : float
        Z-score threshold to exit a position.
    max_holding_days : int
        Force exit after this many days regardless of z-score.
    use_pairs : bool
        If True, interpret two-column price frame as a spread (col0 - β·col1).
    """

    def __init__(
        self,
        window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_holding_days: int = 10,
        use_pairs: bool = False,
    ):
        super().__init__(
            name="mean_reversion",
            params={
                "window": window,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "max_holding_days": max_holding_days,
                "use_pairs": use_pairs,
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

        use_pairs = self._get_param("use_pairs", False)
        if use_pairs and close.shape[1] >= 2:
            series = self._compute_spread(close.iloc[:, 0], close.iloc[:, 1])
            symbol = f"{close.columns[0]}/{close.columns[1]}"
        else:
            series = close.iloc[:, 0]
            symbol = str(close.columns[0])

        return self._signals_from_series(series, symbol)

    def _signals_from_series(
        self, series: pd.Series, symbol: str
    ) -> List[Signal]:
        window = self._get_param("window", 20)
        entry_z = self._get_param("entry_z", 2.0)
        exit_z = self._get_param("exit_z", 0.5)
        max_hold = self._get_param("max_holding_days", 10)

        roll_mean = series.rolling(window).mean()
        roll_std = series.rolling(window).std().replace(0, np.nan)
        zscore = (series - roll_mean) / roll_std

        signals: List[Signal] = []
        position: Optional[SignalType] = None
        entry_day = 0

        for i, (date, z) in enumerate(zscore.items()):
            if np.isnan(z):
                continue
            price = float(series.loc[date])
            days_held = i - entry_day

            # Force exit on max holding period
            if position is not None and days_held >= max_hold:
                signals.append(
                    Signal(
                        timestamp=date.to_pydatetime(),
                        symbol=symbol,
                        signal_type=SignalType.EXIT,
                        price=price,
                        metadata={"reason": "max_holding_days", "zscore": float(z)},
                    )
                )
                position = None
                continue

            # Exit on mean reversion
            if position is not None and abs(z) < exit_z:
                signals.append(
                    Signal(
                        timestamp=date.to_pydatetime(),
                        symbol=symbol,
                        signal_type=SignalType.EXIT,
                        price=price,
                        metadata={"reason": "mean_reversion", "zscore": float(z)},
                    )
                )
                position = None
                continue

            # Entry signals
            if position is None:
                if z < -entry_z:
                    position = SignalType.LONG
                    entry_day = i
                    signals.append(
                        Signal(
                            timestamp=date.to_pydatetime(),
                            symbol=symbol,
                            signal_type=SignalType.LONG,
                            confidence=min(1.0, abs(z) / (entry_z * 2)),
                            target_weight=1.0,
                            price=price,
                            metadata={"zscore": float(z)},
                        )
                    )
                elif z > entry_z:
                    position = SignalType.SHORT
                    entry_day = i
                    signals.append(
                        Signal(
                            timestamp=date.to_pydatetime(),
                            symbol=symbol,
                            signal_type=SignalType.SHORT,
                            confidence=min(1.0, abs(z) / (entry_z * 2)),
                            target_weight=1.0,
                            price=price,
                            metadata={"zscore": float(z)},
                        )
                    )

        return signals

    # ------------------------------------------------------------------
    # Pairs helpers
    # ------------------------------------------------------------------

    def _compute_spread(
        self, asset1: pd.Series, asset2: pd.Series
    ) -> pd.Series:
        """Compute OLS hedge ratio and return the spread series."""
        beta = self._ols_beta(asset1, asset2)
        spread = asset1 - beta * asset2
        spread.name = "spread"
        return spread

    @staticmethod
    def _ols_beta(y: pd.Series, x: pd.Series) -> float:
        aligned = pd.concat([y, x], axis=1).dropna()
        if aligned.empty:
            return 1.0
        x_vals = aligned.iloc[:, 1].values
        y_vals = aligned.iloc[:, 0].values
        beta = np.cov(y_vals, x_vals)[0, 1] / np.var(x_vals)
        return float(beta)

    def zscore_series(self, prices: pd.DataFrame) -> pd.Series:
        """Public utility — return the rolling z-score for inspection."""
        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
        window = self._get_param("window", 20)
        mu = close.rolling(window).mean()
        sigma = close.rolling(window).std().replace(0, np.nan)
        return ((close - mu) / sigma).rename("zscore")
