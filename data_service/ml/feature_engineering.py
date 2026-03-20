"""
Feature engineering pipeline.

Generates technical indicators, return-based features, and
cross-sectional rank features from raw OHLCV data.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Build a feature matrix from raw price / volume data.

    Usage
    -----
    >>> fe = FeatureEngineer(windows=[5, 10, 20, 60])
    >>> features = fe.build(prices)          # prices: OHLCV DataFrame
    """

    def __init__(
        self,
        windows: List[int] = None,
        include_volume: bool = True,
        include_macro: bool = False,
        lag_features: int = 5,
    ):
        self.windows = windows or [5, 10, 20, 60, 120]
        self.include_volume = include_volume
        self.include_macro = include_macro
        self.lag_features = lag_features

    def build(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build and return the full feature matrix."""
        feats: Dict[str, pd.Series] = {}

        close = self._get_col(prices, "close")
        high = self._get_col(prices, "high")
        low = self._get_col(prices, "low")
        volume = self._get_col(prices, "volume") if self.include_volume else None

        # Returns
        feats.update(self._return_features(close))

        # Moving averages & deviations
        feats.update(self._ma_features(close))

        # Volatility
        feats.update(self._volatility_features(close))

        # Momentum oscillators
        feats.update(self._momentum_features(close))

        # Volume features
        if volume is not None:
            feats.update(self._volume_features(close, volume))

        # Price range features (require high / low)
        if high is not None and low is not None:
            feats.update(self._range_features(high, low, close))

        # Lags of returns
        feats.update(self._lag_features(close))

        # Macro overlay
        if self.include_macro and macro is not None:
            for col in macro.columns:
                aligned = macro[col].reindex(close.index, method="ffill")
                feats[f"macro_{col}"] = aligned

        df = pd.DataFrame(feats, index=close.index)
        return df.dropna(how="all")

    # ------------------------------------------------------------------
    # Feature families
    # ------------------------------------------------------------------

    def _return_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        feats = {}
        for w in self.windows:
            feats[f"ret_{w}d"] = close.pct_change(w)
        feats["ret_1d"] = close.pct_change(1)
        return feats

    def _ma_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        feats = {}
        for w in self.windows:
            sma = close.rolling(w).mean()
            feats[f"sma_{w}"] = sma
            feats[f"close_over_sma_{w}"] = close / sma - 1
            ema = close.ewm(span=w, adjust=False).mean()
            feats[f"ema_{w}"] = ema
            feats[f"close_over_ema_{w}"] = close / ema - 1
        return feats

    def _volatility_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        feats = {}
        log_ret = np.log(close / close.shift(1))
        for w in self.windows:
            feats[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
        feats["parkinson_vol_20"] = self._parkinson_vol(close, 20)
        return feats

    def _momentum_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        feats = {}
        # RSI
        feats["rsi_14"] = self._rsi(close, 14)
        feats["rsi_28"] = self._rsi(close, 28)
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        feats["macd"] = macd
        feats["macd_signal"] = signal
        feats["macd_hist"] = macd - signal
        return feats

    def _volume_features(
        self, close: pd.Series, volume: pd.Series
    ) -> Dict[str, pd.Series]:
        feats = {}
        for w in self.windows:
            vol_sma = volume.rolling(w).mean()
            feats[f"vol_ratio_{w}d"] = volume / vol_sma
        feats["obv"] = (np.sign(close.diff()) * volume).cumsum()
        feats["vwap_20"] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        feats["close_over_vwap"] = close / feats["vwap_20"] - 1
        return feats

    def _range_features(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Dict[str, pd.Series]:
        feats = {}
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        feats["atr_14"] = tr.rolling(14).mean()
        feats["hl_range"] = (high - low) / close
        feats["close_position"] = (close - low) / (high - low).replace(0, np.nan)
        return feats

    def _lag_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        feats = {}
        ret = close.pct_change(1)
        for lag in range(1, self.lag_features + 1):
            feats[f"ret_lag_{lag}"] = ret.shift(lag)
        return feats

    # ------------------------------------------------------------------
    # Technical indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _parkinson_vol(close: pd.Series, window: int) -> pd.Series:
        log_hl = np.log(close / close.shift(1)) ** 2
        return (log_hl.rolling(window).mean() * 252) ** 0.5

    @staticmethod
    def _get_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
        if col in df.columns:
            return df[col]
        if col.capitalize() in df.columns:
            return df[col.capitalize()]
        return None
