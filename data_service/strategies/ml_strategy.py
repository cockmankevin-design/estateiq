"""
ML-driven strategy.

Wraps any scikit-learn compatible model to produce trade signals from
engineered features. Works with classifiers (long/short/flat) or
regressors (predicted return → threshold-based signal).
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Machine-learning strategy bridge.

    Expects pre-computed features (e.g. from FactorModel) and a fitted
    sklearn-compatible estimator. Can also accept a feature builder
    callable for end-to-end predict-at-inference use.

    Parameters
    ----------
    model : sklearn estimator, optional
        A fitted model with a `predict` (or `predict_proba`) method.
    model_path : str | Path, optional
        Path to a pickled model file (loaded if `model` is None).
    feature_builder : callable, optional
        f(prices: DataFrame) → features: DataFrame
    long_threshold : float
        Predicted return / probability above which we go LONG.
    short_threshold : float
        Predicted return / probability below which we go SHORT.
    rebalance_freq : str
        Pandas offset alias for rebalance dates.
    """

    def __init__(
        self,
        model=None,
        model_path: Optional[str] = None,
        feature_builder: Optional[Callable] = None,
        long_threshold: float = 0.02,
        short_threshold: float = -0.02,
        rebalance_freq: str = "W",
    ):
        super().__init__(
            name="ml_strategy",
            params={
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                "rebalance_freq": rebalance_freq,
            },
        )
        self._model = model
        self._model_path = model_path
        self._feature_builder = feature_builder

        if self._model is None and self._model_path:
            self._load_model()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> List[Signal]:
        if self._model is None:
            raise RuntimeError("No model loaded. Pass a model or model_path.")

        if features is None and self._feature_builder is not None:
            features = self._feature_builder(prices)

        if features is None:
            raise ValueError("features must be provided or feature_builder must be set")

        rebalance_freq = self._get_param("rebalance_freq", "W")
        long_thresh = self._get_param("long_threshold", 0.02)
        short_thresh = self._get_param("short_threshold", -0.02)

        rebalance_dates = features.resample(rebalance_freq).last().index
        signals: List[Signal] = []

        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]

        for date in rebalance_dates:
            if date not in features.index:
                continue
            row = features.loc[[date]].copy()
            row = row.fillna(0.0)

            try:
                prediction = self._predict(row)
            except Exception as exc:
                self._logger.warning("Prediction failed at %s: %s", date, exc)
                continue

            symbol = str(close.name) if hasattr(close, "name") else "ASSET"
            price = float(close.loc[date]) if date in close.index else None

            if prediction >= long_thresh:
                signals.append(
                    Signal(
                        timestamp=date.to_pydatetime(),
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        confidence=min(1.0, float(prediction / (long_thresh * 2))),
                        target_weight=1.0,
                        price=price,
                        metadata={"predicted_return": float(prediction)},
                    )
                )
            elif prediction <= short_thresh:
                signals.append(
                    Signal(
                        timestamp=date.to_pydatetime(),
                        symbol=symbol,
                        signal_type=SignalType.SHORT,
                        confidence=min(1.0, float(-prediction / (-short_thresh * 2))),
                        target_weight=1.0,
                        price=price,
                        metadata={"predicted_return": float(prediction)},
                    )
                )
            else:
                signals.append(
                    Signal(
                        timestamp=date.to_pydatetime(),
                        symbol=symbol,
                        signal_type=SignalType.HOLD,
                        confidence=1.0,
                        target_weight=0.0,
                        price=price,
                        metadata={"predicted_return": float(prediction)},
                    )
                )

        return signals

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _predict(self, X: pd.DataFrame) -> float:
        """
        Return a scalar prediction.
        For classifiers with predict_proba, returns P(class=1) - 0.5
        to centre around zero.
        """
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]
            classes = list(self._model.classes_)
            if 1 in classes:
                return float(proba[classes.index(1)]) - 0.5
        return float(self._model.predict(X)[0])

    def _load_model(self) -> None:
        path = Path(self._model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            self._model = pickle.load(fh)
        self._logger.info("Model loaded from %s", path)

    def save_model(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        with open(path, "wb") as fh:
            pickle.dump(self._model, fh)
        self._logger.info("Model saved to %s", path)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLStrategy":
        """Fit the model if it has a fit method (i.e. not yet trained)."""
        if self._model is None:
            raise RuntimeError("No model set")
        self._model.fit(X, y)
        return self
