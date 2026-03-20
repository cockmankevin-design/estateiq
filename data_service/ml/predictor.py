"""
End-to-end prediction pipeline.

Orchestrates feature engineering → model training → inference
for both batch (historical) and online (live bar) prediction.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_engineering import FeatureEngineer
from .models import ModelRegistry, ModelType

logger = logging.getLogger(__name__)


class Predictor:
    """
    High-level prediction pipeline.

    Typical workflow
    ----------------
    1. predictor = Predictor(ModelType.XGBOOST)
    2. predictor.fit(prices, target="next_1d_return")
    3. predictions = predictor.predict(new_prices)

    Parameters
    ----------
    model_type : ModelType
        Which model to use.
    task : "regression" | "classification"
    forward_window : int
        Target lookahead in days (e.g. 1 = next-day return).
    feature_windows : list[int]
        Rolling windows for feature construction.
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.XGBOOST,
        task: str = "regression",
        forward_window: int = 1,
        feature_windows: Optional[List[int]] = None,
        train_ratio: float = 0.8,
    ):
        self.model_type = model_type
        self.task = task
        self.forward_window = forward_window
        self.train_ratio = train_ratio

        self._fe = FeatureEngineer(windows=feature_windows or [5, 10, 20, 60])
        self._registry = ModelRegistry()
        self._model = None
        self._feature_cols: List[str] = []
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        **model_kwargs: Any,
    ) -> "Predictor":
        """Fit the full pipeline on historical price data."""
        X, y = self._prepare_dataset(prices, macro)
        if X.empty or len(y) < 10:
            raise ValueError("Insufficient data for training")

        split = int(len(X) * self.train_ratio)
        X_train, y_train = X.iloc[:split], y.iloc[:split]

        self._model = self._registry.build(
            self.model_type, task=self.task, **model_kwargs
        )
        self._model.fit(X_train, y_train)
        self._feature_cols = list(X.columns)
        self._trained = True

        # Validation metrics
        X_val, y_val = X.iloc[split:], y.iloc[split:]
        if not X_val.empty:
            self._log_val_metrics(X_val, y_val)

        return self

    def _prepare_dataset(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        features = self._fe.build(prices, macro=macro)
        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
        target = close.pct_change(self.forward_window).shift(-self.forward_window)
        if self.task == "classification":
            target = (target > 0).astype(int)
        aligned = features.join(target.rename("target")).dropna()
        X = aligned.drop(columns=["target"])
        y = aligned["target"]
        return X, y

    def _log_val_metrics(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        preds = self._model.predict(X_val)
        if self.task == "regression":
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            logger.info("Validation — MAE: %.4f | R²: %.4f", mae, r2)
        else:
            from sklearn.metrics import accuracy_score, roc_auc_score
            acc = accuracy_score(y_val, preds)
            try:
                auc = roc_auc_score(y_val, self._model.predict_proba(X_val)[:, 1])
            except Exception:
                auc = float("nan")
            logger.info("Validation — Accuracy: %.4f | AUC: %.4f", acc, auc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        as_signal: bool = False,
    ) -> pd.Series:
        """
        Run inference on new price data.

        Parameters
        ----------
        prices : DataFrame
            Recent OHLCV data (needs enough history for feature windows).
        as_signal : bool
            If True, return +1/0/-1 instead of raw prediction.

        Returns
        -------
        pd.Series indexed by date
        """
        if not self._trained or self._model is None:
            raise RuntimeError("Predictor not fitted. Call .fit() first.")

        features = self._fe.build(prices, macro=macro)
        features = features.reindex(columns=self._feature_cols, fill_value=0.0).fillna(0.0)
        features = features.dropna()

        preds = self._model.predict(features)
        result = pd.Series(preds, index=features.index, name="prediction")

        if as_signal:
            result = np.sign(result).astype(int)

        return result

    def predict_latest(self, prices: pd.DataFrame) -> float:
        """Return the single most-recent prediction scalar."""
        preds = self.predict(prices)
        if preds.empty:
            raise RuntimeError("No predictions generated")
        return float(preds.iloc[-1])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import pickle
        state = {
            "model": self._model,
            "feature_cols": self._feature_cols,
            "config": {
                "model_type": self.model_type,
                "task": self.task,
                "forward_window": self.forward_window,
                "train_ratio": self.train_ratio,
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        logger.info("Predictor saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "Predictor":
        import pickle
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        cfg = state["config"]
        p = cls(
            model_type=cfg["model_type"],
            task=cfg["task"],
            forward_window=cfg["forward_window"],
            train_ratio=cfg["train_ratio"],
        )
        p._model = state["model"]
        p._feature_cols = state["feature_cols"]
        p._trained = True
        return p
