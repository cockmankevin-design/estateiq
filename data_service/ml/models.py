"""
Model registry and factory.

Provides a unified interface for creating and managing
XGBoost, LightGBM, Random Forest, and LSTM models.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    RIDGE = "ridge"
    LSTM = "lstm"


class ModelRegistry:
    """
    Factory and registry for ML models.

    Usage
    -----
    >>> reg = ModelRegistry()
    >>> model = reg.build(ModelType.XGBOOST, task="regression", n_estimators=300)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """

    _instances: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def build(
        self,
        model_type: ModelType,
        task: str = "regression",
        scale_features: bool = True,
        **hyperparams: Any,
    ) -> Pipeline:
        """
        Build a scikit-learn Pipeline containing an optional scaler
        and the requested estimator.

        Parameters
        ----------
        model_type : ModelType
        task : "regression" | "classification"
        scale_features : bool
            Wrap the estimator with StandardScaler.
        **hyperparams : forwarded to the underlying estimator.
        """
        estimator = self._build_estimator(model_type, task, **hyperparams)
        steps = []
        if scale_features and model_type not in (
            ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.RANDOM_FOREST
        ):
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        pipeline = Pipeline(steps)
        return pipeline

    def _build_estimator(self, model_type: ModelType, task: str, **hp: Any):
        if model_type == ModelType.XGBOOST:
            return self._xgboost(task, **hp)
        if model_type == ModelType.LIGHTGBM:
            return self._lightgbm(task, **hp)
        if model_type == ModelType.RANDOM_FOREST:
            return self._random_forest(task, **hp)
        if model_type == ModelType.RIDGE:
            return self._ridge(task, **hp)
        if model_type == ModelType.LSTM:
            return self._lstm_wrapper(task, **hp)
        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # Estimator builders
    # ------------------------------------------------------------------

    @staticmethod
    def _xgboost(task: str, **hp):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost not installed") from exc
        defaults = dict(n_estimators=200, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
        defaults.update(hp)
        if task == "classification":
            return xgb.XGBClassifier(**defaults, eval_metric="logloss",
                                      use_label_encoder=False)
        return xgb.XGBRegressor(**defaults)

    @staticmethod
    def _lightgbm(task: str, **hp):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm not installed") from exc
        defaults = dict(n_estimators=200, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        verbose=-1)
        defaults.update(hp)
        if task == "classification":
            return lgb.LGBMClassifier(**defaults)
        return lgb.LGBMRegressor(**defaults)

    @staticmethod
    def _random_forest(task: str, **hp):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        defaults = dict(n_estimators=200, max_depth=10, min_samples_leaf=5,
                        n_jobs=-1, random_state=42)
        defaults.update(hp)
        if task == "classification":
            return RandomForestClassifier(**defaults)
        return RandomForestRegressor(**defaults)

    @staticmethod
    def _ridge(task: str, **hp):
        from sklearn.linear_model import LogisticRegression, Ridge
        defaults = {"alpha": 1.0}
        defaults.update(hp)
        if task == "classification":
            return LogisticRegression(**{"C": 1.0, "max_iter": 1000, **hp})
        return Ridge(**defaults)

    @staticmethod
    def _lstm_wrapper(task: str, **hp):
        """Return a scikit-learn compatible LSTM wrapper (requires torch)."""
        return _LSTMWrapper(task=task, **hp)

    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------

    @staticmethod
    def cross_validate_ts(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        metric: str = "neg_mean_squared_error",
    ) -> Tuple[float, float]:
        """Time-series aware cross-validation. Returns (mean, std) of scores."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring=metric, n_jobs=-1)
        return float(scores.mean()), float(scores.std())


# ---------------------------------------------------------------------------
# Lightweight LSTM wrapper (sklearn-compatible)
# ---------------------------------------------------------------------------

class _LSTMWrapper:
    """
    Minimal sklearn-compatible wrapper around a PyTorch LSTM.
    Expects 2-D feature input; handles sequence creation internally.
    """

    def __init__(
        self,
        task: str = "regression",
        seq_len: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        self.task = task
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._net = None

    def fit(self, X, y):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("PyTorch not installed") from exc

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)

        # Build sequences
        Xs, ys = [], []
        for i in range(self.seq_len, len(X_arr)):
            Xs.append(X_arr[i - self.seq_len: i])
            ys.append(y_arr[i])
        if not Xs:
            raise ValueError("Not enough data for LSTM sequences")
        Xt = torch.tensor(np.array(Xs))
        yt = torch.tensor(np.array(ys)).unsqueeze(-1)

        input_size = Xt.shape[-1]
        self._net = nn.LSTM(
            input_size, self.hidden_size, self.num_layers,
            batch_first=True, dropout=self.dropout
        )
        self._fc = nn.Linear(self.hidden_size, 1)
        params = list(self._net.parameters()) + list(self._fc.parameters())
        optim = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(Xt, yt)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self._net.train()
        self._fc.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                out, _ = self._net(xb)
                pred = self._fc(out[:, -1, :])
                loss = loss_fn(pred, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()
        return self

    def predict(self, X):
        import torch

        X_arr = np.array(X, dtype=np.float32)
        # Pad if shorter than seq_len
        if len(X_arr) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X_arr), X_arr.shape[1]), dtype=np.float32)
            X_arr = np.vstack([pad, X_arr])
        seq = torch.tensor(X_arr[-self.seq_len:]).unsqueeze(0)
        self._net.eval()
        self._fc.eval()
        with torch.no_grad():
            out, _ = self._net(seq)
            pred = self._fc(out[:, -1, :])
        return pred.numpy().flatten()
