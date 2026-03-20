"""
Performance metrics for backtesting and live monitoring.

Implements the standard quant finance KPI suite:
Sharpe, Sortino, Calmar, max drawdown, alpha/beta, etc.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class PerformanceMetrics:
    """
    Compute a comprehensive set of risk/return metrics from a returns series.

    Parameters
    ----------
    returns : pd.Series         Daily portfolio returns.
    benchmark : pd.Series       Daily benchmark returns (optional).
    risk_free_rate : float      Annualised risk-free rate.
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.04,
    ):
        self.returns = returns.dropna()
        self.benchmark = benchmark.reindex(self.returns.index).fillna(0) if benchmark is not None else None
        self.risk_free_rate = risk_free_rate
        self._rf_daily = risk_free_rate / TRADING_DAYS

    def compute(self) -> Dict[str, float]:
        """Return the full metrics dictionary."""
        r = self.returns
        metrics: Dict[str, float] = {}

        # Return
        metrics["total_return"] = self._total_return(r)
        metrics["annualised_return"] = self._annualised_return(r)
        metrics["cagr"] = self._cagr(r)

        # Risk
        metrics["annualised_vol"] = self._annualised_vol(r)
        metrics["downside_vol"] = self._downside_vol(r)
        metrics["max_drawdown"] = self._max_drawdown(r)
        metrics["avg_drawdown"] = self._avg_drawdown(r)
        metrics["var_95"] = self._var(r, 0.05)
        metrics["cvar_95"] = self._cvar(r, 0.05)
        metrics["skewness"] = float(r.skew())
        metrics["kurtosis"] = float(r.kurtosis())

        # Risk-adjusted
        metrics["sharpe"] = self._sharpe(r)
        metrics["sortino"] = self._sortino(r)
        metrics["calmar"] = self._calmar(r)
        metrics["omega"] = self._omega(r)

        # Win/loss
        metrics["win_rate"] = float((r > 0).mean())
        metrics["profit_factor"] = self._profit_factor(r)
        metrics["avg_win"] = float(r[r > 0].mean()) if (r > 0).any() else 0.0
        metrics["avg_loss"] = float(r[r < 0].mean()) if (r < 0).any() else 0.0

        # Benchmark-relative (if provided)
        if self.benchmark is not None:
            metrics["alpha"] = self._alpha(r, self.benchmark)
            metrics["beta"] = self._beta(r, self.benchmark)
            metrics["information_ratio"] = self._information_ratio(r, self.benchmark)
            metrics["tracking_error"] = self._tracking_error(r, self.benchmark)
            metrics["up_capture"] = self._capture(r, self.benchmark, up=True)
            metrics["down_capture"] = self._capture(r, self.benchmark, up=False)

        # Streak
        metrics["max_consec_wins"] = self._max_consecutive(r, positive=True)
        metrics["max_consec_losses"] = self._max_consecutive(r, positive=False)

        return {k: round(float(v), 6) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _total_return(r: pd.Series) -> float:
        return float((1 + r).prod() - 1)

    @staticmethod
    def _annualised_return(r: pd.Series) -> float:
        n = len(r)
        return float((1 + r).prod() ** (TRADING_DAYS / max(n, 1)) - 1)

    @staticmethod
    def _cagr(r: pd.Series) -> float:
        n_years = len(r) / TRADING_DAYS
        total = (1 + r).prod()
        return float(total ** (1 / max(n_years, 1e-6)) - 1)

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _annualised_vol(r: pd.Series) -> float:
        return float(r.std() * np.sqrt(TRADING_DAYS))

    def _downside_vol(self, r: pd.Series) -> float:
        downside = r[r < self._rf_daily] - self._rf_daily
        return float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0

    @staticmethod
    def _max_drawdown(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    @staticmethod
    def _avg_drawdown(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    @staticmethod
    def _var(r: pd.Series, alpha: float) -> float:
        return float(np.percentile(r, alpha * 100))

    @staticmethod
    def _cvar(r: pd.Series, alpha: float) -> float:
        cutoff = np.percentile(r, alpha * 100)
        return float(r[r <= cutoff].mean())

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    def _sharpe(self, r: pd.Series) -> float:
        excess = r - self._rf_daily
        std = excess.std()
        return float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    def _sortino(self, r: pd.Series) -> float:
        excess = r - self._rf_daily
        dv = self._downside_vol(r)
        ann_excess = excess.mean() * TRADING_DAYS
        return float(ann_excess / dv) if dv > 0 else 0.0

    def _calmar(self, r: pd.Series) -> float:
        cagr = self._cagr(r)
        mdd = abs(self._max_drawdown(r))
        return float(cagr / mdd) if mdd > 0 else 0.0

    @staticmethod
    def _omega(r: pd.Series, threshold: float = 0.0) -> float:
        gains = r[r > threshold].sum()
        losses = -r[r <= threshold].sum()
        return float(gains / losses) if losses > 0 else np.inf

    # ------------------------------------------------------------------
    # Win/loss metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _profit_factor(r: pd.Series) -> float:
        gross_win = r[r > 0].sum()
        gross_loss = -r[r < 0].sum()
        return float(gross_win / gross_loss) if gross_loss > 0 else np.inf

    @staticmethod
    def _max_consecutive(r: pd.Series, positive: bool) -> float:
        mask = (r > 0) if positive else (r < 0)
        max_streak, current = 0, 0
        for val in mask:
            current = current + 1 if val else 0
            max_streak = max(max_streak, current)
        return float(max_streak)

    # ------------------------------------------------------------------
    # Benchmark-relative
    # ------------------------------------------------------------------

    @staticmethod
    def _beta(r: pd.Series, b: pd.Series) -> float:
        cov = np.cov(r, b)
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0

    def _alpha(self, r: pd.Series, b: pd.Series) -> float:
        beta = self._beta(r, b)
        alpha_daily = r.mean() - self._rf_daily - beta * (b.mean() - self._rf_daily)
        return float(alpha_daily * TRADING_DAYS)

    @staticmethod
    def _information_ratio(r: pd.Series, b: pd.Series) -> float:
        active = r - b
        return float(active.mean() / active.std() * np.sqrt(TRADING_DAYS)) if active.std() > 0 else 0.0

    @staticmethod
    def _tracking_error(r: pd.Series, b: pd.Series) -> float:
        return float((r - b).std() * np.sqrt(TRADING_DAYS))

    @staticmethod
    def _capture(r: pd.Series, b: pd.Series, up: bool) -> float:
        mask = b > 0 if up else b < 0
        if not mask.any():
            return 0.0
        return float(r[mask].mean() / b[mask].mean()) if b[mask].mean() != 0 else 0.0
