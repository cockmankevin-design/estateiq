"""
Multi-factor model.

Combines alpha and risk factors into a single framework for:
  - Factor exposure estimation (regression-based)
  - Risk decomposition (factor vs. idiosyncratic)
  - Alpha attribution
  - Portfolio optimisation inputs
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Result of a factor regression for a single asset."""

    symbol: str
    betas: Dict[str, float]             # factor_name → beta
    alpha: float                        # Jensen's alpha (annualised)
    r_squared: float
    idiosyncratic_vol: float            # annualised
    t_stats: Dict[str, float] = field(default_factory=dict)


class FactorModel:
    """
    Barra-style linear factor model.

    Usage
    -----
    >>> fm = FactorModel(factors=["MKT", "SMB", "HML", "WML"])
    >>> fm.fit(factor_returns, asset_returns)
    >>> exposures = fm.exposures()
    >>> risk_decomp = fm.risk_decomposition("AAPL")
    """

    def __init__(
        self,
        factors: Optional[List[str]] = None,
        risk_free_rate: float = 0.04,
        min_obs: int = 60,
    ):
        self.factors = factors or ["MKT", "SMB", "HML", "WML"]
        self.risk_free_rate = risk_free_rate
        self.min_obs = min_obs

        self._factor_returns: Optional[pd.DataFrame] = None
        self._asset_returns: Optional[pd.DataFrame] = None
        self._exposures: Dict[str, FactorExposure] = {}
        self._factor_cov: Optional[pd.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> "FactorModel":
        """
        Estimate factor exposures via OLS for each asset.

        Parameters
        ----------
        factor_returns : date × factor DataFrame
        asset_returns  : date × symbol DataFrame
        """
        # Align on common dates
        common = factor_returns.index.intersection(asset_returns.index)
        F = factor_returns.loc[common, self.factors].dropna()
        R = asset_returns.loc[F.index].dropna(how="all")

        self._factor_returns = F
        self._asset_returns = R
        self._factor_cov = F.cov() * 252      # annualised

        self._exposures = {}
        for symbol in R.columns:
            r = R[symbol].dropna()
            f = F.loc[r.index]
            if len(r) < self.min_obs:
                logger.warning("Skipping %s — insufficient data (%d obs)", symbol, len(r))
                continue
            self._exposures[symbol] = self._fit_asset(symbol, r, f)

        self._fitted = True
        return self

    def _fit_asset(
        self, symbol: str, r: pd.Series, F: pd.DataFrame
    ) -> FactorExposure:
        from numpy.linalg import lstsq

        # Excess returns
        rf_daily = self.risk_free_rate / 252
        y = r.values - rf_daily
        X = np.column_stack([np.ones(len(F)), F.values])

        betas, residuals, rank, _ = lstsq(X, y, rcond=None)
        alpha_daily = betas[0]
        factor_betas = dict(zip(self.factors, betas[1:]))

        fitted = X @ betas
        resid = y - fitted
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # T-statistics
        n, p = len(y), len(betas)
        sigma2 = ss_res / max(n - p, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        t_stats_arr = betas / np.maximum(se, 1e-12)
        t_stats = {"alpha": float(t_stats_arr[0])}
        t_stats.update(dict(zip(self.factors, t_stats_arr[1:].tolist())))

        return FactorExposure(
            symbol=symbol,
            betas=factor_betas,
            alpha=float(alpha_daily * 252),
            r_squared=float(r2),
            idiosyncratic_vol=float(np.std(resid) * np.sqrt(252)),
            t_stats=t_stats,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def exposures(self) -> pd.DataFrame:
        """Return a symbol × factor DataFrame of betas."""
        self._check_fitted()
        rows = {}
        for sym, exp in self._exposures.items():
            row = dict(exp.betas)
            row["alpha"] = exp.alpha
            row["r_squared"] = exp.r_squared
            row["idio_vol"] = exp.idiosyncratic_vol
            rows[sym] = row
        return pd.DataFrame(rows).T

    def risk_decomposition(self, symbol: str) -> Dict[str, float]:
        """
        Decompose total variance into factor and idiosyncratic components.

        Returns a dict with keys:
          total_var, factor_var, idio_var, factor_pct, idio_pct
        """
        self._check_fitted()
        if symbol not in self._exposures:
            raise KeyError(f"No exposure data for {symbol}")

        exp = self._exposures[symbol]
        beta_vec = np.array([exp.betas.get(f, 0.0) for f in self.factors])
        factor_var = float(beta_vec @ self._factor_cov.values @ beta_vec)
        idio_var = exp.idiosyncratic_vol ** 2
        total_var = factor_var + idio_var

        return {
            "total_var": total_var,
            "factor_var": factor_var,
            "idio_var": idio_var,
            "factor_pct": factor_var / total_var if total_var > 0 else 0.0,
            "idio_pct": idio_var / total_var if total_var > 0 else 0.0,
        }

    def alpha_summary(self, min_t_stat: float = 1.96) -> pd.DataFrame:
        """Return assets with statistically significant alpha."""
        self._check_fitted()
        rows = []
        for sym, exp in self._exposures.items():
            rows.append({
                "symbol": sym,
                "alpha": exp.alpha,
                "t_stat": exp.t_stats.get("alpha", 0.0),
                "r_squared": exp.r_squared,
            })
        df = pd.DataFrame(rows).set_index("symbol")
        return df[df["t_stat"].abs() >= min_t_stat].sort_values("alpha", ascending=False)

    def predict_return(self, symbol: str, factor_scenario: Dict[str, float]) -> float:
        """
        Predict expected return under a custom factor scenario.
        `factor_scenario` maps factor name → expected return.
        """
        self._check_fitted()
        if symbol not in self._exposures:
            raise KeyError(symbol)
        exp = self._exposures[symbol]
        pred = exp.alpha / 252  # daily alpha
        for factor, beta in exp.betas.items():
            pred += beta * factor_scenario.get(factor, 0.0)
        return float(pred * 252)  # annualised

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FactorModel not fitted. Call .fit() first.")
