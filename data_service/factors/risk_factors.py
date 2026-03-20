"""
Risk factor library.

Implements Fama-French 3/5 factor construction, Carhart 4-factor,
macro risk factors, and PCA-based statistical risk factors.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskFactors:
    """
    Builds systematic risk factors used in risk models and attribution.

    Attributes are typically market-wide (not cross-sectional) time series.
    """

    # ------------------------------------------------------------------
    # Fama-French style
    # ------------------------------------------------------------------

    @staticmethod
    def market_factor(
        universe_returns: pd.DataFrame,
        risk_free_rate: pd.Series,
        weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Market excess return (MKT-RF).
        `universe_returns`: date × symbol DataFrame of daily returns.
        """
        mkt = (
            universe_returns.mul(weights, axis=1).sum(axis=1)
            if weights is not None
            else universe_returns.mean(axis=1)
        )
        rf_aligned = risk_free_rate.reindex(mkt.index, method="ffill").fillna(0) / 252
        return (mkt - rf_aligned).rename("MKT")

    @staticmethod
    def smb(
        returns: pd.DataFrame,
        market_cap: pd.DataFrame,
    ) -> pd.Series:
        """Small-Minus-Big (size) factor."""
        median_cap = market_cap.median(axis=1)
        small_mask = market_cap.lt(median_cap, axis=0)
        big_mask = ~small_mask
        smb = (
            returns[small_mask].mean(axis=1) - returns[big_mask].mean(axis=1)
        )
        return smb.rename("SMB")

    @staticmethod
    def hml(
        returns: pd.DataFrame,
        book_to_market: pd.DataFrame,
        pct_hi: float = 0.30,
        pct_lo: float = 0.30,
    ) -> pd.Series:
        """High-Minus-Low (value) factor."""
        hi_thresh = book_to_market.quantile(1 - pct_hi, axis=1)
        lo_thresh = book_to_market.quantile(pct_lo, axis=1)
        hi_mask = book_to_market.gt(hi_thresh, axis=0)
        lo_mask = book_to_market.lt(lo_thresh, axis=0)
        hml = returns[hi_mask].mean(axis=1) - returns[lo_mask].mean(axis=1)
        return hml.rename("HML")

    @staticmethod
    def wml(
        returns: pd.DataFrame,
        momentum: pd.DataFrame,
        pct: float = 0.30,
    ) -> pd.Series:
        """Winners-Minus-Losers (momentum) factor (Carhart, 1997)."""
        win_thresh = momentum.quantile(1 - pct, axis=1)
        los_thresh = momentum.quantile(pct, axis=1)
        win_mask = momentum.gt(win_thresh, axis=0)
        los_mask = momentum.lt(los_thresh, axis=0)
        wml = returns[win_mask].mean(axis=1) - returns[los_mask].mean(axis=1)
        return wml.rename("WML")

    # ------------------------------------------------------------------
    # Macro risk factors
    # ------------------------------------------------------------------

    @staticmethod
    def interest_rate_factor(treasury_returns: pd.Series) -> pd.Series:
        """Changes in 10-year Treasury yield."""
        return treasury_returns.diff().rename("RATE")

    @staticmethod
    def credit_factor(ig_returns: pd.Series, treasury_returns: pd.Series) -> pd.Series:
        """IG credit spread change (investment-grade OAS minus treasury)."""
        return (ig_returns - treasury_returns).rename("CREDIT")

    @staticmethod
    def inflation_factor(cpi_series: pd.Series) -> pd.Series:
        """Realised monthly inflation (YoY CPI change)."""
        return cpi_series.pct_change(12).rename("INFLATION")

    # ------------------------------------------------------------------
    # Statistical risk factors (PCA)
    # ------------------------------------------------------------------

    @staticmethod
    def pca_factors(
        returns: pd.DataFrame,
        n_components: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract latent risk factors via PCA.

        Returns
        -------
        factors : date × n_components DataFrame of factor returns
        loadings : symbol × n_components DataFrame of factor loadings
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        clean = returns.dropna(axis=1, how="any")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(clean)

        pca = PCA(n_components=n_components)
        factor_returns = pca.fit_transform(scaled)
        cols = [f"PC{i+1}" for i in range(n_components)]
        factors_df = pd.DataFrame(factor_returns, index=clean.index, columns=cols)
        loadings_df = pd.DataFrame(pca.components_.T, index=clean.columns, columns=cols)

        logger.info(
            "PCA explained variance: %s",
            [f"{v:.2%}" for v in pca.explained_variance_ratio_],
        )
        return factors_df, loadings_df

    # ------------------------------------------------------------------
    # Factor covariance
    # ------------------------------------------------------------------

    @staticmethod
    def factor_covariance(
        factor_returns: pd.DataFrame,
        method: str = "sample",
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute factor covariance matrix.

        method : "sample" | "ewm" | "shrinkage"
        """
        if method == "sample":
            cov = factor_returns.cov()
        elif method == "ewm":
            if window is None:
                window = 63
            cov = factor_returns.ewm(halflife=window // 2).cov().iloc[-len(factor_returns.columns):]
        elif method == "shrinkage":
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(factor_returns.dropna())
            cov = pd.DataFrame(
                lw.covariance_,
                index=factor_returns.columns,
                columns=factor_returns.columns,
            )
        else:
            raise ValueError(f"Unknown covariance method: {method}")
        return cov
