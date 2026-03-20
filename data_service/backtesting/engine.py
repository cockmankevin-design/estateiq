"""
Event-driven backtesting engine.

Simulates strategy execution bar-by-bar with realistic transaction
costs, slippage, and position sizing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single executed trade."""

    date: datetime
    symbol: str
    side: str            # "buy" | "sell"
    quantity: float
    price: float
    commission: float
    slippage: float

    @property
    def gross_value(self) -> float:
        return self.quantity * self.price

    @property
    def net_cost(self) -> float:
        return self.gross_value + self.commission + self.slippage * self.gross_value


@dataclass
class BacktestResult:
    """Full output of a backtest run."""

    equity_curve: pd.Series          # portfolio value over time
    returns: pd.Series               # daily portfolio returns
    trades: List[Trade]
    positions: pd.DataFrame          # date × symbol weights
    benchmark_returns: Optional[pd.Series]
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    Vectorised + event-driven backtesting engine.

    Parameters
    ----------
    initial_capital : float
    commission_rate : float   (e.g. 0.001 = 10 bps)
    slippage_bps : float      (e.g. 5 = 5 bps)
    risk_free_rate : float    (annualised)
    benchmark : pd.Series     (optional benchmark returns)
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        risk_free_rate: float = 0.04,
        benchmark: Optional[pd.Series] = None,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage_bps / 10_000
        self.risk_free_rate = risk_free_rate
        self.benchmark = benchmark

    def run(
        self,
        strategy: BaseStrategy,
        prices: pd.DataFrame,
        rebalance_freq: str = "ME",
        **strategy_kwargs: Any,
    ) -> BacktestResult:
        """
        Run a strategy backtest.

        Parameters
        ----------
        strategy : BaseStrategy
        prices : pd.DataFrame
            OHLCV (at minimum 'close' column required).
        rebalance_freq : str
            Rebalance frequency alias (e.g. 'ME', 'W', 'Q').
        """
        logger.info(
            "Running backtest: %s | capital=%.0f | %s→%s",
            strategy.name,
            self.initial_capital,
            prices.index[0].date(),
            prices.index[-1].date(),
        )

        # Generate signals
        result = strategy.run(prices, **strategy_kwargs)
        if not result.ok:
            raise RuntimeError(f"Strategy failed: {result.errors}")

        # Build target weights time series
        target_weights = self._build_target_weights(result.positions, prices)

        # Simulate portfolio
        equity, returns, trades, positions = self._simulate(prices, target_weights)

        # Compute performance metrics
        from .performance import PerformanceMetrics
        metrics = PerformanceMetrics(
            returns=returns,
            benchmark=self.benchmark,
            risk_free_rate=self.risk_free_rate,
        ).compute()

        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            trades=trades,
            positions=positions,
            benchmark_returns=self.benchmark,
            metrics=metrics,
            diagnostics={"strategy": strategy.name, "signals": len(result.signals)},
        )

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------

    def _simulate(
        self,
        prices: pd.DataFrame,
        target_weights: pd.DataFrame,
    ):
        close = prices["close"] if "close" in prices.columns else prices
        if isinstance(close, pd.Series):
            close = close.to_frame()

        # Align weights with price dates
        weights = target_weights.reindex(close.index).ffill().fillna(0.0)
        weights = weights.reindex(columns=close.columns, fill_value=0.0)

        equity = pd.Series(index=close.index, dtype=float)
        returns = pd.Series(index=close.index, dtype=float)
        trades: List[Trade] = []
        positions_hist: List[pd.Series] = []

        capital = self.initial_capital
        prev_weights = pd.Series(0.0, index=close.columns)
        equity.iloc[0] = capital
        returns.iloc[0] = 0.0
        positions_hist.append(prev_weights.copy())

        for i in range(1, len(close)):
            date = close.index[i]
            prev_date = close.index[i - 1]

            # Price returns
            px_ret = (close.iloc[i] / close.iloc[i - 1] - 1).fillna(0.0)

            # Portfolio return (before rebalancing)
            port_ret = float((prev_weights * px_ret).sum())
            capital = capital * (1 + port_ret)

            # Rebalancing
            new_weights = weights.iloc[i]
            delta = new_weights - prev_weights
            turnover = delta.abs().sum()

            if turnover > 1e-6:
                commission = capital * turnover * self.commission_rate
                slippage_cost = capital * turnover * self.slippage
                capital -= commission + slippage_cost

                # Record trades
                for sym in close.columns:
                    if abs(delta[sym]) > 1e-6:
                        px = float(close.iloc[i][sym])
                        qty = abs(delta[sym]) * capital / max(px, 1e-8)
                        trades.append(
                            Trade(
                                date=date.to_pydatetime(),
                                symbol=sym,
                                side="buy" if delta[sym] > 0 else "sell",
                                quantity=qty,
                                price=px,
                                commission=commission / max(len(close.columns), 1),
                                slippage=self.slippage,
                            )
                        )

            equity.iloc[i] = capital
            returns.iloc[i] = capital / equity.iloc[i - 1] - 1
            prev_weights = new_weights.copy()
            positions_hist.append(prev_weights.copy())

        positions_df = pd.DataFrame(positions_hist, index=close.index)
        return equity, returns, trades, positions_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_target_weights(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Forward-fill strategy positions onto the full price date range."""
        if positions.empty:
            return pd.DataFrame(0.0, index=prices.index, columns=["CASH"])
        weights = positions.reindex(prices.index, method="ffill").fillna(0.0)
        # Normalise rows so weights sum to at most 1
        row_sum = weights.abs().sum(axis=1).replace(0, 1)
        weights = weights.div(row_sum.clip(lower=1), axis=0)
        return weights
