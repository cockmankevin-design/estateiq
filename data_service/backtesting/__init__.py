"""Backtesting engine — event-driven simulation, performance analytics, reporting."""

from .engine import BacktestEngine, BacktestResult
from .performance import PerformanceMetrics
from .report import BacktestReport

__all__ = ["BacktestEngine", "BacktestResult", "PerformanceMetrics", "BacktestReport"]
