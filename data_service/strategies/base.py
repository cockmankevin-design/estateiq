"""
Abstract strategy base class.

All strategies share a common interface:
  - generate_signals(prices) → list[Signal]
  - on_bar(bar)              → optional online update hook
  - describe()               → human-readable summary
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Single trade signal produced by a strategy."""

    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float = 1.0          # 0..1
    target_weight: float = 0.0       # portfolio weight (0..1 for longs)
    price: Optional[float] = None    # reference price at signal time
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class StrategyResult:
    """Everything a strategy run produces."""

    signals: List[Signal]
    positions: pd.DataFrame       # columns: symbol, weight
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class BaseStrategy(ABC):
    """
    Contract for every trading strategy.

    Concrete strategies must implement `generate_signals`.
    `on_bar` is optional (for live / streaming use cases).
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params: Dict[str, Any] = params or {}
        self._logger = logging.getLogger(f"strategy.{name}")

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> List[Signal]:
        """
        Core signal generation.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV frame, indexed by date, one or more symbol columns.

        Returns
        -------
        list[Signal]
        """

    def on_bar(self, bar: Dict[str, Any]) -> Optional[Signal]:
        """
        Optional hook called on each new live bar.
        Override in streaming-capable strategies.
        """
        return None

    def run(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> StrategyResult:
        """Execute the strategy and wrap output in a StrategyResult."""
        errors: List[str] = []
        signals: List[Signal] = []
        positions = pd.DataFrame()

        try:
            signals = self.generate_signals(prices, **kwargs)
            positions = self._signals_to_positions(signals, prices)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self._logger.error("Strategy '%s' failed: %s", self.name, msg)
            errors.append(msg)

        return StrategyResult(
            signals=signals,
            positions=positions,
            diagnostics={"strategy": self.name, "params": self.params},
            errors=errors,
        )

    def describe(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _signals_to_positions(
        signals: List[Signal], prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert a list of signals into a weight DataFrame (date × symbol)."""
        if not signals:
            return pd.DataFrame()

        rows = []
        for sig in signals:
            weight = sig.target_weight if sig.signal_type == SignalType.LONG else (
                -sig.target_weight if sig.signal_type == SignalType.SHORT else 0.0
            )
            rows.append({"date": sig.timestamp, "symbol": sig.symbol, "weight": weight})

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.pivot(index="date", columns="symbol", values="weight").fillna(0.0)

    def _get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)
