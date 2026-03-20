"""
Stream processor.

Buffers real-time events, assembles OHLCV bars, and triggers
strategy signal generation on each new completed bar.
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, List, Optional

import pandas as pd

from .stream import DataStream, EventType, StreamEvent

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Processes live bar events, maintains a rolling price buffer,
    and calls registered signal handlers.

    Parameters
    ----------
    buffer_size : int
        Number of bars to keep per symbol.
    signal_callback : callable, optional
        Called with (symbol, features_df) on each new bar.
    """

    def __init__(
        self,
        stream: DataStream,
        buffer_size: int = 500,
        signal_callback: Optional[Callable] = None,
    ):
        self.stream = stream
        self.buffer_size = buffer_size
        self.signal_callback = signal_callback

        self._bars: Dict[str, Deque[Dict]] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        self._tick_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self._last_bar_time: Dict[str, Optional[datetime]] = defaultdict(lambda: None)

        # Register handlers
        self.stream.on(EventType.BAR, self._on_bar)
        self.stream.on(EventType.TRADE, self._on_trade)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_bar(self, event: StreamEvent) -> None:
        sym = event.symbol
        bar = {
            "timestamp": event.timestamp,
            "open":   event.data.get("open"),
            "high":   event.data.get("high"),
            "low":    event.data.get("low"),
            "close":  event.data.get("close"),
            "volume": event.data.get("volume"),
        }
        self._bars[sym].append(bar)
        self._last_bar_time[sym] = event.timestamp
        logger.debug("Bar[%s] %s close=%.4f", sym, event.timestamp, bar["close"] or 0)

        if self.signal_callback and len(self._bars[sym]) >= 2:
            df = self.get_price_df(sym)
            try:
                self.signal_callback(sym, df)
            except Exception as exc:
                logger.error("Signal callback error for %s: %s", sym, exc)

    def _on_trade(self, event: StreamEvent) -> None:
        sym = event.symbol
        tick = {
            "timestamp": event.timestamp,
            "price": event.data.get("price"),
            "size":  event.data.get("size"),
        }
        self._tick_buffer[sym].append(tick)
        # Keep only recent ticks
        if len(self._tick_buffer[sym]) > 10_000:
            self._tick_buffer[sym] = self._tick_buffer[sym][-5_000:]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_price_df(self, symbol: str) -> pd.DataFrame:
        """Return buffered bars as a DataFrame for the given symbol."""
        bars = list(self._bars[symbol])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars).set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df.astype(float, errors="ignore")

    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        bars = self._bars[symbol]
        return bars[-1] if bars else None

    def get_vwap(self, symbol: str) -> Optional[float]:
        """Compute VWAP from today's tick buffer."""
        ticks = self._tick_buffer.get(symbol, [])
        today = datetime.utcnow().date()
        today_ticks = [t for t in ticks if t["timestamp"].date() == today
                       if t["price"] and t["size"]]
        if not today_ticks:
            return None
        pv = sum(t["price"] * t["size"] for t in today_ticks)
        vol = sum(t["size"] for t in today_ticks)
        return pv / vol if vol > 0 else None

    def position_snapshot(self) -> Dict[str, Dict]:
        """Return a snapshot of the latest bar for every tracked symbol."""
        return {sym: self.get_latest_bar(sym) for sym in self._bars if self._bars[sym]}

    def all_symbols(self) -> List[str]:
        return list(self._bars.keys())
