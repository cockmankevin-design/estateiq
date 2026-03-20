"""
WebSocket data streaming client.

Supports Alpaca Market Data v2, Polygon.io WebSocket,
and generic WebSocket feeds. Emits typed StreamEvent objects
to registered handler callbacks.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    NEWS = "news"
    STATUS = "status"
    ERROR = "error"


@dataclass
class StreamEvent:
    """Normalised event from any streaming provider."""

    event_type: EventType
    symbol: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict] = None


Handler = Callable[[StreamEvent], None]


class DataStream:
    """
    Async WebSocket stream client.

    Usage
    -----
    >>> stream = DataStream(provider="alpaca", api_key=KEY, api_secret=SECRET)
    >>> stream.subscribe(["AAPL", "MSFT"], on_bar=my_handler)
    >>> asyncio.run(stream.start())
    """

    def __init__(
        self,
        provider: str = "alpaca",
        api_key: str = "",
        api_secret: str = "",
        websocket_url: Optional[str] = None,
        reconnect_interval: float = 5.0,
        max_reconnects: int = 10,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_secret = api_secret
        self.reconnect_interval = reconnect_interval
        self.max_reconnects = max_reconnects

        self._ws_url = websocket_url or self._default_url(provider)
        self._subscriptions: List[str] = []
        self._handlers: Dict[EventType, List[Handler]] = {et: [] for et in EventType}
        self._running = False
        self._reconnects = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(
        self,
        symbols: List[str],
        on_bar: Optional[Handler] = None,
        on_trade: Optional[Handler] = None,
        on_quote: Optional[Handler] = None,
    ) -> "DataStream":
        self._subscriptions.extend(symbols)
        if on_bar:
            self._handlers[EventType.BAR].append(on_bar)
        if on_trade:
            self._handlers[EventType.TRADE].append(on_trade)
        if on_quote:
            self._handlers[EventType.QUOTE].append(on_quote)
        return self

    def on(self, event_type: EventType, handler: Handler) -> "DataStream":
        """Register a handler for a specific event type."""
        self._handlers[event_type].append(handler)
        return self

    async def start(self) -> None:
        """Start the streaming loop with automatic reconnection."""
        self._running = True
        while self._running and self._reconnects <= self.max_reconnects:
            try:
                await self._connect_and_stream()
            except Exception as exc:
                self._reconnects += 1
                logger.warning(
                    "Stream disconnected (attempt %d/%d): %s",
                    self._reconnects, self.max_reconnects, exc,
                )
                if self._running and self._reconnects <= self.max_reconnects:
                    await asyncio.sleep(self.reconnect_interval)

        logger.info("Stream stopped after %d reconnects", self._reconnects)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets not installed") from exc

        async with websockets.connect(self._ws_url, ping_interval=20) as ws:
            logger.info("Connected to %s stream at %s", self.provider, self._ws_url)
            self._reconnects = 0

            # Auth
            await ws.send(json.dumps({"action": "auth", "key": self.api_key, "secret": self.api_secret}))
            # Subscribe
            if self._subscriptions:
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "bars": self._subscriptions,
                    "trades": self._subscriptions,
                    "quotes": self._subscriptions,
                }))

            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    messages = json.loads(raw_msg)
                    if isinstance(messages, list):
                        for msg in messages:
                            event = self._parse_message(msg)
                            if event:
                                self._dispatch(event)
                except Exception as exc:
                    logger.error("Message parse error: %s", exc)

    def _parse_message(self, msg: Dict) -> Optional[StreamEvent]:
        t = msg.get("T", "")
        symbol = msg.get("S", "")
        ts_str = msg.get("t", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.utcnow()
        except ValueError:
            ts = datetime.utcnow()

        if t == "b":    # bar
            return StreamEvent(EventType.BAR, symbol, ts, {
                "open": msg.get("o"), "high": msg.get("h"),
                "low": msg.get("l"), "close": msg.get("c"), "volume": msg.get("v"),
            }, raw=msg)
        if t == "t":    # trade
            return StreamEvent(EventType.TRADE, symbol, ts, {
                "price": msg.get("p"), "size": msg.get("s"), "conditions": msg.get("c"),
            }, raw=msg)
        if t == "q":    # quote
            return StreamEvent(EventType.QUOTE, symbol, ts, {
                "ask_price": msg.get("ap"), "ask_size": msg.get("as"),
                "bid_price": msg.get("bp"), "bid_size": msg.get("bs"),
            }, raw=msg)
        return None

    def _dispatch(self, event: StreamEvent) -> None:
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as exc:
                logger.error("Handler error for %s: %s", event.event_type, exc)

    @staticmethod
    def _default_url(provider: str) -> str:
        urls = {
            "alpaca": "wss://stream.data.alpaca.markets/v2/iex",
            "polygon": "wss://socket.polygon.io/stocks",
        }
        return urls.get(provider, "wss://stream.data.alpaca.markets/v2/iex")
