"""Real-time data streaming and processing."""

from .stream import DataStream, StreamEvent
from .processor import StreamProcessor

__all__ = ["DataStream", "StreamEvent", "StreamProcessor"]
