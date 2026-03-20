"""
SQLAlchemy ORM models and database session management.

Tables:
  - price_bars       : OHLCV market data
  - backtest_runs    : backtest metadata and metrics
  - strategy_signals : generated trade signals
  - factor_scores    : computed alpha/risk factors
  - properties       : real-estate property data
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional

from sqlalchemy import (
    JSON, BigInteger, Column, DateTime, Float, Index,
    Integer, String, Text, create_engine, event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class PriceBar(Base):
    __tablename__ = "price_bars"

    id        = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol    = Column(String(16), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    open      = Column(Float, nullable=False)
    high      = Column(Float, nullable=False)
    low       = Column(Float, nullable=False)
    close     = Column(Float, nullable=False)
    volume    = Column(Float, nullable=True)
    source    = Column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_price_bars_symbol_ts", "symbol", "timestamp", unique=True),
    )

    def __repr__(self):
        return f"<PriceBar {self.symbol} {self.timestamp} close={self.close}>"


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    strategy_name  = Column(String(64), nullable=False)
    params         = Column(JSON, nullable=True)
    start_date     = Column(DateTime, nullable=False)
    end_date       = Column(DateTime, nullable=False)
    initial_capital= Column(Float, nullable=False, default=100_000.0)
    metrics        = Column(JSON, nullable=True)
    notes          = Column(Text, nullable=True)


class StrategySignal(Base):
    __tablename__ = "strategy_signals"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp     = Column(DateTime, nullable=False, index=True)
    strategy_name = Column(String(64), nullable=False, index=True)
    symbol        = Column(String(16), nullable=False)
    signal_type   = Column(String(8), nullable=False)  # LONG / SHORT / EXIT / HOLD
    confidence    = Column(Float, nullable=True)
    target_weight = Column(Float, nullable=True)
    price         = Column(Float, nullable=True)
    metadata      = Column(JSON, nullable=True)


class FactorScore(Base):
    __tablename__ = "factor_scores"

    id         = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, nullable=False, index=True)
    symbol     = Column(String(16), nullable=False, index=True)
    factor     = Column(String(64), nullable=False)
    score      = Column(Float, nullable=False)
    universe   = Column(String(64), nullable=True)


class Property(Base):
    __tablename__ = "properties"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    address       = Column(Text, nullable=False)
    city          = Column(String(64), nullable=True)
    state         = Column(String(8), nullable=True)
    zip_code      = Column(String(10), nullable=True, index=True)
    property_type = Column(String(32), nullable=True)
    price         = Column(Float, nullable=True)
    beds          = Column(Float, nullable=True)
    baths         = Column(Float, nullable=True)
    sqft          = Column(Float, nullable=True)
    roi_estimate  = Column(Float, nullable=True)
    lat           = Column(Float, nullable=True)
    lon           = Column(Float, nullable=True)
    raw_data      = Column(JSON, nullable=True)
    fetched_at    = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class Database:
    """
    Manages SQLAlchemy engine, session factory, and schema creation.

    Usage
    -----
    >>> db = Database("postgresql://user:pass@localhost/estateiq")
    >>> db.create_tables()
    >>> with db.session() as sess:
    ...     sess.add(PriceBar(...))
    ...     sess.commit()
    """

    def __init__(self, url: str = "sqlite:///estateiq.db", echo: bool = False):
        self.url = url
        self._engine = create_engine(url, echo=echo, pool_pre_ping=True)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("Database connected: %s", url.split("@")[-1] if "@" in url else url)

    def create_tables(self) -> None:
        """Create all tables (idempotent — skips existing tables)."""
        Base.metadata.create_all(self._engine)
        logger.info("Database tables ensured")

    def drop_tables(self) -> None:
        """Drop all managed tables. DESTRUCTIVE — use with care."""
        Base.metadata.drop_all(self._engine)
        logger.warning("All tables dropped")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        sess = self._Session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def upsert_price_bars(self, bars: List[PriceBar]) -> int:
        """Insert price bars, ignoring duplicates. Returns number inserted."""
        inserted = 0
        with self.session() as sess:
            for bar in bars:
                existing = sess.query(PriceBar).filter_by(
                    symbol=bar.symbol, timestamp=bar.timestamp
                ).first()
                if existing is None:
                    sess.add(bar)
                    inserted += 1
        return inserted

    def get_price_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> List[PriceBar]:
        with self.session() as sess:
            return (
                sess.query(PriceBar)
                .filter(
                    PriceBar.symbol == symbol,
                    PriceBar.timestamp >= start,
                    PriceBar.timestamp <= end,
                )
                .order_by(PriceBar.timestamp)
                .all()
            )

    def save_backtest(self, run: BacktestRun) -> int:
        with self.session() as sess:
            sess.add(run)
            sess.flush()
            return run.id

    def save_signals(self, signals: List[StrategySignal]) -> None:
        with self.session() as sess:
            sess.bulk_save_objects(signals)
