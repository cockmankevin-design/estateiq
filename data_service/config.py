"""
Central configuration for the EstateIQ data service layer.
All environment variables and default settings live here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "estateiq")
    user: str = os.getenv("DB_USER", "estateiq")
    password: str = os.getenv("DB_PASSWORD", "")
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    ttl_seconds: int = 300  # default cache TTL


@dataclass
class DataFetcherConfig:
    # Market data providers
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    polygon_key: str = os.getenv("POLYGON_KEY", "")
    quandl_key: str = os.getenv("QUANDL_KEY", "")
    fred_key: str = os.getenv("FRED_KEY", "")

    # Real estate specific
    redfin_api_url: str = os.getenv("REDFIN_API_URL", "https://api.redfin.com/v1")
    zillow_key: str = os.getenv("ZILLOW_KEY", "")
    attom_key: str = os.getenv("ATTOM_KEY", "")

    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.5


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001       # 0.1 %
    slippage_bps: float = 5.0            # 5 basis points
    risk_free_rate: float = 0.04         # annualised
    benchmark_symbol: str = "SPY"
    default_lookback_days: int = 252     # one trading year


@dataclass
class MLConfig:
    model_dir: str = os.getenv("MODEL_DIR", "models/")
    feature_store_dir: str = os.getenv("FEATURE_STORE_DIR", "features/")
    train_test_split: float = 0.8
    cv_folds: int = 5
    random_seed: int = 42
    default_model: str = "xgboost"       # xgboost | lightgbm | random_forest | lstm


@dataclass
class WebConfig:
    host: str = os.getenv("WEB_HOST", "0.0.0.0")
    port: int = int(os.getenv("WEB_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(",")
    )


@dataclass
class RealtimeConfig:
    websocket_url: str = os.getenv("WS_URL", "wss://stream.data.alpaca.markets/v2")
    alpaca_key: str = os.getenv("ALPACA_KEY", "")
    alpaca_secret: str = os.getenv("ALPACA_SECRET", "")
    reconnect_interval: float = 5.0
    max_reconnects: int = 10


@dataclass
class AppConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    fetcher: DataFetcherConfig = field(default_factory=DataFetcherConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    web: WebConfig = field(default_factory=WebConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("APP_ENV", "development")


# Singleton config instance
config = AppConfig()
