"""
FastAPI router — REST endpoints for the data service layer.

Endpoints:
  GET  /health                    — service health check
  GET  /prices/{symbol}           — fetch historical OHLCV
  POST /backtest                  — run a strategy backtest
  GET  /signals/{strategy}        — retrieve recent signals
  GET  /factors/{symbol}          — factor exposures for a symbol
  POST /predict                   — ML return prediction
  GET  /properties                — real-estate property listings
  GET  /realtime/snapshot         — latest live position snapshot
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


class BacktestRequest(BaseModel):
    strategy: str = Field(..., description="Strategy name: momentum | mean_reversion | ml")
    symbols: List[str] = Field(..., min_items=1)
    start_date: str = Field(..., description="ISO date YYYY-MM-DD")
    end_date: str = Field(..., description="ISO date YYYY-MM-DD")
    initial_capital: float = Field(100_000.0, gt=0)
    params: Dict[str, Any] = Field(default_factory=dict)


class BacktestResponse(BaseModel):
    run_id: Optional[int]
    strategy: str
    metrics: Dict[str, float]
    equity_curve: List[Dict]   # [{date, value}]
    n_trades: int


class PredictRequest(BaseModel):
    symbol: str
    lookback_days: int = Field(252, ge=30)
    model_type: str = "xgboost"
    forward_window: int = Field(1, ge=1)


class PredictResponse(BaseModel):
    symbol: str
    predicted_return: float
    confidence: float
    signal: int     # +1 / 0 / -1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat())


@router.get("/prices/{symbol}", tags=["Market Data"])
async def get_prices(
    symbol: str,
    start: str = Query(default=None, description="Start date YYYY-MM-DD"),
    end: str = Query(default=None, description="End date YYYY-MM-DD"),
    interval: str = Query(default="1d", description="Bar interval"),
):
    """Fetch OHLCV price data for a symbol."""
    from ..fetchers.market_data import MarketDataFetcher
    from ..config import config

    end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_dt = datetime.fromisoformat(start) if start else end_dt - timedelta(days=365)

    fetcher = MarketDataFetcher(
        alpha_vantage_key=config.fetcher.alpha_vantage_key,
        polygon_key=config.fetcher.polygon_key,
    )
    try:
        result = fetcher.fetch(symbol.upper(), start_dt, end_dt, interval=interval)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if not result.ok:
        raise HTTPException(status_code=502, detail=result.errors)

    df = result.data.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    return {"symbol": symbol.upper(), "data": df.to_dict(orient="records")}


@router.post("/backtest", response_model=BacktestResponse, tags=["Backtesting"])
async def run_backtest(req: BacktestRequest):
    """Run a strategy backtest and return performance metrics."""
    from ..fetchers.market_data import MarketDataFetcher
    from ..strategies.momentum import MomentumStrategy
    from ..strategies.mean_reversion import MeanReversionStrategy
    from ..backtesting.engine import BacktestEngine
    from ..config import config

    start = datetime.fromisoformat(req.start_date)
    end = datetime.fromisoformat(req.end_date)

    # Fetch price data
    fetcher = MarketDataFetcher(alpha_vantage_key=config.fetcher.alpha_vantage_key)
    close_frames = {}
    for sym in req.symbols:
        res = fetcher.fetch(sym.upper(), start, end)
        if res.ok and "close" in res.data.columns:
            close_frames[sym] = res.data["close"]
    if not close_frames:
        raise HTTPException(status_code=502, detail="No price data retrieved")

    import pandas as pd
    prices = pd.DataFrame(close_frames).to_frame("close") if len(close_frames) == 1 else pd.DataFrame(close_frames)
    # Wrap as multi-column OHLCV-like frame
    if "close" not in prices.columns:
        prices = prices.rename(columns={list(close_frames.keys())[0]: "close"})

    # Build strategy
    strategy_map = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
    }
    if req.strategy not in strategy_map:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

    strategy = strategy_map[req.strategy](**req.params)
    engine = BacktestEngine(initial_capital=req.initial_capital)

    try:
        result = engine.run(strategy, prices)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    equity_records = [
        {"date": str(d.date()), "value": round(float(v), 2)}
        for d, v in result.equity_curve.items()
    ]

    return BacktestResponse(
        run_id=None,
        strategy=req.strategy,
        metrics=result.metrics,
        equity_curve=equity_records,
        n_trades=len(result.trades),
    )


@router.post("/predict", response_model=PredictResponse, tags=["ML"])
async def predict_return(req: PredictRequest):
    """Run ML return prediction for a symbol."""
    from datetime import timedelta
    from ..fetchers.market_data import MarketDataFetcher
    from ..ml.models import ModelType
    from ..ml.predictor import Predictor
    from ..config import config

    end = datetime.utcnow()
    start = end - timedelta(days=req.lookback_days + 120)  # extra buffer for features

    fetcher = MarketDataFetcher(alpha_vantage_key=config.fetcher.alpha_vantage_key)
    res = fetcher.fetch(req.symbol.upper(), start, end)
    if not res.ok:
        raise HTTPException(status_code=502, detail=res.errors)

    try:
        model_type = ModelType(req.model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {req.model_type}")

    predictor = Predictor(model_type=model_type, forward_window=req.forward_window)
    try:
        predictor.fit(res.data)
        pred = predictor.predict_latest(res.data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    import numpy as np
    return PredictResponse(
        symbol=req.symbol.upper(),
        predicted_return=round(float(pred), 6),
        confidence=min(1.0, abs(float(pred)) / 0.05),
        signal=int(np.sign(pred)),
    )


@router.get("/properties", tags=["Real Estate"])
async def get_properties(
    zip_code: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    limit: int = Query(50, le=500),
):
    """Query the property database."""
    from ..storage.database import Database, Property
    from ..config import config

    db = Database()
    with db.session() as sess:
        q = sess.query(Property)
        if zip_code:
            q = q.filter(Property.zip_code == zip_code)
        if property_type:
            q = q.filter(Property.property_type == property_type)
        if min_price is not None:
            q = q.filter(Property.price >= min_price)
        if max_price is not None:
            q = q.filter(Property.price <= max_price)
        props = q.limit(limit).all()

    return {
        "count": len(props),
        "properties": [
            {
                "id": p.id, "address": p.address, "city": p.city, "state": p.state,
                "zip_code": p.zip_code, "property_type": p.property_type,
                "price": p.price, "beds": p.beds, "baths": p.baths,
                "sqft": p.sqft, "roi_estimate": p.roi_estimate,
                "lat": p.lat, "lon": p.lon,
            }
            for p in props
        ],
    }
