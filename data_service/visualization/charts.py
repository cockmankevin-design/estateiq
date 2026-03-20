"""
Chart library — Plotly-based interactive visualisations for trading analytics.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Charts:
    """
    Static factory class for Plotly figures.

    All methods return a `plotly.graph_objects.Figure` which can be:
      - Rendered in a Jupyter notebook: `fig.show()`
      - Embedded in a Dash app
      - Exported to HTML / PNG / SVG
    """

    GOLD = "#c9a84c"
    DARK_BG = "#0d0d0f"
    GRID_COLOR = "#2a2a3e"
    TEXT_COLOR = "#e0e0e0"

    @classmethod
    def _base_layout(cls, title: str = "", **kwargs):
        import plotly.graph_objects as go

        return dict(
            title=dict(text=title, font=dict(color=cls.GOLD)),
            paper_bgcolor=cls.DARK_BG,
            plot_bgcolor=cls.DARK_BG,
            font=dict(color=cls.TEXT_COLOR),
            xaxis=dict(gridcolor=cls.GRID_COLOR, linecolor=cls.GRID_COLOR),
            yaxis=dict(gridcolor=cls.GRID_COLOR, linecolor=cls.GRID_COLOR),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Price & equity charts
    # ------------------------------------------------------------------

    @classmethod
    def candlestick(
        cls,
        prices: pd.DataFrame,
        title: str = "Price Chart",
        volume: bool = True,
    ):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        rows = 2 if volume and "volume" in prices.columns else 1
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3] if rows == 2 else [1.0],
            vertical_spacing=0.02,
        )

        fig.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices.get("open", prices.get("Open")),
                high=prices.get("high", prices.get("High")),
                low=prices.get("low", prices.get("Low")),
                close=prices.get("close", prices.get("Close")),
                name="OHLC",
                increasing_line_color="#4caf50",
                decreasing_line_color="#f44336",
            ),
            row=1, col=1,
        )

        if rows == 2:
            vol = prices.get("volume", prices.get("Volume"))
            colors = ["#4caf50" if c >= o else "#f44336"
                      for o, c in zip(prices.get("open", vol), prices.get("close", vol))]
            fig.add_trace(
                go.Bar(x=prices.index, y=vol, marker_color=colors, name="Volume"),
                row=2, col=1,
            )

        fig.update_layout(**cls._base_layout(title))
        return fig

    @classmethod
    def equity_curve(
        cls,
        equity: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Equity Curve",
    ):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name="Strategy",
            line=dict(color=cls.GOLD, width=2),
        ))
        if benchmark is not None:
            bm_equity = (1 + benchmark.reindex(equity.index).fillna(0)).cumprod() * equity.iloc[0]
            fig.add_trace(go.Scatter(
                x=bm_equity.index, y=bm_equity.values,
                mode="lines", name="Benchmark",
                line=dict(color="#888", width=1.5, dash="dash"),
            ))
        fig.update_layout(**cls._base_layout(title))
        return fig

    @classmethod
    def drawdown_chart(cls, returns: pd.Series, title: str = "Drawdown"):
        import plotly.graph_objects as go

        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values * 100,
            fill="tozeroy", mode="lines",
            line=dict(color="#f44336", width=1),
            fillcolor="rgba(244,67,54,0.3)",
            name="Drawdown %",
        ))
        fig.update_layout(**cls._base_layout(title), yaxis_ticksuffix="%")
        return fig

    # ------------------------------------------------------------------
    # Distribution charts
    # ------------------------------------------------------------------

    @classmethod
    def returns_distribution(cls, returns: pd.Series, title: str = "Returns Distribution"):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns.values * 100,
            nbinsx=60,
            marker_color=cls.GOLD,
            opacity=0.8,
            name="Daily Returns",
        ))
        fig.add_vline(x=0, line_color="#fff", line_dash="dash")
        fig.add_vline(
            x=float(returns.mean() * 100),
            line_color="#4caf50",
            line_dash="dot",
            annotation_text=f"Mean: {returns.mean():.3%}",
        )
        fig.update_layout(**cls._base_layout(title), xaxis_title="Return (%)")
        return fig

    # ------------------------------------------------------------------
    # Factor / correlation charts
    # ------------------------------------------------------------------

    @classmethod
    def correlation_heatmap(
        cls,
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
    ):
        import plotly.graph_objects as go

        corr = data.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(**cls._base_layout(title))
        return fig

    @classmethod
    def factor_exposure_bar(
        cls,
        exposures: pd.DataFrame,
        symbol: str,
        title: str = "",
    ):
        import plotly.graph_objects as go

        row = exposures.loc[symbol]
        colors = [cls.GOLD if v > 0 else "#f44336" for v in row.values]
        fig = go.Figure(go.Bar(
            x=row.index.tolist(),
            y=row.values.tolist(),
            marker_color=colors,
        ))
        fig.update_layout(**cls._base_layout(title or f"Factor Exposures — {symbol}"))
        return fig

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    @classmethod
    def rolling_sharpe(
        cls,
        returns: pd.Series,
        window: int = 63,
        risk_free_rate: float = 0.04,
        title: str = "Rolling Sharpe Ratio",
    ):
        import plotly.graph_objects as go

        rf_daily = risk_free_rate / 252
        excess = returns - rf_daily
        roll_sharpe = (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(252)
        fig = go.Figure(go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values,
            mode="lines", line=dict(color=cls.GOLD, width=2),
            name=f"{window}d Rolling Sharpe",
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#4caf50",
                      annotation_text="Sharpe = 1")
        fig.add_hline(y=0.0, line_dash="dot", line_color="#888")
        fig.update_layout(**cls._base_layout(title))
        return fig
