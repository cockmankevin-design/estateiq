"""
Dash-based analytics dashboard.

Launches a web UI for monitoring strategies, factor exposures,
backtests, and real-time positions.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Dashboard:
    """
    Interactive web dashboard powered by Plotly Dash.

    Usage
    -----
    >>> from data_service.visualization import Dashboard
    >>> db = Dashboard(port=8050)
    >>> db.run()   # opens http://localhost:8050
    """

    def __init__(
        self,
        title: str = "EstateIQ Analytics",
        port: int = 8050,
        debug: bool = False,
    ):
        self.title = title
        self.port = port
        self.debug = debug
        self._app = None

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------

    def _build_app(self):
        try:
            import dash
            from dash import Input, Output, dcc, html
        except ImportError as exc:
            raise RuntimeError("Dash not installed. Run: pip install dash") from exc

        app = dash.Dash(__name__, title=self.title)
        app.layout = html.Div(
            style={"backgroundColor": "#0d0d0f", "color": "#e0e0e0", "fontFamily": "DM Sans, sans-serif"},
            children=[
                # Header
                html.Div(
                    style={"background": "#1a1a2e", "padding": "16px 24px",
                           "borderBottom": "2px solid #c9a84c"},
                    children=[
                        html.H1(self.title, style={"color": "#c9a84c", "margin": 0, "fontSize": "1.5rem"}),
                    ],
                ),
                # Tabs
                dcc.Tabs(
                    id="main-tabs",
                    value="backtest",
                    style={"backgroundColor": "#0d0d0f"},
                    colors={"border": "#2a2a3e", "primary": "#c9a84c", "background": "#1a1a2e"},
                    children=[
                        dcc.Tab(label="Backtest",     value="backtest"),
                        dcc.Tab(label="Factors",      value="factors"),
                        dcc.Tab(label="Risk",         value="risk"),
                        dcc.Tab(label="Real-time",    value="realtime"),
                        dcc.Tab(label="ML Signals",   value="ml"),
                    ],
                ),
                html.Div(id="tab-content", style={"padding": "24px"}),

                # Refresh interval
                dcc.Interval(id="refresh", interval=30_000, n_intervals=0),
            ],
        )

        @app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
        def render_tab(tab: str):
            return self._render_tab(tab)

        return app

    # ------------------------------------------------------------------
    # Tab renderers
    # ------------------------------------------------------------------

    def _render_tab(self, tab: str):
        from dash import dcc, html

        if tab == "backtest":
            return html.Div([
                html.H2("Backtest Analytics", style={"color": "#c9a84c"}),
                html.P("Run a backtest from the API and results will appear here."),
                dcc.Graph(id="equity-curve-placeholder"),
            ])

        if tab == "factors":
            return html.Div([
                html.H2("Factor Analysis", style={"color": "#c9a84c"}),
                html.P("Factor exposures and alpha attribution."),
            ])

        if tab == "risk":
            return html.Div([
                html.H2("Risk Dashboard", style={"color": "#c9a84c"}),
                html.P("Portfolio VaR, drawdown, and risk factor decomposition."),
            ])

        if tab == "realtime":
            return html.Div([
                html.H2("Real-time Positions", style={"color": "#c9a84c"}),
                html.P("Live position and P&L monitoring (connects to C++ engine)."),
            ])

        if tab == "ml":
            return html.Div([
                html.H2("ML Signal Monitor", style={"color": "#c9a84c"}),
                html.P("Predicted returns, model confidence, and feature importance."),
            ])

        return html.Div("Unknown tab")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, open_browser: bool = True) -> None:
        """Start the Dash development server."""
        if self._app is None:
            self._app = self._build_app()

        logger.info("Dashboard running at http://localhost:%d", self.port)
        self._app.run(
            host="0.0.0.0",
            port=self.port,
            debug=self.debug,
            use_reloader=False,
        )

    def get_app(self):
        """Return the raw Dash app (for WSGI deployment)."""
        if self._app is None:
            self._app = self._build_app()
        return self._app.server
