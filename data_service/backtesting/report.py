"""
Backtest report generator.

Produces human-readable text summaries and HTML/PDF tearsheets
from BacktestResult objects.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .engine import BacktestResult

logger = logging.getLogger(__name__)


class BacktestReport:
    """
    Generate tearsheet-style reports from a BacktestResult.

    Usage
    -----
    >>> report = BacktestReport(result)
    >>> print(report.summary())
    >>> report.to_html("tearsheet.html")
    """

    def __init__(self, result: BacktestResult, title: str = "Backtest Report"):
        self.result = result
        self.title = title

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        m = self.result.metrics
        r = self.result

        lines = [
            "=" * 60,
            f"  {self.title}",
            "=" * 60,
            f"  Period       : {r.equity_curve.index[0].date()} → {r.equity_curve.index[-1].date()}",
            f"  Initial cap  : ${r.equity_curve.iloc[0]:,.0f}",
            f"  Final value  : ${r.equity_curve.iloc[-1]:,.0f}",
            f"  Total trades : {len(r.trades)}",
            "-" * 60,
            "  RETURNS",
            f"    Total Return      : {m.get('total_return', 0):.2%}",
            f"    CAGR              : {m.get('cagr', 0):.2%}",
            f"    Annualised Return : {m.get('annualised_return', 0):.2%}",
            "-" * 60,
            "  RISK",
            f"    Annualised Vol    : {m.get('annualised_vol', 0):.2%}",
            f"    Max Drawdown      : {m.get('max_drawdown', 0):.2%}",
            f"    VaR (95%)         : {m.get('var_95', 0):.2%}",
            f"    CVaR (95%)        : {m.get('cvar_95', 0):.2%}",
            f"    Skewness          : {m.get('skewness', 0):.3f}",
            f"    Kurtosis          : {m.get('kurtosis', 0):.3f}",
            "-" * 60,
            "  RISK-ADJUSTED",
            f"    Sharpe Ratio      : {m.get('sharpe', 0):.3f}",
            f"    Sortino Ratio     : {m.get('sortino', 0):.3f}",
            f"    Calmar Ratio      : {m.get('calmar', 0):.3f}",
            f"    Omega Ratio       : {m.get('omega', 0):.3f}",
            "-" * 60,
            "  TRADING",
            f"    Win Rate          : {m.get('win_rate', 0):.2%}",
            f"    Profit Factor     : {m.get('profit_factor', 0):.3f}",
            f"    Avg Win           : {m.get('avg_win', 0):.4%}",
            f"    Avg Loss          : {m.get('avg_loss', 0):.4%}",
        ]

        if "alpha" in m:
            lines += [
                "-" * 60,
                "  BENCHMARK-RELATIVE",
                f"    Alpha             : {m.get('alpha', 0):.2%}",
                f"    Beta              : {m.get('beta', 0):.3f}",
                f"    Information Ratio : {m.get('information_ratio', 0):.3f}",
                f"    Tracking Error    : {m.get('tracking_error', 0):.2%}",
                f"    Up Capture        : {m.get('up_capture', 0):.2%}",
                f"    Down Capture      : {m.get('down_capture', 0):.2%}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML tearsheet
    # ------------------------------------------------------------------

    def to_html(self, path: str, include_charts: bool = True) -> None:
        """Write a self-contained HTML tearsheet."""
        charts_html = self._charts_html() if include_charts else ""
        metrics_table = self._metrics_table_html()
        trades_table = self._trades_table_html()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{self.title}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d0d0f; color: #e0e0e0;
          margin: 0; padding: 20px; }}
  h1   {{ color: #c9a84c; border-bottom: 2px solid #c9a84c; padding-bottom: 8px; }}
  h2   {{ color: #c9a84c; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th   {{ background: #1a1a2e; color: #c9a84c; padding: 10px; text-align: left; }}
  td   {{ padding: 8px 10px; border-bottom: 1px solid #2a2a3e; }}
  tr:hover td {{ background: #1a1a2e; }}
  .positive {{ color: #4caf50; }}
  .negative {{ color: #f44336; }}
</style>
</head>
<body>
<h1>{self.title}</h1>
<p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
{charts_html}
<h2>Performance Metrics</h2>
{metrics_table}
<h2>Trade Log</h2>
{trades_table}
</body>
</html>"""

        Path(path).write_text(html, encoding="utf-8")
        logger.info("Tearsheet written to %s", path)

    def _metrics_table_html(self) -> str:
        rows = []
        for k, v in self.result.metrics.items():
            css = ""
            display = f"{v:.4f}"
            if "return" in k or "drawdown" in k or "vol" in k or "var" in k:
                display = f"{v:.2%}"
                css = "positive" if v > 0 else ("negative" if v < 0 else "")
            elif k in ("sharpe", "sortino", "calmar", "information_ratio"):
                css = "positive" if v > 0 else "negative"
            label = k.replace("_", " ").title()
            rows.append(f"<tr><td>{label}</td><td class='{css}'>{display}</td></tr>")

        return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"

    def _trades_table_html(self) -> str:
        trades = self.result.trades[:500]  # cap at 500 rows
        if not trades:
            return "<p>No trades executed.</p>"
        rows = []
        for t in trades:
            side_css = "positive" if t.side == "buy" else "negative"
            rows.append(
                f"<tr>"
                f"<td>{t.date.strftime('%Y-%m-%d')}</td>"
                f"<td>{t.symbol}</td>"
                f"<td class='{side_css}'>{t.side.upper()}</td>"
                f"<td>{t.quantity:.2f}</td>"
                f"<td>${t.price:,.2f}</td>"
                f"<td>${t.commission:,.2f}</td>"
                f"</tr>"
            )
        header = "<tr><th>Date</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Commission</th></tr>"
        return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"

    def _charts_html(self) -> str:
        """Inline Plotly equity curve chart."""
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            equity = self.result.equity_curve
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=equity.index.astype(str),
                    y=equity.values,
                    mode="lines",
                    name="Portfolio",
                    line=dict(color="#c9a84c", width=2),
                )
            )
            if self.result.benchmark_returns is not None:
                bm_equity = (1 + self.result.benchmark_returns.reindex(equity.index).fillna(0)).cumprod() * equity.iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=bm_equity.index.astype(str),
                        y=bm_equity.values,
                        mode="lines",
                        name="Benchmark",
                        line=dict(color="#888", width=1.5, dash="dash"),
                    )
                )
            fig.update_layout(
                title="Equity Curve",
                paper_bgcolor="#0d0d0f",
                plot_bgcolor="#0d0d0f",
                font=dict(color="#e0e0e0"),
            )
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(self, directory: str = ".") -> None:
        """Export equity, returns, and trades to CSV files."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.result.equity_curve.to_csv(d / "equity_curve.csv", header=["equity"])
        self.result.returns.to_csv(d / "returns.csv", header=["return"])
        if self.result.trades:
            pd.DataFrame([vars(t) for t in self.result.trades]).to_csv(d / "trades.csv", index=False)
        logger.info("CSV files written to %s", directory)
