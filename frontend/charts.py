"""Plotly chart builders (pure functions of DataFrames; no Streamlit dependency)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_white"
C_ACTUAL, C_PRED, C_FCST, C_NAIVE = "#1f3b73", "#e8710a", "#188038", "#8a8f98"


def price_chart(df: pd.DataFrame, kind: str = "Candlestick", smas: tuple[int, ...] = (20, 50),
                symbol: str = "") -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    if kind == "Candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                     close=df["close"], name="OHLC"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close",
                                 line=dict(color=C_ACTUAL)), row=1, col=1)
    for k in smas:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"].rolling(k).mean(), name=f"SMA {k}",
                                 line=dict(width=1.2)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="#9aa7c7"),
                  row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=560, xaxis_rangeslider_visible=False,
                      title=f"{symbol} — daily prices (₹, end-of-day)", legend_orientation="h",
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(title_text="₹", row=1, col=1)
    return fig


def forecast_chart(hist: pd.DataFrame, preds: list[dict], symbol: str, tail: int = 90) -> go.Figure:
    h = hist.tail(tail)
    fig = go.Figure(go.Scatter(x=h.index, y=h["close"], name="Actual close",
                               line=dict(color=C_ACTUAL)))
    if preds:
        last_x, last_y = h.index[-1], float(h["close"].iloc[-1])
        xs = [last_x] + [pd.Timestamp(p["target_date_estimate"]) for p in preds]
        ys = [last_y] + [p["predicted_price"] for p in preds]
        if all(p.get("interval") for p in preds):  # shaded forecast ranges (widest first)
            for lv, color in (("80", "rgba(24,128,56,0.12)"), ("50", "rgba(24,128,56,0.24)")):
                up = [last_y] + [p["interval"]["ranges"][lv]["upper"] for p in preds]
                lo = [last_y] + [p["interval"]["ranges"][lv]["lower"] for p in preds]
                fig.add_trace(go.Scatter(x=xs + xs[::-1], y=up + lo[::-1], fill="toself",
                                         fillcolor=color, line=dict(width=0), name=f"{lv}% range",
                                         hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, name="Forecast (NOT actual)", mode="lines+markers",
                                 line=dict(color=C_FCST, dash="dash"),
                                 marker=dict(symbol="diamond", size=10)))
    fig.update_layout(template=TEMPLATE, height=420, title=f"{symbol} — recent closes and forecast",
                      yaxis_title="₹", legend_orientation="h", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def backtest_chart(bt: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["target_date"], y=bt["actual_price"], name="Actual",
                             line=dict(color=C_ACTUAL)))
    fig.add_trace(go.Scatter(x=bt["target_date"], y=bt["predicted_price"],
                             name="Backtest prediction", line=dict(color=C_PRED)))
    fig.update_layout(template=TEMPLATE, height=400, yaxis_title="₹ (adjusted)",
                      title=title, legend_orientation="h", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def error_chart(bt: pd.DataFrame, title: str) -> go.Figure:
    err = bt["predicted_price"] - bt["actual_price"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Error (₹) = predicted − actual", "Error (%)"))
    fig.add_trace(go.Bar(x=bt["target_date"], y=err, name="Error ₹", marker_color=C_PRED),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=bt["target_date"], y=err / bt["actual_price"] * 100, name="Error %",
                             line=dict(color=C_ACTUAL)), row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=420, title=title, showlegend=False,
                      margin=dict(l=10, r=10, t=70, b=10))
    return fig


def metric_bars(models: list[dict], metric: str, title: str, unit: str) -> go.Figure:
    names = [m["algorithm"] for m in models]
    vals = [m["metrics"].get(metric) for m in models]
    colors = [C_FCST if m["is_selected"] else (C_NAIVE if m["algorithm"] == "naive" else C_ACTUAL)
              for m in models]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                           text=[None if v is None else f"{v:.2f}" for v in vals]))
    fig.update_layout(template=TEMPLATE, height=320, title=title, yaxis_title=unit,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def live_error_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for h, g in df.groupby("horizon"):
        fig.add_trace(go.Scatter(x=g["as_of_date"], y=g["pct_error"], mode="lines+markers",
                                 name=f"{h}-day horizon"))
    fig.add_hline(y=0, line_color=C_NAIVE)
    fig.update_layout(template=TEMPLATE, height=360, yaxis_title="Error % (pred − actual)",
                      title="Realised forecast error over time",
                      legend_orientation="h", margin=dict(l=10, r=10, t=50, b=10))
    return fig
