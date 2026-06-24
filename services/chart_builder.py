"""
Chart Builder — Professional stock chart rendering with 16 chart types
and integrated technical indicators.

Chart Types:
    Candlestick, OHLC Bars, Hollow Candles, Heikin Ashi,
    Line, Line with Markers, Step Line,
    Area, HLC Area, Baseline,
    Columns, High-Low, Volume Candles,
    Renko, Kagi, Line Break

Overlay Indicators (on main chart):
    SMA (10, 20, 50, 200), EMA (12, 26), Bollinger Bands, VWAP

Subplot Indicators (below main chart):
    Volume, RSI (14), MACD (12/26/9), Stochastic Oscillator
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from services.predictor import get_ml_predictions

# ═══════════════════════════════════════════════════════════════════════════
# Registry Constants
# ═══════════════════════════════════════════════════════════════════════════

CHART_TYPES: list[str] = [
    "Candlestick",
    "OHLC Bars",
    "Hollow Candles",
    "Heikin Ashi",
    "Line",
    "Line with Markers",
    "Step Line",
    "Area",
    "HLC Area",
    "Baseline",
    "Columns",
    "High-Low",
    "Volume Candles",
    "Renko",
    "Kagi",
    "Line Break",
]

OVERLAY_INDICATORS: list[str] = [
    "SMA 10",
    "SMA 20",
    "SMA 50",
    "SMA 200",
    "EMA 12",
    "EMA 26",
    "Bollinger Bands",
    "VWAP",
]

SUBPLOT_INDICATORS: list[str] = [
    "Volume",
    "RSI (14)",
    "MACD",
    "Stochastic",
]

# ═══════════════════════════════════════════════════════════════════════════
# Color Palette  (TradingView-inspired for professional look)
# ═══════════════════════════════════════════════════════════════════════════

BULLISH       = "#26a69a"
BEARISH       = "#ef5350"
PRICE_LINE    = "#2962ff"
AREA_FILL     = "rgba(41,98,255,0.08)"
GRID_COLOR    = "rgba(0,0,0,0.06)"
AXIS_COLOR    = "rgba(0,0,0,0.10)"
BG_MAIN       = "#ffffff"
BG_PAPER      = "#ffffff"

INDICATOR_COLORS = {
    "SMA 10":  "#f57f17",   # Dark Amber
    "SMA 20":  "#ff6d00",   # Dark Orange
    "SMA 50":  "#e91e63",   # Pink
    "SMA 200": "#9c27b0",   # Purple
    "EMA 12":  "#00838f",   # Dark Cyan
    "EMA 26":  "#283593",   # Dark Indigo
    "BB_MID":  "rgba(126,87,194,0.7)",
    "BB_BAND": "rgba(126,87,194,0.08)",
    "VWAP":    "#ff6f00",   # Amber (visible on white)
}


# ═══════════════════════════════════════════════════════════════════════════
# Indicator Computation
# ═══════════════════════════════════════════════════════════════════════════

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cumvol = df["volume"].cumsum()
    cumtp = (tp * df["volume"]).cumsum()
    return cumtp / (cumvol + 1e-10)


def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(window=d_period).mean()
    return k, d


# ═══════════════════════════════════════════════════════════════════════════
# Heikin Ashi Computation
# ═══════════════════════════════════════════════════════════════════════════

def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    ha_open = [df["open"].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["close"].iloc[i - 1]) / 2)
    ha["open"] = ha_open

    ha["high"] = pd.concat(
        [df["high"], ha["open"], ha["close"]], axis=1
    ).max(axis=1)
    ha["low"] = pd.concat(
        [df["low"], ha["open"], ha["close"]], axis=1
    ).min(axis=1)
    ha["date"] = df["date"].values
    ha["volume"] = df["volume"].values
    return ha


# ═══════════════════════════════════════════════════════════════════════════
# Renko Computation
# ═══════════════════════════════════════════════════════════════════════════

def _renko_bricks(df: pd.DataFrame, brick_pct: float = 1.0):
    """Compute Renko bricks using a percentage-based brick size."""
    close = df["close"].values
    avg_price = np.mean(close)
    brick_size = avg_price * (brick_pct / 100)
    if brick_size <= 0:
        brick_size = 1.0

    bricks = []
    base = close[0]

    for price in close[1:]:
        while price >= base + brick_size:
            bricks.append({"bottom": base, "top": base + brick_size, "dir": "up"})
            base += brick_size
        while price <= base - brick_size:
            bricks.append({"bottom": base - brick_size, "top": base, "dir": "down"})
            base -= brick_size

    return bricks, brick_size


# ═══════════════════════════════════════════════════════════════════════════
# Kagi Computation
# ═══════════════════════════════════════════════════════════════════════════

def _kagi_lines(df: pd.DataFrame, reversal_pct: float = 4.0):
    """Compute Kagi chart line segments."""
    close = df["close"].values
    if len(close) < 2:
        return [], []

    reversal_amount = close[0] * (reversal_pct / 100)
    segments_x = [0]
    segments_y = [close[0]]
    direction = 1 if close[1] >= close[0] else -1
    last_extreme = close[0]
    idx = 1

    for i in range(1, len(close)):
        price = close[i]
        if direction == 1:
            if price > last_extreme:
                last_extreme = price
                segments_x[-1] = idx
                segments_y[-1] = price
            elif price <= last_extreme - reversal_amount:
                segments_x.append(idx)
                segments_y.append(last_extreme)
                segments_x.append(idx)
                segments_y.append(price)
                direction = -1
                last_extreme = price
        else:
            if price < last_extreme:
                last_extreme = price
                segments_x[-1] = idx
                segments_y[-1] = price
            elif price >= last_extreme + reversal_amount:
                segments_x.append(idx)
                segments_y.append(last_extreme)
                segments_x.append(idx)
                segments_y.append(price)
                direction = 1
                last_extreme = price
        idx += 1

    return segments_x, segments_y


# ═══════════════════════════════════════════════════════════════════════════
# Line Break Computation
# ═══════════════════════════════════════════════════════════════════════════

def _line_break(df: pd.DataFrame, num_lines: int = 3):
    """Compute Three-Line Break chart data."""
    close = df["close"].values
    lines = []  # Each line: {open, close, dir}

    if len(close) < 2:
        return lines

    # First line
    lines.append({
        "open": close[0], "close": close[1],
        "dir": "up" if close[1] >= close[0] else "down",
    })

    for i in range(2, len(close)):
        price = close[i]
        lookback = lines[-min(num_lines, len(lines)):]
        high = max(l["close"] for l in lookback)
        low = min(l["open"] if l["dir"] == "up" else l["close"] for l in lookback)

        if price > high:
            lines.append({"open": lines[-1]["close"], "close": price, "dir": "up"})
        elif price < low:
            lines.append({"open": lines[-1]["close"], "close": price, "dir": "down"})

    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Chart Type Renderers
# ═══════════════════════════════════════════════════════════════════════════

def _render_candlestick(fig, df, row, col):
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing=dict(line=dict(color=BULLISH), fillcolor=BULLISH),
        decreasing=dict(line=dict(color=BEARISH), fillcolor=BEARISH),
        name="Price", showlegend=False,
    ), row=row, col=col)


def _render_ohlc(fig, df, row, col):
    fig.add_trace(go.Ohlc(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing=dict(line=dict(color=BULLISH)),
        decreasing=dict(line=dict(color=BEARISH)),
        name="OHLC", showlegend=False,
    ), row=row, col=col)


def _render_hollow_candles(fig, df, row, col):
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing=dict(
            line=dict(color=BULLISH, width=1.5),
            fillcolor="rgba(38,166,154,0.1)",  # Hollow / transparent
        ),
        decreasing=dict(
            line=dict(color=BEARISH, width=1.5),
            fillcolor=BEARISH,  # Filled
        ),
        name="Price", showlegend=False,
    ), row=row, col=col)


def _render_heikin_ashi(fig, df, row, col):
    ha = _heikin_ashi(df)
    fig.add_trace(go.Candlestick(
        x=ha["date"], open=ha["open"], high=ha["high"],
        low=ha["low"], close=ha["close"],
        increasing=dict(line=dict(color=BULLISH), fillcolor=BULLISH),
        decreasing=dict(line=dict(color=BEARISH), fillcolor=BEARISH),
        name="Heikin Ashi", showlegend=False,
    ), row=row, col=col)


def _render_line(fig, df, row, col):
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], mode="lines",
        line=dict(color=PRICE_LINE, width=2),
        name="Close", showlegend=False,
    ), row=row, col=col)


def _render_line_markers(fig, df, row, col):
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], mode="lines+markers",
        line=dict(color=PRICE_LINE, width=1.5),
        marker=dict(size=4, color=PRICE_LINE),
        name="Close", showlegend=False,
    ), row=row, col=col)


def _render_step_line(fig, df, row, col):
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], mode="lines",
        line=dict(color=PRICE_LINE, width=2, shape="hv"),
        name="Close", showlegend=False,
    ), row=row, col=col)


def _render_area(fig, df, row, col):
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], mode="lines",
        line=dict(color=PRICE_LINE, width=2),
        fill="tozeroy", fillcolor=AREA_FILL,
        name="Close", showlegend=False,
    ), row=row, col=col)


def _render_hlc_area(fig, df, row, col):
    # High-Low range as a filled area
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["high"], mode="lines",
        line=dict(color="rgba(0,230,118,0.3)", width=0.5),
        name="High", showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["low"], mode="lines",
        line=dict(color="rgba(255,23,68,0.3)", width=0.5),
        fill="tonexty", fillcolor="rgba(41,98,255,0.06)",
        name="Low", showlegend=False,
    ), row=row, col=col)
    # Close line on top
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], mode="lines",
        line=dict(color=PRICE_LINE, width=2),
        name="Close", showlegend=False,
    ), row=row, col=col)


def _render_baseline(fig, df, row, col):
    baseline = df["close"].iloc[0]
    close = df["close"].values
    dates = df["date"].values

    # Price above baseline (green fill)
    above = np.where(close >= baseline, close, baseline)
    fig.add_trace(go.Scatter(
        x=dates, y=[baseline] * len(dates), mode="lines",
        line=dict(color="rgba(0,0,0,0.18)", width=1, dash="dash"),
        showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=dates, y=above, mode="lines",
        line=dict(color=BULLISH, width=0.5),
        fill="tonexty", fillcolor="rgba(38,166,154,0.15)",
        showlegend=False,
    ), row=row, col=col)

    # Price below baseline (red fill)
    below = np.where(close <= baseline, close, baseline)
    fig.add_trace(go.Scatter(
        x=dates, y=[baseline] * len(dates), mode="lines",
        line=dict(width=0), showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=dates, y=below, mode="lines",
        line=dict(color=BEARISH, width=0.5),
        fill="tonexty", fillcolor="rgba(239,83,80,0.15)",
        showlegend=False,
    ), row=row, col=col)

    # Main close line
    fig.add_trace(go.Scatter(
        x=dates, y=close, mode="lines",
        line=dict(color=PRICE_LINE, width=2),
        name="Close", showlegend=False,
    ), row=row, col=col)


def _render_columns(fig, df, row, col):
    colors = [BULLISH if c >= o else BEARISH
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["close"],
        marker_color=colors, opacity=0.8,
        name="Close", showlegend=False,
    ), row=row, col=col)


def _render_high_low(fig, df, row, col):
    # Vertical lines from low to high
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["date"], r["date"]], y=[r["low"], r["high"]],
            mode="lines",
            line=dict(color=BULLISH if r["close"] >= r["open"] else BEARISH, width=2),
            showlegend=False,
        ), row=row, col=col)
    # Close dots
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], mode="markers",
        marker=dict(size=4, color=PRICE_LINE),
        name="Close", showlegend=False,
    ), row=row, col=col)


def _render_volume_candles(fig, df, row, col):
    # Candlestick
    _render_candlestick(fig, df, row, col)
    # Volume overlay (semi-transparent bars in the same pane)
    if "volume" in df.columns:
        vol_colors = [
            "rgba(38,166,154,0.2)" if c >= o else "rgba(239,83,80,0.2)"
            for c, o in zip(df["close"], df["open"])
        ]
        # Scale volume to ~20% of price range
        price_range = df["high"].max() - df["low"].min()
        vol_max = df["volume"].max()
        if vol_max > 0:
            scaled_vol = df["volume"] / vol_max * price_range * 0.2 + df["low"].min()
            fig.add_trace(go.Bar(
                x=df["date"], y=scaled_vol - df["low"].min(),
                marker_color=vol_colors, opacity=0.4,
                base=df["low"].min(),
                name="Volume", showlegend=False,
            ), row=row, col=col)


def _render_renko(fig, df, row, col):
    bricks, brick_size = _renko_bricks(df)
    if not bricks:
        _render_line(fig, df, row, col)
        return

    xs = list(range(len(bricks)))
    for i, brick in enumerate(bricks):
        color = BULLISH if brick["dir"] == "up" else BEARISH
        fig.add_trace(go.Bar(
            x=[i], y=[brick["top"] - brick["bottom"]],
            base=brick["bottom"],
            marker_color=color,
            marker_line=dict(color=color, width=1),
            width=0.8, showlegend=False,
        ), row=row, col=col)


def _render_kagi(fig, df, row, col):
    seg_x, seg_y = _kagi_lines(df)
    if len(seg_x) < 2:
        _render_line(fig, df, row, col)
        return

    # Draw segments with color based on direction
    for i in range(0, len(seg_x) - 1, 2):
        if i + 1 >= len(seg_y):
            break
        color = BULLISH if seg_y[i + 1] >= seg_y[i] else BEARISH
        width = 3 if seg_y[i + 1] >= seg_y[i] else 1.5
        fig.add_trace(go.Scatter(
            x=[seg_x[i], seg_x[i + 1]],
            y=[seg_y[i], seg_y[i + 1]],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False,
        ), row=row, col=col)


def _render_line_break_chart(fig, df, row, col):
    lines = _line_break(df)
    if not lines:
        _render_line(fig, df, row, col)
        return

    for i, line in enumerate(lines):
        color = BULLISH if line["dir"] == "up" else BEARISH
        fig.add_trace(go.Bar(
            x=[i], y=[abs(line["close"] - line["open"])],
            base=min(line["open"], line["close"]),
            marker_color=color,
            marker_line=dict(color=color, width=1),
            width=0.8, showlegend=False,
        ), row=row, col=col)


# ═══════════════════════════════════════════════════════════════════════════
# Chart Type Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

_CHART_RENDERERS = {
    "Candlestick":       _render_candlestick,
    "OHLC Bars":         _render_ohlc,
    "Hollow Candles":    _render_hollow_candles,
    "Heikin Ashi":       _render_heikin_ashi,
    "Line":              _render_line,
    "Line with Markers": _render_line_markers,
    "Step Line":         _render_step_line,
    "Area":              _render_area,
    "HLC Area":          _render_hlc_area,
    "Baseline":          _render_baseline,
    "Columns":           _render_columns,
    "High-Low":          _render_high_low,
    "Volume Candles":    _render_volume_candles,
    "Renko":             _render_renko,
    "Kagi":              _render_kagi,
    "Line Break":        _render_line_break_chart,
}


# ═══════════════════════════════════════════════════════════════════════════
# Overlay Indicator Renderers
# ═══════════════════════════════════════════════════════════════════════════

def _render_overlay(fig, df, indicator: str, row: int, col: int):
    dates = df["date"]

    if indicator.startswith("SMA"):
        window = int(indicator.split()[-1])
        values = _sma(df["close"], window)
        color = INDICATOR_COLORS.get(indicator, "#ffffff")
        fig.add_trace(go.Scatter(
            x=dates, y=values, mode="lines",
            line=dict(color=color, width=1.5),
            name=indicator,
        ), row=row, col=col)

    elif indicator.startswith("EMA"):
        span = int(indicator.split()[-1])
        values = _ema(df["close"], span)
        color = INDICATOR_COLORS.get(indicator, "#ffffff")
        fig.add_trace(go.Scatter(
            x=dates, y=values, mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            name=indicator,
        ), row=row, col=col)

    elif indicator == "Bollinger Bands":
        upper, mid, lower = _bollinger(df["close"])
        fig.add_trace(go.Scatter(
            x=dates, y=upper, mode="lines",
            line=dict(color=INDICATOR_COLORS["BB_MID"], width=1),
            name="BB Upper",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=dates, y=lower, mode="lines",
            line=dict(color=INDICATOR_COLORS["BB_MID"], width=1),
            fill="tonexty", fillcolor=INDICATOR_COLORS["BB_BAND"],
            name="BB Lower",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=dates, y=mid, mode="lines",
            line=dict(color=INDICATOR_COLORS["BB_MID"], width=1, dash="dash"),
            name="BB Mid", showlegend=False,
        ), row=row, col=col)

    elif indicator == "VWAP":
        if "volume" in df.columns:
            values = _vwap(df)
            fig.add_trace(go.Scatter(
                x=dates, y=values, mode="lines",
                line=dict(color=INDICATOR_COLORS["VWAP"], width=1.5, dash="dashdot"),
                name="VWAP",
            ), row=row, col=col)


# ═══════════════════════════════════════════════════════════════════════════
# Subplot Indicator Renderers
# ═══════════════════════════════════════════════════════════════════════════

def _render_volume_subplot(fig, df, row, col):
    if "volume" not in df.columns:
        return
    colors = [
        "rgba(38,166,154,0.6)" if c >= o else "rgba(239,83,80,0.6)"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"],
        marker_color=colors,
        name="Volume", showlegend=False,
    ), row=row, col=col)


def _render_rsi_subplot(fig, df, row, col):
    rsi_values = _rsi(df["close"])
    fig.add_trace(go.Scatter(
        x=df["date"], y=rsi_values, mode="lines",
        line=dict(color="#bb86fc", width=1.5),
        name="RSI (14)", showlegend=False,
    ), row=row, col=col)

    # Overbought/Oversold zones
    fig.add_hline(y=70, line=dict(color="rgba(239,83,80,0.4)", width=1, dash="dash"),
                  row=row, col=col)
    fig.add_hline(y=30, line=dict(color="rgba(38,166,154,0.4)", width=1, dash="dash"),
                  row=row, col=col)
    fig.add_hline(y=50, line=dict(color="rgba(0,0,0,0.08)", width=1, dash="dot"),
                  row=row, col=col)

    # Shaded zones
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.05)",
                  line_width=0, row=row, col=col)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.05)",
                  line_width=0, row=row, col=col)


def _render_macd_subplot(fig, df, row, col):
    macd_line, signal_line, histogram = _macd(df["close"])
    dates = df["date"]

    # Histogram
    hist_colors = [
        "rgba(38,166,154,0.6)" if h >= 0 else "rgba(239,83,80,0.6)"
        for h in histogram
    ]
    fig.add_trace(go.Bar(
        x=dates, y=histogram,
        marker_color=hist_colors,
        name="Histogram", showlegend=False,
    ), row=row, col=col)

    # MACD Line
    fig.add_trace(go.Scatter(
        x=dates, y=macd_line, mode="lines",
        line=dict(color="#2962ff", width=1.5),
        name="MACD", showlegend=False,
    ), row=row, col=col)

    # Signal Line
    fig.add_trace(go.Scatter(
        x=dates, y=signal_line, mode="lines",
        line=dict(color="#ff9800", width=1.5),
        name="Signal", showlegend=False,
    ), row=row, col=col)

    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.08)", width=1),
                  row=row, col=col)


def _render_stochastic_subplot(fig, df, row, col):
    k, d = _stochastic(df)
    dates = df["date"]

    fig.add_trace(go.Scatter(
        x=dates, y=k, mode="lines",
        line=dict(color="#bb86fc", width=1.5),
        name="%K", showlegend=False,
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=dates, y=d, mode="lines",
        line=dict(color="#ff9800", width=1.5, dash="dot"),
        name="%D", showlegend=False,
    ), row=row, col=col)

    fig.add_hline(y=80, line=dict(color="rgba(239,83,80,0.4)", width=1, dash="dash"),
                  row=row, col=col)
    fig.add_hline(y=20, line=dict(color="rgba(38,166,154,0.4)", width=1, dash="dash"),
                  row=row, col=col)

    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(239,83,80,0.05)",
                  line_width=0, row=row, col=col)
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(38,166,154,0.05)",
                  line_width=0, row=row, col=col)


_SUBPLOT_RENDERERS = {
    "Volume":     _render_volume_subplot,
    "RSI (14)":   _render_rsi_subplot,
    "MACD":       _render_macd_subplot,
    "Stochastic": _render_stochastic_subplot,
}

_SUBPLOT_TITLES = {
    "Volume":     "Volume",
    "RSI (14)":   "RSI (14)",
    "MACD":       "MACD (12, 26, 9)",
    "Stochastic": "Stochastic (14, 3)",
}


# ═══════════════════════════════════════════════════════════════════════════
# Trade Signal Engine — Multi-Indicator Consensus
# ═══════════════════════════════════════════════════════════════════════════

def _compute_trade_signals(
    df: pd.DataFrame,
    min_gap: int = 3,
    min_score: int = 2,
) -> list[dict]:
    """
    Compute BUY / SELL signals using a hybrid system combining the ML predictor
    and multi-indicator consensus scoring.

    Indicators used:
        • ML Predictor     — Linear Regression predicted next-period price
        • RSI (14)         — oversold / overbought zones & crossovers
        • MACD (12,26,9)   — bullish / bearish crossovers
        • SMA 20           — price crossover
        • EMA 12 / 26      — golden / death cross
        • Bollinger Bands   — lower / upper band touch

    Parameters
    ----------
    df       : OHLCV DataFrame
    min_gap  : minimum bars between consecutive signals (prevents clutter)
    min_score: minimum consensus score to emit a signal

    Returns
    -------
    List of signal dicts with keys:
        date, signal ('BUY'/'SELL'), price, low, high, score, reasons
    """
    close = df["close"]
    n = len(df)

    if n < 30:  # Need enough data for indicator warm-up
        return []

    # Pre-compute all indicators once
    rsi_vals   = _rsi(close, 14)
    macd_l, sig_l, _ = _macd(close)
    sma20      = _sma(close, 20)
    ema12      = _ema(close, 12)
    ema26      = _ema(close, 26)
    bb_up, _, bb_lo = _bollinger(close, 20, 2.0)

    # Pre-compute ML predictions
    ml_preds   = get_ml_predictions(df)

    # Calculate daily volatility for dynamic thresholding
    pct_changes = close.pct_change().dropna()
    volatility = pct_changes.std()
    if pd.isna(volatility) or volatility <= 0:
        volatility = 0.01  # fallback 1%
    threshold = 0.5 * volatility

    raw: list[dict] = []

    for i in range(1, n):
        buy  = 0
        sell = 0
        buy_r: list[str]  = []
        sell_r: list[str] = []

        r      = rsi_vals.iloc[i]
        r_prev = rsi_vals.iloc[i - 1]
        c      = close.iloc[i]
        c_prev = close.iloc[i - 1]
        ml_pred = ml_preds.iloc[i] if i < len(ml_preds) else np.nan

        # ── ML Predictor ────────────────────────────────────────────
        ml_bullish = False
        ml_bearish = False
        if not pd.isna(ml_pred) and c > 0:
            pred_change_pct = (ml_pred - c) / c
            if pred_change_pct > threshold:
                buy += 2
                buy_r.append("ML Predictor Bullish")
                ml_bullish = True
            elif pred_change_pct < -threshold:
                sell += 2
                sell_r.append("ML Predictor Bearish")
                ml_bearish = True

        # ── RSI ─────────────────────────────────────────────────────
        if not pd.isna(r):
            if r < 30:
                buy += 1;  buy_r.append("RSI Oversold")
            elif r > 70:
                sell += 1; sell_r.append("RSI Overbought")
            if not pd.isna(r_prev):
                if r_prev >= 30 and r < 30:
                    buy += 1;  buy_r.append("RSI Cross ↓30")
                if r_prev <= 70 and r > 70:
                    sell += 1; sell_r.append("RSI Cross ↑70")

        # ── MACD Crossover ──────────────────────────────────────────
        m, s = macd_l.iloc[i], sig_l.iloc[i]
        mp, sp = macd_l.iloc[i - 1], sig_l.iloc[i - 1]
        if not pd.isna(m) and not pd.isna(mp):
            if m > s and mp <= sp:
                buy += 2;  buy_r.append("MACD Bullish Cross")
            elif m < s and mp >= sp:
                sell += 2; sell_r.append("MACD Bearish Cross")

        # ── SMA 20 Crossover ────────────────────────────────────────
        sm  = sma20.iloc[i]
        smp = sma20.iloc[i - 1]
        if not pd.isna(sm) and not pd.isna(smp):
            if c > sm and c_prev <= smp:
                buy += 1;  buy_r.append("Price ↑ SMA 20")
            elif c < sm and c_prev >= smp:
                sell += 1; sell_r.append("Price ↓ SMA 20")

        # ── EMA 12 / 26 Crossover ──────────────────────────────────
        e12, e26 = ema12.iloc[i], ema26.iloc[i]
        e12p, e26p = ema12.iloc[i - 1], ema26.iloc[i - 1]
        if not pd.isna(e12) and not pd.isna(e12p):
            if e12 > e26 and e12p <= e26p:
                buy += 1;  buy_r.append("EMA Golden Cross")
            elif e12 < e26 and e12p >= e26p:
                sell += 1; sell_r.append("EMA Death Cross")

        # ── Bollinger Band Touch ────────────────────────────────────
        bbu = bb_up.iloc[i]
        bbl = bb_lo.iloc[i]
        if not pd.isna(bbl):
            if c <= bbl:
                buy += 1;  buy_r.append("Below Lower BB")
            elif c >= bbu:
                sell += 1; sell_r.append("Above Upper BB")

        # ── Emit if score meets threshold ───────────────────────────
        # Require ML prediction agreement OR strong technical consensus (score >= 4)
        if buy >= min_score and (ml_bullish or buy >= 4) and buy >= sell:
            raw.append(dict(
                idx=i, date=df["date"].iloc[i], signal="BUY",
                price=float(c), low=float(df["low"].iloc[i]),
                high=float(df["high"].iloc[i]),
                score=buy, reasons=buy_r,
            ))
        elif sell >= min_score and (ml_bearish or sell >= 4) and sell > buy:
            raw.append(dict(
                idx=i, date=df["date"].iloc[i], signal="SELL",
                price=float(c), low=float(df["low"].iloc[i]),
                high=float(df["high"].iloc[i]),
                score=sell, reasons=sell_r,
            ))

    # ── Filter: min gap + prefer alternating buy/sell ───────────────
    filtered: list[dict] = []
    last_idx = -min_gap - 1
    last_sig = None
    for sig in raw:
        gap_ok = sig["idx"] - last_idx >= min_gap
        alt_ok = sig["signal"] != last_sig
        strong = sig["score"] >= 3
        if gap_ok and (alt_ok or strong):
            filtered.append(sig)
            last_idx = sig["idx"]
            last_sig = sig["signal"]

    return filtered


def _render_trade_signals(fig, df: pd.DataFrame, row: int, col: int):
    """
    Render BUY / SELL markers and entry/exit price levels on the chart.
    """
    signals = _compute_trade_signals(df)
    if not signals:
        return

    buys  = [s for s in signals if s["signal"] == "BUY"]
    sells = [s for s in signals if s["signal"] == "SELL"]

    price_range = df["high"].max() - df["low"].min()
    offset = price_range * 0.04  # Marker offset from candle body

    # ── BUY markers (green triangles below candles) ─────────────────
    if buys:
        fig.add_trace(go.Scatter(
            x=[s["date"] for s in buys],
            y=[s["low"] - offset for s in buys],
            mode="markers+text",
            marker=dict(
                symbol="triangle-up", size=14,
                color="#26a69a",
                line=dict(width=1.5, color="#1b7a6e"),
            ),
            text=["BUY"] * len(buys),
            textposition="bottom center",
            textfont=dict(size=9, color="#26a69a", family="Inter, sans-serif"),
            name="⬆ Buy Signal",
            showlegend=True,
            hovertemplate=(
                "<b>⬆ BUY Signal</b><br>"
                "Entry Price: %{customdata[0]:.2f}<br>"
                "Strength: %{customdata[1]}/5<br>"
                "Reasons: %{customdata[2]}"
                "<extra></extra>"
            ),
            customdata=[
                [s["price"], s["score"], " · ".join(s["reasons"])]
                for s in buys
            ],
        ), row=row, col=col)

    # ── SELL markers (red triangles above candles) ──────────────────
    if sells:
        fig.add_trace(go.Scatter(
            x=[s["date"] for s in sells],
            y=[s["high"] + offset for s in sells],
            mode="markers+text",
            marker=dict(
                symbol="triangle-down", size=14,
                color="#ef5350",
                line=dict(width=1.5, color="#c62828"),
            ),
            text=["SELL"] * len(sells),
            textposition="top center",
            textfont=dict(size=9, color="#ef5350", family="Inter, sans-serif"),
            name="⬇ Sell Signal",
            showlegend=True,
            hovertemplate=(
                "<b>⬇ SELL Signal</b><br>"
                "Exit Price: %{customdata[0]:.2f}<br>"
                "Strength: %{customdata[1]}/5<br>"
                "Reasons: %{customdata[2]}"
                "<extra></extra>"
            ),
            customdata=[
                [s["price"], s["score"], " · ".join(s["reasons"])]
                for s in sells
            ],
        ), row=row, col=col)

    # ── Latest Entry / Exit price lines ─────────────────────────────
    if buys:
        latest_buy = buys[-1]
        fig.add_hline(
            y=latest_buy["price"], row=row, col=col,
            line=dict(color="rgba(38,166,154,0.5)", width=1.5, dash="dash"),
            annotation=dict(
                text=f"▶ Entry: {latest_buy['price']:.2f}",
                font=dict(size=10, color="#26a69a"),
                bgcolor="rgba(38,166,154,0.08)",
                bordercolor="rgba(38,166,154,0.3)",
            ),
            annotation_position="top left",
        )

    if sells:
        latest_sell = sells[-1]
        fig.add_hline(
            y=latest_sell["price"], row=row, col=col,
            line=dict(color="rgba(239,83,80,0.5)", width=1.5, dash="dash"),
            annotation=dict(
                text=f"◀ Exit: {latest_sell['price']:.2f}",
                font=dict(size=10, color="#ef5350"),
                bgcolor="rgba(239,83,80,0.08)",
                bordercolor="rgba(239,83,80,0.3)",
            ),
            annotation_position="bottom left",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main Chart Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_stock_chart(
    df: pd.DataFrame,
    chart_type: str = "Candlestick",
    overlays: list[str] | None = None,
    subplots: list[str] | None = None,
    asset_name: str = "",
    currency_sym: str = "$",
    is_dark: bool = False,
) -> go.Figure:
    """
    Build a professional multi-panel stock chart.

    Parameters
    ----------
    df : DataFrame with columns: open, high, low, close, volume, date
    chart_type : One of CHART_TYPES
    overlays : List from OVERLAY_INDICATORS to overlay on the main chart
    subplots : List from SUBPLOT_INDICATORS to show as separate panels
    asset_name : Display name for the chart title
    currency_sym : Currency symbol for axis labels
    is_dark : Whether to render the chart in dark mode

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if overlays is None:
        overlays = []
    if subplots is None:
        subplots = []

    # ── Theme Configuration ─────────────────────────────────────────────
    grid_color = "rgba(255, 255, 255, 0.08)" if is_dark else GRID_COLOR
    axis_color = "rgba(255, 255, 255, 0.15)" if is_dark else AXIS_COLOR
    bg_main = "#181a20" if is_dark else BG_MAIN
    bg_paper = "#181a20" if is_dark else BG_PAPER
    font_color = "#eaecef" if is_dark else "#333"
    legend_color = "#eaecef" if is_dark else "#555"
    legend_bg = "rgba(24, 26, 32, 0.9)" if is_dark else "rgba(255,255,255,0.9)"
    chart_template = "plotly_dark" if is_dark else "plotly_white"

    # ── Layout: determine subplot rows ──────────────────────────────────
    n_sub = len(subplots)
    n_rows = 1 + n_sub

    # Dynamic row heights: main chart gets ~60%, subplots share the rest
    if n_sub == 0:
        row_heights = [1.0]
    elif n_sub == 1:
        row_heights = [0.72, 0.28]
    elif n_sub == 2:
        row_heights = [0.58, 0.21, 0.21]
    elif n_sub == 3:
        row_heights = [0.50, 0.18, 0.16, 0.16]
    else:
        main_h = 0.45
        sub_h = (1 - main_h) / n_sub
        row_heights = [main_h] + [sub_h] * n_sub

    sub_titles = [f"{asset_name}  ·  {chart_type}"]
    for s in subplots:
        sub_titles.append(_SUBPLOT_TITLES.get(s, s))

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=sub_titles,
    )

    # ── Render main chart ───────────────────────────────────────────────
    renderer = _CHART_RENDERERS.get(chart_type, _render_candlestick)

    # For High-Low, limit data points to avoid trace overload
    if chart_type == "High-Low" and len(df) > 120:
        render_df = df.tail(120).copy().reset_index(drop=True)
    else:
        render_df = df

    renderer(fig, render_df, row=1, col=1)

    # ── Render overlay indicators ───────────────────────────────────────
    for indicator in overlays:
        _render_overlay(fig, df, indicator, row=1, col=1)

    # ── Render trade BUY / SELL signals ─────────────────────────────────
    _render_trade_signals(fig, df, row=1, col=1)

    # ── Render subplot indicators ───────────────────────────────────────
    for i, indicator in enumerate(subplots):
        sub_renderer = _SUBPLOT_RENDERERS.get(indicator)
        if sub_renderer:
            sub_renderer(fig, df, row=2 + i, col=1)

    # ── Chart height ────────────────────────────────────────────────────
    base_height = 580
    sub_height = 180
    total_height = base_height + n_sub * sub_height

    # ── Apply professional TradingView-style theme ──────────────────────
    fig.update_layout(
        template=chart_template,
        paper_bgcolor=bg_paper,
        plot_bgcolor=bg_main,
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            color=font_color, size=11,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="center", x=0.5,
            font=dict(size=10, color=legend_color),
            bgcolor=legend_bg,
        ),
        margin=dict(l=60, r=30, t=50, b=30),
        height=total_height,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    # ── Style all axes ──────────────────────────────────────────────────
    # Use 'category' type for x-axis to eliminate weekend/holiday gaps.
    # This prevents paper-thin candles, triangular BB fills, and other
    # rendering artifacts caused by Plotly interpolating missing dates.
    for i in range(1, n_rows + 1):
        x_axis = f"xaxis{i}" if i > 1 else "xaxis"
        y_axis = f"yaxis{i}" if i > 1 else "yaxis"

        x_config = dict(
            type="category",
            showgrid=True,
            gridcolor=grid_color,
            linecolor=axis_color,
            zeroline=False,
            tickangle=-30,
            nticks=12,
            showticklabels=(i == n_rows),  # only show labels on bottom
        )

        y_config = dict(
            showgrid=True,
            gridcolor=grid_color,
            linecolor=axis_color,
            zeroline=False,
            side="right",
        )

        fig.update_layout(**{x_axis: x_config, y_axis: y_config})

    # ── RSI y-axis range ────────────────────────────────────────────────
    for i, indicator in enumerate(subplots):
        if indicator == "RSI (14)":
            y_axis = f"yaxis{2 + i}"
            fig.update_layout(**{y_axis: dict(range=[0, 100])})
        elif indicator == "Stochastic":
            y_axis = f"yaxis{2 + i}"
            fig.update_layout(**{y_axis: dict(range=[0, 100])})

    # ── Range slider on last x-axis ─────────────────────────────────────
    last_xaxis = f"xaxis{n_rows}" if n_rows > 1 else "xaxis"
    fig.update_layout(**{
        last_xaxis: dict(
            rangeslider=dict(visible=True, thickness=0.04),
            showticklabels=True,
        ),
    })

    return fig
