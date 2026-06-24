"""
AI Stock Prediction Engine — Gradio Interface v2.

Premium, user-friendly Gradio UI with:
  • Category-based asset browsing (Indian Stocks, US Stocks, Crypto, etc.)
  • Searchable dropdowns with human-friendly names (no ticker symbols!)
  • Auto-resolution of display names → yFinance tickers
  • Interactive Plotly charts with Actual vs Predicted overlay
  • Technical indicator cards, confidence meter, agent logs
  • Mobile-responsive layout designed for deployment
"""

import asyncio
import logging
import os
import json
import subprocess
from datetime import date, datetime

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from services.data_fetcher import fetch_stock_data, fetch_news_headlines
from services.predictor import run_prediction, compute_confidence
from services.sentiment import analyze_sentiment, build_sentiment_gauge
from services.backtester import run_historical_backtest

from services.chart_builder import (
    build_stock_chart,
    CHART_TYPES,
    OVERLAY_INDICATORS,
    SUBPLOT_INDICATORS,
)
from stock_universe import (
    CATEGORIES,
    ALL_ASSETS,
    TICKER_TO_NAME,
    resolve_ticker,
    get_category_names,
    get_assets_for_category,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Display-friendly timeframes ────────────────────────────────────────────
TIMEFRAME_OPTIONS = {
    "1 Minute":   "1m",
    "2 Minutes":  "2m",
    "5 Minutes":  "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Day":      "1d",
    "1 Week":     "1w",
    "1 Month":    "1mo",
}

# ── Currency helpers ───────────────────────────────────────────────────────
SYMBOL_MAP = {
    "USD": "$", "INR": "₹", "EUR": "€",
    "GBP": "£", "JPY": "¥", "AUD": "A$", "CAD": "C$",
}


def _detect_currency(ticker: str):
    if "-" in ticker:
        parts = ticker.split("-")
        if len(parts) == 2:
            q = parts[1].upper()
            return SYMBOL_MAP.get(q, q), q
    elif ticker.endswith("=X") or ticker.endswith(".NYB"):
        base = ticker.replace("=X", "").replace("-Y.NYB", "")
        if len(base) >= 6:
            q = base[-3:].upper()
            return SYMBOL_MAP.get(q, q), q
        return "$", "USD"
    elif ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "₹", "INR"
    return "$", "USD"


def _detect_market(ticker: str) -> str:
    if "-" in ticker and not ticker.endswith(".NYB"):
        return "Cryptocurrency Exchange"
    if ticker.endswith("=X") or ticker.endswith(".NYB"):
        return "Foreign Exchange Market"
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "National Stock Exchange, India"
    if ticker.startswith("^NSEI") or ticker.startswith("^BSE") or ticker.startswith("^NSE"):
        return "Indian Market Index"
    if ticker.startswith("^"):
        return "Global Market Index"
    if "=F" in ticker:
        return "Futures / Commodities Exchange"
    return "US / Global Capital Market"


def _get_asset_emoji(category: str) -> str:
    emoji_map = {
        "🇮🇳 Indian Stocks": "🏢",
        "🇮🇳 Indian Indices": "📊",
        "🇮🇳 Indian ETFs": "📦",
        "🇺🇸 US Stocks": "🏛️",
        "🇺🇸 US & Global ETFs": "📦",
        "🌍 Global Indices": "🌐",
        "₿ Crypto": "₿",
        "🏆 Commodities & Futures": "⚡",
        "💱 Forex": "💱",
    }
    return emoji_map.get(category, "📈")


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic dropdown update
# ═══════════════════════════════════════════════════════════════════════════

def update_asset_dropdown(category: str):
    """When category changes, update the asset dropdown choices."""
    assets = get_assets_for_category(category)
    default = assets[0] if assets else None
    return gr.update(choices=assets, value=default)


# ═══════════════════════════════════════════════════════════════════════════
# Core prediction function
# ═══════════════════════════════════════════════════════════════════════════

def predict(
    asset_name: str,
    timeframe_label: str,
    category: str,
    chart_type: str = "Candlestick",
    overlay_indicators: list[str] | None = None,
    subplot_indicators: list[str] | None = None,
):
    """
    Run the full prediction pipeline using display name → ticker resolution.
    Returns: (price_html, live_chart, pred_chart, indicators_html, news_html, logs_text)
    """
    if overlay_indicators is None:
        overlay_indicators = []
    if subplot_indicators is None:
        subplot_indicators = []
    if not asset_name:
        raise gr.Error("Please select an asset from the dropdown.")

    # Resolve display name → ticker
    ticker = resolve_ticker(asset_name)
    if ticker is None:
        raise gr.Error(
            f"'{asset_name}' not found in the database. "
            "Please select a valid asset from the dropdown."
        )

    # Resolve timeframe label → code
    timeframe = TIMEFRAME_OPTIONS.get(timeframe_label, "1d")
    currency_sym, currency_code = _detect_currency(ticker)

    # ── Fetch data ──────────────────────────────────────────────────────
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df = loop.run_until_complete(fetch_stock_data(ticker, timeframe))
        loop.close()
    except Exception as exc:
        raise gr.Error(f"Data fetch failed for {asset_name}: {exc}")

    if df.empty:
        raise gr.Error(
            f"No market data available for '{asset_name}' ({ticker}). "
            "The market may be closed, or this timeframe has insufficient data. "
            "Try a different timeframe (e.g. 1 Day or 1 Month)."
        )

    # ── Run ML pipeline ─────────────────────────────────────────────────
    try:
        results = run_prediction(df)
    except ValueError as exc:
        raise gr.Error(
            f"Prediction failed for {asset_name}: {exc}\n\n"
            "💡 Tip: Try a longer timeframe (1 Day or 1 Month) for more data."
        )

    latest        = results["latest_df"]
    forecast      = results["next_day_forecast"]
    r2            = results["r2_score"]
    hist_pred     = results["historical_vs_predicted"]
    current_price = float(latest["close"])
    confidence    = compute_confidence(latest)

    # ── Build Live Market Chart ──────────────────────────────────────────
    live_chart = build_stock_chart(
        df=df,
        chart_type=chart_type or "Candlestick",
        overlays=overlay_indicators,
        subplots=subplot_indicators,
        asset_name=asset_name,
        currency_sym=currency_sym,
    )

    # Direction
    if forecast > current_price * 1.005:
        direction, dir_color, dir_icon = "Bullish", "#26a69a", "▲"
    elif forecast < current_price * 0.995:
        direction, dir_color, dir_icon = "Bearish", "#ef5350", "▼"
    else:
        direction, dir_color, dir_icon = "Neutral", "#ff9800", "●"

    pct_change = ((forecast - current_price) / current_price) * 100
    market_text = _detect_market(ticker)
    emoji = _get_asset_emoji(category)

    # ── Sparkline for Single Asset Predictor Card ──
    spark_closes = df["close"].tail(15).tolist()
    spark_closes = [float(p) for p in spark_closes if not pd.isna(p)]
    single_sparkline = _generate_sparkline_svg(spark_closes, forecast >= current_price, width=120, height=35)

    # ════════════════════════════════════════════════════════════════════
    # 1.  Price & Forecast Card
    # ════════════════════════════════════════════════════════════════════
    conf_bar_color = "#26a69a" if confidence >= 60 else ("#ff9800" if confidence >= 40 else "#ef5350")

    price_html = f"""
    <div style="
        background: #ffffff;
        border-radius: 20px; padding: 36px 40px;
        border: 1px solid #eaecef;
        box-shadow: 0 2px 12px rgba(0,0,0,.06);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    ">
        <!-- Asset Header -->
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; margin-bottom:8px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <span style="font-size:36px;">{emoji}</span>
                <div>
                    <div style="font-size:24px; font-weight:800; color:#1e2329; line-height:1.2;">{asset_name}</div>
                    <div style="font-size:12px; color:#707a8a; margin-top:3px; letter-spacing:.5px;">{ticker} &nbsp;·&nbsp; {market_text}</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                {single_sparkline}
                <span style="
                    background:{dir_color}14; color:{dir_color};
                    padding:8px 20px; border-radius:30px; font-size:14px;
                    font-weight:700; letter-spacing:.6px;
                    border: 1px solid {dir_color}33;
                ">{dir_icon} {direction}</span>
            </div>
        </div>

        <!-- Divider -->
        <div style="height:1px; background:linear-gradient(90deg, transparent, #eaecef, transparent); margin:20px 0;"></div>

        <!-- Metrics Grid -->
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:24px;">
            <div>
                <div style="font-size:10px; color:#707a8a; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Current Price</div>
                <div style="font-size:32px; font-weight:800; color:#1e2329; margin-top:4px; line-height:1.1;">
                    {currency_sym}{current_price:,.2f}
                </div>
                <div style="font-size:11px; color:#929aa5; margin-top:2px;">{currency_code}</div>
            </div>
            <div>
                <div style="font-size:10px; color:#707a8a; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">ML Forecast</div>
                <div style="font-size:32px; font-weight:800; color:{dir_color}; margin-top:4px; line-height:1.1;">
                    {currency_sym}{forecast:,.2f}
                </div>
                <div style="font-size:12px; color:{dir_color}; font-weight:600; margin-top:2px;">
                    {dir_icon} {pct_change:+.2f}%
                </div>
            </div>
            <div>
                <div style="font-size:10px; color:#707a8a; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Confidence</div>
                <div style="font-size:32px; font-weight:800; color:{conf_bar_color}; margin-top:4px; line-height:1.1;">
                    {confidence}%
                </div>
                <!-- Mini confidence bar -->
                <div style="margin-top:8px; background:#f0f2f5; border-radius:6px; height:6px; width:100%; max-width:140px;">
                    <div style="height:100%; width:{confidence}%; background:{conf_bar_color}; border-radius:6px; transition: width .3s;"></div>
                </div>
            </div>
            <div>
                <div style="font-size:10px; color:#707a8a; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Model R²</div>
                <div style="font-size:32px; font-weight:800; color:#2962ff; margin-top:4px; line-height:1.1;">
                    {r2}
                </div>
                <div style="font-size:11px; color:#929aa5; margin-top:2px;">Goodness of Fit</div>
            </div>
        </div>
    </div>
    """

    # ════════════════════════════════════════════════════════════════════
    # 2.  Plotly Chart
    # ════════════════════════════════════════════════════════════════════
    chart_df = pd.DataFrame(hist_pred)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.06,
        subplot_titles=(
            f"{asset_name} — Actual vs ML Predicted",
            "Prediction Error"
        ),
    )

    fig.add_trace(go.Scatter(
        x=chart_df["date"], y=chart_df["actual"],
        mode="lines", name="Actual Price",
        line=dict(color="#2962ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(41,98,255,.06)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=chart_df["date"], y=chart_df["predicted"],
        mode="lines+markers", name="ML Predicted",
        line=dict(color="#e91e63", width=2, dash="dot"),
        marker=dict(size=3, color="#e91e63"),
    ), row=1, col=1)

    chart_df["error"] = chart_df["predicted"] - chart_df["actual"]
    bar_colors = ["rgba(38,166,154,0.5)" if e >= 0 else "rgba(239,83,80,0.5)" for e in chart_df["error"]]
    fig.add_trace(go.Bar(
        x=chart_df["date"], y=chart_df["error"],
        name="Error", marker_color=bar_colors,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#333", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11, color="#555"),
        ),
        margin=dict(l=50, r=30, t=60, b=40),
        height=520,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(0,0,0,.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.06)", linecolor="rgba(0,0,0,.08)")

    # ════════════════════════════════════════════════════════════════════
    # 3.  Technical Indicators & Signal Breakdown
    # ════════════════════════════════════════════════════════════════════
    rsi      = float(latest["rsi_14"])
    sma10    = float(latest["sma_10"])
    sma50    = float(latest["sma_50"])
    macd     = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    bb_upper = float(latest.get("bb_upper", 0))
    bb_lower = float(latest.get("bb_lower", 0))

    rsi_label = "Overbought ⚠️" if rsi > 70 else ("Oversold ⚠️" if rsi < 30 else "Neutral Zone ✅")
    rsi_color = "#ef5350" if rsi > 70 else ("#ef5350" if rsi < 30 else "#26a69a")
    sma_status = "Golden Cross ✅" if sma10 > sma50 else "Death Cross ⚠️"
    sma_col = "#26a69a" if sma10 > sma50 else "#ef5350"
    macd_status = "Bullish ✅" if macd > macd_sig else "Bearish ⚠️"
    macd_col = "#26a69a" if macd > macd_sig else "#ef5350"

    def _card(icon, label, value, sub, sub_color):
        return f"""
        <div style="
            background: #ffffff; border-radius:16px;
            padding:22px 24px; flex:1; min-width:180px;
            border:1px solid #eaecef;
            box-shadow: 0 1px 4px rgba(0,0,0,.04);
        ">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                <span style="font-size:18px;">{icon}</span>
                <span style="font-size:11px; color:#707a8a; text-transform:uppercase; letter-spacing:1px; font-weight:600;">{label}</span>
            </div>
            <div style="font-size:24px; font-weight:700; color:#1e2329; margin-bottom:4px;">{value}</div>
            <div style="font-size:12px; color:{sub_color}; font-weight:600;">{sub}</div>
        </div>
        """

    # Evaluate individual signal states
    signals_list = [
        ("Wilder's RSI (14)", f"{rsi:.2f}", rsi_label.replace(" ⚠️", "").replace(" ✅", ""), rsi_label, rsi_color),
        ("SMA 10 / 50 Crossover", f"{currency_sym}{sma10:,.2f} / {currency_sym}{sma50:,.2f}", "BUY" if sma10 > sma50 else "SELL", sma_status, sma_col),
        ("MACD Crossover", f"{macd:.4f} (Signal: {macd_sig:.4f})", "BUY" if macd > macd_sig else "SELL", macd_status, macd_col),
        ("Bollinger Bands (20, 2σ)", f"{currency_sym}{current_price:,.2f}", "BUY" if current_price <= bb_lower else ("SELL" if current_price >= bb_upper else "NEUTRAL"), f"Upper: {currency_sym}{bb_upper:,.2f} | Lower: {currency_sym}{bb_lower:,.2f}", "#26a69a" if current_price <= bb_lower else ("#ef5350" if current_price >= bb_upper else "#707a8a")),
    ]
    
    table_rows = "".join(f"""
    <tr style="border-bottom: 1px solid #eaecef;">
        <td style="padding: 12px; font-weight: 600; color: #1e2329;">{name}</td>
        <td style="padding: 12px; color: #474d57; font-family: monospace; font-size: 13px;">{val}</td>
        <td style="padding: 12px;">
            <span style="background: {color}14; color: {color}; padding: 4px 12px; border-radius: 12px; font-weight: 700; font-size: 11px;">{sig}</span>
        </td>
        <td style="padding: 12px; color: #707a8a; font-size: 12px;">{desc}</td>
    </tr>
    """ for name, val, sig, desc, color in signals_list)
    
    table_html = f"""
    <div style="background: #ffffff; border-radius: 16px; border: 1px solid #eaecef; padding: 20px; margin-top: 14px; box-shadow: 0 1px 4px rgba(0,0,0,.04); font-family: 'Inter', sans-serif; width: 100%;">
        <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 14px; font-weight: 700; color: #1e2329; text-transform: uppercase; letter-spacing: 1px;">💡 Indicator Signal Breakdown</h3>
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead>
                <tr style="border-bottom: 2px solid #eaecef; color: #707a8a; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">
                    <th style="padding: 12px; font-weight: 700;">Indicator</th>
                    <th style="padding: 12px; font-weight: 700;">Current Value</th>
                    <th style="padding: 12px; font-weight: 700;">Signal</th>
                    <th style="padding: 12px; font-weight: 700;">Trigger Condition</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """
    
    indicators_html = f"""
    <div style="display:flex; gap:14px; flex-wrap:wrap; font-family:'Inter','Segoe UI',sans-serif; width: 100%;">
        {_card("📉", "RSI (14)", f"{rsi:.2f}", rsi_label, rsi_color)}
        {_card("📊", "SMA 10 / 50", f"{currency_sym}{sma10:,.2f} / {currency_sym}{sma50:,.2f}", sma_status, sma_col)}
        {_card("📈", "MACD", f"{macd:.4f}", f"Signal: {macd_sig:.4f} — {macd_status}", macd_col)}
        {_card("🎯", "Bollinger Bands", f"{currency_sym}{bb_upper:,.2f}", f"Lower: {currency_sym}{bb_lower:,.2f}", "#2962ff")}
        {table_html}
    </div>
    """

    # ════════════════════════════════════════════════════════════════════
    # 4.  News
    # ════════════════════════════════════════════════════════════════════
    try:
        loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop2)
        headlines = loop2.run_until_complete(fetch_news_headlines(ticker))
        loop2.close()
    except Exception:
        headlines = ["Headlines currently unavailable."]

    news_items = "".join(
        f"""<div style="
            padding:14px 18px; margin-bottom:10px;
            background:#ffffff; border-radius:12px;
            border-left:3px solid #2962ff; color:#474d57; font-size:13px;
            line-height:1.5; box-shadow: 0 1px 3px rgba(0,0,0,.04);
        ">📰&nbsp; {h}</div>"""
        for h in headlines
    )
    news_html = f'<div style="font-family:\'Inter\',\'Segoe UI\',sans-serif;">{news_items}</div>'

    # ════════════════════════════════════════════════════════════════════
    # 5.  Agent Logs
    # ════════════════════════════════════════════════════════════════════
    today_str = date.today().strftime("%Y-%m-%d")
    logs = [
        f"[THOUGHT]      Analyzing pipeline for {asset_name} ({ticker})  •  {today_str}",
        f"[ACTION]       Detected: {market_text}. Loading regional presets...",
        f"[ACTION]       Downloaded {timeframe_label} data slice. Normalizing columns...",
        f"[ACTION]       Computing Wilder's RSI-14, EMA-MACD, Bollinger Bands...",
        f"[OBSERVATION]  RSI-14 = {rsi:.2f}  |  SMA-10 = {sma10:.2f}  |  SMA-50 = {sma50:.2f}",
        f"[ACTION]       Building lag-3 feature matrix. Isolating forecast row...",
        f"[ACTION]       Training LinearRegression (80/20 chronological split)...",
        f"[OBSERVATION]  Model R² = {r2}  |  Test samples = {len(hist_pred)}",
        f"[OBSERVATION]  Forecast: {currency_sym}{forecast:,.2f}  ({pct_change:+.2f}%)",
        f"[RESULT]       ✅ {direction} with {confidence}% confidence",
    ]
    logs_text = "\n".join(logs)

    return price_html, live_chart, fig, indicators_html, news_html, logs_text


# ═══════════════════════════════════════════════════════════════════════════
# Market Scanner Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def _generate_sparkline_svg(prices, is_positive=True, width=100, height=30):
    if not prices or len(prices) < 2:
        return ""
    try:
        # Normalize prices between 3 and height-3
        min_p, max_p = min(prices), max(prices)
        rng = max_p - min_p if max_p != min_p else 1.0
        
        points = []
        for i, p in enumerate(prices):
            x = (i / (len(prices) - 1)) * width
            y = height - ((p - min_p) / rng) * (height - 6) - 3
            points.append(f"{x},{y}")
            
        color = "#26a69a" if is_positive else "#ef5350"
        # Generate random IDs to avoid conflicts inside DOM
        import uuid
        uid = uuid.uuid4().hex[:6]
        glow_id = f"glow-{uid}"
        grad_id = f"grad-{uid}"
        
        path_data = "M " + " L ".join(points)
        
        # Area path
        area_points = list(points)
        area_points.append(f"{width},{height}")
        area_points.append(f"0,{height}")
        area_path_data = "M " + " L ".join(area_points) + " Z"
        
        svg = f"""
        <svg width="{width}" height="{height}" style="overflow: visible; display: inline-block; vertical-align: middle;">
            <defs>
                <filter id="{glow_id}" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="2" result="blur" />
                    <feComponentTransfer in="blur" result="glow1">
                        <feFuncA type="linear" slope="0.6"/>
                    </feComponentTransfer>
                    <feMerge>
                        <feMergeNode in="glow1" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
                <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="{color}" stop-opacity="0.25"/>
                    <stop offset="100%" stop-color="{color}" stop-opacity="0.0"/>
                </linearGradient>
            </defs>
            <path d="{area_path_data}" fill="url(#{grad_id})" stroke="none" />
            <!-- Glowing background path -->
            <path d="{path_data}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round" opacity="0.45" filter="url(#{glow_id})" />
            <!-- Crisp foreground path -->
            <path d="{path_data}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" />
        </svg>
        """
        return svg
    except Exception:
        return ""


def render_scan_results_html():
    results_path = r"c:\Users\athar\OneDrive\Desktop\Stock Predictor ML\market_scan_results.json"
    if not os.path.exists(results_path):
        return """
        <div style="text-align: center; padding: 50px 20px; font-family: 'Inter', sans-serif;">
            <p style="font-size: 16px; color: #707a8a; font-weight: 500;">No scan results found.</p>
            <p style="font-size: 13px; color: #929aa5; margin-top: 8px; margin-bottom: 24px;">Please click the button below to run the first market scan.</p>
        </div>
        """
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"<p style='color: red; font-family: sans-serif;'>Error reading scan results: {e}</p>"

    scan_time = data.get("scan_time", "Unknown")
    
    # 1. Render Overall Leaders
    def render_leader_card(title, assets, is_tomorrow=False):
        rows = ""
        for i, item in enumerate(assets):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "•"
            perf = item["tomorrow_perf"] if is_tomorrow else item["today_perf"]
            perf_color = "#26a69a" if perf >= 0 else "#ef5350"
            perf_sign = "+" if perf >= 0 else ""
            price_val = item["tomorrow_forecast"] if is_tomorrow else item["price"]
            
            # Detect currency for item
            ticker = item["ticker"]
            if ticker.endswith(".NS") or ticker.endswith(".BO"):
                currency_sym = "₹"
            elif "-" in ticker:
                currency_sym = "$"
            else:
                currency_sym = "$"

            rows += f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; background: #f8f9fa; border-radius: 12px; margin-bottom: 8px; border: 1px solid #eaecef;">
                <div style="display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0;">
                    <span style="font-size: 16px;">{rank_emoji}</span>
                    <div style="min-width: 0; flex: 1;">
                        <div style="font-weight: 700; color: #1e2329; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['name']}</div>
                        <div style="font-size: 11px; color: #707a8a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['ticker']} · {item['category']}</div>
                    </div>
                </div>
                <div style="margin: 0 16px; display: flex; align-items: center; justify-content: center;">
                    {_generate_sparkline_svg(item.get('sparkline_prices', []), perf >= 0, width=80, height=25)}
                </div>
                <div style="text-align: right; min-width: 80px;">
                    <div style="font-weight: 700; color: #1e2329; font-size: 13px;">{currency_sym}{price_val:,.2f}</div>
                    <div style="color: {perf_color}; font-weight: 700; font-size: 11px;">{perf_sign}{perf:.2f}%</div>
                </div>
            </div>
            """
        return f"""
        <div style="flex: 1; min-width: 300px; background: #ffffff; border-radius: 16px; padding: 20px; border: 1px solid #eaecef; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
            <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 14px; font-weight: 800; color: #1e2329; letter-spacing: 0.5px; border-bottom: 2px solid #f0f2f5; padding-bottom: 10px; text-transform: uppercase;">{title}</h3>
            {rows}
        </div>
        """

    overall_html = f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px;">
        {render_leader_card("🏆 Today's Top Performers (Actual)", data.get("overall_top_today", []))}
        {render_leader_card("🔮 Tomorrow's Predicted Leaders (Forecast)", data.get("overall_top_tomorrow", []), is_tomorrow=True)}
    </div>
    """

    # 2. Render Sector/Category-wise Top Performers Grid
    grid_cards = ""
    for cat_name, cat_data in data.get("categories", {}).items():
        # Render top today for category
        today_rows = ""
        for i, item in enumerate(cat_data.get("top_today", [])):
            perf = item["today_perf"]
            perf_color = "#26a69a" if perf >= 0 else "#ef5350"
            perf_sign = "+" if perf >= 0 else ""
            today_rows += f"""
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; margin-bottom: 8px; padding: 4px 0;">
                <div style="display: flex; flex-direction: column; min-width: 0; flex: 1;">
                    <span style="font-weight: 600; color: #474d57; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90px;" title="{item['name']}">{item['name']}</span>
                    <span style="font-size: 10px; color: #929aa5;">{item['price']:,.2f}</span>
                </div>
                <div style="margin: 0 6px; display: flex; align-items: center;">
                    {_generate_sparkline_svg(item.get('sparkline_prices', []), perf >= 0, width=55, height=18)}
                </div>
                <span style="color: {perf_color}; font-weight: 700; min-width: 45px; text-align: right;">{perf_sign}{perf:.2f}%</span>
            </div>
            """
            
        # Render top tomorrow for category
        tomorrow_rows = ""
        for i, item in enumerate(cat_data.get("top_tomorrow", [])):
            perf = item["tomorrow_perf"]
            perf_color = "#26a69a" if perf >= 0 else "#ef5350"
            perf_sign = "+" if perf >= 0 else ""
            tomorrow_rows += f"""
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; margin-bottom: 8px; padding: 4px 0;">
                <div style="display: flex; flex-direction: column; min-width: 0; flex: 1;">
                    <span style="font-weight: 600; color: #474d57; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90px;" title="{item['name']}">{item['name']}</span>
                    <span style="font-size: 10px; color: #929aa5;">{item['tomorrow_forecast']:,.2f}</span>
                </div>
                <div style="margin: 0 6px; display: flex; align-items: center;">
                    {_generate_sparkline_svg(item.get('sparkline_prices', []), perf >= 0, width=55, height=18)}
                </div>
                <span style="color: {perf_color}; font-weight: 700; min-width: 45px; text-align: right;">{perf_sign}{perf:.2f}%</span>
            </div>
            """
            
        grid_cards += f"""
        <div style="background: #ffffff; border-radius: 16px; padding: 18px; border: 1px solid #eaecef; box-shadow: 0 1px 4px rgba(0,0,0,0.03); display: flex; flex-direction: column;">
            <div style="font-size: 13px; font-weight: 800; color: #1e2329; margin-bottom: 12px; border-bottom: 1px solid #f0f2f5; padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">{cat_name}</div>
            <div style="display: flex; gap: 14px; flex: 1;">
                <div style="flex: 1; min-width: 0;">
                    <div style="font-size: 9px; text-transform: uppercase; color: #707a8a; letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 700;">Today</div>
                    {today_rows or "<div style='font-size:11px;color:#929aa5;'>No data</div>"}
                </div>
                <div style="width: 1px; background: #eaecef;"></div>
                <div style="flex: 1; min-width: 0;">
                    <div style="font-size: 9px; text-transform: uppercase; color: #707a8a; letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 700;">Tomorrow</div>
                    {tomorrow_rows or "<div style='font-size:11px;color:#929aa5;'>No data</div>"}
                </div>
            </div>
        </div>
        """

    grid_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; margin-bottom: 20px;">
        {grid_cards}
    </div>
    """

    header_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; font-family: 'Inter', sans-serif; flex-wrap: wrap; gap: 10px;">
        <div>
            <h2 style="margin: 0; font-size: 20px; font-weight: 800; color: #1e2329;">📊 Market Performance Dashboard</h2>
            <p style="margin: 4px 0 0 0; font-size: 12px; color: #707a8a;">Real-time category-wise top actual gainers and tomorrow's ML forecasts (L2 Regularized)</p>
        </div>
        <div style="background: rgba(41,98,255,0.06); border: 1px solid rgba(41,98,255,0.15); color: #2962ff; border-radius: 20px; padding: 6px 14px; font-size: 11px; font-weight: 600; font-family: monospace;">
            Last Scan: {scan_time}
        </div>
    </div>
    """

    return f"""
    <div style="font-family: 'Inter', sans-serif; background: #f0f2f5; padding: 2px;">
        {header_html}
        {overall_html}
        <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 15px; font-weight: 800; color: #1e2329; font-family: 'Inter', sans-serif;">📂 Sectoral Top Performers</h3>
        {grid_html}
    </div>
    """

def run_market_scan_btn():
    import sys
    script_path = r"C:\Users\athar\.gemini\antigravity-ide\brain\8c5cdf5f-8218-43c4-8de3-bf7f62c5c83b\scratch\market_scanner.py"
    
    logs = []
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Launching Market Scanner background task...")
    
    try:
        # Run the script in a subprocess
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )
        
        # Read output in real time
        for line in process.stdout:
            logs.append(line.strip())
            
        process.wait()
        
        if process.returncode == 0:
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Market Scan completed successfully!")
        else:
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Market Scan failed with exit code {process.returncode}.")
    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Exception occurred during scan: {e}")
        
    # Reload and render HTML
    html_content = render_scan_results_html()
    return html_content, "\n".join(logs)


# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Light Theme Canvas ────────────────────────────────────────────── */
.gradio-container {
    background: #f0f2f5 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
}

/* ── Header ─────────────────────────────────────────────────────────── */
#app-header {
    text-align: center;
    padding: 32px 20px 16px;
}
#app-header h1 {
    font-size: 36px; font-weight: 900;
    background: linear-gradient(135deg, #2962ff 0%, #1565c0 40%, #0d47a1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
#app-header .subtitle {
    color: #707a8a; font-size: 13px; margin-top: 8px;
    letter-spacing: .6px; line-height: 1.5;
}
#app-header .badge-row {
    display: flex; justify-content: center; gap: 8px;
    flex-wrap: wrap; margin-top: 14px;
}
#app-header .badge {
    background: rgba(41,98,255,.06);
    border: 1px solid rgba(41,98,255,.15);
    color: #2962ff; font-size: 11px; font-weight: 600;
    padding: 4px 14px; border-radius: 20px;
    letter-spacing: .4px;
}

/* ── Selection Panel ────────────────────────────────────────────────── */
#selection-panel {
    background: #ffffff !important;
    border: 1px solid #eaecef !important;
    border-radius: 20px !important;
    padding: 20px 24px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.04) !important;
}

/* ── Predict Button ─────────────────────────────────────────────────── */
#predict-btn {
    background: linear-gradient(135deg, #2962ff 0%, #1e88e5 50%, #1565c0 100%) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 36px !important;
    box-shadow: 0 4px 16px rgba(41,98,255,.3) !important;
    transition: all .2s ease !important;
    min-height: 52px !important;
    letter-spacing: .3px !important;
}
#predict-btn:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 24px rgba(41,98,255,.45) !important;
}
#predict-btn:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ── Section Titles ─────────────────────────────────────────────────── */
.section-title {
    font-size: 12px !important;
    color: #707a8a !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    font-weight: 700 !important;
    margin: 8px 0 !important;
    padding-left: 4px !important;
}

/* ── Logs (keep dark for terminal feel) ─────────────────────────────── */
#logs-box textarea {
    background: #1e2329 !important;
    color: #0ecb81 !important;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
    font-size: 11.5px !important;
    border: 1px solid #2b3139 !important;
    border-radius: 14px !important;
    line-height: 1.6 !important;
}

/* ── Category Radio ─────────────────────────────────────────────────── */
#category-radio .wrap {
    gap: 6px !important;
}
#category-radio label {
    border-radius: 10px !important;
    font-size: 12px !important;
    padding: 8px 14px !important;
}

/* ── Chart Controls ─────────────────────────────────────────────────── */
#chart-controls {
    background: #ffffff !important;
    border: 1px solid #eaecef !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.04) !important;
}
#chart-controls .wrap {
    gap: 6px !important;
}

/* ── Hide footer ────────────────────────────────────────────────────── */
footer { display: none !important; }
"""


# ═══════════════════════════════════════════════════════════════════════════
# Build the UI
# ═══════════════════════════════════════════════════════════════════════════

HEADER_HTML = """
<div id="app-header">
    <h1>🚀 AI Stock Prediction Engine</h1>
    <p class="subtitle">
        Machine-learning powered price forecasting for Stocks, ETFs, Crypto,
        Commodities, Forex & more
    </p>
    <div class="badge-row">
        <span class="badge">yFinance Data</span>
        <span class="badge">Linear Regression</span>
        <span class="badge">Wilder's RSI</span>
        <span class="badge">MACD</span>
        <span class="badge">Bollinger Bands</span>
        <span class="badge">350+ Assets</span>
    </div>
</div>
"""

# Default category & asset
DEFAULT_CATEGORY = "🇮🇳 Indian Stocks"
DEFAULT_ASSETS = get_assets_for_category(DEFAULT_CATEGORY)
DEFAULT_ASSET = "Reliance Industries"


def build_interface():
    with gr.Blocks(title="AI Stock Prediction Engine") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            with gr.Tab("📈 Single Asset Predictor"):
                # ── Selection Panel ─────────────────────────────────────────────
                with gr.Group(elem_id="selection-panel"):
                    # Category selector
                    gr.HTML('<p class="section-title">📂 Choose Category</p>')
                    category_radio = gr.Radio(
                        choices=get_category_names(),
                        value=DEFAULT_CATEGORY,
                        label="",
                        elem_id="category-radio",
                    )

                    with gr.Row():
                        # Asset dropdown (searchable)
                        asset_dropdown = gr.Dropdown(
                            choices=DEFAULT_ASSETS,
                            value=DEFAULT_ASSET,
                            label="🔍 Search & Select Asset",
                            filterable=True,
                            allow_custom_value=True,
                            scale=4,
                            info="Type to search by company name",
                        )
                        # Timeframe
                        timeframe_dropdown = gr.Dropdown(
                            choices=list(TIMEFRAME_OPTIONS.keys()),
                            value="1 Day",
                            label="⏱️ Timeframe",
                            scale=1,
                        )
                        # Predict button
                        predict_btn = gr.Button(
                            "⚡ Analyze & Predict",
                            elem_id="predict-btn",
                            scale=1,
                        )

                # ── Chart Controls ──────────────────────────────────────────────
                with gr.Group(elem_id="chart-controls"):
                    gr.HTML('<p class="section-title">📊 Chart Settings & Indicators</p>')
                    with gr.Row():
                        chart_type_dropdown = gr.Dropdown(
                            choices=CHART_TYPES,
                            value="Candlestick",
                            label="📈 Chart Type",
                            scale=2,
                        )
                        overlay_checkboxes = gr.CheckboxGroup(
                            choices=OVERLAY_INDICATORS,
                            value=["SMA 20", "Bollinger Bands"],
                            label="🔗 Overlay Indicators",
                            scale=4,
                        )
                        subplot_checkboxes = gr.CheckboxGroup(
                            choices=SUBPLOT_INDICATORS,
                            value=["Volume"],
                            label="📉 Subplot Indicators",
                            scale=3,
                        )

                # ── Output Sections ─────────────────────────────────────────────
                price_output = gr.HTML(label="")

                gr.HTML('<p class="section-title">📊 Live Market Chart</p>')
                live_chart_output = gr.Plot(label="")

                gr.HTML('<p class="section-title">🤖 ML Prediction — Actual vs Predicted</p>')
                pred_chart_output = gr.Plot(label="")

                gr.HTML('<p class="section-title">📊 Technical Indicators</p>')
                indicators_output = gr.HTML(label="")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML('<p class="section-title">📰 Market Headlines</p>')
                        news_output = gr.HTML(label="")
                    with gr.Column(scale=3):
                        gr.HTML('<p class="section-title">🤖 Agent Execution Log</p>')
                        logs_output = gr.Textbox(
                            label="", lines=12, interactive=False,
                            elem_id="logs-box",
                        )

                # ── Quick Picks ─────────────────────────────────────────────────
                gr.HTML('<p class="section-title">⚡ Quick Picks</p>')
                gr.Examples(
                    examples=[
                        ["Reliance Industries",   "1 Day",   "🇮🇳 Indian Stocks"],
                        ["Tata Consultancy Services", "1 Month", "🇮🇳 Indian Stocks"],
                        ["HDFC Bank",             "1 Week",  "🇮🇳 Indian Stocks"],
                        ["Infosys",               "1 Day",   "🇮🇳 Indian Stocks"],
                        ["Apple",                 "1 Day",   "🇺🇸 US Stocks"],
                        ["Bitcoin (BTC)",         "1 Day",   "₿ Crypto"],
                        ["Gold (COMEX)",          "1 Day",   "🏆 Commodities & Futures"],
                        ["NIFTY 50",              "1 Month", "🇮🇳 Indian Indices"],
                        ["USD/INR (Dollar to Rupee)", "1 Day", "💱 Forex"],
                        ["NVIDIA",                "1 Day",   "🇺🇸 US Stocks"],
                    ],
                    inputs=[asset_dropdown, timeframe_dropdown, category_radio],
                    label="",
                )

            with gr.Tab("📊 Market Scanner & Top Performers"):
                # Market scan output HTML
                scanner_html_output = gr.HTML(
                    value=render_scan_results_html(),
                    elem_id="scanner-dashboard"
                )
                
                with gr.Row():
                    run_scan_btn = gr.Button(
                        "🔄 Run Full Market Scan",
                        elem_id="predict-btn", # reuse styles for premium look
                        scale=1
                    )
                    
                scanner_logs_output = gr.Textbox(
                    label="Scanner Execution Log",
                    lines=8,
                    interactive=False,
                    elem_id="logs-box",
                    value="Scanner ready. Click 'Run Full Market Scan' to refresh."
                )

        # ── Wire Events ─────────────────────────────────────────────────
        # Category change → update asset dropdown
        category_radio.change(
            fn=update_asset_dropdown,
            inputs=[category_radio],
            outputs=[asset_dropdown],
        )

        # All inputs for prediction
        all_inputs = [
            asset_dropdown, timeframe_dropdown, category_radio,
            chart_type_dropdown, overlay_checkboxes, subplot_checkboxes,
        ]
        all_outputs = [
            price_output, live_chart_output, pred_chart_output,
            indicators_output, news_output, logs_output,
        ]

        # Predict button
        predict_btn.click(
            fn=predict,
            inputs=all_inputs,
            outputs=all_outputs,
        )
        
        # Run Scan button
        run_scan_btn.click(
            fn=run_market_scan_btn,
            inputs=[],
            outputs=[scanner_html_output, scanner_logs_output]
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.gray,
            font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "sans-serif"],
        ),
    )
