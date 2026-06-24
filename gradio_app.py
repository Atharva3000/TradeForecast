"""
AI Stock Prediction Engine — Gradio Interface v3.

Premium, high-end financial dashboard with:
  • Persistent left-hand glassmorphic sidebar navigation (Dashboard, Advanced Predictor, Global Scanner, Backtest Sandbox, Settings)
  • Command Bar search input with filter pills and popular searches overlay
  • Compact settings control bar directly above charts (Chart Type radio, Indicators dropdown)
  • Dynamic ML models (Ridge Regression, XGBoost Regressor, Deep Learning Neural Net) and customizable lag Lookback
  • Plotly interactive actual vs predicted chart and horizontal bar chart for feature importances
  • Lexicon sentiment analysis speedometer macro sentiment gauge next to Headlines
  • Full walk-forward consensus strategy historical backtester returning Sharpe, drawdown, win rate, and equity curve
  • Multi-chart comparative grid (2-chart or 4-chart side-by-side splits)
"""

import asyncio
import logging
import os
import json
import subprocess
from datetime import date, datetime

import gradio as gr
import pandas as pd
import numpy as np
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
    search_assets,
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
# Feature Importance Chart Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_importance_chart(feat_imp: dict, is_dark: bool = False) -> go.Figure:
    if not feat_imp:
        fig = go.Figure()
        fig.add_annotation(text="No feature importance available for this model", showarrow=False, font=dict(size=14, family="Inter, sans-serif"))
        fig.update_layout(
            template="plotly_dark" if is_dark else "plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig
    
    sorted_imp = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
    ))
    
    text_color = "#eaecef" if is_dark else "#333333"
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.06)"
    
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Segoe UI, sans-serif", color=text_color, size=10),
        margin=dict(l=100, r=20, t=30, b=20),
        height=250,
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=True, gridcolor=grid_color)
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Core prediction function
# ═══════════════════════════════════════════════════════════════════════════


def build_placeholder_chart(message: str, is_dark: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=12, family="Inter, sans-serif", color="#848e9c" if is_dark else "#707a8a"),
        x=0.5, y=0.5,
        xref="paper", yref="paper"
    )
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def _error_fallback(message: str, is_dark: bool = False) -> tuple:
    price_html = f"""
    <div style="
        background: rgba(239, 83, 80, 0.1);
        border-radius: 16px; padding: 20px 24px;
        border: 1px solid rgba(239, 83, 80, 0.3);
        box-shadow: 0 4px 12px rgba(229,62,62,.05);
        font-family: 'Inter', sans-serif;
        color: #ef5350;
    ">
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:24px;">⚠️</span>
            <div>
                <div style="font-weight: 800; font-size: 15px;">Analysis Stopped</div>
                <div style="font-size: 13px; margin-top: 2px; color: #ef5350;">{message}</div>
            </div>
        </div>
    </div>
    """
    
    live_chart = build_placeholder_chart(f"Market Chart: {message}", is_dark)
    pred_chart = build_placeholder_chart(f"ML Forecast: {message}", is_dark)
    imp_fig = build_placeholder_chart(f"ML Feature Importance: {message}", is_dark)
    
    indicators_html = f"""
    <div style="background:var(--bg-card); border-radius:12px; border:1px solid var(--border-color); padding:16px; font-family:'Inter',sans-serif; color:var(--text-muted); text-align:center;">
        Indicator analysis not computed.
    </div>
    """
    
    news_html = f"""
    <div style="background:var(--bg-card); border-radius:12px; border:1px solid var(--border-color); padding:16px; font-family:'Inter',sans-serif; color:var(--text-muted); text-align:center;">
        News Headlines feed not loaded.
    </div>
    """
    
    logs_text = f"[WARNING] Analysis not completed: {message}\n[TIP] Please search for another asset or select a larger timeframe."
    
    sentiment_fig = build_placeholder_chart("Sentiment Gauge: Not loaded", is_dark)
    
    return (
        price_html,
        live_chart,
        pred_chart,
        indicators_html,
        news_html,
        logs_text,
        sentiment_fig,
        imp_fig
    )


# ═══════════════════════════════════════════════════════════════════════════
# Core prediction function
# ═══════════════════════════════════════════════════════════════════════════

def predict(
    asset_name: str,
    timeframe_label: str,
    chart_type: str = "🕯️ Candle",
    selected_indicators: list[str] | None = None,
    model_type: str = "Ridge Regression",
    lag_period: int = 3,
    is_dark: bool = False,
):
    """Run full analysis & model predictions."""
    try:
        if selected_indicators is None:
            selected_indicators = []
        if not asset_name:
            return _error_fallback("Please search and select a valid asset first.", is_dark)

        ticker = resolve_ticker(asset_name)
        if ticker is None:
            return _error_fallback(f"'{asset_name}' ticker could not be resolved from registry.", is_dark)

        # Chart Type Mapping
        chart_type_map = {
            "🕯️ Candle": "Candlestick",
            "📈 Line": "Line",
            "📊 OHLC": "OHLC Bars",
            "✨ Heikin Ashi": "Heikin Ashi"
        }
        actual_chart_type = chart_type_map.get(chart_type, "Candlestick")

        # Split overlay vs subplot
        overlay_indicators = [ind for ind in selected_indicators if ind in OVERLAY_INDICATORS]
        subplot_indicators = [ind for ind in selected_indicators if ind in SUBPLOT_INDICATORS]

        timeframe = TIMEFRAME_OPTIONS.get(timeframe_label, "1d")
        currency_sym, currency_code = _detect_currency(ticker)

        # Fetch
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df = loop.run_until_complete(fetch_stock_data(ticker, timeframe))
        loop.close()

        if df.empty:
            return _error_fallback(f"No historical market data found for '{asset_name}' ({ticker}). Try another timeframe.", is_dark)

        # Predict
        results = run_prediction(df, model_type=model_type, lag_period=int(lag_period))

        latest        = results["latest_df"]
        forecast      = results["next_day_forecast"]
        r2            = results["r2_score"]
        hist_pred     = results["historical_vs_predicted"]
        current_price = float(latest["close"])
        confidence    = compute_confidence(latest)

        # Build Charts
        live_chart = build_stock_chart(
            df=df,
            chart_type=actual_chart_type,
            overlays=overlay_indicators,
            subplots=subplot_indicators,
            asset_name=asset_name,
            currency_sym=currency_sym,
            is_dark=is_dark,
        )

        if forecast > current_price * 1.005:
            direction, dir_color, dir_icon = "Bullish", "#26a69a", "▲"
        elif forecast < current_price * 0.995:
            direction, dir_color, dir_icon = "Bearish", "#ef5350", "▼"
        else:
            direction, dir_color, dir_icon = "Neutral", "#ff9800", "●"

        pct_change = ((forecast - current_price) / current_price) * 100
        market_text = _detect_market(ticker)
        
        # Category resolution
        category = None
        for cat, assets in CATEGORIES.items():
            if asset_name in assets:
                category = cat
                break
        if not category:
            category = "🇮🇳 Indian Stocks"
            
        emoji = _get_asset_emoji(category)

        # Sparkline
        spark_closes = df["close"].tail(15).tolist()
        spark_closes = [float(p) for p in spark_closes if not pd.isna(p)]
        single_sparkline = _generate_sparkline_svg(spark_closes, forecast >= current_price, width=120, height=35)

        # Price card HTML (styled using CSS variables for theme support)
        conf_bar_color = "#26a69a" if confidence >= 60 else ("#ff9800" if confidence >= 40 else "#ef5350")
        price_html = f"""
        <div style="
            background: var(--bg-card);
            border-radius: 20px; padding: 24px 30px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-card);
            font-family: 'Inter', sans-serif;
            transition: background 0.3s ease, border-color 0.3s ease;
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <span style="font-size:36px;">{emoji}</span>
                    <div>
                        <div style="font-size:22px; font-weight:800; color:var(--text-primary); line-height:1.2;">{asset_name}</div>
                        <div style="font-size:12px; color:var(--text-muted); margin-top:3px;">{ticker} &nbsp;·&nbsp; {market_text}</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:16px;">
                    {single_sparkline}
                    <span style="
                        background:{dir_color}14; color:{dir_color};
                        padding:6px 16px; border-radius:30px; font-size:13px;
                        font-weight:700; border: 1px solid {dir_color}33;
                    ">{dir_icon} {direction}</span>
                </div>
            </div>

            <div style="height:1px; background:var(--border-color); margin:16px 0;"></div>

            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:16px;">
                <div>
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; font-weight:700;">Current Price</div>
                    <div style="font-size:26px; font-weight:800; color:var(--text-primary); margin-top:4px;">
                        {currency_sym}{current_price:,.2f}
                    </div>
                    <div style="font-size:11px; color:var(--text-muted);">{currency_code}</div>
                </div>
                <div>
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; font-weight:700;">Tomorrow Forecast</div>
                    <div style="font-size:26px; font-weight:800; color:{dir_color}; margin-top:4px;">
                        {currency_sym}{forecast:,.2f}
                    </div>
                    <div style="font-size:11px; color:{dir_color}; font-weight:700;">
                        {dir_icon} {pct_change:+.2f}%
                    </div>
                </div>
                <div>
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; font-weight:700;">Confidence</div>
                    <div style="font-size:26px; font-weight:800; color:{conf_bar_color}; margin-top:4px;">
                        {confidence}%
                    </div>
                    <div style="margin-top:6px; background:var(--bg-primary); border-radius:6px; height:6px; width:100%; max-width:120px;">
                        <div style="height:100%; width:{confidence}%; background:{conf_bar_color}; border-radius:6px;"></div>
                    </div>
                </div>
                <div>
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; font-weight:700;">Model R² ({model_type})</div>
                    <div style="font-size:26px; font-weight:800; color:#2962ff; margin-top:4px;">
                        {r2}
                    </div>
                    <div style="font-size:11px; color:var(--text-muted);">Lookback Lag: {lag_period} days</div>
                </div>
            </div>
        </div>
        """

        # Plotly Error comparison chart
        if not hist_pred or len(hist_pred) == 0:
            fig = build_placeholder_chart("No prediction history available for residual plotting.", is_dark)
        else:
            chart_df = pd.DataFrame(hist_pred)
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.75, 0.25],
                vertical_spacing=0.06,
                subplot_titles=(
                    "Actual vs ML Forecast History",
                    "Residual Forecast Error"
                ),
            )
            fig.add_trace(go.Scatter(
                x=chart_df["date"], y=chart_df["actual"],
                mode="lines", name="Actual Price",
                line=dict(color="#2962ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(41,98,255,.04)"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=chart_df["date"], y=chart_df["predicted"],
                mode="lines", name="Predicted",
                line=dict(color="#e91e63", width=1.8, dash="dash")
            ), row=1, col=1)
            chart_df["error"] = chart_df["predicted"] - chart_df["actual"]
            bar_colors = ["rgba(38,166,154,0.5)" if e >= 0 else "rgba(239,83,80,0.5)" for e in chart_df["error"]]
            fig.add_trace(go.Bar(
                x=chart_df["date"], y=chart_df["error"],
                name="Error", marker_color=bar_colors
            ), row=2, col=1)
            
            fig.update_layout(
                template="plotly_dark" if is_dark else "plotly_white",
                paper_bgcolor="#181a20" if is_dark else "#ffffff",
                plot_bgcolor="#181a20" if is_dark else "#ffffff",
                font=dict(family="Inter, sans-serif", color="#eaecef" if is_dark else "#333333", size=11),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(color="#eaecef" if is_dark else "#333333"),
                    bgcolor="rgba(24, 26, 32, 0.9)" if is_dark else "rgba(255,255,255,0.9)"
                ),
                margin=dict(l=40, r=20, t=40, b=30),
                height=320,
                hovermode="x unified"
            )
            fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0,0,0,0.06)", linecolor="rgba(255, 255, 255, 0.15)" if is_dark else "rgba(0,0,0,0.1)")
            fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0,0,0,0.06)", linecolor="rgba(255, 255, 255, 0.15)" if is_dark else "rgba(0,0,0,0.1)")

        # Technical Indicators table
        rsi      = float(latest["rsi_14"])
        sma10    = float(latest["sma_10"])
        sma50    = float(latest["sma_50"])
        macd     = float(latest["macd"])
        macd_sig = float(latest["macd_signal"])
        bb_upper = float(latest.get("bb_upper", 0))
        bb_lower = float(latest.get("bb_lower", 0))

        rsi_label = "Overbought ⚠️" if rsi > 70 else ("Oversold ⚠️" if rsi < 30 else "Neutral ✅")
        rsi_color = "#ef5350" if rsi > 70 else ("#ef5350" if rsi < 30 else "#26a69a")
        sma_status = "Golden Cross ✅" if sma10 > sma50 else "Death Cross ⚠️"
        sma_col = "#26a69a" if sma10 > sma50 else "#ef5350"
        macd_status = "Bullish ✅" if macd > macd_sig else "Bearish ⚠️"
        macd_col = "#26a69a" if macd > macd_sig else "#ef5350"

        def _card(icon, label, value, sub, sub_color):
            return f"""
            <div style="
                background: var(--bg-card); border-radius:12px;
                padding:16px 20px; flex:1; min-width:160px;
                border:1px solid var(--border-color);
                box-shadow: var(--shadow-card);
            ">
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:8px;">
                    <span style="font-size:16px;">{icon}</span>
                    <span style="font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; font-weight:700;">{label}</span>
                </div>
                <div style="font-size:20px; font-weight:700; color:var(--text-primary);">{value}</div>
                <div style="font-size:11px; color:{sub_color}; font-weight:600; margin-top:2px;">{sub}</div>
            </div>
            """

        signals_list = [
            ("Wilder's RSI (14)", f"{rsi:.2f}", rsi_label.replace(" ⚠️", "").replace(" ✅", ""), rsi_label, rsi_color),
            ("SMA 10 / 50 Crossover", f"{currency_sym}{sma10:,.2f} / {currency_sym}{sma50:,.2f}", "BUY" if sma10 > sma50 else "SELL", sma_status, sma_col),
            ("MACD Crossover", f"{macd:.4f} (Signal: {macd_sig:.4f})", "BUY" if macd > macd_sig else "SELL", macd_status, macd_col),
            ("Bollinger Bands (20, 2σ)", f"{currency_sym}{current_price:,.2f}", "BUY" if current_price <= bb_lower else ("SELL" if current_price >= bb_upper else "NEUTRAL"), f"Upper: {currency_sym}{bb_upper:,.2f} | Lower: {currency_sym}{bb_lower:,.2f}", "#26a69a" if current_price <= bb_lower else ("#ef5350" if current_price >= bb_upper else "#707a8a")),
        ]
        
        table_rows = "".join(f"""
        <tr style="border-bottom: 1px solid var(--border-color);">
            <td style="padding: 10px; font-weight: 600; color: var(--text-primary);">{name}</td>
            <td style="padding: 10px; color: var(--text-secondary); font-family: monospace; font-size: 12px;">{val}</td>
            <td style="padding: 10px;">
                <span style="background: {color}14; color: {color}; padding: 3px 10px; border-radius: 10px; font-weight: 700; font-size: 11px;">{sig}</span>
            </td>
            <td style="padding: 10px; color: var(--text-muted); font-size: 11px;">{desc}</td>
        </tr>
        """ for name, val, sig, desc, color in signals_list)
        
        table_html = f"""
        <div style="background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border-color); padding: 16px; margin-top: 12px; box-shadow: var(--shadow-card); font-family: 'Inter', sans-serif; width: 100%;">
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="border-bottom: 2px solid var(--border-color); color: var(--text-muted); font-size: 10px; text-transform: uppercase;">
                        <th style="padding: 10px; font-weight: 700;">Indicator</th>
                        <th style="padding: 10px; font-weight: 700;">Value</th>
                        <th style="padding: 10px; font-weight: 700;">Signal</th>
                        <th style="padding: 10px; font-weight: 700;">Breakdown</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
        
        indicators_html = f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; font-family:'Inter',sans-serif; width: 100%;">
            {_card("📉", "RSI (14)", f"{rsi:.2f}", rsi_label, rsi_color)}
            {_card("📊", "SMA 10 / 50", f"{currency_sym}{sma10:,.2f}", sma_status, sma_col)}
            {_card("📈", "MACD", f"{macd:.4f}", macd_status, macd_col)}
            {_card("🎯", "Bollinger Bands", f"{currency_sym}{bb_upper:,.2f}", f"Lower: {currency_sym}{bb_lower:,.2f}", "#2962ff")}
            {table_html}
        </div>
        """

        # Fetch News Headlines
        try:
            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)
            headlines = loop2.run_until_complete(fetch_news_headlines(ticker))
            loop2.close()
        except Exception:
            headlines = ["Headlines currently unavailable."]

        score = analyze_sentiment(headlines)
        sentiment_fig = build_sentiment_gauge(score, is_dark=is_dark)

        news_items = "".join(
            f"""<div style="
                padding:10px 14px; margin-bottom:8px;
                background:var(--bg-subcard); border-radius:10px;
                border:1px solid var(--border-color); color:var(--text-secondary); font-size:12px;
                line-height:1.4; box-shadow: var(--shadow-card);
            ">📰&nbsp; {h}</div>"""
            for h in headlines
        )
        news_html = f'<div style="font-family:\'Inter\',sans-serif; max-height:220px; overflow-y:auto;">{news_items}</div>'

        # Feature importances
        imp_fig = build_feature_importance_chart(results.get("feature_importance", {}), is_dark=is_dark)

        # Logs
        logs = [
            f"[THOUGHT]      Analyzing pipeline for {asset_name} ({ticker})",
            f"[ACTION]       Detected: {market_text}",
            f"[ACTION]       Ingested {timeframe_label} data slice.",
            f"[ACTION]       Computing Wilder's RSI, EMA-MACD, Bollinger Bands...",
            f"[ACTION]       Fitting {model_type} model with lag-{lag_period} features...",
            f"[OBSERVATION]  Model R² Score = {r2}",
            f"[OBSERVATION]  Forecast: {currency_sym}{forecast:,.2f} ({pct_change:+.2f}%)",
            f"[RESULT]       ✅ {direction} with {confidence}% confidence index",
        ]
        logs_text = "\n".join(logs)

        return (
            price_html,
            live_chart,
            fig,
            indicators_html,
            news_html,
            logs_text,
            sentiment_fig,
            imp_fig
        )
    except Exception as exc:
        logger.exception("Prediction handler error")
        return _error_fallback(f"Workspace Runtime Error: {exc}", is_dark)


def load_grid_chart(asset_name, timeframe_label):
    if not asset_name:
        return build_placeholder_chart("Select an asset to load.")
    ticker = resolve_ticker(asset_name)
    if not ticker:
        return build_placeholder_chart(f"Asset '{asset_name}' not resolved.")
    timeframe = TIMEFRAME_OPTIONS.get(timeframe_label, "1d")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df = loop.run_until_complete(fetch_stock_data(ticker, timeframe))
        loop.close()
    except Exception as exc:
        return build_placeholder_chart(f"Fetch failed: {exc}")
    
    if df.empty:
        return build_placeholder_chart("No market data available.")
        
    try:
        currency_sym, _ = _detect_currency(ticker)
        fig = build_stock_chart(
            df=df,
            chart_type="Candlestick",
            overlays=["SMA 20"],
            subplots=["Volume"],
            asset_name=asset_name,
            currency_sym=currency_sym
        )
        return fig
    except Exception as exc:
        return build_placeholder_chart(f"Chart Render failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Backtesting Executor
# ═══════════════════════════════════════════════════════════════════════════

def _backtest_error_fallback(message: str) -> tuple:
    metrics_html = f"""
    <div style="background:#fff5f5; border-radius:12px; border:1px solid #fed7d7; padding:16px; font-family:'Inter',sans-serif; color:#c53030;">
        <strong>Backtest Interrupted:</strong> {message}
    </div>
    """
    fig = build_placeholder_chart(f"Backtest Curve: {message}")
    trades_html = "<p style='color:#707a8a; font-family:Inter; padding:10px;'>Journal empty due to simulation error.</p>"
    return metrics_html, fig, trades_html


def run_backtest_ui(asset_name, timeframe_label, initial_capital, transaction_cost, model_type, lag_period):
    try:
        if not asset_name:
            return _backtest_error_fallback("Please select a target asset first.")
        ticker = resolve_ticker(asset_name)
        if not ticker:
            return _backtest_error_fallback("Asset ticker could not be resolved.")
        timeframe = TIMEFRAME_OPTIONS.get(timeframe_label, "1d")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(fetch_stock_data(ticker, timeframe))
            loop.close()
        except Exception as exc:
            return _backtest_error_fallback(f"Fetch failed: {exc}")
            
        if df.empty or len(df) < 60:
            return _backtest_error_fallback(f"Insufficient data ({len(df) if not df.empty else 0} points). Need at least 60 points.")
            
        res = run_historical_backtest(
            df=df,
            model_type=model_type,
            lag_period=int(lag_period),
            initial_capital=float(initial_capital),
            transaction_cost_pct=float(transaction_cost)
        )
        
        if not res.get("success"):
            return _backtest_error_fallback(res.get("error", "Backtest failed"))
            
        # Build Stats cards
        metrics_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 14px; margin-bottom: 16px; font-family: 'Inter', sans-serif;">
            <div style="background: #ffffff; padding: 16px 20px; border-radius: 12px; border: 1px solid #eaecef; box-shadow: 0 1px 4px rgba(0,0,0,.03);">
                <div style="font-size: 10px; color: #707a8a; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Strategy Return</div>
                <div style="font-size: 24px; font-weight: 800; color: {'#26a69a' if res['strategy_return_pct'] >= 0 else '#ef5350'}; margin-top: 4px;">{res['strategy_return_pct']:+.2f}%</div>
            </div>
            <div style="background: #ffffff; padding: 16px 20px; border-radius: 12px; border: 1px solid #eaecef; box-shadow: 0 1px 4px rgba(0,0,0,.03);">
                <div style="font-size: 10px; color: #707a8a; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Buy & Hold Return</div>
                <div style="font-size: 24px; font-weight: 800; color: {'#26a69a' if res['benchmark_return_pct'] >= 0 else '#ef5350'}; margin-top: 4px;">{res['benchmark_return_pct']:+.2f}%</div>
            </div>
            <div style="background: #ffffff; padding: 16px 20px; border-radius: 12px; border: 1px solid #eaecef; box-shadow: 0 1px 4px rgba(0,0,0,.03);">
                <div style="font-size: 10px; color: #707a8a; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Alpha vs Benchmark</div>
                <div style="font-size: 24px; font-weight: 800; color: {'#26a69a' if res['alpha'] >= 0 else '#ef5350'}; margin-top: 4px;">{res['alpha']:+.2f}%</div>
            </div>
            <div style="background: #ffffff; padding: 16px 20px; border-radius: 12px; border: 1px solid #eaecef; box-shadow: 0 1px 4px rgba(0,0,0,.03);">
                <div style="font-size: 10px; color: #707a8a; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Win Rate</div>
                <div style="font-size: 24px; font-weight: 800; color: #26a69a; margin-top: 4px;">{res['win_rate_pct']:.1f}%</div>
                <div style="font-size: 11px; color: #707a8a;">{res['total_trades']} closed trades</div>
            </div>
            <div style="background: #ffffff; padding: 16px 20px; border-radius: 12px; border: 1px solid #eaecef; box-shadow: 0 1px 4px rgba(0,0,0,.03);">
                <div style="font-size: 10px; color: #707a8a; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Max Drawdown</div>
                <div style="font-size: 24px; font-weight: 800; color: #ef5350; margin-top: 4px;">-{res['max_drawdown_pct']:.2f}%</div>
            </div>
            <div style="background: #ffffff; padding: 16px 20px; border-radius: 12px; border: 1px solid #eaecef; box-shadow: 0 1px 4px rgba(0,0,0,.03);">
                <div style="font-size: 10px; color: #707a8a; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Sharpe Ratio</div>
                <div style="font-size: 24px; font-weight: 800; color: #2962ff; margin-top: 4px;">{res['sharpe_ratio']:.2f}</div>
            </div>
        </div>
        """
        
        # Equity Curve Plotly
        curve_df = pd.DataFrame(res["equity_curve"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_df["date"], y=curve_df["strategy"],
            mode="lines", name="Consensus Strategy",
            line=dict(color="#26a69a", width=2.5),
            fill="tozeroy", fillcolor="rgba(38, 166, 154, 0.04)"
        ))
        fig.add_trace(go.Scatter(
            x=curve_df["date"], y=curve_df["benchmark"],
            mode="lines", name="Buy & Hold Benchmark",
            line=dict(color="#707a8a", width=1.5, dash="dash")
        ))
        fig.update_layout(
            template="plotly_white",
            title=f"Consensus Strategy Equity Curve ({asset_name})",
            font=dict(family="Inter, sans-serif", size=11),
            margin=dict(l=40, r=20, t=40, b=30),
            height=350,
            hovermode="x unified"
        )
        
        # Trades Journal table
        trades_list = res["trades"]
        if not trades_list:
            trades_html = "<p style='color:#707a8a; font-family:Inter; padding:10px;'>No trades generated by consensus triggers.</p>"
        else:
            rows = ""
            for t in trades_list:
                ret = t.get("return_pct", 0.0)
                color = "#26a69a" if ret >= 0 else "#ef5350"
                sign = "+" if ret >= 0 else ""
                rows += f"""
                <tr style="border-bottom: 1px solid #eaecef; font-size:12px;">
                    <td style="padding: 8px 10px; color:#1e2329;">{t.get('entry_date')}</td>
                    <td style="padding: 8px 10px; color:#474d57; font-family:monospace;">${t.get('entry_price'):,.2f}</td>
                    <td style="padding: 8px 10px; color:#1e2329;">{t.get('exit_date')}</td>
                    <td style="padding: 8px 10px; color:#474d57; font-family:monospace;">${t.get('exit_price'):,.2f}</td>
                    <td style="padding: 8px 10px; color:{color}; font-weight:700;">{sign}{ret:.2f}%</td>
                </tr>
                """
            trades_html = f"""
            <div style="background: #ffffff; border-radius: 12px; border: 1px solid #eaecef; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.03); font-family: 'Inter', sans-serif;">
                <h3 style="margin-top:0; margin-bottom:12px; font-size:13px; font-weight:800; color:#1e2329; text-transform:uppercase;">📜 CLOSED TRADES JOURNAL</h3>
                <div style="max-height: 200px; overflow-y:auto;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left;">
                        <thead>
                            <tr style="border-bottom: 2px solid #eaecef; color:#707a8a; font-size:9px; text-transform:uppercase;">
                                <th style="padding: 8px 10px;">Entry Date</th>
                                <th style="padding: 8px 10px;">Entry Price</th>
                                <th style="padding: 8px 10px;">Exit Date</th>
                                <th style="padding: 8px 10px;">Exit Price</th>
                                <th style="padding: 8px 10px;">Trade Return</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
            """
            
        return metrics_html, fig, trades_html
    except Exception as exc:
        logger.exception("Backtest handler error")
        return _backtest_error_fallback(f"Backtest engine crashed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Market Scanner Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def _generate_sparkline_svg(prices, is_positive=True, width=100, height=30):
    if not prices or len(prices) < 2:
        return ""
    try:
        min_p, max_p = min(prices), max(prices)
        rng = max_p - min_p if max_p != min_p else 1.0
        
        points = []
        for i, p in enumerate(prices):
            x = (i / (len(prices) - 1)) * width
            y = height - ((p - min_p) / rng) * (height - 6) - 3
            points.append(f"{x},{y}")
            
        color = "#26a69a" if is_positive else "#ef5350"
        import uuid
        uid = uuid.uuid4().hex[:6]
        glow_id = f"glow-{uid}"
        grad_id = f"grad-{uid}"
        
        path_data = "M " + " L ".join(points)
        
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
            <path d="{path_data}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round" opacity="0.45" filter="url(#{glow_id})" />
            <path d="{path_data}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" />
        </svg>
        """
        return svg
    except Exception:
        return ""


def render_scan_results_html():
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_scan_results.json")
    if not os.path.exists(results_path):
        return """
        <div style="text-align: center; padding: 50px 20px; font-family: 'Inter', sans-serif; background: var(--bg-card); border-radius: 16px; border: 1px solid var(--border-color);">
            <p style="font-size: 16px; color: var(--text-muted); font-weight: 500;">No scan results found.</p>
            <p style="font-size: 13px; color: var(--text-muted); margin-top: 8px; margin-bottom: 24px;">Please click the button below to run the first market scan.</p>
        </div>
        """
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"<p style='color: red; font-family: sans-serif; background: var(--bg-card); border-radius:12px; padding:12px; border:1px solid var(--border-color);'>Error reading scan results: {e}</p>"

    scan_time = data.get("scan_time", "Unknown")
    
    def render_leader_card(title, assets, is_tomorrow=False):
        rows = ""
        for i, item in enumerate(assets):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "•"
            perf = item["tomorrow_perf"] if is_tomorrow else item["today_perf"]
            perf_color = "#26a69a" if perf >= 0 else "#ef5350"
            perf_sign = "+" if perf >= 0 else ""
            price_val = item["tomorrow_forecast"] if is_tomorrow else item["price"]
            
            ticker = item["ticker"]
            currency_sym = "₹" if (ticker.endswith(".NS") or ticker.endswith(".BO")) else "$"

            rows += f"""
            <div class="scanner-stock-row" onclick="setClickedStock('{item['name']}')" style="
                display: flex; align-items: center; justify-content: space-between;
                padding: 12px 16px; background: var(--bg-subcard); border-radius: 12px;
                margin-bottom: 8px; border: 1px solid var(--border-color); cursor: pointer;
                transition: all 0.2s ease;
            ">
                <div style="display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0;">
                    <span style="font-size: 16px;">{rank_emoji}</span>
                    <div style="min-width: 0; flex: 1;">
                        <div style="font-weight: 700; color: var(--text-primary); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['name']}</div>
                        <div style="font-size: 11px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['ticker']} · {item['category']}</div>
                    </div>
                </div>
                <div style="margin: 0 16px; display: flex; align-items: center; justify-content: center;">
                    {_generate_sparkline_svg(item.get('sparkline_prices', []), perf >= 0, width=80, height=25)}
                </div>
                <div style="text-align: right; min-width: 80px;">
                    <div style="font-weight: 700; color: var(--text-primary); font-size: 13px;">{currency_sym}{price_val:,.2f}</div>
                    <div style="color: {perf_color}; font-weight: 700; font-size: 11px;">{perf_sign}{perf:.2f}%</div>
                </div>
            </div>
            """
        return f"""
        <div style="flex: 1; min-width: 300px; background: var(--bg-card); border-radius: 16px; padding: 20px; border: 1px solid var(--border-color); box-shadow: var(--shadow-card);">
            <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 14px; font-weight: 800; color: var(--text-primary); letter-spacing: 0.5px; border-bottom: 2px solid var(--border-color); padding-bottom: 10px; text-transform: uppercase;">{title}</h3>
            {rows}
        </div>
        """

    overall_html = f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px;">
        {render_leader_card("🏆 Today's Top Performers (Actual)", data.get("overall_top_today", []))}
        {render_leader_card("🔮 Tomorrow's Predicted Leaders (Forecast)", data.get("overall_top_tomorrow", []), is_tomorrow=True)}
    </div>
    """

    # For Today's Performers column:
    today_cards = ""
    for cat_name, cat_data in data.get("categories", {}).items():
        today_rows = ""
        for i, item in enumerate(cat_data.get("top_today", [])):
            perf = item["today_perf"]
            perf_color = "#26a69a" if perf >= 0 else "#ef5350"
            perf_sign = "+" if perf >= 0 else ""
            price_val = item["price"]
            ticker = item["ticker"]
            currency_sym = "₹" if (ticker.endswith(".NS") or ticker.endswith(".BO")) else "$"
            
            today_rows += f"""
            <div class="scanner-stock-row" onclick="setClickedStock('{item['name']}')" style="
                display: flex; justify-content: space-between; align-items: center;
                padding: 10px 14px; background: var(--bg-subcard); border-radius: 12px;
                margin-bottom: 8px; border: 1px solid var(--border-color); cursor: pointer;
            ">
                <div style="display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1;">
                    <span style="font-size: 14px;">📈</span>
                    <div style="min-width: 0; flex: 1;">
                        <div style="font-weight: 700; color: var(--text-primary); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['name']}</div>
                        <div style="font-size: 11px; color: var(--text-muted);">{item['ticker']}</div>
                    </div>
                </div>
                <div style="margin: 0 12px; display: flex; align-items: center;">
                    {_generate_sparkline_svg(item.get('sparkline_prices', []), perf >= 0, width=65, height=22)}
                </div>
                <div style="text-align: right; min-width: 70px;">
                    <div style="font-weight: 700; color: var(--text-primary); font-size: 12px;">{currency_sym}{price_val:,.2f}</div>
                    <div style="color: {perf_color}; font-weight: 700; font-size: 11px;">{perf_sign}{perf:.2f}%</div>
                </div>
            </div>
            """
        today_cards += f"""
        <div style="background: var(--bg-card); border-radius: 16px; padding: 18px; border: 1px solid var(--border-color); box-shadow: var(--shadow-card); margin-bottom: 16px;">
            <div style="font-size: 13px; font-weight: 800; color: var(--text-primary); margin-bottom: 12px; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">{cat_name}</div>
            {today_rows or "<div style='font-size:11px;color:var(--text-muted);'>No data</div>"}
        </div>
        """

    # For Tomorrow's Performers column:
    tomorrow_cards = ""
    for cat_name, cat_data in data.get("categories", {}).items():
        tomorrow_rows = ""
        for i, item in enumerate(cat_data.get("top_tomorrow", [])):
            perf = item["tomorrow_perf"]
            perf_color = "#26a69a" if perf >= 0 else "#ef5350"
            perf_sign = "+" if perf >= 0 else ""
            price_val = item["tomorrow_forecast"]
            ticker = item["ticker"]
            currency_sym = "₹" if (ticker.endswith(".NS") or ticker.endswith(".BO")) else "$"
            
            tomorrow_rows += f"""
            <div class="scanner-stock-row" onclick="setClickedStock('{item['name']}')" style="
                display: flex; justify-content: space-between; align-items: center;
                padding: 10px 14px; background: var(--bg-subcard); border-radius: 12px;
                margin-bottom: 8px; border: 1px solid var(--border-color); cursor: pointer;
            ">
                <div style="display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1;">
                    <span style="font-size: 14px;">🔮</span>
                    <div style="min-width: 0; flex: 1;">
                        <div style="font-weight: 700; color: var(--text-primary); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['name']}</div>
                        <div style="font-size: 11px; color: var(--text-muted);">{item['ticker']}</div>
                    </div>
                </div>
                <div style="margin: 0 12px; display: flex; align-items: center;">
                    {_generate_sparkline_svg(item.get('sparkline_prices', []), perf >= 0, width=65, height=22)}
                </div>
                <div style="text-align: right; min-width: 70px;">
                    <div style="font-weight: 700; color: var(--text-primary); font-size: 12px;">{currency_sym}{price_val:,.2f}</div>
                    <div style="color: {perf_color}; font-weight: 700; font-size: 11px;">{perf_sign}{perf:.2f}%</div>
                </div>
            </div>
            """
        tomorrow_cards += f"""
        <div style="background: var(--bg-card); border-radius: 16px; padding: 18px; border: 1px solid var(--border-color); box-shadow: var(--shadow-card); margin-bottom: 16px;">
            <div style="font-size: 13px; font-weight: 800; color: var(--text-primary); margin-bottom: 12px; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">{cat_name}</div>
            {tomorrow_rows or "<div style='font-size:11px;color:var(--text-muted);'>No data</div>"}
        </div>
        """

    grid_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 24px; margin-bottom: 20px;">
        <div>
            <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 15px; font-weight: 800; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                <span>🏢</span> Today's Category Performers
            </h3>
            {today_cards}
        </div>
        <div>
            <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 15px; font-weight: 800; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                <span>🔮</span> Tomorrow's Forecasted Leaders
            </h3>
            {tomorrow_cards}
        </div>
    </div>
    """

    header_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; font-family: 'Inter', sans-serif; flex-wrap: wrap; gap: 10px;">
        <div>
            <h2 style="margin: 0; font-size: 20px; font-weight: 800; color: var(--text-primary);">📊 Global Macro Scanner</h2>
            <p style="margin: 4px 0 0 0; font-size: 12px; color: var(--text-muted);">Top performing financial assets and forecasts across global markets</p>
        </div>
        <div style="background: rgba(41,98,255,0.06); border: 1px solid rgba(41,98,255,0.15); color: #2962ff; border-radius: 20px; padding: 6px 14px; font-size: 11px; font-weight: 600; font-family: monospace;">
            Last Scan: {scan_time}
        </div>
    </div>
    """

    return f"""
    <div style="font-family: 'Inter', sans-serif; background: var(--bg-primary); padding: 2px; transition: background 0.3s ease;">
        {header_html}
        {overall_html}
        <h3 style="margin-top: 0; margin-bottom: 20px; font-size: 16px; font-weight: 800; color: var(--text-primary); font-family: 'Inter', sans-serif; border-bottom: 1px solid var(--border-color); padding-bottom: 8px;">📂 Sectoral Top Performers</h3>
        {grid_html}
    </div>
    """


def run_market_scan_btn():
    from services.market_scanner import run_scan
    import asyncio

    logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Launching Market Scanner..."]

    try:
        asyncio.run(run_scan())
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Market Scan completed successfully!")
    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Exception: {e}")

    html_content = render_scan_results_html()
    return html_content, "\n".join(logs)



# ═══════════════════════════════════════════════════════════════════════════
# Styling & Theme Layer (Light Theme Glassmorphic TV Preset)
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
:root {
    --bg-primary: #f4f6f9;
    --bg-card: #ffffff;
    --border-color: #eaecef;
    --text-primary: #1e2329;
    --text-secondary: #474d57;
    --text-muted: #707a8a;
    --bg-sidebar: rgba(255, 255, 255, 0.75);
    --bg-hover: rgba(37, 99, 235, 0.08);
    --border-sidebar: rgba(226, 232, 240, 0.8);
    --shadow-card: 0 2px 12px rgba(0,0,0,.04);
    --bg-subcard: #f8f9fa;
    --transition-theme: background 0.3s ease, border-color 0.3s ease, color 0.3s ease;
}

.dark {
    --bg-primary: #0b0e11;
    --bg-card: #181a20;
    --border-color: #2b3139;
    --text-primary: #eaecef;
    --text-secondary: #c9d1d9;
    --text-muted: #848e9c;
    --bg-sidebar: rgba(24, 26, 32, 0.95);
    --bg-hover: rgba(59, 130, 246, 0.15);
    --border-sidebar: rgba(43, 49, 57, 0.8);
    --shadow-card: 0 4px 24px rgba(0,0,0,.3);
    --bg-subcard: #1e2329;
}

.gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    max-width: 1500px !important;
    transition: var(--transition-theme);
}

#app-header {
    text-align: center;
    padding: 24px 20px 12px;
}
#app-header h1 {
    font-size: 32px; font-weight: 900;
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.dark #app-header h1 {
    background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#app-header .subtitle {
    color: var(--text-secondary); font-size: 13px; margin-top: 6px;
}

/* Sidebar Styling */
#sidebar-column {
    background: var(--bg-sidebar) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid var(--border-sidebar) !important;
    padding: 24px 16px !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
    margin-right: 14px;
    transition: var(--transition-theme);
}

.nav-btn {
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 8px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.nav-btn:hover {
    background: var(--bg-hover) !important;
    color: #2563eb !important;
    transform: translateX(4px) !important;
}
.dark .nav-btn:hover {
    color: #60a5fa !important;
}

/* Premium Card Panels */
.glass-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: var(--shadow-card) !important;
    transition: var(--transition-theme);
}

/* Horizontal setting control bar */
#chart-controls {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.02) !important;
    margin-bottom: 12px !important;
    transition: var(--transition-theme);
}

#predict-btn {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.2s ease !important;
}
#predict-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
}
.dark #predict-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
}
.dark #predict-btn:hover {
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.3) !important;
}

#search-overlay {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    box-shadow: var(--shadow-card) !important;
    margin-top: 6px !important;
    transition: var(--transition-theme);
}

#logs-box textarea {
    background: #111827 !important;
    color: #10b981 !important;
    font-family: 'Consolas', monospace !important;
    font-size: 11px !important;
    border-radius: 10px !important;
    line-height: 1.5 !important;
}
.dark #logs-box textarea {
    background: #0d1117 !important;
    border-color: var(--border-color) !important;
}

.section-title {
    font-size: 11px !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 700 !important;
    margin: 8px 0 !important;
}

/* Scanner stock row interactive styling */
.scanner-stock-row {
    transition: all 0.2s ease;
}
.scanner-stock-row:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
    border-color: #2563eb !important;
}
.dark .scanner-stock-row:hover {
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    border-color: #3b82f6 !important;
}

#clicked-stock-trigger { display: none !important; }
footer { display: none !important; }
"""

HEADER_HTML = """
<div id="app-header">
    <h1>🚀 Premium AI Financial Analytics</h1>
    <p class="subtitle">
        High-fidelity predictive forecasting dashboard fueled by Machine Learning & Multi-Indicator Consensus Engine
    </p>
</div>
"""

DEFAULT_ASSET = "Reliance Industries"


# ═══════════════════════════════════════════════════════════════════════════
# Interface Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_interface():
    head_js = """
    <script>
    function setClickedStock(assetName) {
        var container = document.getElementById("clicked-stock-trigger");
        if (container) {
            var inputEl = container.querySelector("textarea") || container.querySelector("input");
            if (inputEl) {
                inputEl.value = assetName;
                var event = new Event('input', { bubbles: true });
                inputEl.dispatchEvent(event);
            }
        }
    }
    </script>
    """
    with gr.Blocks(title="AI Stock Prediction Engine", css=CUSTOM_CSS, head=head_js) as demo:
        gr.HTML(HEADER_HTML)

        # Text trigger that is hidden via CSS overlays, allowing macro scanner row clicks to execute Python logic
        clicked_stock_trigger = gr.Textbox(visible=True, elem_id="clicked-stock-trigger", value="")

        with gr.Row():
            # Left glassmorphic sidebar
            with gr.Column(scale=1, min_width=250, elem_id="sidebar-column"):
                gr.Markdown("### 🧭 WORKSPACE SECTIONS")
                btn_dashboard = gr.Button("🏠 Financial Dashboard", variant="primary", elem_classes=["nav-btn"])
                btn_advanced  = gr.Button("🧠 Advanced Predictor", variant="secondary", elem_classes=["nav-btn"])
                btn_scanner   = gr.Button("🔍 Global Scanner", variant="secondary", elem_classes=["nav-btn"])
                btn_backtest  = gr.Button("🧪 Backtest Sandbox", variant="secondary", elem_classes=["nav-btn"])
                btn_settings  = gr.Button("⚙️ Settings & System", variant="secondary", elem_classes=["nav-btn"])
                
                gr.HTML("<div style='height: 20px;'></div>")
                gr.Markdown("### ⚙️ PREFERENCES")
                dark_mode_toggle = gr.Checkbox(label="🌙 Dark Mode Theme", value=False, elem_id="dark-mode-toggle")
                
                gr.HTML("<div style='height: 80px;'></div>")
                gr.Markdown("<div style='text-align: center; color: #9ca3af; font-size: 11px;'>Engine v3.0 (XGBoost Active)</div>")

            # Main canvas
            with gr.Column(scale=4, elem_id="main-canvas"):
                
                # ===========================================================
                # PANEL 1: DASHBOARD
                # ===========================================================
                with gr.Column(visible=True) as panel_dashboard:
                    # Layout Selector
                    layout_mode = gr.Radio(
                        choices=["🏠 Detailed Single Asset View", "📊 Comparative Grid View"],
                        value="🏠 Detailed Single Asset View",
                        label="Dashboard Mode Layout",
                        show_label=False
                    )
                    
                    # ── Option 1: Detailed Single Asset workspace ──
                    with gr.Group(visible=True) as workspace_single:
                        
                        # Selection Block
                        with gr.Group(elem_id="selection-panel", elem_classes=["glass-card"]):
                            gr.HTML('<p class="section-title">🔍 Unified Command Bar Search</p>')
                            with gr.Row():
                                search_input = gr.Textbox(
                                    placeholder="🔍 Search asset display names (e.g. Reliance, Apple, Bitcoin, Gold)...",
                                    label="Search Input",
                                    show_label=False,
                                    scale=4
                                )
                                timeframe_dropdown = gr.Dropdown(
                                    choices=list(TIMEFRAME_OPTIONS.keys()),
                                    value="1 Day",
                                    label="Timeframe",
                                    show_label=False,
                                    scale=1
                                )
                                predict_btn = gr.Button("⚡ Analyze Asset", elem_id="predict-btn", scale=1)
                                
                            # Search suggestions / categories overlay
                            with gr.Group(visible=False, elem_id="search-overlay") as overlay_group:
                                with gr.Row():
                                    gr.Markdown("**Quick Filters:**")
                                    btn_filter_stocks = gr.Button("🏢 Indian Stocks", size="sm")
                                    btn_filter_us = gr.Button("🏛️ US Stocks", size="sm")
                                    btn_filter_crypto = gr.Button("₿ Crypto", size="sm")
                                    btn_filter_forex = gr.Button("💱 Forex", size="sm")
                                with gr.Row():
                                    gr.Markdown("**Popular Searches:**")
                                    pop_1 = gr.Button("Reliance Industries", size="sm")
                                    pop_2 = gr.Button("Apple", size="sm")
                                    pop_3 = gr.Button("Bitcoin (BTC)", size="sm")
                                    pop_4 = gr.Button("Gold (COMEX)", size="sm")
                                with gr.Row():
                                    close_overlay_btn = gr.Button("✕ Close Search Assistant", size="sm", variant="stop")
                                    
                            # Resolved asset dropdown (holds search matches)
                            search_results = gr.Dropdown(
                                choices=sorted(list(ALL_ASSETS.keys())),
                                value=DEFAULT_ASSET,
                                label="Active Selection Target",
                                show_label=True,
                                info="Confirm selected target below"
                            )

                        # Compact Settings control bar
                        with gr.Group(elem_id="chart-controls"):
                            with gr.Row():
                                chart_type_radio = gr.Radio(
                                    choices=["🕯️ Candle", "📈 Line", "📊 OHLC", "✨ Heikin Ashi"],
                                    value="🕯️ Candle",
                                    label="Chart Type Settings",
                                    show_label=False,
                                    scale=2
                                )
                                indicators_dropdown = gr.Dropdown(
                                    choices=OVERLAY_INDICATORS + SUBPLOT_INDICATORS,
                                    value=["SMA 20", "Bollinger Bands", "Volume"],
                                    multiselect=True,
                                    label="Applied Technical Indicators Overlay",
                                    show_label=False,
                                    scale=4
                                )

                        # Outputs Summary Card
                        price_output = gr.HTML(label="")
                        
                        # Main Side-by-Side Charts (Today on Left, Tomorrow forecast on Right)
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.HTML('<p class="section-title">📊 Live Market Chart (Today)</p>')
                                live_chart_output = gr.Plot(label="", elem_classes=["glass-card"])
                            with gr.Column(scale=1):
                                gr.HTML('<p class="section-title">🔮 ML Forecast vs Actual Price (Tomorrow)</p>')
                                pred_chart_output = gr.Plot(label="", elem_classes=["glass-card"])
                        
                        # Secondary indicators and feeds encapsulated in clean tabs
                        with gr.Tabs(elem_classes=["glass-card"]) as details_tabs:
                            with gr.Tab("💡 Consensus Signals"):
                                indicators_output = gr.HTML(label="")
                                
                            with gr.Tab("🧠 AI Predictor Diagnostics"):
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        gr.HTML('<p class="section-title">📊 Model Feature Coefficients / Importance</p>')
                                        feature_importance_output = gr.Plot(label="")
                                    with gr.Column(scale=2):
                                        gr.HTML('<p class="section-title">🤖 Agent Execution Logs</p>')
                                        logs_output = gr.Textbox(
                                            label="", lines=10, interactive=False,
                                            elem_id="logs-box",
                                        )
                                        
                            with gr.Tab("📰 Market Sentiment & News"):
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        gr.HTML('<p class="section-title">📊 Lexical Macro Sentiment</p>')
                                        sentiment_gauge_output = gr.Plot(label="")
                                    with gr.Column(scale=3):
                                        gr.HTML('<p class="section-title">📰 Financial News Feed</p>')
                                        news_output = gr.HTML(label="")

                    # ── Option 2: Comparative Grid workspace ──
                    with gr.Group(visible=False) as workspace_grid:
                        gr.HTML('<p class="section-title">📊 Comparative Split-Screen Workspace</p>')
                        grid_layout_type = gr.Radio(
                            choices=["2-Chart Split", "4-Chart Split"],
                            value="2-Chart Split",
                            label="Grid Dimension",
                            show_label=False
                        )
                        
                        # Row 1 (Workspace 1 & 2)
                        with gr.Row() as grid_row_1:
                            with gr.Column(elem_classes=["glass-card"]) as box_1:
                                gr.Markdown("#### 📈 Split Pane 1")
                                asset_1 = gr.Dropdown(choices=sorted(list(ALL_ASSETS.keys())), value="Reliance Industries", filterable=True, label="Asset")
                                tf_1 = gr.Dropdown(choices=list(TIMEFRAME_OPTIONS.keys()), value="1 Day", label="Timeframe")
                                plot_1 = gr.Plot()
                                btn_1 = gr.Button("🔄 Render Pane 1", size="sm", variant="secondary")
                            with gr.Column(elem_classes=["glass-card"]) as box_2:
                                gr.Markdown("#### 📈 Split Pane 2")
                                asset_2 = gr.Dropdown(choices=sorted(list(ALL_ASSETS.keys())), value="Apple", filterable=True, label="Asset")
                                tf_2 = gr.Dropdown(choices=list(TIMEFRAME_OPTIONS.keys()), value="1 Day", label="Timeframe")
                                plot_2 = gr.Plot()
                                btn_2 = gr.Button("🔄 Render Pane 2", size="sm", variant="secondary")
                                
                        # Row 2 (Workspace 3 & 4)
                        with gr.Row(visible=False) as grid_row_2:
                            with gr.Column(elem_classes=["glass-card"]) as box_3:
                                gr.Markdown("#### 📈 Split Pane 3")
                                asset_3 = gr.Dropdown(choices=sorted(list(ALL_ASSETS.keys())), value="Bitcoin (BTC)", filterable=True, label="Asset")
                                tf_3 = gr.Dropdown(choices=list(TIMEFRAME_OPTIONS.keys()), value="1 Day", label="Timeframe")
                                plot_3 = gr.Plot()
                                btn_3 = gr.Button("🔄 Render Pane 3", size="sm", variant="secondary")
                            with gr.Column(elem_classes=["glass-card"]) as box_4:
                                gr.Markdown("#### 📈 Split Pane 4")
                                asset_4 = gr.Dropdown(choices=sorted(list(ALL_ASSETS.keys())), value="Gold (COMEX)", filterable=True, label="Asset")
                                tf_4 = gr.Dropdown(choices=list(TIMEFRAME_OPTIONS.keys()), value="1 Day", label="Timeframe")
                                plot_4 = gr.Plot()
                                btn_4 = gr.Button("🔄 Render Pane 4", size="sm", variant="secondary")

                # ===========================================================
                # PANEL 2: ADVANCED PREDICTOR
                # ===========================================================
                with gr.Column(visible=False) as panel_advanced:
                    gr.Markdown("## 🧠 Advanced Model Tuning & Lag Configuration")
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.Markdown("### Prediction Pipeline Customizer")
                        with gr.Row():
                            model_type_dropdown = gr.Dropdown(
                                choices=["Ridge Regression", "XGBoost Regressor", "Deep Learning (MLP Neural Net)"],
                                value="Ridge Regression",
                                label="Active Regressor Algorithm"
                            )
                            lag_period_slider = gr.Slider(
                                minimum=1, maximum=10, step=1,
                                value=3,
                                label="Lag Lookback Periods (Days)"
                            )
                        
                        gr.Markdown("#### Algorithm Specific Parameters")
                        with gr.Row():
                            ridge_alpha = gr.Slider(minimum=0.1, maximum=100.0, step=0.5, value=10.0, label="Ridge L2 Alpha Penalty")
                            xgb_estimators = gr.Slider(minimum=10, maximum=300, step=10, value=100, label="XGBoost Trees Count")
                            xgb_depth = gr.Slider(minimum=2, maximum=8, step=1, value=3, label="XGBoost Max Tree Depth")
                            
                        gr.Markdown("Changes will affect the ML forecasts rendered inside the main Financial Dashboard and Backtest Sandbox.")

                # ===========================================================
                # PANEL 3: GLOBAL SCANNER
                # ===========================================================
                with gr.Column(visible=False) as panel_scanner:
                    scanner_html_output = gr.HTML(
                        value=render_scan_results_html(),
                        elem_id="scanner-dashboard"
                    )
                    with gr.Row():
                        run_scan_btn = gr.Button("🔄 Execute Macro Scanner Refresh", elem_id="predict-btn")
                    
                    scanner_logs_output = gr.Textbox(
                        label="Scanner Process Log",
                        lines=6,
                        interactive=False,
                        elem_id="logs-box",
                        value="Scan engine standing by. Click the button to trigger walk-forward scan."
                    )

                # ===========================================================
                # PANEL 4: BACKTEST SANDBOX
                # ===========================================================
                with gr.Column(visible=False) as panel_backtest:
                    gr.Markdown("## 🧪 Consensus Strategy Sandbox Backtester")
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.Markdown("### Setup Strategy Parameters & Initial Conditions")
                        with gr.Row():
                            backtest_asset = gr.Dropdown(choices=sorted(list(ALL_ASSETS.keys())), value="Reliance Industries", filterable=True, label="Target Asset Name")
                            backtest_tf = gr.Dropdown(choices=list(TIMEFRAME_OPTIONS.keys()), value="1 Day", label="Historical Candle Timeframe")
                        with gr.Row():
                            backtest_capital = gr.Number(value=10000.0, label="Starting Capital ($)")
                            backtest_fee = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.1, label="Execution slippage + Commission (%)")
                        with gr.Row():
                            backtest_model = gr.Dropdown(choices=["Ridge Regression", "XGBoost Regressor", "Deep Learning (MLP Neural Net)"], value="Ridge Regression", label="Underlying Forecast Engine")
                            backtest_lag = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Lookback Lag Depth")
                            
                        run_backtest_btn = gr.Button("🧪 Run Historical Walk-Forward Simulation", elem_id="predict-btn")
                        
                    backtest_metrics_output = gr.HTML(label="")
                    backtest_chart_output = gr.Plot(label="", elem_classes=["glass-card"])
                    backtest_trades_output = gr.HTML(label="")

                # ===========================================================
                # PANEL 5: SETTINGS
                # ===========================================================
                with gr.Column(visible=False) as panel_settings:
                    gr.Markdown("## ⚙️ System Settings & Control Panel")
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.Markdown("### Application Preferences")
                        gr.Checkbox(value=True, label="Enable Multi-threaded Socket Pipeline Ingestion")
                        gr.Checkbox(value=False, label="Force Strict Numeric Precision Fallbacks")
                        gr.Slider(minimum=10, maximum=120, value=30, label="Headline Lexical Cache Invalidation (Seconds)")
                        gr.Button("Restore System to Defaults", variant="stop")

        # ── Wire Workspace Nav Events ──────────────────────────────────────────
        def show_panel(panel_name):
            return [
                gr.update(visible=(panel_name == "dashboard")),
                gr.update(visible=(panel_name == "advanced")),
                gr.update(visible=(panel_name == "scanner")),
                gr.update(visible=(panel_name == "backtest")),
                gr.update(visible=(panel_name == "settings")),
                
                gr.update(variant="primary" if panel_name == "dashboard" else "secondary"),
                gr.update(variant="primary" if panel_name == "advanced" else "secondary"),
                gr.update(variant="primary" if panel_name == "scanner" else "secondary"),
                gr.update(variant="primary" if panel_name == "backtest" else "secondary"),
                gr.update(variant="primary" if panel_name == "settings" else "secondary"),
            ]

        nav_outputs = [
            panel_dashboard, panel_advanced, panel_scanner, panel_backtest, panel_settings,
            btn_dashboard, btn_advanced, btn_scanner, btn_backtest, btn_settings
        ]
        
        btn_dashboard.click(fn=lambda: show_panel("dashboard"), inputs=[], outputs=nav_outputs)
        btn_advanced.click(fn=lambda: show_panel("advanced"), inputs=[], outputs=nav_outputs)
        btn_scanner.click(fn=lambda: show_panel("scanner"), inputs=[], outputs=nav_outputs)
        btn_backtest.click(fn=lambda: show_panel("backtest"), inputs=[], outputs=nav_outputs)
        btn_settings.click(fn=lambda: show_panel("settings"), inputs=[], outputs=nav_outputs)

        # ── Wire Layout Workspace Toggles ──────────────────────────────────────
        layout_mode.change(
            fn=lambda mode: (
                gr.update(visible=(mode == "🏠 Detailed Single Asset View")),
                gr.update(visible=(mode == "📊 Comparative Grid View"))
            ),
            inputs=[layout_mode],
            outputs=[workspace_single, workspace_grid]
        )

        grid_layout_type.change(
            fn=lambda dim: gr.update(visible=(dim == "4-Chart Split")),
            inputs=[grid_layout_type],
            outputs=[grid_row_2]
        )

        # ── Wire Command Search & Suggestion Panel ─────────────────────────────
        search_input.focus(fn=lambda: gr.update(visible=True), outputs=[overlay_group])
        close_overlay_btn.click(fn=lambda: gr.update(visible=False), outputs=[overlay_group])

        def filter_choices(query):
            if not query:
                return gr.update(choices=sorted(list(ALL_ASSETS.keys())), value=DEFAULT_ASSET)
            matches = search_assets(query)
            if matches:
                return gr.update(choices=matches, value=matches[0])
            return gr.update(choices=[], value=None)

        search_input.change(fn=filter_choices, inputs=[search_input], outputs=[search_results])

        def filter_by_cat(cat_name):
            assets = get_assets_for_category(cat_name)
            return "", gr.update(choices=assets, value=assets[0]), gr.update(visible=False)

        btn_filter_stocks.click(fn=lambda: filter_by_cat("🇮🇳 Indian Stocks"), outputs=[search_input, search_results, overlay_group])
        btn_filter_us.click(fn=lambda: filter_by_cat("🇺🇸 US Stocks"), outputs=[search_input, search_results, overlay_group])
        btn_filter_crypto.click(fn=lambda: filter_by_cat("₿ Crypto"), outputs=[search_input, search_results, overlay_group])
        btn_filter_forex.click(fn=lambda: filter_by_cat("💱 Forex"), outputs=[search_input, search_results, overlay_group])

        def select_popular(asset_name):
            return "", gr.update(choices=sorted(list(ALL_ASSETS.keys())), value=asset_name), gr.update(visible=False)

        pop_1.click(fn=lambda: select_popular("Reliance Industries"), outputs=[search_input, search_results, overlay_group])
        pop_2.click(fn=lambda: select_popular("Apple"), outputs=[search_input, search_results, overlay_group])
        pop_3.click(fn=lambda: select_popular("Bitcoin (BTC)"), outputs=[search_input, search_results, overlay_group])
        pop_4.click(fn=lambda: select_popular("Gold (COMEX)"), outputs=[search_input, search_results, overlay_group])

        search_results.change(fn=lambda: gr.update(visible=False), outputs=[overlay_group])

        # ── Wire Prediction Execution ──────────────────────────────────────────
        all_predict_inputs = [
            search_results, timeframe_dropdown, chart_type_radio,
            indicators_dropdown, model_type_dropdown, lag_period_slider,
            dark_mode_toggle
        ]
        all_predict_outputs = [
            price_output, live_chart_output, pred_chart_output,
            indicators_output, news_output, logs_output,
            sentiment_gauge_output, feature_importance_output
        ]
        
        predict_btn.click(fn=predict, inputs=all_predict_inputs, outputs=all_predict_outputs)

        # ── Wire Theme Toggling ────────────────────────────────────────────────
        dark_mode_toggle.change(
            fn=None,
            inputs=[dark_mode_toggle],
            js="""(is_dark) => {
                if (is_dark) {
                    document.body.classList.add('dark');
                    document.documentElement.classList.add('dark');
                } else {
                    document.body.classList.remove('dark');
                    document.documentElement.classList.remove('dark');
                }
                return is_dark;
            }"""
        )
        
        # When theme is toggled, also re-render Plotly charts in python
        dark_mode_toggle.change(fn=predict, inputs=all_predict_inputs, outputs=all_predict_outputs)

        # ── Wire Split Grid Load Commands ──────────────────────────────────────
        btn_1.click(fn=load_grid_chart, inputs=[asset_1, tf_1], outputs=[plot_1])
        btn_2.click(fn=load_grid_chart, inputs=[asset_2, tf_2], outputs=[plot_2])
        btn_3.click(fn=load_grid_chart, inputs=[asset_3, tf_3], outputs=[plot_3])
        btn_4.click(fn=load_grid_chart, inputs=[asset_4, tf_4], outputs=[plot_4])

        # ── Wire Backtester sandbox ────────────────────────────────────────────
        run_backtest_btn.click(
            fn=run_backtest_ui,
            inputs=[backtest_asset, backtest_tf, backtest_capital, backtest_fee, backtest_model, backtest_lag],
            outputs=[backtest_metrics_output, backtest_chart_output, backtest_trades_output]
        )

        # ── Wire Scanner Refresh ───────────────────────────────────────────────
        run_scan_btn.click(
            fn=run_market_scan_btn,
            inputs=[],
            outputs=[scanner_html_output, scanner_logs_output]
        )

        # ── Wire Scanner Click-to-Dashboard Handler ───────────────────────────
        def handle_stock_click(clicked_stock, timeframe_label, chart_type, selected_indicators, model_type, lag_period, is_dark):
            if not clicked_stock:
                return [gr.update()] * (len(all_predict_outputs) + len(nav_outputs) + 2)
            
            # Predict clicked asset
            predict_res = predict(
                asset_name=clicked_stock,
                timeframe_label=timeframe_label,
                chart_type=chart_type,
                selected_indicators=selected_indicators,
                model_type=model_type,
                lag_period=lag_period,
                is_dark=is_dark
            )
            
            # Switch visible panel to dashboard
            nav_res = show_panel("dashboard")
            
            # Returns predictions + navigation state + updates search_results dropdown + clears clicked_stock_trigger
            return list(predict_res) + nav_res + [clicked_stock, ""]

        clicked_stock_trigger.change(
            fn=handle_stock_click,
            inputs=[clicked_stock_trigger, timeframe_dropdown, chart_type_radio, indicators_dropdown, model_type_dropdown, lag_period_slider, dark_mode_toggle],
            outputs=all_predict_outputs + nav_outputs + [search_results, clicked_stock_trigger]
        )

    return demo


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
