"""
Market Scanner Service — Scans the full stock universe and writes results.

Runs the prediction pipeline on a representative subset of assets across
all categories (Indian Stocks, US Stocks, Crypto, Forex, Commodities),
computing today's performance, tomorrow's forecast, confidence score,
and RSI/MACD/SMA/BB signals for each asset.

Results are written to market_scan_results.json in the project root.
This module can be run directly as a script or imported.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Ensure project root is on the path when run as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # services/ -> project root
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from services.data_fetcher import fetch_stock_data, fetch_news_headlines
from services.predictor import run_prediction, compute_confidence, compute_sma, compute_rsi, compute_macd
from services.sentiment import analyze_sentiment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset sample per category (keep small for scan speed)
# ---------------------------------------------------------------------------
SCAN_UNIVERSE: dict[str, list[tuple[str, str]]] = {
    "Indian Stocks": [
        ("Reliance Industries",   "RELIANCE.NS"),
        ("Tata Consultancy Svcs", "TCS.NS"),
        ("HDFC Bank",             "HDFCBANK.NS"),
        ("Infosys",               "INFY.NS"),
        ("ICICI Bank",            "ICICIBANK.NS"),
        ("State Bank of India",   "SBIN.NS"),
    ],
    "US Stocks": [
        ("Apple Inc",             "AAPL"),
        ("NVIDIA Corp",           "NVDA"),
        ("Microsoft Corp",        "MSFT"),
        ("Tesla Inc",             "TSLA"),
        ("Alphabet Inc",          "GOOGL"),
        ("Meta Platforms",        "META"),
    ],
    "Crypto": [
        ("Bitcoin",               "BTC-USD"),
        ("Ethereum",              "ETH-USD"),
        ("Solana",                "SOL-USD"),
        ("BNB",                   "BNB-USD"),
    ],
    "Commodities": [
        ("Gold",                  "GC=F"),
        ("Silver",                "SI=F"),
        ("Crude Oil (WTI)",       "CL=F"),
    ],
    "Forex": [
        ("EUR/USD",               "EURUSD=X"),
        ("USD/INR",               "INR=X"),
        ("GBP/USD",               "GBPUSD=X"),
    ],
}

RESULTS_PATH = os.path.join(_PROJECT_ROOT, "market_scan_results.json")


def _detect_currency_symbol(ticker: str) -> str:
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "₹"
    if "-" in ticker:
        parts = ticker.split("-")
        if len(parts) == 2:
            sym_map = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}
            return sym_map.get(parts[1].upper(), "$")
    return "$"


def _compute_signals(df: pd.DataFrame) -> dict:
    """Compute RSI, SMA cross, MACD, and BB signals for the latest row."""
    if df.empty or len(df) < 60:
        return {}

    close = df["close"]
    rsi = compute_rsi(close, 14).iloc[-1]
    sma10 = compute_sma(close, 10).iloc[-1]
    sma50 = compute_sma(close, 50).iloc[-1]
    macd_line, sig_line, _ = compute_macd(close)
    macd_val  = macd_line.iloc[-1]
    sig_val   = sig_line.iloc[-1]

    bb_mid   = close.rolling(20).mean().iloc[-1]
    bb_std   = close.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    price    = close.iloc[-1]

    rsi_signal  = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL")
    sma_signal  = "BUY" if sma10 > sma50 else "SELL"
    macd_signal = "BUY" if macd_val > sig_val else "SELL"
    bb_signal   = "NEUTRAL" if bb_lower <= price <= bb_upper else ("SELL" if price >= bb_upper else "BUY")

    return {
        "RSI (14)": {
            "value": round(rsi, 2),
            "signal": rsi_signal,
            "desc": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral Zone (30-70)"),
        },
        "SMA 10 / 50": {
            "value": f"{sma10:.2f} / {sma50:.2f}",
            "signal": sma_signal,
            "desc": "Golden Cross (SMA 10 > SMA 50)" if sma_signal == "BUY" else "Death Cross (SMA 10 < SMA 50)",
        },
        "MACD": {
            "value": f"{macd_val:.4f} (Sig: {sig_val:.4f})",
            "signal": macd_signal,
            "desc": "Bullish Crossover (MACD > Signal)" if macd_signal == "BUY" else "Bearish Crossover (MACD < Signal)",
        },
        "Bollinger Bands": {
            "value": f"{price:.2f} (Bands: {bb_lower:.2f}-{bb_upper:.2f})",
            "signal": bb_signal,
            "desc": "Price within bands" if bb_signal == "NEUTRAL" else (
                "Above upper band" if price >= bb_upper else "Below lower band"
            ),
        },
    }


async def _scan_asset(
    name: str,
    ticker: str,
    category: str,
    timeframe: str = "1M",
) -> dict | None:
    """Scan a single asset and return a result dict, or None on failure."""
    try:
        df = await fetch_stock_data(ticker, timeframe)
        if df.empty or len(df) < 60:
            logger.warning("Skipping %s — insufficient data", ticker)
            return None

        pipeline = run_prediction(df)
        latest   = pipeline["latest_df"]
        price    = float(latest["close"])
        forecast = pipeline["next_day_forecast"]
        r2       = pipeline["r2_score"]

        # Today perf (compare last two closes)
        if len(df) >= 2:
            prev_close = float(df["close"].iloc[-2])
            today_perf = ((price - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
        else:
            today_perf = 0.0

        tomorrow_perf = ((forecast - price) / price * 100) if price != 0 else 0.0
        confidence    = compute_confidence(latest)
        signals       = _compute_signals(df)

        # Compute trend direction signal (Bullish, Bearish, Neutral)
        direction = "Neutral"
        if forecast > price * 1.005:
            direction = "Bullish"
        elif forecast < price * 0.995:
            direction = "Bearish"

        # Fetch news and compute sentiment score
        try:
            headlines = await fetch_news_headlines(ticker)
            sentiment_score = round(analyze_sentiment(headlines), 4)
        except Exception:
            sentiment_score = 0.0

        # Sparkline: last 15 closes
        sparkline = list(df["close"].tail(15).round(4).values)

        return {
            "ticker":            ticker,
            "name":              name,
            "category":          category,
            "price":             round(price, 4),
            "today_perf":        round(today_perf, 4),
            "tomorrow_forecast": round(forecast, 4),
            "tomorrow_perf":     round(tomorrow_perf, 4),
            "confidence":        confidence,
            "confidence_score":  confidence,
            "r2":                r2,
            "signals":           signals,
            "signal":            direction,
            "sentiment_score":   sentiment_score,
            "sparkline_prices":  sparkline,
        }
    except Exception as exc:
        logger.error("Error scanning %s: %s", ticker, exc)
        return None


async def run_scan(timeframe: str = "1M") -> dict:
    """Run the full scan and return the structured results dict."""
    logger.info("Market Scan started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    categories: dict[str, dict] = {}
    all_results: list[dict] = []

    for cat_name, assets in SCAN_UNIVERSE.items():
        cat_results = []
        for name, ticker in assets:
            result = await _scan_asset(name, ticker, cat_name, timeframe)
            if result:
                cat_results.append(result)
                all_results.append(result)
            print(f"  Scanned: {ticker}", flush=True)

        # Sort results by absolute expected performance for a better list experience
        cat_results_sorted = sorted(cat_results, key=lambda x: abs(x["tomorrow_perf"]), reverse=True)
        categories[cat_name] = cat_results_sorted

    overall_top_today    = sorted(all_results, key=lambda x: x["today_perf"],    reverse=True)[:10]
    overall_top_tomorrow = sorted(all_results, key=lambda x: x["tomorrow_perf"], reverse=True)[:10]

    output = {
        "scan_time":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "categories":           categories,
        "overall_top_today":    overall_top_today,
        "overall_top_tomorrow": overall_top_tomorrow,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    logger.info("Market Scan complete. Results saved to %s", RESULTS_PATH)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    asyncio.run(run_scan())
    print("Scan complete.")
