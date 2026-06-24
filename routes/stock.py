"""
REST Endpoints — Stock prediction routes.

Exposes:
    GET /api/predict/{ticker}?timeframe=1M
    GET /api/scan
    POST /api/scan/refresh

Supports US and Indian market tickers automatically:
    - Tickers ending in .NS (NSE) or .BO (BSE) → currency ₹ / INR
    - All other tickers → currency $ / USD
"""

from datetime import date
import asyncio
import os
import time
import logging
from typing import Optional
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, Query

from services.data_fetcher import fetch_stock_data, fetch_news_headlines, VALID_TIMEFRAMES, INTRADAY_TIMEFRAMES
from services.predictor import run_prediction, compute_confidence
from services.sentiment import analyze_sentiment
from services.auth_db import register_user, authenticate_user, update_user_profile

# Resolve the results file relative to this file (routes/ -> project root)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_RESULTS_PATH = os.path.join(_BASE_DIR, "market_scan_results.json")

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory TTL cache for predictions to prevent thread starvation and yfinance blockages
PREDICTION_CACHE = {}  # (ticker, timeframe, model) -> (timestamp, data)
CACHE_TTL = 300  # 5 minutes

class RegisterRequest(BaseModel):
    username: str
    password: str
    name: Optional[str] = None
    email: Optional[str] = None
    trading_experience: Optional[str] = None
    investment_capital: Optional[float] = None
    country: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class ProfileUpdateRequest(BaseModel):
    username: str
    name: Optional[str] = None
    email: Optional[str] = None
    trading_experience: Optional[str] = None
    investment_capital: Optional[float] = None
    country: Optional[str] = None



@router.get("/api/predict/batch")
async def get_predictions_batch(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
    timeframe: str = Query(default="1M"),
    model: str = Query(default="linear"),
):
    """
    Get prediction data for multiple tickers in a single batch request.
    Returns a dictionary of ticker -> light prediction data.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results = {}

    async def fetch_and_predict(t):
        try:
            res = await get_prediction(t, timeframe, model)
            results[t] = {
                "status": "success",
                "current_price": res["current_price"],
                "forecast_price": res["forecast_price"],
                "prediction_direction": res["prediction_direction"],
                "confidence_score": res["confidence_score"],
                "currency_symbol": res["currency_symbol"],
            }
        except Exception as e:
            results[t] = {"status": "error", "message": str(e)}

    await asyncio.gather(*(fetch_and_predict(t) for t in ticker_list))
    return results


@router.get("/api/predict/{ticker}")
async def get_prediction(
    ticker: str,
    timeframe: str = "1M",
    model: str = "linear",
):
    """
    Run the full prediction pipeline for a stock ticker.

    Returns current price, directional forecast, confidence score,
    technical indicators, historical-vs-predicted chart data, and
    dynamically generated agent execution logs.
    """
    ticker = ticker.upper()

    # ---- Cache lookup ------------------------------------------------------
    cache_key = (ticker, timeframe, model)
    now_ts = time.time()
    timeframe_ttl = 300 if timeframe in INTRADAY_TIMEFRAMES else 14400  # 5 min for intraday, 4 hours for daily/weekly/monthly
    if cache_key in PREDICTION_CACHE:
        cached_time, cached_data = PREDICTION_CACHE[cache_key]
        if now_ts - cached_time < timeframe_ttl:
            logger.info("Prediction Cache HIT for %s (%s, %s)", ticker, timeframe, model)
            return cached_data

    # ---- Validate timeframe ------------------------------------------------
    if timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid timeframe '{timeframe}'. "
                f"Must be one of: {VALID_TIMEFRAMES}"
            ),
        )

    # ---- Detect market region & currency -----------------------------------
    currency_symbol = "$"
    currency_code   = "USD"
    
    SYMBOL_MAP = {
        "USD": "$",
        "INR": "₹",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "AUD": "A$",
        "CAD": "C$",
    }
    
    if "-" in ticker:
        # Crypto format: e.g. BTC-USD, ETH-INR, ETH-EUR
        parts = ticker.split("-")
        if len(parts) == 2:
            quote = parts[1].upper()
            currency_code = quote
            currency_symbol = SYMBOL_MAP.get(quote, quote)
    elif ticker.endswith("=X"):
        # Forex format: e.g. EURUSD=X, USDINR=X
        base_pair = ticker[:-2]  # Strip '=X'
        if len(base_pair) >= 6:
            quote = base_pair[-3:].upper()  # Last 3 characters before '=X'
            currency_code = quote
            currency_symbol = SYMBOL_MAP.get(quote, quote)
    else:
        # Standard Stocks format (Indian vs US/Global)
        is_indian_market = ticker.endswith(".NS") or ticker.endswith(".BO")
        currency_symbol = "₹" if is_indian_market else "$"
        currency_code   = "INR" if is_indian_market else "USD"

    # ---- Fetch historical data ---------------------------------------------
    df = await fetch_stock_data(ticker, timeframe)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Ticker '{ticker}' is invalid or returned no data. "
                "For Indian markets, append .NS (NSE) or .BO (BSE) — "
                "e.g. RELIANCE.NS, TCS.BO."
            ),
        )

    # ---- Run prediction pipeline -------------------------------------------
    # Map model selection to internal predictor model names
    model_mapping = {
        "linear": "Ridge Regression",
        "ensemble": "XGBoost",
        "deep": "Deep Learning (MLP Neural Net)"
    }
    internal_model = model_mapping.get(model.lower(), "Ridge Regression")

    try:
        pipeline_results = run_prediction(df, model_type=internal_model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    latest_row    = pipeline_results["latest_df"]
    current_price = float(latest_row["close"])
    forecast      = pipeline_results["next_day_forecast"]

    # ---- Resolve full stock/company name -----------------------------------
    try:
        from stock_universe import TICKER_TO_NAME
        stock_name = TICKER_TO_NAME.get(ticker)
        if not stock_name:
            import yfinance as yf
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            stock_name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        stock_name = ticker

    # ---- Determine directional bias ----------------------------------------
    direction = "Neutral"
    if forecast > current_price * 1.005:
        direction = "Bullish"
    elif forecast < current_price * 0.995:
        direction = "Bearish"

    confidence = compute_confidence(latest_row)

    # ---- Fetch mock news ---------------------------------------------------
    headlines = await fetch_news_headlines(ticker)

    # ---- Generate dynamic agent execution logs -----------------------------
    today_str   = date.today().strftime("%Y-%m-%d")
    
    if "-" in ticker:
        market_text = "Cryptocurrency Exchange"
    elif ticker.endswith("=X"):
        market_text = "Foreign Exchange Market"
    elif ticker.endswith(".NS") or ticker.endswith(".BO"):
        market_text = "Indian National Exchange"
    else:
        market_text = "US / Global Capital Market"

    agent_logs = [
        f"[THOUGHT] Analyzing pipeline structure for {ticker} on {today_str}...",
        f"[ACTION] Detected exchange: {market_text}. Loading regional presets...",
        f"[ACTION] Downloaded {timeframe} dataset slice. Flattening multi-level indices...",
        f"[ACTION] Computing Wilder's alpha-decay RSI-14 and Industrial EMA MACD loops...",
        f"[OBSERVATION] RSI-14: {float(latest_row['rsi_14']):.2f}. Feeding into quantitative core metrics...",
        f"[ACTION] Isolating forecast row before NaN cleanup to prevent data leakage...",
        f"[ACTION] Running predictive trend engine on chronological split...",
        f"[OBSERVATION] Model convergence score on historical backtesting: {pipeline_results['r2_score']}",
        f"[OBSERVATION] Trajectory maps as {direction}. Forecast: {currency_symbol}{forecast:.2f}",
        f"[RESULT] Consensus: {direction} with {confidence}% confidence.",
    ]

    # ---- Run comprehensive technical analysis ------------------------------
    try:
        from services.technical_analysis import full_technical_analysis
        analysis = full_technical_analysis(df, current_price)
    except Exception as exc:
        logger.error("Error computing comprehensive technical analysis: %s", exc)
        analysis = {
            "indicators": [],
            "success_rates": [],
            "consensus": {
                "action": "HOLD / WAIT",
                "confidence": "Low",
                "duration": "Watch — Wait for Breakout",
                "rationale": f"Analysis failed: {str(exc)}",
                "score": 0,
                "buy_count": 0,
                "sell_count": 0,
                "hold_count": 0,
                "total": 0,
                "best_indicator": "—",
                "best_accuracy": 0,
            },
            "fibonacci_levels": {},
            "pivot_points": {},
        }

    # ---- Structured response -----------------------------------------------
    # ---- Compute sentiment score on headlines --------------------------------
    sentiment_score = round(analyze_sentiment(headlines), 4)

    sentiment_label = "Neutral"
    if sentiment_score > 0.15:
        sentiment_label = "Positive"
    elif sentiment_score < -0.15:
        sentiment_label = "Negative"

    def safe_val(v):
        import numpy as np
        try:
            fv = float(v)
            return None if np.isnan(fv) or np.isinf(fv) else round(fv, 4)
        except Exception:
            return None

    response_data = {
        "ticker":                 ticker,
        "stock_name":             stock_name,
        "current_price":          current_price,
        "forecast_price":         round(forecast, 2),
        "currency_symbol":        currency_symbol,
        "currency_code":          currency_code,
        "prediction_direction":   direction,
        "confidence_score":       int(confidence),
        "sentiment_score":        sentiment_score,
        "sentiment_label":        sentiment_label,
        "technical_indicators": {
            "rsi_14":      safe_val(latest_row.get("rsi_14")),
            "sma_10":      safe_val(latest_row.get("sma_10")),
            "sma_50":      safe_val(latest_row.get("sma_50")),
            "macd":        safe_val(latest_row.get("macd")),
            "macd_signal": safe_val(latest_row.get("macd_signal")),
        },
        "indicators":             analysis["indicators"],
        "success_rates":          analysis["success_rates"],
        "consensus":              analysis["consensus"],
        "fibonacci_levels":       analysis["fibonacci_levels"],
        "pivot_points":           analysis["pivot_points"],
        "historical_vs_predicted": pipeline_results["historical_vs_predicted"],
        "news_headlines":          headlines,
        "agent_logs":              agent_logs,
        "agent_log":               agent_logs,
    }
    
    PREDICTION_CACHE[cache_key] = (time.time(), response_data)
    return response_data


# ---------------------------------------------------------------------------
# Global Market Scanner Endpoints
# ---------------------------------------------------------------------------
import json
from fastapi import BackgroundTasks


async def _run_scanner_async():
    """Run the self-contained market scanner service."""
    from services.market_scanner import run_scan
    try:
        await run_scan()
        logger.info("Market Scanner finished successfully.")
    except Exception as e:
        logger.error("Error running Market Scanner: %s", e)


def _run_scanner_sync():
    """Synchronous wrapper — called from BackgroundTasks."""
    asyncio.run(_run_scanner_async())


@router.get("/api/scan")
async def get_scan_results():
    """Serve the saved market scan results."""
    # Check if a refreshed scan exists in /tmp first, fallback to bundled scan
    tmp_path = "/tmp/market_scan_results.json"
    target_path = tmp_path if os.path.exists(tmp_path) else SCAN_RESULTS_PATH
    
    if not os.path.exists(target_path):
        return {
            "scan_time": "Never Scanned",
            "categories": {},
            "overall_top_today": [],
            "overall_top_tomorrow": [],
        }
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scan results: {str(e)}")


@router.post("/api/scan/refresh")
async def refresh_scan(background_tasks: BackgroundTasks):
    """Trigger a market scan in the background."""
    background_tasks.add_task(_run_scanner_sync)
    return {"status": "scanning", "message": "Market scan started in background."}


@router.get("/api/ticker-prices")
async def get_ticker_prices(
    tickers: str = Query(default="RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,SBIN.NS,BAJFINANCE.NS,WIPRO.NS,AXISBANK.NS,TATAMOTORS.NS,MARUTI.NS,^NSEI,^BSESN,^NSEBANK")
):
    """
    Fetch the latest prices for a comma-separated list of tickers.
    """
    import yfinance as yf
    import pandas as pd
    
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results = {}
    
    async def fetch_price(ticker):
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: yf.download(ticker, period="5d", interval="1d", progress=False)
            )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df_clean = df.dropna(subset=["Close"])
                if len(df_clean) >= 1:
                    latest_close = float(df_clean["Close"].iloc[-1])
                    prev_close = float(df_clean["Close"].iloc[-2]) if len(df_clean) > 1 else latest_close
                    change = latest_close - prev_close
                    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
                    results[ticker] = {
                        "price": round(latest_close, 2),
                        "change": round(change, 2),
                        "pct_change": round(pct_change, 2)
                    }
                else:
                    results[ticker] = {"price": None, "error": "Insufficient data"}
            else:
                results[ticker] = {"price": None, "error": "No data returned"}
        except Exception as e:
            results[ticker] = {"price": None, "error": str(e)}

    await asyncio.gather(*(fetch_price(t) for t in ticker_list))
    return results

async def preload_sector_predictions():
    """Warm up the prediction cache for sector watchlist tickers."""
    tickers = [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS",
        "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "ADANIENT.NS",
        "MARUTI.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "TITAN.NS"
    ]
    logger.info("Pre-warming prediction cache for %d sector tickers...", len(tickers))
    
    # Preload sequentially with a short sleep to be gentle on yfinance API
    for ticker in tickers:
        try:
            await get_prediction(ticker, "1M")
            logger.info("Cache preloaded successfully for: %s", ticker)
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error("Failed to preload cache for %s: %s", ticker, e)
    logger.info("Sector watchlist cache pre-warming complete.")


# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/auth/register")
async def api_register(req: RegisterRequest):
    try:
        user = register_user(
            username=req.username,
            password=req.password,
            name=req.name,
            email=req.email,
            trading_experience=req.trading_experience,
            investment_capital=req.investment_capital,
            country=req.country
        )
        return {"status": "success", "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/api/auth/login")
async def api_login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"status": "success", "user": user}

@router.post("/api/auth/update")
async def api_update(req: ProfileUpdateRequest):
    try:
        user = update_user_profile(
            username=req.username,
            name=req.name,
            email=req.email,
            trading_experience=req.trading_experience,
            investment_capital=req.investment_capital,
            country=req.country
        )
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "success", "user": user}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/api/stocks")
async def get_all_stocks():
    """Return all available stock tickers and names from the universe."""
    try:
        from stock_universe import TICKER_TO_NAME
        return TICKER_TO_NAME
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading stock universe: {str(e)}")

