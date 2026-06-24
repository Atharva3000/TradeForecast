"""
Data Engine — Async yFinance downloads and mock news aggregation.

Handles:
- Intraday timeframes (1min, 3min, 5min, 10min, 15min, 30min) mapped to
  the closest available yfinance interval.
- Daily timeframes (1D, 1W, 1M) with sufficient lookback for SMA-50.
- MultiIndex column flattening for single-ticker downloads.
- DatetimeIndex → string conversion for safe JSON serialization.
"""

import asyncio
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TIMEFRAME_CONFIG: dict[str, dict[str, str]] = {
    # Intraday
    "1m":    {"interval": "1m",  "period": "5d"},
    "3m":    {"interval": "2m",  "period": "10d"},
    "5m":    {"interval": "5m",  "period": "30d"},
    "15m":   {"interval": "15m", "period": "30d"},
    "30m":   {"interval": "30m", "period": "30d"},
    "45m":   {"interval": "90m", "period": "60d"},
    "1h":    {"interval": "1h",  "period": "60d"},
    "2h":    {"interval": "1h",  "period": "120d"},
    "3h":    {"interval": "1h",  "period": "180d"},
    "4h":    {"interval": "1h",  "period": "240d"},
    # Daily / Weekly / Monthly
    "1d":    {"interval": "1d",  "period": "1y"},
    "1w":    {"interval": "1wk", "period": "3y"},
    "1mo":   {"interval": "1mo", "period": "5y"},
    # Legacy / Backwards compatibility keys
    "1min":  {"interval": "1m",  "period": "5d"},
    "3min":  {"interval": "2m",  "period": "10d"},
    "5min":  {"interval": "5m",  "period": "30d"},
    "15min": {"interval": "15m", "period": "30d"},
    "30min": {"interval": "30m", "period": "30d"},
    "1D":    {"interval": "1d",  "period": "1y"},
    "1W":    {"interval": "1wk", "period": "3y"},
    "1M":    {"interval": "1mo", "period": "5y"},
}

VALID_TIMEFRAMES: list[str] = list(TIMEFRAME_CONFIG.keys())
INTRADAY_TIMEFRAMES: set[str] = {
    "1m", "3m", "5m", "15m", "30m", "45m", "1h", "2h", "3h", "4h",
    "1min", "3min", "5min", "15min", "30min"
}

def _download_stock_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """Synchronous yfinance download — never call directly from async code."""
    config = TIMEFRAME_CONFIG.get(timeframe)
    if config is None:
        raise ValueError(
            f"Unsupported timeframe: '{timeframe}'. "
            f"Valid options: {VALID_TIMEFRAMES}"
        )

    df = yf.download(
        ticker,
        period=config["period"],
        interval=config["interval"],
        progress=False,
        auto_adjust=True,
    )
    return df

async def fetch_stock_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch historical price data for *ticker* at the given *timeframe*.
    Returns a cleaned DataFrame with lowercase columns and a ``date`` string.
    """
    try:
        df = await asyncio.to_thread(_download_stock_data, ticker, timeframe)

        if df is None or df.empty:
            logger.warning(
                "No data returned for ticker '%s' (timeframe=%s)", ticker, timeframe
            )
            return pd.DataFrame()

        # Flatten MultiIndex columns (yfinance single-ticker quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Convert DatetimeIndex to a string column
        if timeframe in INTRADAY_TIMEFRAMES:
            df["date"] = df.index.strftime("%Y-%m-%d %H:%M")
        else:
            df["date"] = df.index.strftime("%Y-%m-%d")

        df = df.reset_index(drop=True)
        return df

    except Exception as exc:
        logger.error(
            "Failed to fetch data for '%s' (timeframe=%s): %s",
            ticker, timeframe, exc,
            exc_info=True,
        )
        return pd.DataFrame()

async def fetch_news_headlines(ticker: str) -> list[str]:
    """Return 3 mock but contextually-relevant asset news headlines."""
    today = datetime.now().strftime("%B %d, %Y")
    
    # Dynamic name stripping for Forex, Crypto, and Indian markets
    base_ticker = ticker.replace(".NS", "").replace(".BO", "").replace("=X", "")
    
    return [
        f"Macro indicators flag shift in {base_ticker} liquidity pools — {today}",
        f"Algorithmic order books note structural support patterns forming across {base_ticker} vectors",
        f"Technical market summary: {base_ticker} momentum parameters signal tightening variance bands",
    ]