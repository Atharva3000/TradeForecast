"""
Comprehensive Technical Analysis Service.

Computes 19 classic technical indicators for a given OHLCV DataFrame
and returns:
  - indicators: list of {name, value, signal, strength, reason}
  - success_rates: historical per-indicator accuracy on this stock
  - consensus: weighted buy/sell/hold recommendation + duration
  - fibonacci_levels: key Fibonacci retracement levels
  - pivot_points: classic daily pivot levels
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LOW-LEVEL INDICATOR MATHS
# ─────────────────────────────────────────────────────────────

def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average (linearly weighted)."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average: WMA(2×WMA(n/2) − WMA(n), √n)."""
    half = max(1, period // 2)
    sqrt_p = max(1, int(np.sqrt(period)))
    return _wma(2 * _wma(series, half) - _wma(series, period), sqrt_p)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9
          ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal


def _bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0
               ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    return sma + n_std * sd, sma, sma - n_std * sd


def _stochastic(df: pd.DataFrame, k: int = 14, d: int = 3
                ) -> Tuple[pd.Series, pd.Series]:
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    pct_k = 100 * (df["close"] - lo) / (hi - lo + 1e-10)
    return pct_k, pct_k.rolling(d).mean()


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi = df["high"].rolling(period).max()
    lo = df["low"].rolling(period).min()
    return -100 * (hi - df["close"]) / (hi - lo + 1e-10)


def _adx(df: pd.DataFrame, period: int = 14
         ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    hi, lo, cl = df["high"], df["low"], df["close"]
    pdm = hi.diff().clip(lower=0)
    mdm = (-lo.diff()).clip(lower=0)
    tr = pd.concat(
        [hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)
    mdi = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    return dx.ewm(alpha=1 / period, adjust=False).mean(), pdi, mdi


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df.get("volume", pd.Series(dtype=float))
    if vol is not None and not vol.empty and vol.sum() > 0:
        vol = vol.replace(0, np.nan).ffill()
        return (tp * vol).cumsum() / vol.cumsum()
    return tp.rolling(14).mean()


def _fib_levels(high: float, low: float) -> Dict[str, float]:
    d = high - low
    return {
        "0%":    round(high, 2),
        "23.6%": round(high - 0.236 * d, 2),
        "38.2%": round(high - 0.382 * d, 2),
        "50%":   round(high - 0.500 * d, 2),
        "61.8%": round(high - 0.618 * d, 2),
        "78.6%": round(high - 0.786 * d, 2),
        "100%":  round(low, 2),
    }


def _pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    p = (high + low + close) / 3
    return {
        "pivot": round(p, 2),
        "r1": round(2 * p - low, 2),
        "r2": round(p + (high - low), 2),
        "r3": round(high + 2 * (p - low), 2),
        "s1": round(2 * p - high, 2),
        "s2": round(p - (high - low), 2),
        "s3": round(low - 2 * (high - p), 2),
    }


def _safe(series: pd.Series, idx: int = -1) -> Optional[float]:
    """Return float or None if NaN/missing."""
    try:
        v = float(series.iloc[idx])
        return v if not np.isnan(v) else None
    except Exception:
        return None


def _ind(name: str, value: str, signal: str, strength: str, reason: str) -> dict:
    return {
        "name": name, "value": value,
        "signal": signal, "strength": strength, "reason": reason,
    }


# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────

def full_technical_analysis(df: pd.DataFrame, current_price: float) -> dict:
    """
    Compute 19 technical indicators and return signals + consensus.

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close (and optionally volume).
         Index must be monotonically ordered (oldest → newest).
    current_price : float – latest close price.

    Returns
    -------
    dict with keys: indicators, success_rates, consensus, fibonacci_levels, pivot_points
    """
    # Normalise column names and drop any rows with NaN price values
    df = df.rename(columns=str.lower).dropna(subset=["close", "high", "low"]).copy().reset_index(drop=True)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    cp    = current_price
    results: List[dict] = []

    # ── 1. RSI (14) ────────────────────────────────────────────
    rsi_s = _rsi(close)
    rv = _safe(rsi_s) or 50
    if rv <= 30:
        results.append(_ind("RSI (14)", f"{rv:.1f}", "Buy", "Strong",
            f"RSI at {rv:.1f} is deeply oversold (<30). Price tends to reverse upward from this zone — high-probability bounce setup."))
    elif rv >= 70:
        results.append(_ind("RSI (14)", f"{rv:.1f}", "Sell", "Strong",
            f"RSI at {rv:.1f} is overbought (>70). Momentum is overextended; a pullback or trend reversal is probable."))
    elif rv < 45:
        results.append(_ind("RSI (14)", f"{rv:.1f}", "Sell", "Moderate",
            f"RSI at {rv:.1f} is below 45 — bearish momentum. Not yet oversold but sellers remain in control."))
    elif rv > 55:
        results.append(_ind("RSI (14)", f"{rv:.1f}", "Buy", "Moderate",
            f"RSI at {rv:.1f} is above 55 — bullish momentum zone. Buyers are dominant; uptrend likely continuing."))
    else:
        results.append(_ind("RSI (14)", f"{rv:.1f}", "Hold", "Weak",
            f"RSI at {rv:.1f} is in the neutral band (45–55). No momentum edge — wait for a breakout from this range."))

    # ── 2. MACD (12, 26, 9) ───────────────────────────────────
    ml, sl, hl = _macd(close)
    mv  = _safe(ml) or 0
    sv  = _safe(sl) or 0
    hv  = _safe(hl) or 0
    hv2 = _safe(hl, -2) or 0
    if mv > sv:
        if hv >= hv2:
            results.append(_ind("MACD (12,26,9)", f"{mv:.2f}", "Buy", "Strong",
                f"MACD ({mv:.2f}) > Signal ({sv:.2f}) with accelerating histogram — sustained bullish momentum."))
        else:
            results.append(_ind("MACD (12,26,9)", f"{mv:.2f}", "Buy", "Moderate",
                f"MACD ({mv:.2f}) > Signal ({sv:.2f}) but histogram decelerating — bullish, yet momentum fading."))
    else:
        if hv <= hv2:
            results.append(_ind("MACD (12,26,9)", f"{mv:.2f}", "Sell", "Strong",
                f"MACD ({mv:.2f}) < Signal ({sv:.2f}) with deepening histogram — strong bearish momentum."))
        else:
            results.append(_ind("MACD (12,26,9)", f"{mv:.2f}", "Sell", "Moderate",
                f"MACD ({mv:.2f}) < Signal ({sv:.2f}) but histogram shrinking — bearish, yet losing steam."))

    # ── 3. Cross MACD (5, 13, 1) — Fast MACD ─────────────────
    fml, fsl, _ = _macd(close, fast=5, slow=13, sig=1)
    fmv = _safe(fml) or 0; fsv = _safe(fsl) or 0
    results.append(_ind(
        "Cross MACD (5,13,1)", f"{fmv:.2f}",
        "Buy" if fmv > fsv else "Sell", "Moderate",
        ("Fast MACD (5,13,1) bullish — short-term buying pressure is building rapidly." if fmv > fsv
         else "Fast MACD (5,13,1) bearish — short-term selling pressure is dominant.")))

    # ── 4. Bollinger Bands (20, 2) ────────────────────────────
    bbu_s, bbm_s, bbl_s = _bollinger(close)
    bbu = _safe(bbu_s) or cp; bbm = _safe(bbm_s) or cp; bbl = _safe(bbl_s) or cp
    if cp >= bbu:
        results.append(_ind("Bollinger Bands (20,2)", f"U:{bbu:.1f}", "Sell", "Strong",
            f"Price (₹{cp:.1f}) at/above upper BB (₹{bbu:.1f}) — statistically overbought; mean-reversion to ₹{bbm:.1f} expected."))
    elif cp <= bbl:
        results.append(_ind("Bollinger Bands (20,2)", f"L:{bbl:.1f}", "Buy", "Strong",
            f"Price (₹{cp:.1f}) at/below lower BB (₹{bbl:.1f}) — statistically oversold; bounce toward ₹{bbm:.1f} likely."))
    elif bbu > bbm and (cp - bbm) / (bbu - bbm + 1e-5) > 0.7:
        results.append(_ind("Bollinger Bands (20,2)", f"U:{bbu:.1f} M:{bbm:.1f}", "Sell", "Moderate",
            f"Price (₹{cp:.1f}) near upper BB — elevated reversal risk. Consider trimming positions."))
    elif bbm > bbl and (bbm - cp) / (bbm - bbl + 1e-5) > 0.7:
        results.append(_ind("Bollinger Bands (20,2)", f"L:{bbl:.1f} M:{bbm:.1f}", "Buy", "Moderate",
            f"Price (₹{cp:.1f}) near lower BB — elevated bounce probability. Dip-buy opportunity."))
    else:
        results.append(_ind("Bollinger Bands (20,2)", f"M:{bbm:.1f}", "Hold", "Weak",
            f"Price (₹{cp:.1f}) within bands (₹{bbl:.1f}–₹{bbu:.1f}) — no extreme signal; market is range-bound."))

    # ── 5. SMA 10 ─────────────────────────────────────────────
    s10v = _safe(close.rolling(10).mean()) or cp
    results.append(_ind(
        "SMA 10", f"₹{s10v:.2f}",
        "Buy" if cp > s10v else "Sell", "Moderate",
        (f"Price (₹{cp:.1f}) > SMA 10 (₹{s10v:.1f}) — short-term trend is bullish; momentum intact." if cp > s10v
         else f"Price (₹{cp:.1f}) < SMA 10 (₹{s10v:.1f}) — short-term downtrend; selling pressure present.")))

    # ── 6. SMA 50 ─────────────────────────────────────────────
    s50v = _safe(close.rolling(50).mean()) or cp
    results.append(_ind(
        "SMA 50", f"₹{s50v:.2f}",
        "Buy" if cp > s50v else "Sell", "Strong",
        (f"Price (₹{cp:.1f}) > SMA 50 (₹{s50v:.1f}) — medium-term bull trend; institutional buy zone." if cp > s50v
         else f"Price (₹{cp:.1f}) < SMA 50 (₹{s50v:.1f}) — medium-term bear trend; avoid fresh longs.")))

    # ── 7. SMA 200 ────────────────────────────────────────────
    s200v = _safe(close.rolling(200).mean()) or cp
    results.append(_ind(
        "SMA 200", f"₹{s200v:.2f}",
        "Buy" if cp > s200v else "Sell", "Strong",
        (f"Price (₹{cp:.1f}) > SMA 200 (₹{s200v:.1f}) — long-term bull market signal; suitable for positional accumulation." if cp > s200v
         else f"Price (₹{cp:.1f}) < SMA 200 (₹{s200v:.1f}) — long-term bear signal; only swing trades suitable.")))

    # ── 8. EMA 9 ──────────────────────────────────────────────
    e9v = _safe(close.ewm(span=9, adjust=False).mean()) or cp
    results.append(_ind(
        "EMA 9", f"₹{e9v:.2f}",
        "Buy" if cp > e9v else "Sell", "Moderate",
        (f"Price (₹{cp:.1f}) > EMA 9 (₹{e9v:.1f}) — fast EMA confirms immediate bullish bias; good for intraday entries." if cp > e9v
         else f"Price (₹{cp:.1f}) < EMA 9 (₹{e9v:.1f}) — fast EMA shows selling momentum; avoid intraday longs.")))

    # ── 9. EMA 21 ─────────────────────────────────────────────
    e21v = _safe(close.ewm(span=21, adjust=False).mean()) or cp
    results.append(_ind(
        "EMA 21", f"₹{e21v:.2f}",
        "Buy" if cp > e21v else "Sell", "Moderate",
        (f"Price (₹{cp:.1f}) > EMA 21 (₹{e21v:.1f}) — intermediate momentum is bullish; confirms short-term uptrend." if cp > e21v
         else f"Price (₹{cp:.1f}) < EMA 21 (₹{e21v:.1f}) — intermediate momentum is bearish; downtrend in progress.")))

    # ── 10. EMA 200 ───────────────────────────────────────────
    e200v = _safe(close.ewm(span=200, adjust=False).mean()) or cp
    results.append(_ind(
        "EMA 200", f"₹{e200v:.2f}",
        "Buy" if cp > e200v else "Sell", "Strong",
        (f"Price (₹{cp:.1f}) > EMA 200 (₹{e200v:.1f}) — long-term bull market. Best for accumulation by long-term investors." if cp > e200v
         else f"Price (₹{cp:.1f}) < EMA 200 (₹{e200v:.1f}) — long-term bear zone; only nimble short-term trades appropriate.")))

    # ── 11. DMA (SMA 20, displaced +5) ───────────────────────
    dma_raw = close.rolling(20).mean().shift(5)
    dmav = _safe(dma_raw) or cp
    results.append(_ind(
        "DMA (SMA20+5)", f"₹{dmav:.2f}",
        "Buy" if cp > dmav else "Sell", "Moderate",
        (f"Price (₹{cp:.1f}) > DMA (₹{dmav:.1f}) — 5-period lag confirms established uptrend reliably." if cp > dmav
         else f"Price (₹{cp:.1f}) < DMA (₹{dmav:.1f}) — displaced MA confirms downtrend is established.")))

    # ── 12. HMA 9 ─────────────────────────────────────────────
    hma_s = _hma(close, 9)
    hmav  = _safe(hma_s) or cp
    hmav2 = _safe(hma_s, -2) or hmav
    results.append(_ind(
        "HMA 9", f"₹{hmav:.2f}",
        "Buy" if hmav > hmav2 else "Sell", "Moderate",
        (f"HMA 9 rising (₹{hmav2:.1f} → ₹{hmav:.1f}) — Hull MA (low-lag) signals upward momentum confirmed." if hmav > hmav2
         else f"HMA 9 falling (₹{hmav2:.1f} → ₹{hmav:.1f}) — Hull MA leads price lower; bearish momentum ahead.")))

    # ── 13. Stochastic Oscillator (14, 3) ─────────────────────
    skv_s, sdv_s = _stochastic(df)
    skv = _safe(skv_s) or 50; sdv = _safe(sdv_s) or 50
    if skv < 20 and sdv < 20:
        results.append(_ind("Stochastic (14,3)", f"%K:{skv:.1f} %D:{sdv:.1f}", "Buy", "Strong",
            f"Both %K ({skv:.1f}) and %D ({sdv:.1f}) below 20 — deeply oversold; high-probability reversal zone."))
    elif skv > 80 and sdv > 80:
        results.append(_ind("Stochastic (14,3)", f"%K:{skv:.1f} %D:{sdv:.1f}", "Sell", "Strong",
            f"Both %K ({skv:.1f}) and %D ({sdv:.1f}) above 80 — deeply overbought; high-probability reversal zone."))
    elif skv > sdv:
        results.append(_ind("Stochastic (14,3)", f"%K:{skv:.1f} %D:{sdv:.1f}", "Buy", "Moderate",
            f"%K ({skv:.1f}) crossed above %D ({sdv:.1f}) — bullish stochastic crossover confirmed."))
    elif skv < sdv:
        results.append(_ind("Stochastic (14,3)", f"%K:{skv:.1f} %D:{sdv:.1f}", "Sell", "Moderate",
            f"%K ({skv:.1f}) crossed below %D ({sdv:.1f}) — bearish stochastic crossover confirmed."))
    else:
        results.append(_ind("Stochastic (14,3)", f"%K:{skv:.1f} %D:{sdv:.1f}", "Hold", "Weak",
            f"Stochastic lines converging (%K: {skv:.1f}, %D: {sdv:.1f}) — indecisive, no clear edge."))

    # ── 14. Williams %R (14) ──────────────────────────────────
    wrv = _safe(_williams_r(df)) or -50
    if wrv < -80:
        results.append(_ind("Williams %R (14)", f"{wrv:.1f}", "Buy", "Strong",
            f"Williams %R at {wrv:.1f} (<-80) — deeply oversold; contrarian buy signal with high reversal probability."))
    elif wrv > -20:
        results.append(_ind("Williams %R (14)", f"{wrv:.1f}", "Sell", "Strong",
            f"Williams %R at {wrv:.1f} (>-20) — overbought; contrarian sell signal for short-term reversal."))
    elif wrv > -50:
        results.append(_ind("Williams %R (14)", f"{wrv:.1f}", "Sell", "Weak",
            f"Williams %R at {wrv:.1f} is in upper half — mild bearish bias."))
    else:
        results.append(_ind("Williams %R (14)", f"{wrv:.1f}", "Buy", "Weak",
            f"Williams %R at {wrv:.1f} is in lower half — mild bullish bias."))

    # ── 15. CCI (20) ──────────────────────────────────────────
    cciv = _safe(_cci(df)) or 0
    if cciv < -100:
        results.append(_ind("CCI (20)", f"{cciv:.1f}", "Buy", "Strong",
            f"CCI at {cciv:.1f} (<-100) — price significantly below typical value; strong mean-reversion buy opportunity."))
    elif cciv > 100:
        results.append(_ind("CCI (20)", f"{cciv:.1f}", "Sell", "Strong",
            f"CCI at {cciv:.1f} (>+100) — price significantly above typical value; strong mean-reversion sell signal."))
    elif cciv > 0:
        results.append(_ind("CCI (20)", f"{cciv:.1f}", "Buy", "Weak",
            f"CCI at {cciv:.1f} is above zero — mild bullish momentum."))
    else:
        results.append(_ind("CCI (20)", f"{cciv:.1f}", "Sell", "Weak",
            f"CCI at {cciv:.1f} is below zero — mild bearish momentum."))

    # ── 16. ADX (14) ──────────────────────────────────────────
    adx_s, pdi_s, mdi_s = _adx(df)
    adxv = _safe(adx_s) or 20; pdiv = _safe(pdi_s) or 25; mdiv = _safe(mdi_s) or 25
    if adxv > 25 and pdiv > mdiv:
        results.append(_ind("ADX (14)", f"ADX:{adxv:.1f}", "Buy", "Strong",
            f"ADX ({adxv:.1f}) confirms strong trend; +DI ({pdiv:.1f}) > -DI ({mdiv:.1f}) — ride the bullish trend."))
    elif adxv > 25 and mdiv > pdiv:
        results.append(_ind("ADX (14)", f"ADX:{adxv:.1f}", "Sell", "Strong",
            f"ADX ({adxv:.1f}) confirms strong trend; -DI ({mdiv:.1f}) > +DI ({pdiv:.1f}) — bearish trend is dominant."))
    elif adxv < 20:
        results.append(_ind("ADX (14)", f"ADX:{adxv:.1f}", "Hold", "Weak",
            f"ADX ({adxv:.1f}) < 20 — weak or no trend; market is ranging. Avoid trend-following strategies."))
    else:
        results.append(_ind("ADX (14)", f"ADX:{adxv:.1f}", "Hold", "Moderate",
            f"ADX ({adxv:.1f}) between 20–25 — trend is developing; +DI:{pdiv:.1f} -DI:{mdiv:.1f}. Wait for confirmation."))

    # ── 17. VWAP ──────────────────────────────────────────────
    vwapv = _safe(_vwap(df)) or cp
    results.append(_ind(
        "VWAP", f"₹{vwapv:.2f}",
        "Buy" if cp > vwapv else "Sell", "Moderate",
        (f"Price (₹{cp:.1f}) > VWAP (₹{vwapv:.1f}) — institutional/smart money bias is bullish; safe intraday long." if cp > vwapv
         else f"Price (₹{cp:.1f}) < VWAP (₹{vwapv:.1f}) — institutional selling pressure; avoid intraday longs.")))

    # ── 18. Pivot Points ──────────────────────────────────────
    rec = df.tail(5)
    ppH = float(rec["high"].max()); ppL = float(rec["low"].min()); ppC = float(rec["close"].iloc[-1])
    pp = _pivot_points(ppH, ppL, ppC)
    if cp > pp["r1"]:
        results.append(_ind("Pivot Points", f"P:{pp['pivot']:.1f} R1:{pp['r1']:.1f}", "Buy", "Strong",
            f"Price (₹{cp:.1f}) broke above R1 (₹{pp['r1']:.1f}) — bullish breakout above first resistance. Target R2 ₹{pp['r2']:.1f}."))
    elif cp < pp["s1"]:
        results.append(_ind("Pivot Points", f"P:{pp['pivot']:.1f} S1:{pp['s1']:.1f}", "Sell", "Strong",
            f"Price (₹{cp:.1f}) broke below S1 (₹{pp['s1']:.1f}) — bearish breakdown below first support. Risk to S2 ₹{pp['s2']:.1f}."))
    elif cp > pp["pivot"]:
        results.append(_ind("Pivot Points", f"P:{pp['pivot']:.1f}", "Buy", "Moderate",
            f"Price (₹{cp:.1f}) above Pivot (₹{pp['pivot']:.1f}) — bullish bias; resistance at R1 ₹{pp['r1']:.1f}."))
    else:
        results.append(_ind("Pivot Points", f"P:{pp['pivot']:.1f}", "Sell", "Moderate",
            f"Price (₹{cp:.1f}) below Pivot (₹{pp['pivot']:.1f}) — bearish bias; support at S1 ₹{pp['s1']:.1f}."))

    # ── 19. Fibonacci Retracement ─────────────────────────────
    rec60 = df.tail(60)
    fH = float(rec60["high"].max()); fL = float(rec60["low"].min())
    fib = _fib_levels(fH, fL)
    tol = (fH - fL) * 0.02  # 2 % tolerance
    f38 = fib["38.2%"]; f50 = fib["50%"]; f618 = fib["61.8%"]; f236 = fib["23.6%"]
    if abs(cp - f618) <= tol:
        results.append(_ind("Fibonacci Retracement", f"61.8%:₹{f618:.1f}", "Buy", "Strong",
            f"Price (₹{cp:.1f}) at golden-ratio 61.8% Fib support (₹{f618:.1f}) — highest-probability reversal zone; excellent risk-reward."))
    elif abs(cp - f38) <= tol:
        results.append(_ind("Fibonacci Retracement", f"38.2%:₹{f38:.1f}", "Buy", "Strong",
            f"Price (₹{cp:.1f}) at 38.2% Fib support (₹{f38:.1f}) — key historical reversal level; strong support."))
    elif abs(cp - f50) <= tol:
        results.append(_ind("Fibonacci Retracement", f"50%:₹{f50:.1f}", "Hold", "Moderate",
            f"Price (₹{cp:.1f}) at 50% Fibonacci (₹{f50:.1f}) — psychological midpoint; directional breakout awaited."))
    elif cp > f236:
        results.append(_ind("Fibonacci Retracement", f"23.6%:₹{f236:.1f}", "Sell", "Moderate",
            f"Price (₹{cp:.1f}) above shallow 23.6% retracement (₹{f236:.1f}) — weak pullback suggests further decline may follow."))
    elif cp < f618:
        results.append(_ind("Fibonacci Retracement", f"61.8%:₹{f618:.1f}", "Sell", "Strong",
            f"Price (₹{cp:.1f}) below 61.8% Fib level (₹{f618:.1f}) — deep retracement; bearish structural damage."))
    else:
        results.append(_ind("Fibonacci Retracement", f"50%:₹{f50:.1f}", "Hold", "Weak",
            f"Price (₹{cp:.1f}) between Fibonacci levels — awaiting directional confirmation."))

    # ── Compute historical success rates ──────────────────────
    success = _compute_success_rates(df)

    # ── Build consensus ───────────────────────────────────────
    consensus = _build_consensus(results, success)

    def _clean_nans(d: dict) -> dict:
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}

    return {
        "indicators": results,
        "success_rates": success,
        "consensus": consensus,
        "fibonacci_levels": _clean_nans(fib),
        "pivot_points": _clean_nans(pp),
    }


# ─────────────────────────────────────────────────────────────
# HISTORICAL ACCURACY TRACKER
# ─────────────────────────────────────────────────────────────

def _compute_success_rates(df: pd.DataFrame) -> List[dict]:
    """
    Backtest each indicator's directional accuracy on the provided history.
    Compares next-day close direction vs indicator signal at each day.
    """
    close = df["close"].reset_index(drop=True)
    high  = df["high"].reset_index(drop=True)
    low   = df["low"].reset_index(drop=True)
    n = len(close)
    if n < 50:
        return []

    start = 40  # leave room for longer-window indicators

    # Pre-compute all series
    rsi_all = _rsi(close)
    ml_all, sl_all, _ = _macd(close)
    s10_all = close.rolling(10).mean()
    s50_all = close.rolling(50).mean()
    e9_all  = close.ewm(span=9, adjust=False).mean()
    bbu_all, bbm_all, bbl_all = _bollinger(close)
    df_reset = pd.DataFrame({"high": high, "low": low, "close": close})
    stk_all, _ = _stochastic(df_reset)
    cci_all = _cci(df_reset)
    wr_all  = _williams_r(df_reset)

    def _get(s, i):
        try:
            v = float(s.iloc[i])
            return v if not np.isnan(v) else None
        except Exception:
            return None

    tracking: Dict[str, List[Tuple[int, int]]] = {
        "RSI (14)": [], "MACD (12,26,9)": [], "SMA 10": [], "SMA 50": [],
        "EMA 9": [], "Stochastic (14,3)": [], "CCI (20)": [],
        "Bollinger Bands": [], "Williams %R": [],
    }

    for i in range(start, n - 1):
        cp_i   = float(close.iloc[i])
        actual = 1 if float(close.iloc[i + 1]) > cp_i else -1

        # RSI
        r = _get(rsi_all, i)
        if r is not None:
            tracking["RSI (14)"].append((1 if r > 50 else -1, actual))

        # MACD
        mv = _get(ml_all, i); sv = _get(sl_all, i)
        if mv is not None and sv is not None:
            tracking["MACD (12,26,9)"].append((1 if mv > sv else -1, actual))

        # SMA 10
        s10 = _get(s10_all, i)
        if s10: tracking["SMA 10"].append((1 if cp_i > s10 else -1, actual))

        # SMA 50
        s50 = _get(s50_all, i)
        if s50: tracking["SMA 50"].append((1 if cp_i > s50 else -1, actual))

        # EMA 9
        e9 = _get(e9_all, i)
        if e9: tracking["EMA 9"].append((1 if cp_i > e9 else -1, actual))

        # Stochastic
        sk = _get(stk_all, i)
        if sk is not None:
            tracking["Stochastic (14,3)"].append((1 if sk < 50 else -1, actual))

        # CCI
        cc = _get(cci_all, i)
        if cc is not None:
            tracking["CCI (20)"].append((1 if cc > 0 else -1, actual))

        # Bollinger
        bbu = _get(bbu_all, i); bbm = _get(bbm_all, i); bbl = _get(bbl_all, i)
        if all(v is not None for v in [bbu, bbm, bbl]):
            pred = -1 if cp_i >= bbu else (1 if cp_i <= bbl else (1 if cp_i < bbm else -1))
            tracking["Bollinger Bands"].append((pred, actual))

        # Williams %R
        wr = _get(wr_all, i)
        if wr is not None:
            tracking["Williams %R"].append((1 if wr < -50 else -1, actual))

    rates = []
    for name, sigs in tracking.items():
        if len(sigs) < 5:
            continue
        correct = sum(1 for p, a in sigs if p == a)
        rates.append({
            "name": name,
            "accuracy": round(correct / len(sigs) * 100, 1),
            "total_signals": len(sigs),
        })

    rates.sort(key=lambda x: x["accuracy"], reverse=True)
    return rates


# ─────────────────────────────────────────────────────────────
# CONSENSUS BUILDER
# ─────────────────────────────────────────────────────────────

def _build_consensus(results: List[dict], success_rates: List[dict]) -> dict:
    """
    Weighted vote across all 19 indicators:
      Strong = 3, Moderate = 2, Weak = 1
    Score normalised to [-100, +100].
    """
    weights = {"Strong": 3, "Moderate": 2, "Weak": 1}
    score = 0; max_score = 0; buy = sell = hold = 0

    for r in results:
        w = weights.get(r["strength"], 1)
        if r["signal"] == "Buy":
            score += w; buy += 1
        elif r["signal"] == "Sell":
            score -= w; sell += 1
        else:
            hold += 1
        max_score += w

    norm = score / max_score if max_score else 0  # -1 … +1
    total = buy + sell + hold

    if norm > 0.40:
        action = "BUY"
        confidence = "High" if norm > 0.65 else "Moderate"
        if norm > 0.65:
            duration = "3–6 Months (Medium Term)"
            rationale = (
                f"{buy}/{total} indicators signal BUY with a strong weighted score of "
                f"{norm*100:.0f}/100. Multi-indicator bullish consensus — "
                "accumulate on dips, set stop-loss below SMA 50/S1. "
                "Target: next Fibonacci resistance or R2 pivot."
            )
        else:
            duration = "1–3 Months (Short–Medium Term)"
            rationale = (
                f"{buy}/{total} indicators lean bullish (score: {norm*100:.0f}/100). "
                "Moderate consensus — consider partial entry (50%) now and "
                "add more on dips to key support. Keep stop-loss tight."
            )
    elif norm < -0.40:
        action = "SELL / EXIT"
        confidence = "High" if norm < -0.65 else "Moderate"
        if norm < -0.65:
            duration = "Exit Now / Short Term Short"
            rationale = (
                f"{sell}/{total} indicators signal SELL with bearish score "
                f"{norm*100:.0f}/100. Strong multi-indicator bearish consensus — "
                "exit long positions immediately. Re-enter only after at least "
                "3 consecutive bullish signals confirm trend reversal."
            )
        else:
            duration = "Reduce Positions / Hold Off"
            rationale = (
                f"{sell}/{total} indicators lean bearish (score: {norm*100:.0f}/100). "
                "Moderate bearish bias — reduce exposure by 50%, set tighter "
                "stops on remaining positions, await trend-reversal confirmation."
            )
    else:
        action = "HOLD / WAIT"
        confidence = "Low"
        duration = "Watch — Wait for Breakout"
        rationale = (
            f"Mixed signals — {buy} Buy, {sell} Sell, {hold} Neutral. "
            "The market is in a consolidation phase. "
            "Wait for at least 3 strong-strength indicators to align "
            "in the same direction before entering a new position."
        )

    best = success_rates[0] if success_rates else None

    return {
        "action": action,
        "confidence": confidence,
        "duration": duration,
        "rationale": rationale,
        "score": round(norm * 100, 1),
        "buy_count": buy,
        "sell_count": sell,
        "hold_count": hold,
        "total": total,
        "best_indicator": best["name"] if best else "—",
        "best_accuracy": best["accuracy"] if best else 0,
    }
