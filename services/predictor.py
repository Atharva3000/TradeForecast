"""
Engineering & Math Layer — Technical indicators and ML prediction pipeline.

Implements:
- SMA (Simple Moving Average)
- RSI via Wilder's exponential smoothing (alpha = 1/period)
- MACD with standard industrial EMA (span-based)
- Bollinger Bands (20-period, 2σ)
- LinearRegression forecast with lag-3 features and 80/20 chrono split

Critical design decision (per user feedback):
    The very last DataFrame row is isolated as the forecast input *before*
    dropping NaN target rows.  This prevents the pipeline from silently
    discarding the only row that can predict tomorrow's price.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False



# =====================================================================
# Technical Indicators
# =====================================================================

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average over *window* periods."""
    return series.rolling(window=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using **Wilder's Exponential Smoothing**.

    Formula
    -------
    RSI = 100 − (100 / (1 + RS))

    where RS = SmoothedAvgGain / SmoothedAvgLoss and the smoothing uses
    an EMA with α = 1 / period  (Wilder's decay factor).

    A tiny epsilon (1e-10) is added to the denominator to prevent
    division-by-zero on perfectly flat price segments.
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: alpha = 1 / period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence with standard industrial EMA.

    Returns
    -------
    (macd_line, signal_line, histogram)

    Unlike Wilder's smoothing, MACD uses the traditional EMA factor
    α = 2 / (span + 1), which Pandas applies automatically via
    ``Series.ewm(span=...)``.
    """
    ema_fast = series.ewm(span=fast_span, adjust=False).mean()
    ema_slow = series.ewm(span=slow_span, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# =====================================================================
# Confidence Heuristic
# =====================================================================

def compute_confidence(latest_row: pd.Series) -> int:
    """
    Heuristic 0-100 score derived from technical-indicator alignment.

    Scoring rubric (20 pts each, max 100):
        1. RSI in neutral zone  [40 ≤ RSI ≤ 60]
        2. Price above SMA-50
        3. SMA-10 > SMA-50      (golden-cross region)
        4. MACD above signal line
        5. Price within Bollinger Bands (replaces flat-volatility penalty
           per user feedback — a breakout with heavy volume should not be
           penalised by a raw std-dev check)
    """
    score = 0

    rsi        = latest_row.get("rsi_14", 50)
    sma_10     = latest_row.get("sma_10", 0)
    sma_50     = latest_row.get("sma_50", 0)
    macd       = latest_row.get("macd", 0)
    macd_sig   = latest_row.get("macd_signal", 0)
    close      = latest_row.get("close", 0)
    bb_upper   = latest_row.get("bb_upper", close)
    bb_lower   = latest_row.get("bb_lower", close)

    if 40 <= rsi <= 60:
        score += 20
    if close > sma_50:
        score += 20
    if sma_10 > sma_50:
        score += 20
    if macd > macd_sig:
        score += 20
    if bb_lower <= close <= bb_upper:
        score += 20

    return max(0, min(100, score))


# =====================================================================
# Prediction Pipeline
# =====================================================================

FEATURE_COLS: list[str] = [
    "sma_10", "sma_50", "rsi_14",
    "macd", "macd_signal",
    "close_lag_1", "close_lag_2", "close_lag_3",
]


def run_prediction(df: pd.DataFrame, model_type: str = "Ridge Regression", lag_period: int = 3) -> dict:
    """
    End-to-end prediction pipeline.

    Steps
    -----
    1. Compute technical indicators on the full DataFrame.
    2. Create lag features dynamically up to lag_period.
    3. Define target as the *next* period's close (shift -1).
    4. Isolate the last row for forecasting.
    5. Drop NaN rows, split 80/20 chronologically (no shuffle).
    6. Fit the selected model (Ridge, XGBoost with GBR fallback, or MLP Neural Net).
    7. Return predictions, forecast, R2, historical data, and feature importances.
    """
    df = df.dropna(subset=["close"]).copy().reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(
            f"Insufficient data points ({len(df)}). "
            "Need at least 60 for reliable indicator computation."
        )

    # --- Technical indicators -----------------------------------------------
    df["sma_10"] = compute_sma(df["close"], 10)
    df["sma_50"] = compute_sma(df["close"], 50)
    df["rsi_14"] = compute_rsi(df["close"], 14)

    macd_line, signal_line, histogram = compute_macd(df["close"])
    df["macd"]           = macd_line
    df["macd_signal"]    = signal_line
    df["macd_histogram"] = histogram

    # Bollinger Bands (20-period, 2σ)
    bb_mid = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    # --- Dynamic Lag features -------------------------------------------------
    feature_cols = [
        "sma_10", "sma_50", "rsi_14",
        "macd", "macd_signal",
    ]
    for i in range(1, lag_period + 1):
        col_name = f"close_lag_{i}"
        df[col_name] = df["close"].shift(i)
        feature_cols.append(col_name)

    # --- Target: next-period close ------------------------------------------
    df["target"] = df["close"].shift(-1)

    # === CRITICAL: isolate the last row BEFORE dropping NaN targets ==========
    forecast_row = df.iloc[[-1]].copy()
    latest_indicators = df.iloc[-1].copy()

    # --- Clean for training -------------------------------------------------
    train_df = df.dropna(subset=feature_cols + ["target"]).copy()

    if len(train_df) < 10:
        raise ValueError(
            f"After cleaning, only {len(train_df)} usable rows remain. "
            "Need at least 10 for model training."
        )

    # --- 80 / 20 chronological split ----------------------------------------
    split_idx = int(len(train_df) * 0.8)
    train_set = train_df.iloc[:split_idx]
    test_set  = train_df.iloc[split_idx:]

    X_train = train_set[feature_cols].values
    y_train = train_set["target"].values
    X_test  = test_set[feature_cols].values
    y_test  = test_set["target"].values

    # --- Model Selection ----------------------------------------------------
    if model_type == "XGBoost" or model_type == "XGBoost Regressor":
        if XGB_AVAILABLE:
            model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    elif model_type == "Deep Learning (MLP Neural Net)" or model_type == "Deep Learning (Neural Net)":
        model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    else:
        model = Ridge(alpha=10.0)

    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)

    # --- Forecast next period -----------------------------------------------
    X_forecast = forecast_row[feature_cols].values.reshape(1, -1)

    if np.isnan(X_forecast).any():
        raise ValueError(
            "Forecast features contain NaN — the dataset is too short "
            "for the selected timeframe and indicator windows."
        )

    next_day_forecast = float(model.predict(X_forecast)[0])

    # Predict for the entire historical training set
    all_features = train_df[feature_cols].values
    all_predictions = model.predict(all_features)

    # --- Build historical_vs_predicted array (full history) -----------------
    historical_vs_predicted: list[dict] = []
    for i, (_, row) in enumerate(train_df.iterrows()):
        actual = round(float(row["close"]), 2)
        
        # Extract or fallback for open, high, low
        open_val = row.get("open", actual)
        high_val = row.get("high", actual)
        low_val = row.get("low", actual)
        
        # Safe checks for NaN values
        if pd.isna(open_val): open_val = actual
        if pd.isna(high_val): high_val = max(actual, open_val)
        if pd.isna(low_val): low_val = min(actual, open_val)
        
        historical_vs_predicted.append({
            "date":      str(row["date"]),
            "open":      round(float(open_val), 2),
            "high":      round(float(high_val), 2),
            "low":       round(float(low_val), 2),
            "actual":    actual,
            "predicted": round(float(all_predictions[i]), 2),
        })

    # Return model feature importances/coefficients if possible
    feature_importance = {}
    try:
        if hasattr(model, "coef_"):
            for col, val in zip(feature_cols, model.coef_):
                feature_importance[col] = float(val)
        elif hasattr(model, "feature_importances_"):
            for col, val in zip(feature_cols, model.feature_importances_):
                feature_importance[col] = float(val)
    except Exception:
        pass

    return {
        "latest_df":               latest_indicators,
        "next_day_forecast":       next_day_forecast,
        "r2_score":                round(r2, 4),
        "historical_vs_predicted": historical_vs_predicted,
        "feature_importance":      feature_importance
    }


def get_ml_predictions(df: pd.DataFrame, model_type: str = "Ridge Regression", lag_period: int = 3) -> pd.Series:
    """
    Train the model and predict the next-period close
    for all rows where features are available.
    """
    if len(df) < 60:
        return pd.Series(np.nan, index=df.index)

    df_feat = df.copy()

    # --- Technical indicators -----------------------------------------------
    df_feat["sma_10"] = compute_sma(df_feat["close"], 10)
    df_feat["sma_50"] = compute_sma(df_feat["close"], 50)
    df_feat["rsi_14"] = compute_rsi(df_feat["close"], 14)

    macd_line, signal_line, _ = compute_macd(df_feat["close"])
    df_feat["macd"]           = macd_line
    df_feat["macd_signal"]    = signal_line

    # --- Dynamic Lag features -------------------------------------------------
    feature_cols = [
        "sma_10", "sma_50", "rsi_14",
        "macd", "macd_signal",
    ]
    for i in range(1, lag_period + 1):
        col_name = f"close_lag_{i}"
        df_feat[col_name] = df_feat["close"].shift(i)
        feature_cols.append(col_name)

    # --- Target: next-period close ------------------------------------------
    df_feat["target"] = df_feat["close"].shift(-1)

    # Features mask (non-NaN features)
    feature_mask = df_feat[feature_cols].notna().all(axis=1)

    # Rows with both features and target for training
    train_mask = feature_mask & df_feat["target"].notna()

    if train_mask.sum() < 10:
        return pd.Series(np.nan, index=df.index)

    X_train = df_feat.loc[train_mask, feature_cols].values
    y_train = df_feat.loc[train_mask, "target"].values

    if model_type == "XGBoost" or model_type == "XGBoost Regressor":
        if XGB_AVAILABLE:
            model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    elif model_type == "Deep Learning (MLP Neural Net)" or model_type == "Deep Learning (Neural Net)":
        model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    else:
        model = Ridge(alpha=10.0)

    model.fit(X_train, y_train)

    # Predict for all rows where features are valid
    predictions = pd.Series(np.nan, index=df.index)
    X_pred = df_feat.loc[feature_mask, feature_cols].values
    predictions.loc[feature_mask] = model.predict(X_pred)

    return predictions


