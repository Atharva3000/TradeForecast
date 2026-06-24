import numpy as np
import pandas as pd
from services.predictor import get_ml_predictions, compute_sma, compute_rsi, compute_macd

def run_historical_backtest(
    df: pd.DataFrame,
    model_type: str = "Ridge Regression",
    lag_period: int = 3,
    initial_capital: float = 10000.0,
    transaction_cost_pct: float = 0.1,
) -> dict:
    """
    Simulate a trading strategy historically on the asset's data.
    
    Strategy Rules:
    - Multi-indicator consensus score computed daily:
      * RSI (14): +1 if <30 (oversold), -1 if >70 (overbought)
      * SMA 10/50: +1 if SMA 10 > SMA 50, -1 if SMA 10 < SMA 50
      * MACD: +1 if MACD > Signal, -1 if MACD < Signal
      * BB: +1 if close <= BB lower, -1 if close >= BB upper
      * ML Forecast: +2 if predicted change > 0.5%, -2 if predicted change < -0.5%
    - BUY when consensus >= 3 (Bullish consensus)
    - SELL when consensus <= -2 or RSI > 75 (Bearish/Overbought exit)
    """
    df = df.dropna(subset=["close"]).copy().reset_index(drop=True)
    n = len(df)
    if n < 60:
        return {
            "error": "Insufficient data to backtest. Need at least 60 days.",
            "success": False
        }

    # 1. Compute Indicators
    df["sma_10"] = compute_sma(df["close"], 10)
    df["sma_50"] = compute_sma(df["close"], 50)
    df["rsi_14"] = compute_rsi(df["close"], 14)
    macd_l, sig_l, _ = compute_macd(df["close"])
    df["macd"] = macd_l
    df["macd_signal"] = sig_l
    
    bb_mid = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    # Get ML Predictions
    try:
        ml_preds = get_ml_predictions(df, model_type, lag_period)
    except Exception:
        ml_preds = pd.Series(df["close"].shift(-1), index=df.index) # fallback

    # Compute daily consensus scores
    scores = np.zeros(n)
    for i in range(1, n):
        score = 0
        close_p = df.loc[i, "close"]
        rsi = df.loc[i, "rsi_14"]
        sma10 = df.loc[i, "sma_10"]
        sma50 = df.loc[i, "sma_50"]
        macd = df.loc[i, "macd"]
        macd_sig = df.loc[i, "macd_signal"]
        bb_up = df.loc[i, "bb_upper"]
        bb_lo = df.loc[i, "bb_lower"]
        pred = ml_preds.iloc[i]
        
        # RSI
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1
            
        # SMA Crossover
        if sma10 > sma50:
            score += 1
        else:
            score -= 1
            
        # MACD
        if macd > macd_sig:
            score += 1
        else:
            score -= 1
            
        # Bollinger Bands
        if close_p <= bb_lo:
            score += 1
        elif close_p >= bb_up:
            score -= 1
            
        # ML
        if not pd.isna(pred) and close_p > 0:
            change = (pred - close_p) / close_p
            if change > 0.005:
                score += 2
            elif change < -0.005:
                score -= 2
                
        scores[i] = score

    df["consensus_score"] = scores

    # 2. Trading Simulation
    cash = initial_capital
    shares = 0.0
    position = 0 # 0 = cash, 1 = shares
    portfolio_values = []
    benchmark_shares = initial_capital / df.loc[0, "close"]
    
    trades = [] # Keep track of trade returns for win rate
    buy_price = 0.0
    
    # We skip first 50 rows due to indicators warming up
    warm_up = 50
    
    for i in range(n):
        close_p = float(df.loc[i, "close"])
        date_str = str(df.loc[i, "date"])
        score = scores[i]
        rsi = float(df.loc[i, "rsi_14"])
        
        if i < warm_up:
            portfolio_values.append({
                "date": date_str,
                "strategy": cash,
                "benchmark": benchmark_shares * close_p
            })
            continue

        # Buy Signal
        if position == 0 and score >= 3:
            # Buy shares
            fee = cash * (transaction_cost_pct / 100)
            net_cash = cash - fee
            shares = net_cash / close_p
            cash = 0.0
            position = 1
            buy_price = close_p
            trades.append({"entry_date": date_str, "entry_price": buy_price})
            
        # Sell Signal
        elif position == 1 and (score <= -2 or rsi > 75):
            # Sell shares
            gross_cash = shares * close_p
            fee = gross_cash * (transaction_cost_pct / 100)
            cash = gross_cash - fee
            shares = 0.0
            position = 0
            if trades:
                trades[-1]["exit_date"] = date_str
                trades[-1]["exit_price"] = close_p
                trades[-1]["return_pct"] = ((close_p - buy_price) / buy_price) * 100

        # Calculate current daily portfolio value
        curr_val = cash + (shares * close_p)
        portfolio_values.append({
            "date": date_str,
            "strategy": curr_val,
            "benchmark": benchmark_shares * close_p
        })

    # 3. Calculate Performance Metrics
    final_state = portfolio_values[-1]
    strat_final = final_state["strategy"]
    bench_final = final_state["benchmark"]
    
    strat_return_pct = ((strat_final - initial_capital) / initial_capital) * 100
    bench_return_pct = ((bench_final - initial_capital) / initial_capital) * 100
    alpha = strat_return_pct - bench_return_pct
    
    # Win Rate
    completed_trades = [t for t in trades if "exit_price" in t]
    profitable_trades = [t for t in completed_trades if t["exit_price"] > t["entry_price"]]
    win_rate = (len(profitable_trades) / len(completed_trades) * 100) if completed_trades else 0.0
    
    # Max Drawdown
    peak = initial_capital
    max_dd = 0.0
    for state in portfolio_values:
        val = state["strategy"]
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd * 100
    
    # Sharpe Ratio (daily returns based)
    strat_series = pd.Series([s["strategy"] for s in portfolio_values])
    daily_returns = strat_series.pct_change().dropna()
    mean_ret = daily_returns.mean()
    std_dev = daily_returns.std()
    
    # Annualized Sharpe (assuming 252 trading days)
    sharpe = (mean_ret / (std_dev + 1e-10)) * np.sqrt(252) if len(daily_returns) > 1 and std_dev > 0 else 0.0

    return {
        "success": True,
        "initial_capital": initial_capital,
        "final_capital": strat_final,
        "strategy_return_pct": round(strat_return_pct, 2),
        "benchmark_return_pct": round(bench_return_pct, 2),
        "alpha": round(alpha, 2),
        "win_rate_pct": round(win_rate, 1),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": len(completed_trades),
        "trades": completed_trades,
        "equity_curve": portfolio_values
    }

