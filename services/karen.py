import re
import os
import json
import logging
import urllib.request
import asyncio
from typing import Optional
import yfinance as yf

from stock_universe import ALL_ASSETS, TICKER_TO_NAME
from services.paper_db import get_portfolio, execute_order, reset_portfolio, set_portfolio_cash
from services.data_fetcher import fetch_stock_data
from services.predictor import run_prediction, compute_confidence
from services.rag import retrieve_context

logger = logging.getLogger(__name__)

# Predefined shortcuts for popular assets to speed up resolution
SHORTCUTS = {
    "nifty": "^NSEI",
    "nifty 50": "^NSEI",
    "nifty50": "^NSEI",
    "sensex": "^BSESN",
    "bank nifty": "^NSEBANK",
    "banknifty": "^NSEBANK",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infy": "INFY.NS",
    "infosys": "INFY.NS",
    "wipro": "WIPRO.NS",
    "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "sbin": "SBIN.NS",
    "state bank of india": "SBIN.NS",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD"
}

def resolve_input_symbol(symbol_str: str) -> Optional[str]:
    """
    Intelligently map user-typed strings (tickers or friendly names) to Yahoo Finance tickers.
    """
    symbol_str = symbol_str.strip().lower()
    
    # 1. Exact shortcut match
    if symbol_str in SHORTCUTS:
        return SHORTCUTS[symbol_str]
        
    # 2. Direct uppercase in TICKER_TO_NAME
    upper_sym = symbol_str.upper()
    if upper_sym in TICKER_TO_NAME:
        return upper_sym
        
    # 3. Direct uppercase with .NS or .BO suffix
    if (upper_sym + ".NS") in TICKER_TO_NAME:
        return upper_sym + ".NS"
    if (upper_sym + ".BO") in TICKER_TO_NAME:
        return upper_sym + ".BO"
        
    # 4. Check for display name exact match (case-insensitive)
    for name, ticker in ALL_ASSETS.items():
        if name.lower() == symbol_str:
            return ticker
            
    # 5. Fuzzy match: substring check in display names
    best_match = None
    best_len = 9999
    for name, ticker in ALL_ASSETS.items():
        if symbol_str in name.lower():
            # Prefer shorter display names to avoid greedy matching
            if len(name) < best_len:
                best_len = len(name)
                best_match = ticker
                
    if best_match:
        return best_match
        
    # 6. Fallback: if it looks like a ticker, return it uppercase
    if len(symbol_str) <= 10 and symbol_str.replace(".", "").replace("-", "").isalnum():
        return upper_sym
        
    return None

def get_current_ticker_price(ticker: str) -> Optional[float]:
    """Helper to fetch the current market price of a stock using fast_info or history download."""
    try:
        t = yf.Ticker(ticker)
        lp = t.fast_info.last_price
        if lp is not None:
            return float(lp)
            
        # Fallback to 5-day daily close
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if not df.empty:
            import pandas as pd
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return float(df.dropna(subset=["Close"])["Close"].iloc[-1])
    except Exception as e:
        logger.error("Failed to fetch price for ticker %s in Karen assistant: %s", ticker, e)
    return None

async def get_prediction_for_karen(ticker: str) -> dict:
    """Fetches stock historical data and runs the standard consensus prediction pipeline."""
    df = await fetch_stock_data(ticker, "1M")
    if df.empty:
        raise ValueError(f"Ticker '{ticker}' returned empty historical dataframe.")
        
    # Run the machine learning pipeline (Ridge Regression by default)
    pipeline_results = run_prediction(df, model_type="Ridge Regression")
    latest_row = pipeline_results["latest_df"]
    current_price = float(latest_row["close"])
    forecast = float(pipeline_results["next_day_forecast"])
    
    diff = forecast - current_price
    direction = "up" if diff >= 0 else "down"
    conf = float(compute_confidence(latest_row))
    
    return {
        "current_price": current_price,
        "forecast_price": forecast,
        "prediction_direction": direction,
        "confidence_score": conf,
        "consensus_signal": "BUY" if direction == "up" and conf > 55 else "SELL" if direction == "down" and conf > 55 else "HOLD"
    }

def get_live_scan_results() -> dict:
    """Reads scan results from market_scan_results.json."""
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(_BASE_DIR, "market_scan_results.json")
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def format_val(val: float, currency: str) -> str:
    """Formats numeric value to Indian or US style currency layout."""
    if currency == "₹":
        return f"₹{val:,.2f}"
    return f"${val:,.2f}"

def call_gemini_api(api_key: str, prompt: str, system_instruction: str) -> str:
    """Direct HTTP POST request to Google Gemini API to prevent dependency issues."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_instruction}\n\nUser Request & Context:\n{prompt}"}
                ]
            }
        ]
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = response.read().decode("utf-8")
            data = json.loads(res_body)
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "I received your message, but the AI response was empty. Please try again."
    except Exception as e:
        logger.error("Gemini API call failed in Karen assistant: %s", e)
        return f"*(Gemini API connection error)*: I could not contact my brain servers. Falling back to platform intelligence."

async def generate_karen_response(
    username: str,
    message: str,
    history: list,
    active_ticker: Optional[str] = None
) -> dict:
    """
    Main orchestrator for Karen AI. Identifies action intents, fetches database/market context,
    runs local or LLM generation, and executes transactions.
    """
    msg_clean = message.strip()
    username = username.strip().lower()
    
    action_executed = None
    ticker_affected = None
    context_data = {}
    
    # -------------------------------------------------------------------------
    # RAG RETRIEVAL (Retrieve platform documentation matching user query)
    # -------------------------------------------------------------------------
    rag_docs = retrieve_context(msg_clean, limit=1)
    rag_context = ""
    if rag_docs:
        rag_context = "\n\nRelevant Platform Documentation Context:\n"
        for doc in rag_docs:
            rag_context += f"### {doc['title']}\n{doc['content']}\n\n"

    # -------------------------------------------------------------------------
    # 1. INTENT ANALYSIS & ACTION EXECUTION
    # -------------------------------------------------------------------------
    
    # --- A. BUY ORDER INTENT ---
    buy_match = re.search(r"(?i)\bbuy\s+(\d+(?:\.\d+)?)\s*(?:shares?\s+of)?\s*([a-zA-Z0-9.\-\^]+(?:\s+[a-zA-Z0-9.\-\^]+)*)", msg_clean)
    buy_match_alt = re.search(r"(?i)\bbuy\s+([a-zA-Z0-9.\-\^]+(?:\s+[a-zA-Z0-9.\-\^]+)*)\s+(\d+(?:\.\d+)?)", msg_clean)
    
    if buy_match or buy_match_alt:
        qty_str = buy_match.group(1) if buy_match else buy_match_alt.group(2)
        stock_str = buy_match.group(2) if buy_match else buy_match_alt.group(1)
        
        try:
            quantity = float(qty_str)
            ticker = resolve_input_symbol(stock_str)
            
            if not ticker:
                return {
                    "response": f"I couldn't find a stock matching **'{stock_str}'**. Please specify the exact ticker (e.g., `RELIANCE.NS` or `AAPL`).",
                    "system_status": "ticker_unresolved"
                }
                
            price = get_current_ticker_price(ticker)
            if not price:
                return {
                    "response": f"I resolved **'{stock_str}'** to `{ticker}`, but failed to fetch its current market price. Please try again.",
                    "system_status": "price_lookup_failed"
                }
                
            res = execute_order(username, ticker, "BUY", quantity, price, "Market")
            action_executed = "trade_buy"
            ticker_affected = ticker
            
            friendly_name = TICKER_TO_NAME.get(ticker, ticker)
            p_data = get_portfolio(username)
            currency = p_data["currency"]
            total_cost = quantity * price
            
            local_response = (
                f"### ✅ Order Executed Successfully!\n\n"
                f"I've placed a **BUY** order for **{quantity:,.2f} shares** of **{friendly_name}** (`{ticker}`) "
                f"at a market price of **{format_val(price, currency)}** per share.\n\n"
                f"- **Transaction Cost**: {format_val(total_cost, currency)}\n"
                f"- **New Cash Balance**: {format_val(p_data['cash_balance'], currency)}\n\n"
                f"Your open positions table has been updated in the background!"
            )
            context_data = {"trade": res, "portfolio": p_data}
            
        except ValueError as e:
            return {
                "response": f"⚠️ **Trade Rejected**: {str(e)}",
                "system_status": "trade_rejected"
            }
        except Exception as e:
            return {
                "response": f"⚠️ **Trade Execution Error**: {str(e)}",
                "system_status": "error"
            }
            
    # --- B. SELL ORDER INTENT ---
    elif (sell_match := re.search(r"(?i)\bsell\s+(\d+(?:\.\d+)?)\s*(?:shares?\s+of)?\s*([a-zA-Z0-9.\-\^]+(?:\s+[a-zA-Z0-9.\-\^]+)*)", msg_clean)) or \
         (sell_match_alt := re.search(r"(?i)\bsell\s+([a-zA-Z0-9.\-\^]+(?:\s+[a-zA-Z0-9.\-\^]+)*)\s+(\d+(?:\.\d+)?)", msg_clean)):
        
        qty_str = sell_match.group(1) if sell_match else sell_match_alt.group(2)
        stock_str = sell_match.group(2) if sell_match else sell_match_alt.group(1)
        
        try:
            quantity = float(qty_str)
            ticker = resolve_input_symbol(stock_str)
            
            if not ticker:
                return {
                    "response": f"I couldn't find a stock matching **'{stock_str}'**. Please specify the exact ticker (e.g., `RELIANCE.NS` or `AAPL`).",
                    "system_status": "ticker_unresolved"
                }
                
            price = get_current_ticker_price(ticker)
            if not price:
                return {
                    "response": f"I resolved **'{stock_str}'** to `{ticker}`, but failed to fetch its current market price. Please try again.",
                    "system_status": "price_lookup_failed"
                }
                
            res = execute_order(username, ticker, "SELL", quantity, price, "Market")
            action_executed = "trade_sell"
            ticker_affected = ticker
            
            friendly_name = TICKER_TO_NAME.get(ticker, ticker)
            p_data = get_portfolio(username)
            currency = p_data["currency"]
            total_revenue = quantity * price
            
            local_response = (
                f"### ✅ Order Executed Successfully!\n\n"
                f"I've placed a **SELL** order for **{quantity:,.2f} shares** of **{friendly_name}** (`{ticker}`) "
                f"at a market price of **{format_val(price, currency)}** per share.\n\n"
                f"- **Transaction Revenue**: {format_val(total_revenue, currency)}\n"
                f"- **New Cash Balance**: {format_val(p_data['cash_balance'], currency)}\n\n"
                f"Your positions have been updated in the background!"
            )
            context_data = {"trade": res, "portfolio": p_data}
            
        except ValueError as e:
            return {
                "response": f"⚠️ **Trade Rejected**: {str(e)}",
                "system_status": "trade_rejected"
            }
        except Exception as e:
            return {
                "response": f"⚠️ **Trade Execution Error**: {str(e)}",
                "system_status": "error"
            }
            
    # --- C. PORTFOLIO INTENT ---
    elif re.search(r"(?i)\b(portfolio|positions|holdings|balance|cash|performance)\b", msg_clean):
        p_data = get_portfolio(username)
        currency = p_data["currency"]
        
        positions = p_data["positions"]
        cash = p_data["cash_balance"]
        
        positions_value = 0.0
        unrealized_pnl = 0.0
        
        pos_list = []
        for pos in positions:
            ticker = pos["ticker"]
            price = get_current_ticker_price(ticker) or pos["average_price"]
            curr_val = price * pos["quantity"]
            pnl = (price - pos["average_price"]) * pos["quantity"]
            cost_basis = pos["average_price"] * pos["quantity"]
            pnl_pct = (pnl / cost_basis * 100) if cost_basis != 0 else 0
            
            positions_value += curr_val
            unrealized_pnl += pnl
            
            friendly_name = TICKER_TO_NAME.get(ticker, ticker)
            pos_list.append(
                f"| **{friendly_name}** (`{ticker}`) | {pos['quantity']:.1f} | {format_val(pos['average_price'], currency)} | "
                f"{format_val(price, currency)} | {format_val(curr_val, currency)} | "
                f"<span style='color:{'#00e290' if pnl >= 0 else '#ff8080'}'>{format_val(pnl, currency)} ({pnl_pct:+.2f}%)</span> |"
            )
            
        port_val = cash + positions_value
        
        local_response = (
            f"### 💼 Your Paper Trading Portfolio\n\n"
            f"- **Net Portfolio Value**: **{format_val(port_val, currency)}**\n"
            f"- **Unused Cash Balance**: **{format_val(cash, currency)}**\n"
            f"- **Open Positions Value**: **{format_val(positions_value, currency)}**\n"
            f"- **Total Unrealized PnL**: **<span style='color:{'#00e290' if unrealized_pnl >= 0 else '#ff8080'}'>{format_val(unrealized_pnl, currency)}</span>**\n\n"
        )
        
        if pos_list:
            local_response += (
                f"#### Open Positions\n"
                f"| Asset | Shares | Avg Cost | Last Price | Total Value | Unrealized PnL |\n"
                f"| :--- | :--- | :--- | :--- | :--- | :--- |\n"
                + "\n".join(pos_list)
            )
        else:
            local_response += "*You have no active open positions. Type `Buy 10 Reliance` to start paper trading!*"
            
        context_data = {"portfolio": p_data, "port_val": port_val, "positions_value": positions_value, "unrealized_pnl": unrealized_pnl}

    # --- D. PREDICTION INTENT ---
    elif (pred_match := re.search(r"(?i)\b(?:predict|forecast|prediction|target|outlook|analyze|analysis|chart)\s+(?:for\s+|of\s+)?([a-zA-Z0-9.\-\^]+(?:\s+[a-zA-Z0-9.\-\^]+)*)", msg_clean)):
        stock_str = pred_match.group(1)
        ticker = resolve_input_symbol(stock_str)
        
        if not ticker:
            return {
                "response": f"I couldn't find a stock matching **'{stock_str}'**. Please specify the exact ticker (e.g., `RELIANCE.NS` or `AAPL`).",
                "system_status": "ticker_unresolved"
            }
            
        try:
            pred_res = await get_prediction_for_karen(ticker)
            friendly_name = TICKER_TO_NAME.get(ticker, ticker)
            
            curr_p = pred_res["current_price"]
            fore_p = pred_res["forecast_price"]
            diff = fore_p - curr_p
            diff_pct = (diff / curr_p * 100) if curr_p != 0 else 0
            conf = pred_res["confidence_score"]
            direction = pred_res["prediction_direction"]
            
            currency = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^") else "$"
            
            local_response = (
                f"### 📊 AI Prediction consensus: **{friendly_name}** (`{ticker}`)\n\n"
                f"- **Current Market Price**: **{format_val(curr_p, currency)}**\n"
                f"- **Predicted 30-Day Target**: **{format_val(fore_p, currency)}**\n"
                f"- **Expected Forecast Direction**: **{direction.upper()}** ({diff_pct:+.2f}%)\n"
                f"- **AI Confidence Score**: **{conf:.1f}%**\n"
                f"- **Signals Consensus**: **{pred_res.get('consensus_signal', 'BUY').upper()}**\n\n"
                f"**Confidence Analysis**: The consensus system combines a Ridge technical trend, pattern ensemble, and Deep Neural consensus. "
                f"It indicates a confidence of {conf:.1f}% on this directional target."
            )
            action_executed = "predict"
            ticker_affected = ticker
            context_data = {"prediction": pred_res, "ticker": ticker, "name": friendly_name}
        except Exception as e:
            local_response = f"I resolved the stock to `{ticker}`, but could not run predictions: {str(e)}"

    # --- E. SCANNER INTENT ---
    elif re.search(r"(?i)\b(scanner|scan|bullish|bearish|setups|screener|top picks|recommend)\b", msg_clean):
        scan = get_live_scan_results()
        
        local_response = "### 🔍 Market Scanner Findings\n\n"
        if scan and "overall_top_today" in scan:
            tops = scan.get("overall_top_today", [])
            local_response += f"**Latest Scan Run**: {scan.get('scan_time', 'Recently')}\n\n"
            local_response += "#### Today's Top Bullish Setup Recommendations:\n"
            local_response += "| Ticker | Display Name | Current Price | Trend Consensus | Confidence |\n| :--- | :--- | :--- | :--- | :--- |\n"
            
            for item in tops[:5]:
                ticker = item.get("ticker")
                friendly_name = TICKER_TO_NAME.get(ticker, item.get("name", ticker))
                curr_price = item.get("current_price", 0.0)
                conf = item.get("confidence_score", 0.0)
                direction = item.get("prediction_direction", "UP")
                
                currency = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^") else "$"
                
                local_response += f"| `{ticker}` | **{friendly_name}** | {format_val(curr_price, currency)} | {direction.upper()} | {conf:.1f}% |\n"
        else:
            local_response += "*No market scan data has been loaded or run yet. Go to the Scanner tab to run a live scan.*"
            
        context_data = {"scan": scan}

    # --- F. SET CASH BALANCE INTENT ---
    elif (set_cash_match := re.search(r"(?i)\b(?:set|change)\s+(?:my\s+)?(?:cash|balance)\s+(?:to\s+)?(?:rs\.?|rs|usd|\$|₹)?\s*(\d+(?:\.\d+)?)\b", msg_clean)):
        cash_val = float(set_cash_match.group(1))
        
        try:
            res = set_portfolio_cash(username, cash_val)
            action_executed = "set_cash"
            
            p_data = get_portfolio(username)
            currency = p_data["currency"]
            local_response = (
                f"### 💸 Capital Settings Updated!\n\n"
                f"I've successfully updated your virtual cash balance to **{format_val(cash_val, currency)}** in the database.\n"
                f"Your paper trading order pads and portfolios have been synchronized in real-time."
            )
            context_data = {"cash": cash_val, "portfolio": p_data}
        except Exception as e:
            local_response = f"Failed to update cash balance: {str(e)}"

    # --- G. RESET PORTFOLIO INTENT ---
    elif re.search(r"(?i)\b(reset|clear|wipe)\s+(?:my\s+)?(portfolio|account|history|trades)\b", msg_clean):
        try:
            res = reset_portfolio(username)
            action_executed = "reset_portfolio"
            
            p_data = get_portfolio(username)
            currency = p_data["currency"]
            local_response = (
                f"### 🔄 Account Reset Successfully!\n\n"
                f"I've deleted all trade logs, liquidations, and open positions, resetting your balance back to "
                f"**{format_val(p_data['cash_balance'], currency)}**.\n\n"
                f"Everything is squeaky clean and ready for your next trade!"
            )
            context_data = {"portfolio": p_data}
        except Exception as e:
            local_response = f"Failed to reset account: {str(e)}"

    # --- H. HELP / FAQ INTENT ---
    elif re.search(r"(?i)\b(help|how|who|faq|question|explain|guide|karen)\b", msg_clean) and not rag_docs:
        local_response = (
            f"### 🤖 TradeForecast AI Support (I'm Karen!)\n\n"
            f"I can help you navigate the platform, check indicators, analyze predictions, and place paper trades! "
            f"Here are some examples of what you can ask me to do:\n\n"
            f"* **Paper Trading Actions**:\n"
            f"  - `Buy 10 Reliance` (Fuzzy resolves names to tickers)\n"
            f"  - `Sell 5 AAPL` (Executes sell orders directly)\n"
            f"  - `Show my portfolio` (Outputs cash, open positions, unrealized profits)\n"
            f"  - `Set my cash to 500000` (Sets custom virtual capital limit)\n"
            f"  - `Reset my account` (Wipes history and positions)\n\n"
            f"* **Market Scans & AI Consensuses**:\n"
            f"  - `Show scan results` / `Bullish stocks` (Lists top setup recommendations)\n"
            f"  - `Predict TCS` / `Analyze Apple` (Queries 30-day forecast and confidence)\n\n"
            f"**FAQ**: *Is this real money?* No, all transactions use virtual capital for practice and research. *How are predictions calculated?* "
            f"Using a Ridge regression scale-fit pipeline with 10-day SMA, 50-day SMA, RSI, MACD, and lag metrics."
        )

    # --- I. NO SPECIFIC INTENT MATCHED (GENERAL CHAT) ---
    else:
        local_response = None

    # -------------------------------------------------------------------------
    # 2. RESPONSE GENERATION (HYBRID LLM OR PLATFORM CONTEXT FALLBACK)
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key:
        system_instruction = (
            "You are Karen, a smart, highly professional, and slightly sassy AI Trading Assistant and support system "
            "for the TradeForecast stock prediction platform. "
            "Your job is to answer user queries, guide them on trading, analyze data, and assist them. "
            "Always output response using clean, professional GitHub-flavored markdown. Use bullet points and tables when displaying data. "
            "Keep answers concise. Refer to the active user session and context parameters below when formatting answers.\n\n"
            f"Active User: {username}\n"
            f"Current Ticker Context: {active_ticker or 'None'}\n"
            f"Database Context JSON: {json.dumps(context_data)}"
            f"{rag_context}"
        )
        
        hist_str = ""
        for h in history[-10:]:
            role = "User" if h.get("role") == "user" else "Karen"
            hist_str += f"{role}: {h.get('content')}\n"
            
        prompt = f"{hist_str}User: {msg_clean}\nKaren:"
        
        gemini_response = call_gemini_api(api_key, prompt, system_instruction)
        
        if gemini_response and not gemini_response.startswith("*(Gemini API connection error)*"):
            return {
                "response": gemini_response,
                "action_executed": action_executed,
                "ticker_affected": ticker_affected,
                "system_status": "llm_generated"
            }
            
    # Fallback to RAG document if local_response is None and RAG matches
    if not local_response and rag_docs:
        doc = rag_docs[0]
        local_response = (
            f"### 📖 TradeForecast Documentation: **{doc['title']}**\n\n"
            f"{doc['content']}\n\n"
            f"*Hope this helps! Let me know if you have other questions about our platform.*"
        )

    if local_response:
        return {
            "response": local_response,
            "action_executed": action_executed,
            "ticker_affected": ticker_affected,
            "system_status": "rule_generated"
        }
        
    default_text = (
        f"Hi! I'm Karen, your AI trading assistant. I didn't quite capture a trading command. "
        f"I can buy/sell stocks, analyze portfolios, show top scanner recommendations, and fetch predictions. "
        f"Try asking:\n\n"
        f"- *'Buy 10 shares of Reliance'* or *'Sell 5 AAPL'*\n"
        f"- *'Show my portfolio'* / *'Reset my account'*\n"
        f"- *'What is the prediction for TCS?'* / *'Bullish stocks'* / *'Help'*"
    )
    return {
        "response": default_text,
        "system_status": "default_fallback"
    }
