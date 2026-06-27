import re
import math


# Common English stopwords to clean up search queries
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll",
    "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

# The internal TradeForecast platform documentation knowledge base
DOCUMENTS = [
    {
        "id": "ml_predictions",
        "title": "Machine Learning & Price Predictions",
        "content": (
            "TradeForecast uses a quantitative ML pipeline to generate 30-day stock price predictions. "
            "The primary model is a Ridge Regression estimator. To prevent high-value indices (such as Nifty 50 or Sensex) "
            "from experiencing extreme predicted swings or negative price anomalies, all technical indicators (including "
            "sma_10, sma_50, rsi_14, macd, macd_signal, and historical lag values) are scaled using a scikit-learn StandardScaler. "
            "Additionally, a safety circuit breaker clips the forecast price boundary: predicted targets are restricted to "
            "remain strictly between 50% and 150% of the current stock price, guaranteeing that forecasts are realistic "
            "and never result in negative numbers."
        ),
        "keywords": ["scaler", "ridge", "regression", "standardscaler", "negative", "forecast", "prediction", "circuit", "breaker", "clipping", "swing"]
    },
    {
        "id": "paper_trading_rules",
        "title": "Paper Trading Simulator Rules & Executions",
        "content": (
            "The Paper Trading simulator lets users trade stocks and indices dynamically using virtual cash. "
            "All market orders execute at the asset's active real-time market price. To support precise liquidation "
            "and prevent precision mismatch errors (e.g. from float rounding issues), the simulator uses a floating-point "
            "tolerance check (current_qty - quantity < -1e-5) when liquidating positions. This ensures that users can "
            "sell their exact share holdings without getting rejected. Subtraction bounds are clipped at max(0.0, current_qty - quantity) "
            "to prevent negative holdings. Indian indices and assets default to Indian Rupee (₹) currency mapping and lakh/crore "
            "financial digit formatting, whereas US assets default to US Dollar ($) and millions formatting."
        ),
        "keywords": ["paper", "trading", "sell", "buy", "order", "liquidation", "float", "precision", "holding", "rupee", "dollar", "currency", "lakh"]
    },
    {
        "id": "database_persistence",
        "title": "Database Persistence & Vercel Sync Recovery",
        "content": (
            "TradeForecast stores all user accounts, profile details, and paper trading histories in a SQLite database. "
            "For containerized deployments (Docker), if the directory '/app/data' exists, the DB path is automatically routed to "
            "'/app/data/users.db' to guarantee persistent volume mounts. In serverless/stateless environments like Vercel (where "
            "SQLite cold restarts wipe database writes), a synchronization safeguard is implemented. Every successful trade, "
            "portfolio update, and cash setting is backed up in the user's browser localStorage. On session startup or tab load, "
            "the frontend syncs this backup to the backend via '/api/paper/sync', restoring the SQLite database balance and history "
            "records instantly. Default tester credentials (username: tester, password: testpassword123) are automatically seeded on db init."
        ),
        "keywords": ["database", "sqlite", "users.db", "persistence", "docker", "volume", "sync", "syncing", "restoration", "vercel", "tester", "credentials"]
    },
    {
        "id": "tradingview_widget",
        "title": "TradingView Interactive Widget & Charts",
        "content": (
            "The Stock Predictions panel provides a dual-chart mode: the Lightweight Charts tab showing AI consensus targets, "
            "and the Interactive Tool tab rendering the official TradingView widget. Because the free TradingView iframe "
            "blocks or delays live NSE/BSE Indian equities on third-party sites due to licensing regulations, TradeForecast "
            "displays a notice banner alongside Indian tickers, offering an external link directly to the official TradingView symbol site. "
            "Tickers are translated dynamically from Yahoo Finance to TradingView format: '^NSEI' translates to 'NSE:NIFTY', "
            "'^BSESN' to 'BSE:SENSEX', and '.NS' suffixes resolve to 'NSE:'. Toggling light/dark themes re-renders the widget with "
            "the appropriate palette styling."
        ),
        "keywords": ["tradingview", "widget", "chart", "iframe", "embed", "nse", "bse", "indian", "translation", "nifty", "sensex", "theme"]
    },
    {
        "id": "authentication_persistence",
        "title": "User Authentication & Remember Me Persistence",
        "content": (
            "TradeForecast partitions user session authentication based on the 'Remember me' checkbox state. "
            "If the user logs in or registers with 'Remember me' checked, their authentication credentials and active "
            "session token are saved in localStorage, persisting indefinitely across browser exits and window restarts. "
            "If 'Remember me' is unchecked, session details are stored in sessionStorage, which is ephemeral and is "
            "wiped automatically as soon as the user closes the browser tab, protecting account security on public devices."
        ),
        "keywords": ["authentication", "auth", "login", "remember", "session", "localstorage", "sessionstorage", "persistence"]
    },
    {
        "id": "low_latency_pricing",
        "title": "Low-Latency Live Stock Pricing",
        "content": (
            "To support rapid trade execution calculations on the dashboard, TradeForecast uses a low-latency price "
            "tracking router. Price requests call Yahoo Finance's 'fast_info' property, returning live trading quotes "
            "in less than 50ms (a 95% latency reduction compared to standard multi-day history downloads). In case of "
            "exchange feed outages, the system automatically falls back to parsing a standard 5-day daily Close dataframe, "
            "ensuring trade systems remain operational at all times."
        ),
        "keywords": ["pricing", "price", "fast_info", "latency", "low-latency", "quote", "yfinance", "fallback", "quote", "outage"]
    },
    {
        "id": "technical_indicators",
        "title": "Technical Indicators & Consensus Strategy",
        "content": (
            "TradeForecast calculates several momentum, trend, and support indicators to form a trade consensus. "
            "Moving Average Convergence Divergence (MACD) uses a fast period of 12, slow period of 26, and signal smoothing of 9. "
            "Relative Strength Index (RSI) is calculated over a 14-day lookback window, flagging assets as overbought above 70 "
            "and oversold below 30. Simple Moving Averages (SMA) are computed for 10-day (short-term) and 50-day (long-term) periods, "
            "where an upward cross of the 10-day over the 50-day represents a bullish golden cross, and a downward cross represents "
            "a bearish death cross. Support/Resistance pivots use standard High-Low-Close math."
        ),
        "keywords": ["macd", "rsi", "sma", "moving", "average", "golden", "cross", "indicator", "technical", "momentum", "overbought", "oversold"]
    },
    {
        "id": "account_controls",
        "title": "User Account & Portfolio Controls",
        "content": (
            "Users can manage their profile and virtual funds directly via the Settings panel and chat assistant. "
            "The 'Default Investment Capital' field in Settings sets the starting paper cash balance. In chat, sending a command "
            "like 'Set my cash to 100000' will immediately update the SQLite paper_portfolio balance table for the active user. "
            "To wipe all positions and start fresh, the 'Reset Account' button triggers a complete database purge of active positions "
            "and trade order history records for the user's account, restoring the default capital value."
        ),
        "keywords": ["capital", "reset", "cash", "balance", "virtual", "settings", "purge", "funds", "balance", "profile"]
    },
    {
        "id": "backtester_math",
        "title": "Backtesting Metrics & Simulation Math",
        "content": (
            "The TradeForecast backtester runs historical close data chronologically to evaluate AI trading strategies. "
            "It computes key portfolio performance metrics: 1. Win Rate, calculated as the percentage of closed trades resulting in "
            "a positive net return. 2. Maximum Drawdown, measuring the largest peak-to-trough percentage decline in portfolio value. "
            "3. Sharpe Ratio, which calculates the excess return per unit of volatility (assuming a risk-free rate of 0% for simulation). "
            "All portfolio metrics update in real-time as new virtual positions are closed."
        ),
        "keywords": ["backtest", "backtester", "sharpe", "drawdown", "win", "rate", "simulation", "performance", "return", "volatility"]
    }
]

def tokenize(text: str) -> list[str]:
    """Clean, lowercase, and tokenize a string, removing english stopwords."""
    text = text.lower()
    # Strip punctuation and keep alpha-numeric terms
    text = re.sub(r"[^a-z0-9\s\-_^]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

def retrieve_context(query: str, limit: int = 2) -> list[dict]:
    """
    Ranks documents from our knowledge base matching words in query.
    Returns the top most relevant passages.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
        
    scores = []
    for doc in DOCUMENTS:
        # Combine title, content, and keywords for a rich search pool
        doc_text = f"{doc['title']} {doc['content']} {' '.join(doc['keywords'])}"
        doc_tokens = tokenize(doc_text)
        if not doc_tokens:
            continue
            
        # Calculate term frequencies inside doc
        tf_doc = {}
        for token in doc_tokens:
            tf_doc[token] = tf_doc.get(token, 0) + 1
            
        # Score terms overlap
        score = 0.0
        title_tokens = tokenize(doc["title"])
        
        for token in query_tokens:
            # Term matches inside title get a significant boost
            weight = 3.0 if token in title_tokens else 1.0
            
            # TF score
            if token in tf_doc:
                score += tf_doc[token] * weight
                
        # Length normalization
        norm_score = score / math.sqrt(len(doc_tokens)) if score > 0 else 0.0
        scores.append((norm_score, doc))
        
    # Sort by descending score
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return top limit documents with score > 0
    return [item[1] for item in scores if item[0] > 0][:limit]
