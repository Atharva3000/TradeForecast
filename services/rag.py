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
