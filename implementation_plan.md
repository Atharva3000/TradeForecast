# AI Stock Prediction Engine — Backend API

Build a production-ready, asynchronous FastAPI backend that provides stock prediction capabilities via yFinance data ingestion, Scikit-Learn model inference, and structured JSON responses tailored for a Recharts-powered frontend.

---

## Proposed Directory Structure

```
Stock Predictor ML/
├── main.py                      # FastAPI app init, CORS, global middleware
├── requirements.txt             # Pinned dependencies
├── routes/
│   └── stock.py                 # GET /api/predict/{ticker} endpoint
└── services/
    ├── data_fetcher.py          # yFinance data download + mock news aggregator
    └── predictor.py             # Feature engineering, technicals, Linear Regression
```

---

## Proposed Changes

### Core Application

#### [NEW] [main.py](file:///c:/Users/athar/OneDrive/Desktop/Stock%20Predictor%20ML/main.py)

- Create `FastAPI` app instance with metadata (title, version, description).
- Configure `CORSMiddleware` allowing **all origins** (`["*"]`), all methods, all headers — for frontend integration.
- Add a request-timing middleware that logs request durations.
- Mount the stock router under the `/api` prefix.
- Include a root `/` health-check endpoint returning `{ "status": "ok" }`.

---

### Data Engine

#### [NEW] [services/data_fetcher.py](file:///c:/Users/athar/OneDrive/Desktop/Stock%20Predictor%20ML/services/data_fetcher.py)

**`async fetch_stock_data(ticker: str, start_date: str, end_date: str) → pd.DataFrame`**
- Use `asyncio.to_thread()` to offload the blocking `yfinance.download()` call.
- Map the `timeframe` query param to calendar date ranges:

  | Timeframe | Meaning | Calendar Range |
  |-----------|---------|----------------|
  | `1D`      | 1 Day   | Last 30 trading days (need enough data for indicators) |
  | `1W`      | 1 Week  | Last 90 days |
  | `1M`      | 1 Month | Last 365 days |

- Validate the returned DataFrame is non-empty; raise `HTTPException(404)` with a descriptive message on invalid tickers.
- Catch `Exception` broadly as a fallback for rate-limit or network errors, logging the traceback and raising `HTTPException(503)`.

**`async fetch_news_headlines(ticker: str) → list[str]`**
- Return 3 mock but contextually-relevant headlines templated with the ticker symbol and current date.
- Example: `"Analysts upgrade {ticker} on strong Q2 earnings momentum"`.

---

### Engineering & Math Layer

#### [NEW] [services/predictor.py](file:///c:/Users/athar/OneDrive/Desktop/Stock%20Predictor%20ML/services/predictor.py)

**Index Cleaning**
- Flatten any `MultiIndex` columns returned by yFinance (common when downloading single tickers with recent yfinance versions).
- Reset and normalize column names to lowercase: `open, high, low, close, volume`.

**Technical Indicators**
- `compute_sma(series, window)` → Simple Moving Average.
- `compute_rsi(series, period=14)` → RSI using the standard average-gain / average-loss formula.
- `compute_macd(series)` → MACD line (EMA12 − EMA26) and signal line (EMA9 of MACD).

**Prediction Pipeline — `run_prediction(df: pd.DataFrame) → dict`**
1. Engineer features from the cleaned DataFrame:
   - `sma_10`, `sma_50`, `rsi_14`, `macd`, `macd_signal`
   - Lagged close prices: `close_lag_1`, `close_lag_2`, `close_lag_3`
2. Drop NaN rows created by rolling windows / lags.
3. **Target**: next-day close price (`close` shifted by −1).
4. **Split**: 80% train / 20% test (chronological, no shuffle).
5. **Model**: `sklearn.linear_model.LinearRegression` — fit on train, predict on test.
6. Build the `historical_vs_predicted` array: `[{ date, actual, predicted }]` — covering the test set.
7. Determine `prediction_direction`:
   - Compare the last predicted value to the current close.
   - Bullish if predicted > current × 1.005, Bearish if < current × 0.995, else Neutral.

**Confidence Score — `compute_confidence(df, rsi, sma_10, sma_50, macd, macd_signal) → int`**
- Heuristic 0–100 score derived from alignment of indicators:
  - RSI in neutral zone (40–60) → +20 pts
  - Price above SMA-50 → +20 pts
  - SMA-10 > SMA-50 (golden cross zone) → +20 pts
  - MACD above signal line → +20 pts
  - Low recent volatility (std of last 10 closes < 2% of mean) → +20 pts
- Clamp result to `[0, 100]`.

---

### REST Endpoints

#### [NEW] [routes/stock.py](file:///c:/Users/athar/OneDrive/Desktop/Stock%20Predictor%20ML/routes/stock.py)

**`GET /api/predict/{ticker}?timeframe=1M`**

| Parameter   | Type   | Default | Validation |
|-------------|--------|---------|------------|
| `ticker`    | path   | —       | Required, uppercased |
| `timeframe` | query  | `"1M"`  | Must be one of `1D`, `1W`, `1M` |

**Response schema** (200 OK):
```json
{
  "current_price": 182.45,
  "prediction_direction": "Bullish",
  "confidence_score": 72,
  "technical_indicators": {
    "rsi_14": 55.3,
    "sma_10": 180.2,
    "sma_50": 175.8,
    "macd": 1.45,
    "macd_signal": 1.12
  },
  "historical_vs_predicted": [
    { "date": "2026-05-20", "actual": 181.0, "predicted": 180.5 },
    ...
  ],
  "agent_logs": [
    "[THOUGHT] Analyzing RSI trend for AAPL...",
    "[ACTION] Pulling 365 days of historical data...",
    "[ACTION] Computing SMA-10, SMA-50, RSI-14, MACD...",
    "[ACTION] Engineering 3-day lag features...",
    "[ACTION] Training LinearRegression on 80% split...",
    "[OBSERVATION] Model R² on test set: 0.94",
    "[OBSERVATION] RSI at 55.3 — neutral zone",
    "[OBSERVATION] SMA-10 > SMA-50 — bullish crossover",
    "[THOUGHT] Technical indicators align with upward momentum.",
    "[RESULT] Prediction: Bullish with 72% confidence."
  ]
}
```

Error responses: `404` (invalid ticker), `422` (bad timeframe), `503` (upstream failure).

---

### Dependencies

#### [NEW] [requirements.txt](file:///c:/Users/athar/OneDrive/Desktop/Stock%20Predictor%20ML/requirements.txt)

```
fastapi
uvicorn[standard]
yfinance
scikit-learn
pandas
numpy
```

---

## Verification Plan

### Automated Tests
1. Install dependencies: `pip install -r requirements.txt`
2. Start server: `uvicorn main:app --reload`
3. Hit `GET /api/predict/AAPL?timeframe=1M` and validate:
   - HTTP 200 response
   - All required keys present in JSON
   - `historical_vs_predicted` array is non-empty
   - `confidence_score` is an integer in `[0, 100]`
4. Hit `GET /api/predict/INVALIDTICKER123` → expect 404
5. Capture and report any server error logs.

### Manual Verification
- User can test with different tickers (`MSFT`, `GOOGL`, `TSLA`) and timeframes (`1D`, `1W`, `1M`).
