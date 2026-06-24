# Implementation Plan — Connecting Backend to Agentic Alpha Dashboard

Integrate the newly verified Stock Predictor ML backend with the Stitch mobile dashboard screen: **AI Trading Terminal - Dynamic Currency** (`projects/3208942245280913659/screens/df73c264f1154d2a9b90403a41499e51`).

---

## Proposed Changes

### Stitch Screen Integration

#### [MODIFY] [Stitch Screen: AI Trading Terminal - Dynamic Currency](projects/3208942245280913659/screens/df73c264f1154d2a9b90403a41499e51)

1. **TopAppBar Search Input**:
   - Replace the static "Search" icon and "Terminal" heading combination with an active input field for stock tickers.
   - Text input ID: `#ticker-input`.
   - Default value: `"AAPL"`.
   - Pressing Enter or changing the text triggers a prediction analysis from the backend.

2. **Interactive Timeframe Selection**:
   - Re-label chart timeframe indicators to match backend capabilities: `5min`, `15min`, `30min`, `1D`, `1M`.
   - Give them active states (class styling) and make them clickable to query the API with different intervals.

3. **Global Loading Overlay**:
   - Bind the "Run Agent Analysis" button (`#run-analysis-btn`) and text/timeframe changes to show the `#global-loader` overlay.
   - Temporarily blur screen and show skeleton loaders during API fetch.

4. **API Integration Logic**:
   - Implement `fetchPrediction(ticker, timeframe)` pointing to:
     `http://localhost:8000/api/predict/${ticker}?timeframe=${timeframe}`.
   - Fall back graceful handling for errors: If the ticker is invalid or backend is down, output error messages directly to the ReAct Terminal and hide the loader.

5. **Dynamic Metrics Updates**:
   - **Price Card**: Dynamically render price with appropriate currency formatting (e.g. `₹ 2,450.00` for Indian tickers, `$ 180.20` for US markets) using `currency_symbol` from the API.
   - **Agent Strategy**: Update strategy banner (`#agent-strategy`) to match `prediction_direction` (Bullish / Bearish / Neutral) and apply dynamic Tailwind classes:
     - `Bullish` → Green text (`text-secondary`), subtext: `"Strong Buy Sentiment Detected"`.
     - `Bearish` → Red text (`text-error`), subtext: `"Short/Sell Signal Triggered"`.
     - `Neutral` → Gray text (`text-on-surface-variant`), subtext: `"Hold/Consolidation Phase"`.
   - **Confidence Score**: Update text (`#confidence-percentage`) and animate the SVG radial gauge path (`#confidence-ring`) by adjusting its `strokeDashoffset`:
     `213.6 * (1 - confidence_score / 100)`.
   - **Sentiment Gauge**: Rotate the needle (`#sentiment-needle`) dynamically using CSS transforms:
     `rotate(${ (direction === 'Bullish' ? 1 : -1) * (confidence_score / 100) * 80 }deg)`.

6. **ReAct Terminal Logs**:
   - Parse `agent_logs` from the API.
   - Clear existing mock logs and inject them line-by-line using a staggered interval (`setTimeout` of ~600ms per line) to simulate a live thinking agent.
   - Apply semantic colors (italics for thought, primary blue for action, yellow/orange for observations, green/red for consensus).

7. **Technical Indicators Table**:
   - Overwrite table body (`#indicators-tbody`) with values from `technical_indicators`:
     - **RSI-14**: Displays raw value; lists signal as `"Overbought"` (>70), `"Oversold"` (<30), or `"Neutral"`.
     - **MACD**: Displays MACD & Signal values; lists signal as `"Bullish Divergence"` (if MACD > Signal) or `"Bearish Divergence"`.
     - **SMA Cross**: Displays SMA-10 vs SMA-50; lists signal as `"Golden Cross"` (SMA-10 > SMA-50) or `"Death Cross"`.

8. **Market Intelligence Feed**:
   - Clear the hardcoded items in `#news-feed-list`.
   - Inject the news headlines returned from the API dynamically.

9. **SVG Trajectory Chart**:
   - Replace the static double curves with dynamic curves.
   - Calculate min/max values from the `historical_vs_predicted` array.
   - Map elements over an `800x200` coordinate space.
   - Draw two distinct paths:
     1. **Historical Prices** (`#price-chart-actual`): Semi-transparent grey/blue.
     2. **Predicted Prices** (`#price-chart-predicted`): Vibrant electric blue.
   - Animate the paths on load.
   - Set the chart ceiling/floor labels (`#chart-y-top` and `#chart-y-bottom`) to the exact min/max prices formatted with the currency symbol.

---

## Verification Plan

### Automated/Local Sandbox Testing
1. Run the FastAPI backend server on port 8000:
   `python -m uvicorn main:app --reload --port 8000`.
2. Inspect the modified Stitch screen mockup.
3. Perform the following tests on the UI:
   - Search for **US Ticker** (e.g. `AAPL`, `MSFT`) → Verify prices load in `$`, chart renders actual vs predicted, terminal displays live steps.
   - Search for **Indian Ticker** (e.g. `RELIANCE.NS`, `TCS.NS`) → Verify prices load in `₹`, technical metrics render, news feed updates.
   - Cycle through timeframes (`5min`, `15min`, `1M`) → Verify that the interval query parameter updates correctly.
   - Search for an invalid ticker → Verify that the search fails gracefully, showing error logs in the ReAct terminal without crashing.
