"""
AI Stock Prediction Engine — FastAPI Application Entry Point.

Configures:
- CORS middleware allowing all origins for frontend integration
- Request-duration logging middleware
- Health-check root endpoint
- Stock prediction router mounted at top level
"""

import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from routes.stock import router as stock_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TradeForecast Engine",
    version="1.0.0",
    description=(
        "Asynchronous prediction API powered by real-time quantitative data streams "
        "and multi-indicator consensus systems. Supports US and Indian markets."
    ),
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for frontend integration
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s completed in %.1f ms (status %d)",
        request.method,
        request.url.path,
        elapsed_ms,
        response.status_code,
    )
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(stock_router)


@app.on_event("startup")
async def startup_event():
    import asyncio
    from services.db import init_db
    from routes.stock import preload_sector_predictions
    
    # Initialize the database (SQLite)
    init_db()
    
    # Preload watchlist prediction data
    asyncio.create_task(preload_sector_predictions())



# ---------------------------------------------------------------------------
# Frontend Dashboard Delivery & Health Check
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "engine": "TradeForecast Engine v1.0.0"}
