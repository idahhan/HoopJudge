"""Ball possession analysis FastAPI application.

Run with::

    uvicorn app.main:app --reload --port 8000

Or from the repo root::

    python -m uvicorn app.main:app --reload
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ball import router as ball_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

app = FastAPI(
    title="Ball Possession API",
    description=(
        "Phase 4 of the myogait basketball analysis pipeline. "
        "Detects ball location and classifies possession state per frame "
        "using geometric proximity logic."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ball_router, prefix="/api")


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "ball-possession-api", "version": "0.1.0"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
