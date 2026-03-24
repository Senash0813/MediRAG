from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import get_settings
from .state import build_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.rag_state = build_state(settings, force_reindex=False)
    yield


app = FastAPI(title="Internal Medicine RAG API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware to allow requests from the demo HTML page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo purposes
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

app.include_router(router)
