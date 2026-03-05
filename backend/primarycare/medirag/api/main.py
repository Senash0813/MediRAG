from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medirag.api.schemas import QueryRequest, QueryResponse
from medirag.config import load_settings
from medirag.pipeline.orchestrator import init_assets, run_pipeline

app = FastAPI(title="MediRAG Backend")

# CORS: required for browser-based clients (e.g., Next.js dev server) to call this API.
# Postman doesn't enforce CORS, but browsers do.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # If you access the Next dev server via the LAN, you can add it here later.
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

_assets = None


@app.on_event("startup")
def _startup():
    global _assets
    settings = load_settings()
    _assets = init_assets(settings)


@app.post("/query4", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = run_pipeline(assets=_assets, query=req.query, k=req.top_k)
        return QueryResponse(**result)
    except Exception:
        # Option A: never crash the request on upstream/client issues.
        # We still log details server-side to debug the real root cause.
        logger.exception("/query failed")
        return QueryResponse(
            query=req.query,
            verified_docs=[],
            instruction_obj={},
            final_prompt="",
            answer_text=(
                'The problem is out of scope right now (or the LLM request was rejected). '
                'Please try again, or select a different question.'
            ),
        )
