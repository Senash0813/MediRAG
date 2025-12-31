from __future__ import annotations

import logging

from fastapi import FastAPI

from medirag.api.schemas import QueryRequest, QueryResponse
from medirag.config import load_settings
from medirag.pipeline.orchestrator import init_assets, run_pipeline

app = FastAPI(title="MediRAG Backend")

logger = logging.getLogger(__name__)

_assets = None


@app.on_event("startup")
def _startup():
    global _assets
    settings = load_settings()
    _assets = init_assets(settings)


@app.post("/query", response_model=QueryResponse)
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
