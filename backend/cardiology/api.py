from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os

from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
from main import generate_final_answer, generate_out_of_domain_answer, l2_normalize

app = FastAPI(title="MediRAG Cardiology API")


@app.on_event("startup")
def _warm_vectorstore() -> None:
    # Load once per process; subsequent calls are served from cache.
    # This avoids re-loading large models/indexes on every request.
    load_vectorstore()

# Allow browser requests from the frontend (development origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    alpha: Optional[float] = 0.5
    # If provided, overrides the default in-domain threshold.
    # NOTE: For LangChain FAISS, the returned `score` is typically an L2 distance
    # (smaller is more similar). Tune this value on your data.
    domain_max_distance: Optional[float] = None
    # Text gate uses the embedding model attached to the FAISS vectorstore
    # (Instructor-large during indexing) for best consistency.
    domain_max_distance_text: Optional[float] = None
    debug: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]
    # Optional debug fields (returned only if req.debug=True)
    domain_in_domain: Optional[bool] = None
    domain_best_score: Optional[float] = None
    domain_max_distance: Optional[float] = None
    domain_best_score_text: Optional[float] = None
    domain_max_distance_text: Optional[float] = None


def _default_domain_max_distance() -> float:
    # Keep this lenient by default to avoid false out-of-domain rejections.
    # You can tune it via env var without code changes.
    return float(os.getenv("CARDIOLOGY_DOMAIN_MAX_DISTANCE", "0.22"))


def _default_domain_max_distance_text() -> float:
    # Default tuned independently because raw Instructor embeddings (used in the index)
    # may have a different distance scale than projected InBEDDER vectors.
    return float(os.getenv("CARDIOLOGY_DOMAIN_MAX_DISTANCE_TEXT", "0.22"))


def _domain_gate_text(vectorstore, query: str, max_distance: float) -> tuple[bool, Optional[float]]:
    """Return (in_domain, best_score) using the vectorstore's embedding model.

    This is the most consistent gate because it uses the same embedding model
    that was used to build the FAISS index.
    """
    try:
        res = vectorstore.similarity_search_with_score(query=query, k=1)
    except Exception:
        return True, None

    if not res:
        return False, None

    best_score = res[0][1]
    try:
        best_score_f = float(best_score)
    except Exception:
        return True, None

    return (best_score_f <= max_distance), best_score_f


def _domain_gate(vectorstore, query_embedding: np.ndarray, max_distance: float) -> tuple[bool, Optional[float]]:
    """Return (in_domain, best_score).

    Uses a fast FAISS top-1 lookup against the *entire* cardiology KB.
    """
    try:
        res = vectorstore.similarity_search_with_score_by_vector(
            embedding=query_embedding.tolist(),
            k=1,
        )
    except Exception:
        # If the gate fails, fall back to running the pipeline rather than blocking users.
        return True, None

    if not res:
        return False, None

    best_score = res[0][1]
    try:
        best_score_f = float(best_score)
    except Exception:
        return True, None

    return (best_score_f <= max_distance), best_score_f


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query2", response_model=QueryResponse)
def run_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # Embedding + projection (your InBEDDER-based semantic representation)
    proj_emb = embed_and_project(req.query)
    proj_emb = l2_normalize(np.array(proj_emb))

    # Semantic in-domain gate: if query doesn't match cardiology KB,
    # skip the expensive HyDE+fusion+retrieval pipeline.
    vectorstore = load_vectorstore()

    # Primary gate: use the index's native embedding model (Instructor) for consistency.
    max_distance_text = (
        req.domain_max_distance_text
        if req.domain_max_distance_text is not None
        else _default_domain_max_distance_text()
    )
    in_domain_text, best_score_text = _domain_gate_text(vectorstore, req.query, max_distance=max_distance_text)

    # Secondary gate (optional fallback): projected InBEDDER vector.
    max_distance = req.domain_max_distance if req.domain_max_distance is not None else _default_domain_max_distance()
    in_domain_vec, best_score = _domain_gate(vectorstore, proj_emb, max_distance=max_distance)

    # Decision: trust the text gate when it produced a score; otherwise fall back to vector gate.
    if best_score_text is not None:
        in_domain = in_domain_text
    else:
        in_domain = in_domain_vec

    if not in_domain:
        answer = generate_out_of_domain_answer(req.query)
        if req.debug:
            return QueryResponse(
                answer=answer,
                retrieved_docs=[],
                domain_in_domain=in_domain,
                domain_best_score=best_score,
                domain_max_distance=max_distance,
                domain_best_score_text=best_score_text,
                domain_max_distance_text=max_distance_text,
            )
        return QueryResponse(answer=answer, retrieved_docs=[])

    # HyDE
    hyde_emb, hyde_docs = generate_hypothetical_docs(req.query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    # Fusion
    final_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=req.alpha)
    final_emb = l2_normalize(np.array(final_emb))

    # Retrieval
    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding=final_emb.tolist(),
            k=req.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieval error: {e}")

    if not docs_and_scores:
        return QueryResponse(answer="No relevant documents found.", retrieved_docs=[])

    retrieved_docs_text = [doc.page_content for doc, _ in docs_and_scores]
    docs_only = [doc for doc, _ in docs_and_scores]

    # Final answer
    answer = generate_final_answer(req.query, docs_only)

    if req.debug:
        return QueryResponse(
            answer=answer,
            retrieved_docs=retrieved_docs_text,
            domain_in_domain=in_domain,
            domain_best_score=best_score,
            domain_max_distance=max_distance,
            domain_best_score_text=best_score_text,
            domain_max_distance_text=max_distance_text,
        )

    return QueryResponse(answer=answer, retrieved_docs=retrieved_docs_text)
