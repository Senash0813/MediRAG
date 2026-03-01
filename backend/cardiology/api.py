from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
from query_classifier import classify_query
from main import generate_final_answer, l2_normalize

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


class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]
    query_intent: str
    confidence: float
    is_cardiology: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query2", response_model=QueryResponse)
def run_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    vectorstore = load_vectorstore()

    # ==========================================
    # STEP 1: CLASSIFY QUERY INTENT
    # ==========================================
    intent, confidence, metadata = classify_query(req.query, vectorstore=vectorstore)
    is_cardiology = intent == "CARDIOLOGY"

    # ==========================================
    # HANDLE NON-CARDIOLOGY QUERIES
    # ==========================================
    if not is_cardiology:
        answer = generate_final_answer(
            req.query,
            [],
            intent=intent,
            confidence=confidence
        )
        
        return QueryResponse(
            answer=answer,
            retrieved_docs=[],
            query_intent=intent,
            confidence=confidence,
            is_cardiology=False
        )

    # ==========================================
    # CARDIOLOGY PIPELINE (WITH HYDE)
    # ==========================================

    # Step 1: Embedding + projection
    proj_emb = embed_and_project(req.query)
    proj_emb = l2_normalize(np.array(proj_emb))

    # Step 2: HyDE Generation (only for cardiology)
    hyde_emb, hyde_docs = generate_hypothetical_docs(req.query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    # Step 3: Fusion
    final_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=req.alpha)
    final_emb = l2_normalize(np.array(final_emb))

    # Step 4: Retrieval
    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding=final_emb.tolist(),
            k=req.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieval error: {e}")

    if not docs_and_scores:
        return QueryResponse(
            answer="No relevant documents found in the cardiology database.",
            retrieved_docs=[],
            query_intent=intent,
            confidence=confidence,
            is_cardiology=True
        )

    retrieved_docs_text = [doc.page_content for doc, _ in docs_and_scores]
    docs_only = [doc for doc, _ in docs_and_scores]

    # Step 5: Final answer
    answer = generate_final_answer(
        req.query,
        docs_only,
        intent=intent,
        confidence=confidence
    )

    return QueryResponse(
        answer=answer,
        retrieved_docs=retrieved_docs_text,
        query_intent=intent,
        confidence=confidence,
        is_cardiology=True
    )
