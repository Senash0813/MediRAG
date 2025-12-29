from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
from main import generate_final_answer, l2_normalize

app = FastAPI(title="MediRAG Cardiology API")

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # Embedding + projection
    proj_emb = embed_and_project(req.query)
    proj_emb = l2_normalize(np.array(proj_emb))

    # HyDE
    hyde_emb, hyde_docs = generate_hypothetical_docs(req.query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    # Fusion
    final_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=req.alpha)
    final_emb = l2_normalize(np.array(final_emb))

    # Retrieval
    vectorstore = load_vectorstore()
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

    return QueryResponse(answer=answer, retrieved_docs=retrieved_docs_text)
