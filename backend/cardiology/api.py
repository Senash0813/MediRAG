from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import os
import time

from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
from main import (
    generate_final_answer,
    generate_out_of_domain_answer,
    generate_general_intent_answer,
    generate_greeting_answer,
    detect_general_intent,
    is_greeting_intent,
    l2_normalize,
    _get_phi_pipeline,   # IMPORTANT: used to download/load Phi during startup
    MODEL_NAME,
)

# ==================================================
# APP INITIALIZATION
# ==================================================

app = FastAPI(title="MediRAG Cardiology API")

MODEL_READY = False
VECTORSTORE_READY = False
STARTUP_ERROR: Optional[str] = None


@app.on_event("startup")
def startup_event() -> None:
    """
    Warm up expensive resources when RunPod starts.

    This prevents the first frontend request from downloading/loading
    the HuggingFace Phi model and timing out behind Cloudflare.
    """
    global MODEL_READY, VECTORSTORE_READY, STARTUP_ERROR

    print("\n========================================")
    print("🚀 Starting MediRAG Cardiology API")
    print("========================================")

    try:
        start = time.time()

        print("[Startup] Loading FAISS vectorstore...")
        load_vectorstore()
        VECTORSTORE_READY = True
        print("[Startup] ✅ Vectorstore loaded")

        print(f"[Startup] Downloading/loading HuggingFace model: {MODEL_NAME}")
        _get_phi_pipeline()
        MODEL_READY = True
        print("[Startup] ✅ Phi model loaded")

        elapsed = time.time() - start
        print(f"[Startup] ✅ Backend ready in {elapsed:.2f} seconds")
        print("========================================\n")

    except Exception as e:
        STARTUP_ERROR = str(e)
        MODEL_READY = False
        print("[Startup] ❌ Startup failed:", e)
        print("========================================\n")


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# REQUEST / RESPONSE MODELS
# ==================================================

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    alpha: Optional[float] = 0.5

    # For domain filtering. LangChain FAISS score is usually L2 distance:
    # smaller score = more similar.
    domain_max_distance: Optional[float] = None
    domain_max_distance_text: Optional[float] = None

    debug: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]

    # Optional debug fields
    domain_in_domain: Optional[bool] = None
    domain_best_score: Optional[float] = None
    domain_max_distance: Optional[float] = None
    domain_best_score_text: Optional[float] = None
    domain_max_distance_text: Optional[float] = None


class StageOutput(BaseModel):
    stage_name: str
    stage_number: int
    description: str
    data: Dict[str, Any]


class StagedQueryResponse(BaseModel):
    query: str
    stages: List[StageOutput]
    final_answer: str
    is_in_domain: bool


# ==================================================
# BASIC ENDPOINTS
# ==================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "MediRAG Cardiology API is running",
        "model": MODEL_NAME,
        "model_ready": MODEL_READY,
        "vectorstore_ready": VECTORSTORE_READY,
        "endpoints": ["/health", "/query2", "/query_stages", "/docs"],
    }


@app.get("/health")
def health():
    if STARTUP_ERROR:
        return {
            "status": "error",
            "model": MODEL_NAME,
            "model_ready": MODEL_READY,
            "vectorstore_ready": VECTORSTORE_READY,
            "error": STARTUP_ERROR,
        }

    return {
        "status": "ready" if MODEL_READY and VECTORSTORE_READY else "loading",
        "model": MODEL_NAME,
        "model_ready": MODEL_READY,
        "vectorstore_ready": VECTORSTORE_READY,
    }


# ==================================================
# DOMAIN GATE HELPERS
# ==================================================

def _default_domain_max_distance() -> float:
    return float(os.getenv("CARDIOLOGY_DOMAIN_MAX_DISTANCE", "0.22"))


def _default_domain_max_distance_text() -> float:
    return float(os.getenv("CARDIOLOGY_DOMAIN_MAX_DISTANCE_TEXT", "0.22"))


def _domain_gate_text(vectorstore, query: str, max_distance: float) -> tuple[bool, Optional[float]]:
    """
    Domain gate using vectorstore's native embedding model.
    This is usually the most consistent because it matches the FAISS index embedding.
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

    return best_score_f <= max_distance, best_score_f


def _domain_gate(vectorstore, query_embedding: np.ndarray, max_distance: float) -> tuple[bool, Optional[float]]:
    """
    Secondary domain gate using projected query embedding.
    """
    try:
        res = vectorstore.similarity_search_with_score_by_vector(
            embedding=query_embedding.tolist(),
            k=1,
        )
    except Exception:
        return True, None

    if not res:
        return False, None

    best_score = res[0][1]

    try:
        best_score_f = float(best_score)
    except Exception:
        return True, None

    return best_score_f <= max_distance, best_score_f


# ==================================================
# MAIN QUERY ENDPOINT
# ==================================================

@app.post("/query2", response_model=QueryResponse)
def run_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # Prevent request during startup loading
    if not MODEL_READY or not VECTORSTORE_READY:
        raise HTTPException(
            status_code=503,
            detail="Backend is still loading model/vectorstore. Please wait and retry.",
        )

    # Short general intents should not run full RAG pipeline
    intent = detect_general_intent(req.query)
    if intent is not None:
        return QueryResponse(
            answer=generate_general_intent_answer(intent),
            retrieved_docs=[],
        )

    # Load vectorstore from cache
    vectorstore = load_vectorstore()

    # Query embedding + projection
    proj_emb = embed_and_project(req.query)
    proj_emb = l2_normalize(np.array(proj_emb))

    # Domain gate: native text embedding
    max_distance_text = (
        req.domain_max_distance_text
        if req.domain_max_distance_text is not None
        else _default_domain_max_distance_text()
    )

    in_domain_text, best_score_text = _domain_gate_text(
        vectorstore,
        req.query,
        max_distance=max_distance_text,
    )

    # Secondary projected vector gate
    max_distance = (
        req.domain_max_distance
        if req.domain_max_distance is not None
        else _default_domain_max_distance()
    )

    in_domain_vec, best_score = _domain_gate(
        vectorstore,
        proj_emb,
        max_distance=max_distance,
    )

    # Trust text gate if score exists
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
    alpha = req.alpha if req.alpha is not None else 0.5
    final_emb = fuse_embeddings(
        proj_query=proj_emb,
        hyde_query=hyde_emb,
        alpha=alpha,
    )
    final_emb = l2_normalize(np.array(final_emb))

    # Retrieval
    k = req.k if req.k is not None else 5

    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding=final_emb.tolist(),
            k=k,
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


# ==================================================
# STAGED PIPELINE ENDPOINT
# ==================================================

@app.post("/query_stages", response_model=StagedQueryResponse)
def run_query_stages(req: QueryRequest):
    """
    Detailed staged pipeline execution for demonstration.

    Stages:
    1. Input query
    2. Domain gate
    3. inBEDDER + Projection Head
    4. HyDE
    5. Embedding Fusion
    6. FAISS Retrieval
    7. Answer Generation
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    if not MODEL_READY or not VECTORSTORE_READY:
        raise HTTPException(
            status_code=503,
            detail="Backend is still loading model/vectorstore. Please wait and retry.",
        )

    # General intents
    intent = detect_general_intent(req.query)
    if intent is not None:
        stages = [
            StageOutput(
                stage_number=1,
                stage_name="Input Query",
                description="User's input question to the medical RAG system",
                data={
                    "query": req.query,
                    "query_length": len(req.query),
                    "query_word_count": len(req.query.split()),
                },
            ),
            StageOutput(
                stage_number=2,
                stage_name="General Intent",
                description="Detect short general intent and return canned response without retrieval",
                data={
                    "intent": intent,
                    "action": "RETURN_CANNED_RESPONSE",
                },
            ),
        ]

        return StagedQueryResponse(
            query=req.query,
            stages=stages,
            final_answer=generate_general_intent_answer(intent),
            is_in_domain=True,
        )

    stages: List[StageOutput] = []

    # ==================== STAGE 1 ====================
    stages.append(
        StageOutput(
            stage_number=1,
            stage_name="Input Query",
            description="User's input question to the medical RAG system",
            data={
                "query": req.query,
                "query_length": len(req.query),
                "query_word_count": len(req.query.split()),
            },
        )
    )

    # ==================== STAGE 2 ====================
    vectorstore = load_vectorstore()

    max_distance_text = (
        req.domain_max_distance_text
        if req.domain_max_distance_text is not None
        else _default_domain_max_distance_text()
    )

    in_domain_text, best_score_text = _domain_gate_text(
        vectorstore,
        req.query,
        max_distance=max_distance_text,
    )

    stages.append(
        StageOutput(
            stage_number=2,
            stage_name="Domain Gate",
            description="Determines if the query is within cardiology domain using FAISS similarity",
            data={
                "is_in_domain": in_domain_text,
                "best_similarity_score": float(best_score_text) if best_score_text is not None else None,
                "threshold": float(max_distance_text),
                "decision": "IN_DOMAIN" if in_domain_text else "OUT_OF_DOMAIN",
                "explanation": (
                    f"Query is considered {'IN' if in_domain_text else 'OUT OF'} domain. "
                    f"Best score: {best_score_text:.4f}, threshold: {max_distance_text:.4f}"
                    if best_score_text is not None
                    else "Domain gate check completed"
                ),
            },
        )
    )

    if not in_domain_text:
        out_of_domain_answer = generate_out_of_domain_answer(req.query)

        return StagedQueryResponse(
            query=req.query,
            stages=stages,
            final_answer=out_of_domain_answer,
            is_in_domain=False,
        )

    # ==================== STAGE 3 ====================
    proj_emb = embed_and_project(req.query)
    proj_emb = l2_normalize(np.array(proj_emb))

    stages.append(
        StageOutput(
            stage_number=3,
            stage_name="inBEDDER + Projection Head",
            description="Query embedding using InBEDDER model with learned projection to target space",
            data={
                "embedding_dimension": int(proj_emb.shape[0]),
                "embedding_norm": float(np.linalg.norm(proj_emb)),
                "embedding_sample": proj_emb[:10].tolist(),
                "model": "InBEDDER-RoBERTa-Large",
                "projection": "768-d linear projection head",
                "explanation": (
                    f"Generated {proj_emb.shape[0]}-dimensional normalized embedding "
                    "using InBEDDER encoder with trained projection layer"
                ),
            },
        )
    )

    # ==================== STAGE 4 ====================
    hyde_emb, hyde_docs = generate_hypothetical_docs(
        req.query,
        num_return_sequences=4,
    )
    hyde_emb = l2_normalize(np.array(hyde_emb))

    unique_hyde_docs = []

    for doc in hyde_docs:
        content = doc.split("\n", 1)[1] if "\n" in doc else doc

        is_unique = True

        for existing_doc in unique_hyde_docs:
            existing_content = (
                existing_doc.split("\n", 1)[1]
                if "\n" in existing_doc
                else existing_doc
            )

            if content[:150].strip() == existing_content[:150].strip():
                is_unique = False
                break

        if is_unique:
            unique_hyde_docs.append(doc)

    stages.append(
        StageOutput(
            stage_number=4,
            stage_name="HyDE - Hypothetical Document Generation",
            description="Generate hypothetical medical documents and embed them using Instructor model",
            data={
                "num_hypothetical_docs_generated": len(hyde_docs),
                "num_unique_docs": len(unique_hyde_docs),
                "hypothetical_documents": [
                    doc[:200] + "..." if len(doc) > 200 else doc
                    for doc in unique_hyde_docs
                ],
                "hyde_embedding_dimension": int(hyde_emb.shape[0]),
                "hyde_embedding_norm": float(np.linalg.norm(hyde_emb)),
                "hyde_embedding_sample": hyde_emb[:10].tolist(),
                "model": "sciFive-cardiology-generator + Instructor-large",
                "explanation": (
                    f"Generated {len(hyde_docs)} hypothetical documents "
                    f"({len(unique_hyde_docs)} unique), embedded each, and averaged to create HyDE embedding"
                ),
            },
        )
    )

    # ==================== STAGE 5 ====================
    alpha = req.alpha if req.alpha is not None else 0.5

    final_emb = fuse_embeddings(
        proj_query=proj_emb,
        hyde_query=hyde_emb,
        alpha=alpha,
    )
    final_emb = l2_normalize(np.array(final_emb))

    stages.append(
        StageOutput(
            stage_number=5,
            stage_name="Embedding Fusion",
            description="Combine inBEDDER and HyDE embeddings using weighted fusion",
            data={
                "fusion_alpha": float(alpha),
                "inbedder_weight": float(alpha),
                "hyde_weight": float(1 - alpha),
                "fused_embedding_dimension": int(final_emb.shape[0]),
                "fused_embedding_norm": float(np.linalg.norm(final_emb)),
                "fused_embedding_sample": final_emb[:10].tolist(),
                "explanation": (
                    f"Fused embeddings with α={alpha:.2f} "
                    f"(InBEDDER) and (1-α)={1-alpha:.2f} (HyDE), then L2-normalized"
                ),
            },
        )
    )

    # ==================== STAGE 6 ====================
    k = req.k if req.k is not None else 5

    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding=final_emb.tolist(),
            k=k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS retrieval error: {e}")

    if not docs_and_scores:
        stages.append(
            StageOutput(
                stage_number=6,
                stage_name="FAISS Retrieval",
                description="Retrieve top-k most similar documents from vector database",
                data={
                    "num_retrieved": 0,
                    "retrieved_documents": [],
                    "explanation": "No documents retrieved from FAISS index",
                },
            )
        )

        return StagedQueryResponse(
            query=req.query,
            stages=stages,
            final_answer="No relevant documents found.",
            is_in_domain=True,
        )

    retrieved_items = []

    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        retrieved_items.append(
            {
                "rank": rank,
                "score": float(score),
                "content_preview": (
                    doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content
                ),
                "content_length": len(doc.page_content),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            }
        )

    stages.append(
        StageOutput(
            stage_number=6,
            stage_name="FAISS Retrieval",
            description="Retrieve top-k most similar documents from cardiology knowledge base",
            data={
                "k": k,
                "num_retrieved": len(docs_and_scores),
                "retrieved_documents": retrieved_items,
                "average_score": float(np.mean([score for _, score in docs_and_scores])),
                "best_score": float(docs_and_scores[0][1]),
                "explanation": f"Retrieved top {k} documents using FAISS L2 distance similarity search",
            },
        )
    )

    # ==================== STAGE 7 ====================
    docs_only = [doc for doc, _ in docs_and_scores]
    final_answer = generate_final_answer(req.query, docs_only)

    stages.append(
        StageOutput(
            stage_number=7,
            stage_name="Answer Generation",
            description="Generate final answer using HuggingFace Phi with retrieved context",
            data={
                "answer": final_answer,
                "answer_length": len(final_answer),
                "answer_word_count": len(final_answer.split()),
                "num_context_docs": len(docs_only),
                "model": f"HuggingFace Transformers ({MODEL_NAME})",
                "explanation": f"Generated answer using {len(docs_only)} retrieved documents as context",
            },
        )
    )

    return StagedQueryResponse(
        query=req.query,
        stages=stages,
        final_answer=final_answer,
        is_in_domain=True,
    )