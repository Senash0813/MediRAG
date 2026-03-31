from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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
# Permissive CORS for local demo - allows file:// URLs and any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo purposes
    allow_credentials=False,  # Must be False when using wildcard origins
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


# ========================================
# STAGED PIPELINE API FOR DEMONSTRATION
# ========================================

class StageOutput(BaseModel):
    """Output from a single stage of the pipeline."""
    stage_name: str
    stage_number: int
    description: str
    data: Dict[str, Any]


class StagedQueryResponse(BaseModel):
    """Complete staged pipeline output."""
    query: str
    stages: List[StageOutput]
    final_answer: str
    is_in_domain: bool


@app.post("/query_stages", response_model=StagedQueryResponse)
def run_query_stages(req: QueryRequest):
    """
    Detailed staged pipeline execution for demonstration purposes.
    
    Returns outputs from each stage:
    1. Input query
    2. Domain gate
    3. inBEDDER + Projection Head
    4. HyDE Hypothetical Document generation
    5. Embedding fusion
    6. FAISS Retrieval
    7. Answer generation
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    stages = []
    
    # ==================== STAGE 1: INPUT QUERY ====================
    stages.append(StageOutput(
        stage_number=1,
        stage_name="Input Query",
        description="User's input question to the medical RAG system",
        data={
            "query": req.query,
            "query_length": len(req.query),
            "query_word_count": len(req.query.split())
        }
    ))
    
    # ==================== STAGE 2: DOMAIN GATE ====================
    vectorstore = load_vectorstore()
    
    max_distance_text = (
        req.domain_max_distance_text
        if req.domain_max_distance_text is not None
        else _default_domain_max_distance_text()
    )
    
    in_domain_text, best_score_text = _domain_gate_text(
        vectorstore, req.query, max_distance=max_distance_text
    )
    
    stages.append(StageOutput(
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
            )
        }
    ))
    
    # If out of domain, return early
    if not in_domain_text:
        out_of_domain_answer = generate_out_of_domain_answer(req.query)
        return StagedQueryResponse(
            query=req.query,
            stages=stages,
            final_answer=out_of_domain_answer,
            is_in_domain=False
        )
    
    # ==================== STAGE 3: inBEDDER + PROJECTION HEAD ====================
    proj_emb = embed_and_project(req.query)
    proj_emb = l2_normalize(np.array(proj_emb))
    
    stages.append(StageOutput(
        stage_number=3,
        stage_name="inBEDDER + Projection Head",
        description="Query embedding using InBEDDER model with learned projection to target space",
        data={
            "embedding_dimension": int(proj_emb.shape[0]),
            "embedding_norm": float(np.linalg.norm(proj_emb)),
            "embedding_sample": proj_emb[:10].tolist(),  # First 10 dimensions for preview
            "model": "InBEDDER-RoBERTa-Large",
            "projection": "768-d linear projection head",
            "explanation": f"Generated {proj_emb.shape[0]}-dimensional normalized embedding using InBEDDER encoder with trained projection layer"
        }
    ))
    
    # ==================== STAGE 4: HyDE ====================
    hyde_emb, hyde_docs = generate_hypothetical_docs(req.query, num_return_sequences=4)
    hyde_emb = l2_normalize(np.array(hyde_emb))
    
    # Deduplicate hypothetical documents for display only 
    # Uses similarity-based deduplication to catch near-duplicates
    unique_hyde_docs = []
    for doc in hyde_docs:
        # Extract just the generated content (after the instruction prefix)
        # Instruction format: "Represent the cardiology document for retrieval:\n<content>"
        content = doc.split('\n', 1)[1] if '\n' in doc else doc
        
        # Check if this content is substantially different from existing ones
        is_unique = True
        for existing_doc in unique_hyde_docs:
            existing_content = existing_doc.split('\n', 1)[1] if '\n' in existing_doc else existing_doc
            
            # Compare first 150 chars to determine similarity (catches near-duplicates)
            # Strip whitespace to ignore formatting differences
            if content[:150].strip() == existing_content[:150].strip():
                is_unique = False
                break
        
        if is_unique:
            unique_hyde_docs.append(doc)
    
    stages.append(StageOutput(
        stage_number=4,
        stage_name="HyDE - Hypothetical Document Generation",
        description="Generate hypothetical medical documents and embed them using Instructor model",
        data={
            "num_hypothetical_docs_generated": len(hyde_docs),
            "num_unique_docs": len(unique_hyde_docs),
            "hypothetical_documents": [doc[:200] + "..." if len(doc) > 200 else doc for doc in unique_hyde_docs],
            "hyde_embedding_dimension": int(hyde_emb.shape[0]),
            "hyde_embedding_norm": float(np.linalg.norm(hyde_emb)),
            "hyde_embedding_sample": hyde_emb[:10].tolist(),
            "model": "sciFive-cardiology-generator + Instructor-large",
            "explanation": f"Generated {len(hyde_docs)} hypothetical documents ({len(unique_hyde_docs)} unique), embedded each, and averaged to create HyDE embedding"
        }
    ))
    
    # ==================== STAGE 5: EMBEDDING FUSION ====================
    alpha = req.alpha if req.alpha is not None else 0.5
    final_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=alpha)
    final_emb = l2_normalize(np.array(final_emb))
    
    stages.append(StageOutput(
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
            "explanation": f"Fused embeddings with α={alpha:.2f} (InBEDDER) and (1-α)={1-alpha:.2f} (HyDE), then L2-normalized"
        }
    ))
    
    # ==================== STAGE 6: FAISS RETRIEVAL ====================
    k = req.k if req.k is not None else 5
    
    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding=final_emb.tolist(),
            k=k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS retrieval error: {e}")
    
    if not docs_and_scores:
        stages.append(StageOutput(
            stage_number=6,
            stage_name="FAISS Retrieval",
            description="Retrieve top-k most similar documents from vector database",
            data={
                "num_retrieved": 0,
                "retrieved_documents": [],
                "explanation": "No documents retrieved from FAISS index"
            }
        ))
        
        return StagedQueryResponse(
            query=req.query,
            stages=stages,
            final_answer="No relevant documents found.",
            is_in_domain=True
        )
    
    retrieved_items = []
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        retrieved_items.append({
            "rank": rank,
            "score": float(score),
            "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "content_length": len(doc.page_content),
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
        })
    
    stages.append(StageOutput(
        stage_number=6,
        stage_name="FAISS Retrieval",
        description="Retrieve top-k most similar documents from cardiology knowledge base",
        data={
            "k": k,
            "num_retrieved": len(docs_and_scores),
            "retrieved_documents": retrieved_items,
            "average_score": float(np.mean([score for _, score in docs_and_scores])),
            "best_score": float(docs_and_scores[0][1]),
            "explanation": f"Retrieved top {k} documents using FAISS L2 distance similarity search"
        }
    ))
    
    # ==================== STAGE 7: ANSWER GENERATION ====================
    docs_only = [doc for doc, _ in docs_and_scores]
    final_answer = generate_final_answer(req.query, docs_only)
    
    stages.append(StageOutput(
        stage_number=7,
        stage_name="Answer Generation",
        description="Generate final answer using LLM with retrieved context",
        data={
            "answer": final_answer,
            "answer_length": len(final_answer),
            "answer_word_count": len(final_answer.split()),
            "num_context_docs": len(docs_only),
            "model": "Ollama (phi:2.7b)",
            "explanation": f"Generated answer using {len(docs_only)} retrieved documents as context"
        }
    ))
    
    return StagedQueryResponse(
        query=req.query,
        stages=stages,
        final_answer=final_answer,
        is_in_domain=True
    )
