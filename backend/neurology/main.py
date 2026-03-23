from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from config import *
# from models.query_rewriter import QueryRewriter
# from models.embedder import Embedder
# from models.llm_rephraser import LLMRephraser
# from retrieval.faiss_retriever import FaissRetriever
# from retrieval.bm25_retriever import BM25Retriever
# from retrieval.hybrid_rrf import rrf_fusion
# from config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL

# app = FastAPI(title="RAG Backend v1")

# Add CORS middleware to allow frontend requests


# # Load everything ONCE
# query_rewriter = QueryRewriter(QUERY_REWRITER_DIR)
# embedder = Embedder(EMBEDDING_MODEL_NAME)
# faiss_retriever = FaissRetriever(FAISS_INDEX_PATH, METADATA_PATH)
# bm25_retriever = BM25Retriever(BM25_INDEX_PATH, METADATA_PATH)
# llm = LLMRephraser(
#     model_name=OLLAMA_MODEL_NAME,
#     base_url=OLLAMA_BASE_URL
# )


# class QueryRequest(BaseModel):
#     question: str


# @app.post("/query")
# def query_rag(req: QueryRequest):
#     user_query = req.question

#     rewritten = query_rewriter.rewrite(user_query)
#     embedding = embedder.encode(user_query)

#     faiss_results = faiss_retriever.retrieve(
#         embedding, FAISS_TOP_K
#     )
#     bm25_results = bm25_retriever.retrieve(
#         rewritten, BM25_TOP_K
#     )

#     fused = rrf_fusion(
#         faiss_results, bm25_results, RRF_K
#     )[:FINAL_TOP_K]

#     final_answer = llm.rephrase(user_query, fused)

#     return {
#         "question": user_query,
#         "answer": final_answer,
#         "retrieved_answers": fused
#     }

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from config import *
from models.query_rewriter import QueryRewriter
from models.embedder import Embedder
from models.llm_rephraser import LLMRephraser
from models.slm_gatekeeper import SLMGatekeeper
from models.slm_blueprint import SLMBlueprintGenerator
from models.instruction_reranker import InstructionReranker
from models.shared_slm_manager import SharedSLMManager
from retrieval.faiss_retriever import FaissRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_rrf import rrf_fusion

query_rewriter = QueryRewriter(QUERY_REWRITER_DIR)
embedder = Embedder(EMBEDDING_MODEL_NAME, hf_token=SLM1_HF_TOKEN)
faiss_retriever = FaissRetriever(FAISS_INDEX_PATH, METADATA_PATH)
bm25_retriever = BM25Retriever(BM25_INDEX_PATH, METADATA_PATH)
llm = LLMRephraser(model_name=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL)
instruction_reranker = InstructionReranker(model_name=RERANKER_MODEL_NAME, top_k=RERANKER_TOP_K)

shared_slm = SharedSLMManager(
    base_model=SLM1_BASE_MODEL, 
    hf_token=SLM1_HF_TOKEN,
    force_gpu=FORCE_GPU,
    use_4bit=USE_4BIT_QUANTIZATION
)
slm_gatekeeper = SLMGatekeeper(shared_slm)
slm_blueprint = SLMBlueprintGenerator(shared_slm)

@asynccontextmanager
async def lifespan(app: FastAPI):
    shared_slm.initialize(
        gatekeeper_adapter_path=SLM1_ADAPTER_PATH,
        blueprint_adapter_path=SLM2_ADAPTER_PATH
    )
    yield

app = FastAPI(title="RAG Backend v1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "null"  # Allows file:// protocol
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_rag(req: QueryRequest):
    user_query = req.question
    rewritten = query_rewriter.rewrite(user_query)
    embedding = embedder.encode(user_query)
    faiss_results = faiss_retriever.retrieve(embedding, FAISS_TOP_K)
    bm25_results = bm25_retriever.retrieve(rewritten, BM25_TOP_K)
    fused = rrf_fusion(faiss_results, bm25_results, RRF_K)[:FINAL_TOP_K]
    filtered_chunks = slm_gatekeeper.filter_chunks(user_query, fused)
    blueprint = slm_blueprint.generate_blueprint(user_query)
    ranked_chunks = instruction_reranker.rerank(user_query, blueprint, filtered_chunks)
    final_answer = llm.rephrase(user_query, ranked_chunks)
    return {
        "question": user_query,
        "blueprint": blueprint,
        "answer": final_answer,
        "retrieved_answers": ranked_chunks
    }

@app.post("/chunkrag/detailed")
def query_rag_detailed(req: QueryRequest):
    """
    Detailed RAG pipeline endpoint showing outputs at each stage.
    Perfect for demonstrations and explaining the pipeline flow.
    """
    user_query = req.question
    
    # Stage 1: Original Query
    stage1_output = {
        "stage": "1. Original Query",
        "description": "User's input question",
        "output": {
            "query": user_query
        }
    }
    
    # Stage 2: Query Rewriting & Embedding
    rewritten_query = query_rewriter.rewrite(user_query)
    query_embedding = embedder.encode(user_query)
    stage2_output = {
        "stage": "2. Query Rewriting & Embedding",
        "description": "Rewrite query for better retrieval and generate embedding vector",
        "output": {
            "original_query": user_query,
            "rewritten_query": rewritten_query,
            "embedding_dimension": len(query_embedding[0]),
            "embedding_sample": query_embedding[0][:5].tolist()  # Show first 5 dimensions
        }
    }
    
    # Stage 3: Hybrid Retrieval (BM25 + FAISS + Fusion)
    faiss_results = faiss_retriever.retrieve(query_embedding, FAISS_TOP_K)
    bm25_results = bm25_retriever.retrieve(rewritten_query, BM25_TOP_K)
    fused_results = rrf_fusion(faiss_results, bm25_results, RRF_K)[:FINAL_TOP_K]
    stage3_output = {
        "stage": "3. Hybrid Retrieval",
        "description": "Retrieve relevant chunks using BM25 (keyword) and FAISS (semantic), then fuse with RRF",
        "output": {
            "faiss_retrieved": len(faiss_results),
            "bm25_retrieved": len(bm25_results),
            "fused_top_k": len(fused_results),
            "fused_chunks": [
                {
                    "qa_id": chunk.get("qa_id"),
                    "question": chunk.get("question", ""),
                    "answer_preview": chunk.get("answer", "")[:200] + "...",
                    "rrf_score": chunk.get("rrf_score", 0)
                }
                for chunk in fused_results
            ]
        }
    }
    
    # Stage 4: SLM 1 - Gatekeeper (Filter chunks)
    filtered_chunks = slm_gatekeeper.filter_chunks(user_query, fused_results)
    stage4_output = {
        "stage": "4. SLM 1 - Gatekeeper",
        "description": "Filter chunks using constraint-aware SLM to remove irrelevant or contradictory content",
        "output": {
            "input_chunks": len(fused_results),
            "filtered_chunks": len(filtered_chunks),
            "chunks_removed": len(fused_results) - len(filtered_chunks),
            "passed_chunks": [
                {
                    "qa_id": chunk.get("qa_id"),
                    "question": chunk.get("question", ""),
                    "answer_preview": chunk.get("answer", "")[:200] + "...",
                    "gate_decision": chunk.get("gate_decision", "PASS")
                }
                for chunk in filtered_chunks
            ]
        }
    }
    
    # Stage 5: SLM 2 - Blueprint Generator
    blueprint = slm_blueprint.generate_blueprint(user_query)
    stage5_output = {
        "stage": "5. SLM 2 - Blueprint Generator",
        "description": "Generate clinical blueprint outlining what a strong answer must contain",
        "output": {
            "blueprint": blueprint,
            "num_requirements": len(blueprint) if isinstance(blueprint, list) else 1
        }
    }
    
    # Stage 6: Instruction-Following Re-ranker
    ranked_chunks = instruction_reranker.rerank(user_query, blueprint, filtered_chunks)
    stage6_output = {
        "stage": "6. Instruction-Following Re-ranker",
        "description": "Score and rank chunks based on blueprint guidance, return top 3",
        "output": {
            "input_chunks": len(filtered_chunks),
            "output_top_k": len(ranked_chunks),
            "ranked_chunks": [
                {
                    "rank": idx + 1,
                    "qa_id": chunk.get("qa_id"),
                    "question": chunk.get("question", ""),
                    "answer_preview": chunk.get("answer", "")[:200] + "...",
                    "rerank_score": chunk.get("rerank_score", 0)
                }
                for idx, chunk in enumerate(ranked_chunks)
            ]
        }
    }
    
    # Stage 7: Final LLM - Generate Answer
    final_answer = llm.rephrase(user_query, ranked_chunks)
    stage7_output = {
        "stage": "7. Final LLM - Answer Generation",
        "description": "Generate comprehensive answer using top-ranked chunks",
        "output": {
            "final_answer": final_answer,
            "chunks_used": len(ranked_chunks)
        }
    }
    
    # Return all stages
    return {
        "pipeline_stages": [
            stage1_output,
            stage2_output,
            stage3_output,
            stage4_output,
            stage5_output,
            stage6_output,
            stage7_output
        ],
        "summary": {
            "original_query": user_query,
            "rewritten_query": rewritten_query,
            "total_retrieved": len(fused_results),
            "after_gatekeeper": len(filtered_chunks),
            "final_top_k": len(ranked_chunks),
            "final_answer": final_answer
        }
    }

