# from fastapi import FastAPI
# from pydantic import BaseModel

# from config import *
# from models.query_rewriter import QueryRewriter
# from models.embedder import Embedder
# from models.llm_rephraser import LLMRephraser
# from models.slm_gatekeeper import SLMGatekeeper

# # --- NEW IMPORTS ---
# from models.slm_blueprint import SLMBlueprintGenerator
# from models.instruction_reranker import InstructionReranker

# from retrieval.faiss_retriever import FaissRetriever
# from retrieval.bm25_retriever import BM25Retriever
# from retrieval.hybrid_rrf import rrf_fusion

# app = FastAPI(title="RAG Backend v1")

# # Load everything ONCE
# query_rewriter = QueryRewriter(QUERY_REWRITER_DIR)
# embedder = Embedder(EMBEDDING_MODEL_NAME)
# faiss_retriever = FaissRetriever(FAISS_INDEX_PATH, METADATA_PATH)
# bm25_retriever = BM25Retriever(BM25_INDEX_PATH, METADATA_PATH)
# llm = LLMRephraser(
#     model_name=OLLAMA_MODEL_NAME,
#     base_url=OLLAMA_BASE_URL
# )

# # Initialize SLM 1 Gatekeeper
# slm_gatekeeper = SLMGatekeeper(
#     base_model=SLM1_BASE_MODEL,
#     adapter_path=SLM1_ADAPTER_PATH,
#     hf_token=SLM1_HF_TOKEN
# )

# # --- NEW: Initialize SLM 2 Blueprint Generator ---
# slm_blueprint = SLMBlueprintGenerator(
#     base_model=SLM2_BASE_MODEL,
#     adapter_path=SLM2_ADAPTER_PATH,
#     hf_token=SLM2_HF_TOKEN
# )

# # --- NEW: Initialize Instruction Reranker ---
# instruction_reranker = InstructionReranker(
#     model_name=RERANKER_MODEL_NAME,
#     top_k=RERANKER_TOP_K
# )

# class QueryRequest(BaseModel):
#     question: str

# @app.post("/query")
# def query_rag(req: QueryRequest):
#     user_query = req.question

#     # 1. Retrieval Phase
#     rewritten = query_rewriter.rewrite(user_query)
#     embedding = embedder.encode(user_query)

#     faiss_results = faiss_retriever.retrieve(embedding, FAISS_TOP_K)
#     bm25_results = bm25_retriever.retrieve(rewritten, BM25_TOP_K)

#     fused = rrf_fusion(faiss_results, bm25_results, RRF_K)[:FINAL_TOP_K]

#     # 2. Hard Filter Phase (SLM 1 Gatekeeper)
#     filtered_chunks = slm_gatekeeper.filter_chunks(user_query, fused)

#     # --- NEW: 3. Blueprint Planning Phase (SLM 2) ---
#     blueprint = slm_blueprint.generate_blueprint(user_query)

#     # --- NEW: 4. Soft Scoring Phase (Instruction Reranker) ---
#     ranked_chunks = instruction_reranker.rerank(user_query, blueprint, filtered_chunks)

#     # 5. Final Answer Generation
#     # Pass the surgically ranked Top-K chunks to the final LLM
#     final_answer = llm.rephrase(user_query, ranked_chunks)

#     return {
#         "question": user_query,
#         "blueprint": blueprint,           # Included so you can audit the SLM 2 logic
#         "answer": final_answer,
#         "retrieved_answers": ranked_chunks # Returns the highly-ranked chunks with their new scores
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

shared_slm = SharedSLMManager(base_model=SLM1_BASE_MODEL, hf_token=SLM1_HF_TOKEN)
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