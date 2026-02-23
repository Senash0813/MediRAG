from fastapi import FastAPI
from pydantic import BaseModel

from config import *
from models.query_rewriter import QueryRewriter
from models.embedder import Embedder
from models.llm_rephraser import LLMRephraser
from models.slm_gatekeeper import SLMGatekeeper
from retrieval.faiss_retriever import FaissRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_rrf import rrf_fusion
# from config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL

app = FastAPI(title="RAG Backend v1")

# Load everything ONCE
query_rewriter = QueryRewriter(QUERY_REWRITER_DIR)
embedder = Embedder(EMBEDDING_MODEL_NAME)
faiss_retriever = FaissRetriever(FAISS_INDEX_PATH, METADATA_PATH)
bm25_retriever = BM25Retriever(BM25_INDEX_PATH, METADATA_PATH)
llm = LLMRephraser(
    model_name=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL
)

# Initialize SLM Gatekeeper (lazy-loaded on first use)
slm_gatekeeper = SLMGatekeeper(
    base_model=SLM1_BASE_MODEL,
    adapter_path=SLM1_ADAPTER_PATH,
    hf_token=SLM1_HF_TOKEN
)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_rag(req: QueryRequest):
    user_query = req.question

    rewritten = query_rewriter.rewrite(user_query)
    embedding = embedder.encode(user_query)

    faiss_results = faiss_retriever.retrieve(
        embedding, FAISS_TOP_K
    )
    bm25_results = bm25_retriever.retrieve(
        rewritten, BM25_TOP_K
    )

    fused = rrf_fusion(
        faiss_results, bm25_results, RRF_K
    )[:FINAL_TOP_K]

    filtered_chunks = slm_gatekeeper.filter_chunks(user_query, fused)

    final_answer = llm.rephrase(user_query, filtered_chunks)

    return {
        "question": user_query,
        "answer": final_answer,
        "retrieved_answers": filtered_chunks  # Return only passed chunks
        # "retrieval_stats": {
        #     "total_retrieved": len(fused),
        #     "passed_gatekeeper": len(filtered_chunks),
        #     "filtered_out": len(fused) - len(filtered_chunks)
        # }
    }
