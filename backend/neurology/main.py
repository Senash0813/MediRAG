from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import *
from models.query_rewriter import QueryRewriter
from models.embedder import Embedder
from models.llm_rephraser import LLMRephraser
from retrieval.faiss_retriever import FaissRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_rrf import rrf_fusion
from config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL

app = FastAPI(title="RAG Backend v1")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything ONCE
query_rewriter = QueryRewriter(QUERY_REWRITER_DIR)
embedder = Embedder(EMBEDDING_MODEL_NAME)
faiss_retriever = FaissRetriever(FAISS_INDEX_PATH, METADATA_PATH)
bm25_retriever = BM25Retriever(BM25_INDEX_PATH, METADATA_PATH)
llm = LLMRephraser(
    model_name=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL
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

    final_answer = llm.rephrase(user_query, fused)

    return {
        "question": user_query,
        "answer": final_answer,
        "retrieved_answers": fused
    }
