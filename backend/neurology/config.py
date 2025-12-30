# config.py

FAISS_INDEX_PATH = "data/faiss.index"
BM25_INDEX_PATH = "data/bm25.pkl"
METADATA_PATH = "data/metadata.pkl"

EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

QUERY_REWRITER_DIR = "C:\\Users\\USER\\Desktop\\MediRAG\\chunk-rag pipeline\\flan_t5_neuro_rewriter"  # your local T5
# FINAL_LLM_DIR = "models_local/final_llm"                # your local LLM
OLLAMA_MODEL_NAME = "gemma2:2b"  
OLLAMA_BASE_URL = "http://localhost:11434"


FAISS_TOP_K = 20
BM25_TOP_K = 50
FINAL_TOP_K = 10
RRF_K = 60
