# config.py

FAISS_INDEX_PATH = "data/faiss.index"
BM25_INDEX_PATH = "data/bm25.pkl"
METADATA_PATH = "data/metadata.pkl"

EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

QUERY_REWRITER_DIR = "C:\\Users\\USER\\Desktop\\MediRAG\\chunk-rag pipeline\\flan_t5_neuro_rewriter"  # update this path to point to your local query rewriter model
# FINAL_LLM_DIR = "models_local/final_llm"                # your local LLM
OLLAMA_MODEL_NAME = "phi:2.7b"  
OLLAMA_BASE_URL = "http://localhost:11434"

# SLM1 Gatekeeper Configuration
SLM1_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # This should have the correct Ollama model name
SLM1_ADAPTER_PATH = "C:\\Users\\USER\\Desktop\\MediRAG\\chunk-rag pipeline\\adapters\\slm1_lora_adapter_2"  # update this path to point to SLM 1 adapter
SLM1_HF_TOKEN = "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"

FAISS_TOP_K = 20
BM25_TOP_K = 50
FINAL_TOP_K = 10
RRF_K = 60
