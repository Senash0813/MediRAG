# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

FAISS_INDEX_PATH = "data/faiss.index"
BM25_INDEX_PATH = "data/bm25.pkl"
METADATA_PATH = "data/metadata.pkl"

EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

#QUERY_REWRITER_DIR = "/Users/senash/Desktop/MediRAG/backend/neurology/model_weights/flan_t5_neuro_rewriter"
QUERY_REWRITER_DIR = str(BASE_DIR / "model_weights/flan_t5_neuro_rewriter")  # your local T5
# FINAL_LLM_DIR = "models_local/final_llm"                # your local LLM
OLLAMA_MODEL_NAME = "phi:2.7b"  
OLLAMA_BASE_URL = "http://localhost:11434"

# SLM1 Gatekeeper Configuration
SLM1_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # This should have the correct Ollama model name
SLM1_ADAPTER_PATH = str(BASE_DIR / "model_weights/slm1_lora_adapter_2")  # update this path to point to SLM 1 adapter
SLM1_HF_TOKEN = "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"

SLM2_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct" 
SLM2_ADAPTER_PATH = str(BASE_DIR / "model_weights/slm2-lora-adapter") # Update this path!
SLM2_HF_TOKEN = SLM1_HF_TOKEN # Reusing the token from SLM 1

# --- Instruction-Following Re-ranker Configuration ---
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANKER_TOP_K = 3 # The number of chunks we want to send to the final LLM

FAISS_TOP_K = 20
BM25_TOP_K = 50
FINAL_TOP_K = 10
RRF_K = 60
