from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


KB_PATH = Path(_env("MEDIRAG_KB_PATH", str(BACKEND_DIR / "miriad_balanced_300.json")))

EMBEDDING_MODEL_NAME = _env("MEDIRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATOR_MODEL_NAME = _env("MEDIRAG_GENERATOR_MODEL", "google/flan-t5-base")

INDEX_PATH = Path(_env("MEDIRAG_INDEX_PATH", str(DATA_DIR / "kb_faiss.index")))
META_PATH = Path(_env("MEDIRAG_META_PATH", str(DATA_DIR / "kb_metadata.json")))

EMBEDDING_DIM = int(_env("MEDIRAG_EMBEDDING_DIM", "384"))
TOP_K = int(_env("MEDIRAG_TOP_K", "5"))

# Notebook default is 0 (GPU in Colab). Locally, you can set MEDIRAG_DEVICE=-1 for CPU.
DEVICE = int(_env("MEDIRAG_DEVICE", "0"))

SIM_THRESHOLD = float(_env("MEDIRAG_SIM_THRESHOLD", "0.7"))

UNVERIFIED_MESSAGE = _env(
    "MEDIRAG_UNVERIFIED_MESSAGE",
    "The generated answer could not be verified by retrieved evidence.",
)

# CORS
CORS_ORIGINS = [o.strip() for o in _env("MEDIRAG_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
