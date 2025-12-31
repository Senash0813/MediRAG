from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Data / index paths
    backend_dir: Path

    jsonl_path: Path
    emb_path: Path
    index_path: Path

    # Retrieval / scoring
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    current_year: int = 2025
    top_k: int = 5

    # Semantic Scholar
    s2_api_key: str | None = None

    # LM Studio (OpenAI-compatible)
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "qwen2.5-3b-instruct"  # must match what LM Studio exposes


def load_settings() -> Settings:
    # This package lives under backend/primarycare/medirag.
    # Default data lives one level above "primarycare" (i.e., backend/).
    primarycare_dir = Path(__file__).resolve().parents[1]
    backend_root_dir = primarycare_dir.parent

    backend_dir = primarycare_dir

    jsonl_path = Path(
        os.getenv(
            "MEDIRAG_JSONL_PATH",
            str(backend_root_dir / "primarycare" / "sampled_2500_per_specialty.jsonl"),
        )
    )

    # Keep the notebook's concept of cached files, but place them under backend/.cache by default.
    cache_dir = Path(os.getenv("MEDIRAG_CACHE_DIR", str(backend_dir / ".cache")))
    emb_path = Path(os.getenv("MEDIRAG_EMB_PATH", str(cache_dir / "embeddings_minilm.npy")))
    index_path = Path(os.getenv("MEDIRAG_INDEX_PATH", str(cache_dir / "faiss_minilm.index")))

    s2_api_key = os.getenv("S2_API_KEY")

    lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    lmstudio_model = os.getenv("LMSTUDIO_MODEL", "qwen2.5-3b-instruct")

    return Settings(
        backend_dir=backend_dir,
        jsonl_path=jsonl_path,
        emb_path=emb_path,
        index_path=index_path,
        s2_api_key=s2_api_key,
        lmstudio_base_url=lmstudio_base_url,
        lmstudio_model=lmstudio_model,
    )
