from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data / storage
    kb_file: Path = Field(default=Path("data/kb.json"), description="KB JSON/JSONL path")
    embeddings_dir: Path = Field(
        default=Path("storage/embeddings"), description="Folder where FAISS + metadata are stored"
    )

    kb_max_docs: int | None = Field(
        default=None,
        description="Optional safety cap for very large KB files (set via KB_MAX_DOCS)",
    )

    # Embeddings + retrieval
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformers embedding model",
    )
    embedding_dim: int = Field(default=384)
    top_k: int = Field(default=5)

    embedding_batch_size: int = Field(
        default=64,
        description="Batch size for embedding during indexing (set via EMBED_BATCH_SIZE)",
    )

    # Ollama (generator + judge)
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_generator_model: str = Field(
        default="phi", description='Ollama model name (e.g. "phi", "phi2", "phi3")'
    )
    ollama_judge_model: str = Field(
        default="phi", description="Separate Ollama model for judge; can be same as generator"
    )

    # Judge backend
    judge_backend: str = Field(
        default="ollama",
        description="LLM-as-judge backend: 'ollama' (default) or 'hf' (Transformers pipeline)",
    )
    hf_judge_model: str = Field(
        default="google/flan-t5-base",
        description="Hugging Face model id for judge when JUDGE_BACKEND=hf",
    )
    hf_judge_max_new_tokens: int = Field(
        default=32,
        description="Max new tokens for HF judge generation (when JUDGE_BACKEND=hf)",
    )

    # Domain / scope gate (passage-level index)
    enable_domain_gate: bool = Field(default=True)
    scope_top_k: int = Field(default=10)
    scope_min_top1: float = Field(default=0.35)
    scope_min_avg_topk: float = Field(default=0.30)
    scope_min_cohesion: float = Field(default=0.20)
    scope_max_sentences_per_passage: int = Field(default=3)

    # Verification features
    enable_ner: bool = Field(default=True)
    enable_nli: bool = Field(default=True)
    transformers_device: int = Field(
        default=-1,
        description="Transformers pipeline device: -1=CPU, 0=first CUDA GPU",
    )

    transformer_ner_model: str = Field(default="d4data/biomedical-ner-all")
    nli_model: str = Field(default="cnut1648/biolinkbert-large-mnli-snli")

    # Risk scoring / thresholds
    risk_threshold: int = Field(default=20)

    # Answer-level verification
    answer_nli_supported_th: float = Field(default=0.80)
    answer_nli_unsupported_th: float = Field(default=0.45)
    max_evidence_chars: int = Field(default=1500)

    # Confidence thresholds
    fast_verified_th: float = Field(default=0.80)
    fast_hallucinated_th: float = Field(default=0.45)
    full_verified_th: float = Field(default=0.75)
    full_hallucinated_th: float = Field(default=0.50)
    sim_for_regen: float = Field(default=0.60)

    # Generation parameters
    gen_max_tokens: int = Field(default=256)
    regen_max_tokens: int = Field(default=128)

    # Filenames inside embeddings_dir
    kb_index_filename: str = Field(default="kb_faiss.index")
    kb_meta_filename: str = Field(default="kb_metadata.json")
    scope_index_filename: str = Field(default="scope_faiss.index")
    scope_meta_filename: str = Field(default="scope_metadata.json")


@lru_cache
def get_settings() -> Settings:
    return Settings()
