from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
	"""Application configuration.

	Values can be overridden via environment variables, e.g.:
	- MEDIRAG_PASSAGES_PATH
	- MEDIRAG_CENTROIDS_PATH
	- MEDIRAG_FAISS_INDEX_PATH
	- MEDIRAG_EMBEDDINGS_PATH
	- MEDIRAG_S2_CACHE_PATH
	- MEDIRAG_LMSTUDIO_BASE_URL
	- MEDIRAG_LMSTUDIO_MODEL
	- MEDIRAG_S2_API_KEY
	"""

	base_dir: Path = Field(
		default_factory=lambda: Path(__file__).resolve().parents[3],
		description="Workspace root (auto-detected).",
	)

	# Data and resources
	resources_dir: Path = Field(
		default_factory=lambda: Path(__file__).resolve().parents[2] / "resources",
		description="Base directory for local resources (JSONL, centroids, FAISS, caches).",
	)

	passages_path: Path = Field(
		default_factory=lambda: Path(__file__).resolve().parents[2]
		/ "resources"
		/ "selected_specialties.jsonl",
		description="Path to passages JSONL file (primary corpus).",
	)

	centroids_path: Path = Field(
		default_factory=lambda: Path(__file__).resolve().parents[2]
		/ "resources"
		/ "centroids"
		/ "specialty_centroids.pkl",
		description="Path to specialty centroids pickle file.",
	)

	faiss_index_path: Path = Field(
		default_factory=lambda: Path(__file__).resolve().parents[2]
		/ "resources"
		/ "faiss"
		/ "index.faiss",
		description="Path to FAISS index file.",
	)

	embeddings_path: Path = Field(
		default_factory=lambda: Path(__file__).resolve().parents[2]
		/ "resources"
		/ "faiss"
		/ "embeddings.npy",
		description="Path to cached embeddings array.",
	)

	s2_cache_path: Path = Field(
		default_factory=lambda: Path(__file__).resolve().parents[2]
		/ "resources"
		/ "caches"
		/ "semantic_scholar_cache.json",
		description="Path to Semantic Scholar metadata cache.",
	)

	# Retrieval parameters
	default_top_k: int = Field(5, description="Default number of final documents to return.")
	semantic_top_n: int = Field(10, description="Number of semantic candidates before verification.")
	scope_threshold: float = Field(0.2, description="Similarity threshold for in-scope specialty detection.")

	# Semantic Scholar (required according to current requirement)
	s2_base_url: str = Field(
		"https://api.semanticscholar.org/graph/v1/paper",
		description="Semantic Scholar base URL.",
	)
	s2_api_key: Optional[str] = Field(
		default=None,
		description="Semantic Scholar API key (required for enrichment).",
		env="S2_API_KEY",
	)

	# LLM Provider selection
	llm_provider: Literal["lmstudio", "ollama"] = Field(
		"lmstudio",
		description="Which LLM provider to use: 'lmstudio' or 'ollama'.",
		env="MEDIRAG_LLM_PROVIDER",
	)

	# LLM / LM Studio
	lmstudio_base_url: str = Field(
		"http://127.0.0.1:1234",
		description="Base URL for LM Studio HTTP API.",
	)
	lmstudio_model: str = Field(
		"qwen2.5-3b-instruct:2",
		description="Model identifier served by LM Studio.",
	)
	ollama_base_url: str = Field(
		"http://127.0.0.1:11434",
		description="Base URL for Ollama HTTP API.",
	)
	ollama_model: str = Field(
		"qwen2.5:3b",
		description="Model identifier to use with Ollama.",
	)
	lm_max_new_tokens: int = Field(700, description="Max tokens for LLM generations.")
	lm_temperature: float = Field(0.0, description="Temperature for deterministic generations.")

	class Config:
		env_prefix = "MEDIRAG_"
		case_sensitive = False

	@property
	def llm_base_url(self) -> str:
		"""Return the active LLM base URL based on provider selection."""
		if self.llm_provider == "ollama":
			return self.ollama_base_url
		return self.lmstudio_base_url

	@property
	def llm_model(self) -> str:
		"""Return the active LLM model identifier based on provider selection."""
		if self.llm_provider == "ollama":
			return self.ollama_model
		return self.lmstudio_model


def load_settings() -> AppSettings:
    """Load application settings from environment with sane defaults."""

    settings = AppSettings()
    
    # Debug: print the actual path and LLM provider being used
    print(f"DEBUG: passages_path = {settings.passages_path}")
    print(f"DEBUG: LLM Provider = {settings.llm_provider}")
    print(f"DEBUG: LLM Base URL = {settings.llm_base_url}")
    print(f"DEBUG: LLM Model = {settings.llm_model}")

    # Enforce Semantic Scholar configuration as "required" for now.
    if not settings.s2_api_key:
        raise RuntimeError(
            "Semantic Scholar API key (S2_API_KEY or MEDIRAG_S2_API_KEY) is required but not set."
        )

    return settings

