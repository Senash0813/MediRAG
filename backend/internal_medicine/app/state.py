from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
from sentence_transformers import SentenceTransformer

from .core.config import Settings
from .services.embeddings_store import (
    build_kb_faiss_index,
    build_scope_index_from_kb,
    ensure_dir,
    load_kb_faiss_index,
    load_scope_index,
)
from .services.kb import load_kb
from .services.ollama_client import OllamaClient


@dataclass
class AppState:
    settings: Settings

    kb_docs: List[Dict[str, Any]]
    metadata: List[Dict[str, Any]]

    embedder: SentenceTransformer
    index: Any

    scope_index: Any
    scope_meta: List[Dict[str, Any]]

    ollama: OllamaClient

    # Optional NLP/verification models
    nlp_sci: Any = None
    nlp_bc5cdr: Any = None
    biomed_ner: Any = None
    nli_pipeline: Any = None
    judge_t2t: Any = None

    # Feature availability (after graceful loading)
    ner_ready: bool = False
    nli_ready: bool = False
    scispacy_ready: bool = False
    judge_ready: bool = False

    # KB load diagnostics
    kb_load_error: str | None = None

    @property
    def kb_index_path(self) -> Path:
        return self.settings.embeddings_dir / self.settings.kb_index_filename

    @property
    def kb_meta_path(self) -> Path:
        return self.settings.embeddings_dir / self.settings.kb_meta_filename

    @property
    def scope_index_path(self) -> Path:
        return self.settings.embeddings_dir / self.settings.scope_index_filename

    @property
    def scope_meta_path(self) -> Path:
        return self.settings.embeddings_dir / self.settings.scope_meta_filename


def _try_load_spacy_models(state: AppState) -> None:
    try:
        import spacy  # type: ignore

        # SciSpacy models are optional; only load if present
        try:
            state.nlp_sci = spacy.load("en_core_sci_sm")
            state.nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")
            state.scispacy_ready = True
        except Exception:
            state.nlp_sci = None
            state.nlp_bc5cdr = None
            state.scispacy_ready = False
    except Exception:
        state.nlp_sci = None
        state.nlp_bc5cdr = None
        state.scispacy_ready = False


def _try_load_transformers_pipelines(state: AppState) -> None:
    wants_hf_judge = str(getattr(state.settings, "judge_backend", "ollama")).strip().lower() == "hf"
    # If judge backend is Ollama, judge is available regardless of Transformers.
    state.judge_ready = not wants_hf_judge
    if not (state.settings.enable_ner or state.settings.enable_nli or wants_hf_judge):
        return

    try:
        from transformers import pipeline  # type: ignore

        if state.settings.enable_ner:
            try:
                state.biomed_ner = pipeline(
                    "ner",
                    model=state.settings.transformer_ner_model,
                    aggregation_strategy="simple",
                    device=state.settings.transformers_device,
                )
                state.ner_ready = True
            except Exception:
                state.biomed_ner = None
                state.ner_ready = False

        if state.settings.enable_nli:
            try:
                state.nli_pipeline = pipeline(
                    "text-classification",
                    model=state.settings.nli_model,
                    device=state.settings.transformers_device,
                )
                state.nli_ready = True
            except Exception:
                state.nli_pipeline = None
                state.nli_ready = False

        if wants_hf_judge:
            try:
                state.judge_t2t = pipeline(
                    "text2text-generation",
                    model=state.settings.hf_judge_model,
                    device=state.settings.transformers_device,
                )
                state.judge_ready = True
            except Exception:
                state.judge_t2t = None
                state.judge_ready = False

    except Exception:
        state.biomed_ner = None
        state.nli_pipeline = None
        state.judge_t2t = None
        state.ner_ready = False
        state.nli_ready = False
        state.judge_ready = False


def _safe_load_kb_docs(settings: Settings) -> tuple[List[Dict[str, Any]], str | None]:
    try:
        # If KB is extremely large and no explicit cap is set, apply a conservative default.
        # This prevents runaway memory usage from multi-GB JSON arrays.
        max_docs = settings.kb_max_docs
        if max_docs is None:
            try:
                size_b = settings.kb_file.stat().st_size
                if size_b > 1024 * 1024 * 1024:  # > 1GB
                    max_docs = 200_000
                    print(
                        "⚠️ KB is very large (>1GB). Applying default KB_MAX_DOCS=200000 for safety. "
                        "Set KB_MAX_DOCS in .env to override."
                    )
            except Exception:
                pass

        kb_docs = load_kb([str(settings.kb_file)], max_docs=max_docs)
        err = None
    except FileNotFoundError:
        print(f"⚠️ KB file not found: {settings.kb_file}. Starting with empty KB.")
        kb_docs = []
        err = f"KB file not found: {settings.kb_file}"
    except Exception as e:
        print(f"⚠️ Failed to load KB from {settings.kb_file}: {e}. Starting with empty KB.")
        kb_docs = []
        err = str(e)

    print(f"📚 KB loaded docs: {len(kb_docs)} | KB_FILE={settings.kb_file}")
    return kb_docs, err


def _load_or_build_kb_index(
    *,
    settings: Settings,
    kb_docs: List[Dict[str, Any]],
    force_reindex: bool,
):
    if len(kb_docs) == 0:
        print("ℹ️ Skipping KB indexing because KB is empty.")
        embedder = SentenceTransformer(settings.embedding_model_name)
        index = faiss.IndexFlatIP(settings.embedding_dim)
        metadata: List[Dict[str, Any]] = []
        return index, metadata, embedder

    index_path = settings.embeddings_dir / settings.kb_index_filename
    meta_path = settings.embeddings_dir / settings.kb_meta_filename

    if (not force_reindex) and index_path.exists() and meta_path.exists():
        print(f"✅ Loading existing KB FAISS index from {index_path}")
        index, metadata = load_kb_faiss_index(index_path=index_path, meta_path=meta_path)
        embedder = SentenceTransformer(settings.embedding_model_name)
        return index, metadata, embedder

    print(f"🔧 Building KB FAISS index for {len(kb_docs)} docs using {settings.embedding_model_name}...")
    index, metadata, embedder = build_kb_faiss_index(
        docs=kb_docs,
        embedding_model_name=settings.embedding_model_name,
        embedding_dim=settings.embedding_dim,
        index_path=index_path,
        meta_path=meta_path,
        batch_size=settings.embedding_batch_size,
    )
    print(f"✅ KB index saved: {index_path}")
    return index, metadata, embedder


def _load_or_build_scope_index(*, state: AppState, force_reindex: bool) -> None:
    settings = state.settings
    if len(state.kb_docs) == 0:
        print("ℹ️ Skipping scope index because KB is empty.")
        state.scope_index = faiss.IndexFlatIP(settings.embedding_dim)
        state.scope_meta = []
        return

    if (not force_reindex) and (state.scope_index_path.exists() and state.scope_meta_path.exists()):
        print(f"✅ Loading existing scope index from {state.scope_index_path}")
        state.scope_index, state.scope_meta = load_scope_index(
            index_path=state.scope_index_path,
            meta_path=state.scope_meta_path,
        )
        return

    print("🔧 Building scope index (passage-level) from KB...")
    state.scope_index, state.scope_meta = build_scope_index_from_kb(
        kb_docs=state.kb_docs,
        embedder=state.embedder,
        embedding_dim=settings.embedding_dim,
        index_path=state.scope_index_path,
        meta_path=state.scope_meta_path,
        max_sentences_per_passage=settings.scope_max_sentences_per_passage,
        nlp_sci=state.nlp_sci,
    )
    print(f"✅ Scope index saved: {state.scope_index_path}")


def build_state(settings: Settings, *, force_reindex: bool = False) -> AppState:
    ensure_dir(settings.embeddings_dir)

    kb_docs, kb_err = _safe_load_kb_docs(settings)

    index = None
    metadata: List[Dict[str, Any]] = []

    embedder: Optional[SentenceTransformer] = None

    index, metadata, embedder = _load_or_build_kb_index(
        settings=settings,
        kb_docs=kb_docs,
        force_reindex=force_reindex,
    )

    state = AppState(
        settings=settings,
        kb_docs=kb_docs,
        metadata=metadata,
        embedder=embedder,
        index=index,
        scope_index=None,
        scope_meta=[],
        ollama=OllamaClient(settings.ollama_host),
        kb_load_error=kb_err,
    )

    # Optional models
    _try_load_spacy_models(state)
    _try_load_transformers_pipelines(state)

    _load_or_build_scope_index(state=state, force_reindex=force_reindex)

    return state
