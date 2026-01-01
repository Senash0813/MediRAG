from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

import spacy

from app.core import settings
from app.services.faiss_index import build_faiss_index, load_index_and_metadata
from app.services.kb_loader import load_kb


@dataclass(frozen=True)
class AppResources:
    kb_docs: List[Dict[str, Any]]
    index: Any
    metadata: List[Dict[str, Any]]
    embedder: SentenceTransformer
    rag_generator: Any
    nlp_sci: Any
    nlp_bc5cdr: Any
    biomed_ner: Any
    nli_pipeline: Any
    device: int


_lock = threading.Lock()
_cached: Optional[AppResources] = None
_cached_error: Optional[str] = None


def _resolve_device(requested_device: int) -> int:
    if requested_device >= 0 and not torch.cuda.is_available():
        return -1
    return requested_device


def load_resources() -> AppResources:
    # 1) KB
    kb = load_kb([str(settings.KB_PATH)])

    # 2) FAISS index + metadata
    if not settings.INDEX_PATH.exists() or not settings.META_PATH.exists():
        index = build_faiss_index(
            kb,
            embedding_model_name=settings.EMBEDDING_MODEL_NAME,
            index_path=str(settings.INDEX_PATH),
            meta_path=str(settings.META_PATH),
            embedding_dim=settings.EMBEDDING_DIM,
        )
        metadata = [{"id": d["id"], "text": d["text"], "meta": d["meta"]} for d in kb]
    else:
        index, metadata = load_index_and_metadata(str(settings.INDEX_PATH), str(settings.META_PATH))

    # 3) Models
    device = _resolve_device(settings.DEVICE)

    embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(settings.GENERATOR_MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.GENERATOR_MODEL_NAME)
    rag_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

    nlp_sci = spacy.load("en_core_sci_sm")
    nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")

    biomed_ner = pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        device=device,
    )

    nli_pipeline = pipeline(
        "text-classification",
        model="roberta-large-mnli",
        device=device,
    )

    return AppResources(
        kb_docs=kb,
        index=index,
        metadata=metadata,
        embedder=embedder,
        rag_generator=rag_generator,
        nlp_sci=nlp_sci,
        nlp_bc5cdr=nlp_bc5cdr,
        biomed_ner=biomed_ner,
        nli_pipeline=nli_pipeline,
        device=device,
    )


def get_resources() -> AppResources:
    global _cached, _cached_error
    if _cached is not None:
        return _cached

    with _lock:
        if _cached is not None:
            return _cached

        try:
            _cached = load_resources()
            _cached_error = None
        except Exception as e:
            _cached_error = repr(e)
            raise

        return _cached


def get_resources_error() -> Optional[str]:
    return _cached_error
