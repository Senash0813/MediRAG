from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_kb_faiss_index(
    *,
    docs: List[Dict[str, Any]],
    embedding_model_name: str,
    embedding_dim: int,
    index_path: Path,
    meta_path: Path,
    batch_size: int = 64,
) -> Tuple[faiss.Index, List[Dict[str, Any]], SentenceTransformer]:
    """Build and persist FAISS index + metadata, returning (index, metadata, embedder)."""

    embedder = SentenceTransformer(embedding_model_name)

    index = faiss.IndexFlatIP(embedding_dim)

    # Batch to avoid huge peak memory on large KBs
    try:
        from tqdm.auto import tqdm  # type: ignore

        iterator = tqdm(docs, desc="Indexing KB", unit="doc")
    except Exception:
        iterator = docs

    metadata: List[Dict[str, Any]] = []
    buf_texts: List[str] = []
    buf_docs: List[Dict[str, Any]] = []

    for d in iterator:
        buf_texts.append(d["text"])
        buf_docs.append(d)

        if len(buf_texts) >= int(batch_size):
            embs = embedder.encode(
                buf_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            index.add(embs.astype("float32"))
            metadata.extend(
                [{"id": x["id"], "text": x["text"], "meta": x.get("meta", {})} for x in buf_docs]
            )
            buf_texts.clear()
            buf_docs.clear()

    if buf_texts:
        embs = embedder.encode(
            buf_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        index.add(embs.astype("float32"))
        metadata.extend([{"id": x["id"], "text": x["text"], "meta": x.get("meta", {})} for x in buf_docs])

    faiss.write_index(index, str(index_path))

    # Avoid constructing a gigantic string in memory for large KBs
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    return index, metadata, embedder


def load_kb_faiss_index(*, index_path: Path, meta_path: Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata


def split_sentences_regex(text: str) -> List[str]:
    import re

    sent_split = re.compile(r"(?<=[.!?])\s+")
    return [s.strip() for s in sent_split.split(text or "") if s.strip()]


def chunk_into_passages(
    *,
    text: str,
    max_sentences: int,
    nlp_sci=None,
) -> List[str]:
    """Split doc text into small passages; tries SciSpacy if provided."""

    sents: List[str] = []
    try:
        if nlp_sci is not None:
            doc = nlp_sci(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    except Exception:
        sents = []

    if not sents:
        sents = split_sentences_regex(text)

    chunks: List[str] = []
    for i in range(0, len(sents), max_sentences):
        chunk = " ".join(sents[i : i + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_scope_index_from_kb(
    *,
    kb_docs: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    embedding_dim: int,
    index_path: Path,
    meta_path: Path,
    max_sentences_per_passage: int,
    nlp_sci=None,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    passages: List[str] = []
    scope_meta: List[Dict[str, Any]] = []

    for d in kb_docs:
        doc_id = d.get("id")
        doc_text = d.get("text", "")
        doc_meta = d.get("meta", {})

        chunks = chunk_into_passages(
            text=doc_text,
            max_sentences=max_sentences_per_passage,
            nlp_sci=nlp_sci,
        )

        for j, chunk in enumerate(chunks):
            passages.append(chunk)
            scope_meta.append(
                {
                    "doc_id": doc_id,
                    "passage_id": f"{doc_id}_p{j}",
                    "text": chunk,
                    "meta": doc_meta,
                }
            )

    if not passages:
        raise ValueError("Scope index build failed: no passages created")

    embs = embedder.encode(passages, convert_to_numpy=True, normalize_embeddings=True)

    scope_index = faiss.IndexFlatIP(embedding_dim)
    scope_index.add(np.asarray(embs, dtype="float32"))

    faiss.write_index(scope_index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(scope_meta, f, ensure_ascii=False)

    return scope_index, scope_meta


def load_scope_index(*, index_path: Path, meta_path: Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    scope_index = faiss.read_index(str(index_path))
    scope_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return scope_index, scope_meta


def faiss_search(*, index: faiss.Index, query_emb: np.ndarray, top_k: int) -> Tuple[List[float], List[int]]:
    D, I = index.search(np.asarray(query_emb, dtype="float32"), top_k)
    scores = [float(s) for s in (D[0] if len(D) else [])]
    idxs = [int(i) for i in (I[0] if len(I) else [])]
    return scores, idxs
