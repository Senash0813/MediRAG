from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer


def build_faiss_index(
    docs: List[Dict[str, Any]],
    embedding_model_name: str,
    index_path: str,
    meta_path: str,
    embedding_dim: int,
) -> faiss.Index:
    # 1) load embedding model
    print("Loading embedding model:", embedding_model_name)
    embedder = SentenceTransformer(embedding_model_name)

    # 2) build embeddings in batches
    texts = [d["text"] for d in docs]
    print(f"Computing embeddings for {len(texts)} docs...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 3) create faiss index (IndexFlatIP since we normalized embeddings => inner product equals cosine)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings.astype("float32"))
    print("FAISS index size:", index.ntotal)

    # 4) save index and metadata mapping
    faiss.write_index(index, index_path)
    # save metadata: ids and original texts + meta
    metadata = [{"id": d["id"], "text": d["text"], "meta": d["meta"]} for d in docs]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved index to {index_path} and metadata to {meta_path}")

    return index


def load_index_and_metadata(index_path: str, meta_path: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata
