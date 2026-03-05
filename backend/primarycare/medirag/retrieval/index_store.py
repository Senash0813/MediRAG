from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalAssets:
    embedder: SentenceTransformer
    index: faiss.Index


def load_or_build_faiss_index(
    *,
    embed_model_name: str,
    passages: List[str],
    emb_path: Path,
    index_path: Path,
) -> RetrievalAssets:
    """Structural refactor of the notebook's embedder/index cell.

    Logic intentionally preserved:
    - SentenceTransformer embedder
    - normalize_embeddings=True
    - IndexFlatIP
    - Load cached artifacts if both paths exist
    """

    embedder = SentenceTransformer(embed_model_name)

    if emb_path.exists() and index_path.exists():
        embeddings = np.load(str(emb_path))
        index = faiss.read_index(str(index_path))
        _ = embeddings  # kept to preserve notebook parity (debugging / sanity) without changing behavior
        return RetrievalAssets(embedder=embedder, index=index)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = embedder.encode(
        passages,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(str(emb_path), embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    return RetrievalAssets(embedder=embedder, index=index)
