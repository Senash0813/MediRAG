from __future__ import annotations

from typing import Any, Dict, List


def retrieve_top_k_unique(*, query: str, data: List[Dict[str, Any]], embedder, index, k: int, oversample: int = 3):
    """Notebook function `retrieve_top_k_unique` moved verbatim (logic preserved)."""

    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb, k * oversample)

    seen_papers = {}

    for score, idx in zip(scores[0], indices[0]):
        doc = data[idx]
        pid = doc["paper_id"]

        if pid not in seen_papers or score > seen_papers[pid]["semantic_score"]:
            seen_papers[pid] = {**doc, "semantic_score": float(score)}

        if len(seen_papers) >= k:
            break

    return list(seen_papers.values())
