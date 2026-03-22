from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .embeddings_store import faiss_search


def scope_check_similarity(
    *,
    query: str,
    embedder: SentenceTransformer,
    scope_index,
    top_k: int,
) -> Tuple[float, float, List[float], List[int]]:
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = faiss_search(index=scope_index, query_emb=q_emb.astype("float32"), top_k=top_k)

    top1 = scores[0] if scores else 0.0
    avgk = (sum(scores) / len(scores)) if scores else 0.0

    return top1, avgk, scores, idxs


def scope_check_cohesion(
    *,
    idxs: List[int],
    embedder: SentenceTransformer,
    scope_meta: List[dict],
    k: int,
) -> float:
    idxs = [i for i in idxs[:k] if 0 <= i < len(scope_meta)]
    if len(idxs) < 2:
        return 0.0

    texts = [scope_meta[i]["text"] for i in idxs]
    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    sims: List[float] = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sims.append(float(np.dot(embs[i], embs[j])))

    return float(sum(sims) / len(sims)) if sims else 0.0


def domain_check(
    *,
    query: str,
    embedder: SentenceTransformer,
    scope_index,
    scope_meta: List[dict],
    min_top1: float,
    min_avg_topk: float,
    min_cohesion: float,
    top_k: int,
) -> Tuple[bool, Dict[str, Any]]:
    top1, avgk, scores, idxs = scope_check_similarity(
        query=query,
        embedder=embedder,
        scope_index=scope_index,
        top_k=top_k,
    )

    if (top1 < min_top1) or (avgk < min_avg_topk):
        return False, {
            "reason": "low_similarity",
            "top1": top1,
            "avg_topk": avgk,
            "scores_top5": scores[:5],
        }

    cohesion = scope_check_cohesion(
        idxs=idxs,
        embedder=embedder,
        scope_meta=scope_meta,
        k=min(5, top_k),
    )

    if cohesion < min_cohesion:
        return False, {
            "reason": "low_cohesion",
            "top1": top1,
            "avg_topk": avgk,
            "cohesion": cohesion,
            "scores_top5": scores[:5],
        }

    return True, {
        "reason": "in_domain",
        "top1": top1,
        "avg_topk": avgk,
        "cohesion": cohesion,
        "scores_top5": scores[:5],
    }
