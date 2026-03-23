from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from medirag.data.faiss_index import FaissAssets
from medirag.retrieval.embedder import Embedder


def retrieve_top_n_unique(
	query: str,
	*,
	n: int,
	oversample: int,
	embedder: Embedder,
	faiss_assets: FaissAssets,
) -> List[Dict[str, Any]]:
	"""Retrieve top-n unique documents using FAISS and semantic similarity.

	Mirrors the notebook's retrieve_top_n_unique but uses injected assets.
	"""

	q_emb = embedder.embed_queries([query])

	scores, indices = faiss_assets.index.search(q_emb, n * oversample)

	seen_papers: Dict[Any, Dict[str, Any]] = {}

	passages = faiss_assets.passages

	for score, idx in zip(scores[0], indices[0]):
		if idx < 0 or idx >= len(passages):
			continue

		doc = dict(passages[int(idx)])
		doc["semantic_score"] = float(score)
		pid = doc.get("paper_id")

		if pid not in seen_papers or score > seen_papers[pid]["semantic_score"]:
			seen_papers[pid] = doc

		if len(seen_papers) >= n:
			break

	return list(seen_papers.values())

