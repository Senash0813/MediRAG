from __future__ import annotations

from typing import Dict

import numpy as np

from medirag.domain.models import ScopeDecision
from medirag.retrieval.embedder import Embedder


def check_specialty_scope(
	query: str,
	*,
	embedder: Embedder,
	specialty_centroids: Dict[str, np.ndarray],
	threshold: float,
) -> ScopeDecision:
	"""Compute similarity of query to specialty centroids and decide in-scope.

	Mirrors the notebook's check_specialty_scope logic but uses injected
	Embedder and centroids instead of globals.
	"""

	q_emb = embedder.embed_queries([query])[0]

	best_specialty = None
	best_score = -1.0

	for specialty, centroid in specialty_centroids.items():
		if centroid is None:
			continue

		centroid_arr = np.asarray(centroid, dtype=np.float32)
		score = float(np.dot(q_emb, centroid_arr))

		if score > best_score:
			best_score = score
			best_specialty = specialty

	in_scope = best_score >= float(threshold)

	# Simple debug print to help tune scope_threshold during development.
	print(
		f"[SCOPE] query='{query}' best_specialty={best_specialty} "
		f"score={best_score:.3f} threshold={threshold} in_scope={in_scope}",
	)

	return ScopeDecision(in_scope=in_scope, best_specialty=best_specialty, score=best_score)

