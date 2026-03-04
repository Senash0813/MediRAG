from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
	"""Thin wrapper around SentenceTransformer for query and passage embeddings."""

	def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
		self._model = SentenceTransformer(model_name)

	def embed_queries(self, texts: List[str]) -> np.ndarray:
		return self._model.encode(
			texts,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)

	def embed_passages(self, texts: List[str]) -> np.ndarray:
		return self._model.encode(
			texts,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=True,
		)

