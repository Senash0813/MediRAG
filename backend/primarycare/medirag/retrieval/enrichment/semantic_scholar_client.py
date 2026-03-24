from __future__ import annotations

import re
import time
from typing import Any, Dict
from difflib import SequenceMatcher

import requests
import numpy as np

from medirag.config import AppSettings
from medirag.data.loaders import save_semantic_scholar_cache
from medirag.retrieval.embedder import Embedder


def _norm_title(t: str) -> str:
	t = (t or "").lower()
	t = re.sub(r"[^a-z0-9\s]", " ", t)
	t = re.sub(r"\s+", " ", t).strip()
	return t


def _token_jaccard(a: str, b: str) -> float:
	A = set(_norm_title(a).split())
	B = set(_norm_title(b).split())
	if not A or not B:
		return 0.0
	return len(A & B) / len(A | B)


def compute_title_similarity(local_title: str, s2_title: str) -> float:
	"""
	Compute title similarity using hybrid approach.
	
	Combines:
	- 60% token overlap (Jaccard similarity)
	- 40% character-level similarity (Levenshtein ratio)
	
	This provides more granular scores (0.0-1.0) compared to pure Jaccard,
	which tends to round to 1.0 for similar titles with all matching tokens.
	
	Args:
		local_title: Title from local database
		s2_title: Title from Semantic Scholar
	
	Returns:
		float: Similarity score between 0.0 and 1.0
	"""
	if not local_title or not s2_title:
		return 0.0
	
	# Normalize titles
	t1 = _norm_title(local_title)
	t2 = _norm_title(s2_title)
	
	if not t1 or not t2:
		return 0.0
	
	# Token-based Jaccard similarity (catches word-level matches)
	tokens1 = set(t1.split())
	tokens2 = set(t2.split())
	if not tokens1 or not tokens2:
		return 0.0
	
	jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
	
	# Character-level Levenshtein ratio (catches typos, small differences)
	levenshtein = SequenceMatcher(None, t1, t2).ratio()
	
	# Weighted combination: 60% token overlap + 40% character similarity
	hybrid_score = 0.6 * jaccard + 0.4 * levenshtein
	
	return round(hybrid_score, 3)


def _extract_external_ids(s2_doc: Dict[str, Any]) -> set[str]:
	ext = s2_doc.get("externalIds") or {}
	out: set[str] = set()
	for k, v in ext.items():
		if v:
			out.add(f"{k}:{str(v).strip().lower()}")
	return out


def _local_external_ids(local_doc: Dict[str, Any]) -> set[str]:
	out: set[str] = set()
	for k in ("doi", "pmid", "arxiv_id", "pubmed_id"):
		v = local_doc.get(k)
		if v:
			out.add(f"{k}:{str(v).strip().lower()}")
	return out


def verify_document(local_doc: Dict[str, Any], s2_doc: Dict[str, Any], query: str = "", embedder: Embedder | None = None) -> Dict[str, Any]:
	local_title = local_doc.get("paper_title", "")
	s2_title = s2_doc.get("title") or ""

	# Compute query-to-title semantic similarity to show answerability
	if query and embedder and local_title:
		try:
			# Use embed_queries which already normalizes embeddings
			query_emb = embedder.embed_queries([query])[0]
			title_emb = embedder.embed_queries([local_title])[0]
			title_similarity = float(np.dot(query_emb, title_emb))
		except Exception:
			title_similarity = 0.0
	else:
		# Fallback to local-to-S2 title matching if query/embedder not provided
		title_similarity = compute_title_similarity(local_title, s2_title)

	s2_ids = _extract_external_ids(s2_doc)
	local_ids = _local_external_ids(local_doc)
	id_match = bool(s2_ids and local_ids and (s2_ids & local_ids))

	# More strict threshold: 0.70 instead of 0.55 for better precision
	title_match = title_similarity >= 0.70
	verification: Dict[str, Any] = {
		"title_match": title_match,
		"title_similarity": title_similarity,
		"external_id_match": id_match,
	}

	citations = s2_doc.get("citationCount", 0)
	verification["authority_level"] = (
		"high" if citations >= 100 else "medium" if citations >= 20 else "low"
	)

	year = s2_doc.get("year")
	if year:
		age = 2025 - year
		verification["freshness"] = (
			"current" if age <= 5 else "acceptable" if age <= 10 else "outdated"
		)
	else:
		verification["freshness"] = "unknown"

	verification["influential"] = s2_doc.get("influentialCitationCount", 0) >= 10
	return verification


def fallback_s2_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"title": doc.get("paper_title"),
		"year": doc.get("year"),
		"venue": doc.get("venue"),
		"citationCount": 0,
		"influentialCitationCount": 0,
		"publicationTypes": [],
		"_fallback": True,
	}


def _extract_corpus_id(paper_url: str | None) -> str | None:
	if not paper_url or "CorpusID:" not in paper_url:
		return None
	return paper_url.split("CorpusID:")[-1]


def fetch_s2_metadata(
	doc: Dict[str, Any],
	*,
	settings: AppSettings,
	cache: Dict[str, Any],
) -> Dict[str, Any]:
	corpus_id = _extract_corpus_id(doc.get("paper_url"))
	if not corpus_id:
		return fallback_s2_metadata(doc)

	key = f"CorpusID:{corpus_id}"
	if key in cache:
		return cache[key]

	url = f"{settings.s2_base_url}/{key}"
	headers: Dict[str, str] = {}
	if settings.s2_api_key:
		headers["x-api-key"] = settings.s2_api_key

	params = {
		"fields": ",".join(
			[
				"title",
				"year",
				"venue",
				"citationCount",
				"influentialCitationCount",
				"publicationTypes",
				"externalIds",
			]
		)
	}

	resp = requests.get(url, headers=headers, params=params, timeout=30)
	time.sleep(1.05)

	if resp.status_code != 200:
		return fallback_s2_metadata(doc)

	js = resp.json()
	cache[key] = js
	save_semantic_scholar_cache(settings, cache)

	return js

