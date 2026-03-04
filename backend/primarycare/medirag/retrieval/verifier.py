from __future__ import annotations

from typing import Any, Dict, List

from tqdm import tqdm

from medirag.config import AppSettings
from medirag.domain.models import VerifiedDoc
from medirag.domain.scoring import (
	assign_risk_flags,
	compute_evidence_score,
	final_doc_score,
	get_quality_tier,
)
from medirag.retrieval.enrichment.semantic_scholar_client import (
	fetch_s2_metadata,
	verify_document,
)
from medirag.retrieval.embedder import Embedder
from medirag.data.faiss_index import FaissAssets
from medirag.retrieval.retriever import retrieve_top_n_unique


def compute_query_title_similarity(query: str, title: str, embedder: Embedder) -> float:
	"""Compute semantic similarity between query and paper title.
	
	This shows how well the paper title indicates it can answer the query.
	Used in Phase-1 to demonstrate title-level answerability.
	"""
	if not title or not query:
		return 0.0
	
	# Encode both query and title (already normalized by embed_queries)
	query_emb = embedder.embed_queries([query])[0]
	title_emb = embedder.embed_queries([title])[0]
	
	# Compute cosine similarity (dot product of normalized vectors)
	similarity = float(query_emb @ title_emb)
	
	return round(similarity, 3)


def verify_and_rank_final_flat(
	query: str,
	*,
	settings: AppSettings,
	embedder: Embedder,
	faiss_assets: FaissAssets,
	s2_cache: Dict[str, Any],
	semantic_n: int,
	final_k: int,
) -> List[VerifiedDoc]:
	"""End-to-end retrieval, enrichment, and quality reranking.

	Ports the notebook's verify_and_rank_final_flat using the structured
	components in this package.
	"""

	retrieved = retrieve_top_n_unique(
		query,
		n=semantic_n,
		oversample=3,
		embedder=embedder,
		faiss_assets=faiss_assets,
	)

	verified_docs: List[VerifiedDoc] = []

	for doc in tqdm(retrieved, disable=True):
		s2_meta = fetch_s2_metadata(doc, settings=settings, cache=s2_cache)

		pub_types = s2_meta.get("publicationTypes") or []
		evidence_score = compute_evidence_score(pub_types)

		verification = verify_document(doc, s2_meta)
		verification["evidence_score"] = evidence_score

		# Compute query-to-title similarity for Phase-1 answerability
		paper_title = s2_meta.get("title") or doc.get("paper_title") or ""
		query_title_similarity = compute_query_title_similarity(query, paper_title, embedder)
		
		# Compute combined answerability score (Phase-1 dual-level assessment)
		# Weight: 70% passage-level, 30% title-level
		semantic_score = float(doc.get("semantic_score", 0.0) or 0.0)
		answerability_score = (0.7 * semantic_score) + (0.3 * query_title_similarity)

		final_score_val = final_doc_score(
			answerability_score,
			citation_count=int(s2_meta.get("citationCount", 0) or 0),
			year=s2_meta.get("year"),
			influential_citation_count=int(s2_meta.get("influentialCitationCount", 0) or 0),
			evidence_score=evidence_score,
			title_match=bool(verification.get("title_match")),
			external_id_match=bool(verification.get("external_id_match")),
		)

		quality_tier = get_quality_tier(final_score_val)

		flattened = VerifiedDoc(
			qa_id=doc.get("qa_id"),
			paper_id=doc.get("paper_id"),
			title=paper_title,
			passage_text=doc.get("passage_text"),
			semantic_score=round(semantic_score, 3),
			final_score=final_score_val,
			quality_tier=quality_tier,
			citation_count=int(s2_meta.get("citationCount", 0) or 0),
			influential_citation_count=int(s2_meta.get("influentialCitationCount", 0) or 0),
			year=s2_meta.get("year"),
			publication_types=pub_types,
			evidence_score=evidence_score,
			risk_flags=[],  # filled below
			title_match=bool(verification.get("title_match")),
			title_similarity=query_title_similarity,
			answerability_score=round(answerability_score, 3),
			external_id_match=bool(verification.get("external_id_match")),
			authority_level=str(verification.get("authority_level", "low")),
			freshness=str(verification.get("freshness", "unknown")),
			influential=bool(verification.get("influential")),
		)

		# assign deterministic risk flags
		flattened.risk_flags = assign_risk_flags(
			{
				"freshness": flattened.freshness,
				"citation_count": flattened.citation_count,
				"evidence_score": flattened.evidence_score,
				"quality_tier": flattened.quality_tier,
			}
		)

		verified_docs.append(flattened)

	verified_docs.sort(key=lambda x: x.final_score, reverse=True)
	return verified_docs[:final_k]

