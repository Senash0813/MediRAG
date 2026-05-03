from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from medirag.config import AppSettings
from medirag.data.faiss_index import FaissAssets, load_or_build_faiss_assets
from medirag.data.loaders import (
	load_passages,
	load_semantic_scholar_cache,
	load_specialty_centroids,
)
from medirag.domain.scope_gate import check_specialty_scope
from medirag.llm.answer import run_answer_llm
from medirag.llm.client import LMStudioClient
from medirag.llm.instructor import run_instructor_llm
from medirag.llm.prompts import compute_verification_level, render_final_prompt, select_prompt_template, _doc_id
from medirag.retrieval.embedder import Embedder
from medirag.retrieval.verifier import verify_and_rank_final_flat
from medirag.retrieval.retriever import retrieve_top_n_unique
from medirag.retrieval.enrichment.semantic_scholar_client import fetch_s2_metadata, verify_document
from medirag.domain.scoring import compute_evidence_score, final_doc_score, get_quality_tier, assign_risk_flags
from medirag.domain.models import VerifiedDoc, InstructionObject
from tqdm import tqdm


@dataclass
class PipelineAssets:
	settings: AppSettings
	passages: List[Dict[str, Any]]
	specialty_centroids: Dict[str, Any]
	embedder: Embedder
	faiss_assets: FaissAssets
	s2_cache: Dict[str, Any]
	lm_client: LMStudioClient


def init_assets(settings: AppSettings) -> PipelineAssets:
	"""Initialize and return shared pipeline assets (one-time on startup)."""

	passages = load_passages(settings)
	centroids, _counts, _model_name = load_specialty_centroids(settings)
	s2_cache = load_semantic_scholar_cache(settings)

	embedder = Embedder()
	faiss_assets = load_or_build_faiss_assets(settings, embedder=embedder, passages=passages)

	lm_client = LMStudioClient(settings)

	return PipelineAssets(
		settings=settings,
		passages=passages,
		specialty_centroids=centroids,
		embedder=embedder,
		faiss_assets=faiss_assets,
		s2_cache=s2_cache,
		lm_client=lm_client,
	)


def _make_simplified_prompt_for_demo(
	query: str,
	verified_docs: List[VerifiedDoc],
	instruction: InstructionObject,
	max_docs: int = 5
) -> str:
	"""Create a simplified rendered prompt for demo purposes.
	
	Shows only plan + rules, replaces full passages with doc headers.
	"""
	template = select_prompt_template(instruction)
	
	# Get the selected documents (same logic as make_context_block)
	planned_ids = [str(item.get("doc_id")) for item in instruction.context_plan if isinstance(item, dict)]
	id_to_doc = {str(_doc_id(d)): d for d in verified_docs}
	
	selected: List[VerifiedDoc] = []
	seen: set[str] = set()
	
	for pid in planned_ids:
		d = id_to_doc.get(str(pid))
		if d is not None and str(_doc_id(d)) not in seen:
			selected.append(d)
			seen.add(str(_doc_id(d)))
	
	for d in verified_docs:
		if len(selected) >= max_docs:
			break
		did = str(_doc_id(d))
		if did not in seen:
			selected.append(d)
			seen.add(did)
	
	if not selected:
		selected = verified_docs[:max_docs or 3]
	
	# Create simplified context block (headers only, no passages)
	simplified_blocks: List[str] = []
	for d in selected:
		did = _doc_id(d)
		simplified_blocks.append(
			f"[DOC_ID: {did}]\n"
			f"TITLE: {d.title}\n"
			f"YEAR: {d.year}\n"
			f"FINAL_SCORE: {d.final_score}  QUALITY_TIER: {d.quality_tier}\n"
			f"CITATIONS: {d.citation_count}  INFLUENTIAL: {d.influential_citation_count}\n"
			f"PUB_TYPES: {d.publication_types}\n"
			f"EVIDENCE_SCORE: {d.evidence_score}\n"
			f"RISK_FLAGS: {d.risk_flags}\n"
			f"PASSAGE_TEXT: [Passage included - hidden for demo brevity]"
		)
	
	simplified_context = "\n---\n".join(simplified_blocks)
	
	# Format the template with simplified context
	return template.format(
		query=query,
		required_sections=instruction.required_sections,
		constraints=instruction.constraints,
		context_plan=instruction.context_plan,
		context_block=simplified_context,
	)


def run_pipeline(*, assets: PipelineAssets, query: str, k: int) -> Dict[str, Any]:
	"""Run the end-to-end pipeline for a single query."""

	settings = assets.settings

	# 1) Scope gate
	scope = check_specialty_scope(
		query,
		embedder=assets.embedder,
		specialty_centroids=assets.specialty_centroids,
		threshold=settings.scope_threshold,
	)

	# Base payload always includes scope debug so callers can inspect it
	scope_debug: Dict[str, Any] = {
		"in_scope": scope.in_scope,
		"best_specialty": scope.best_specialty,
		"score": scope.score,
		"threshold": settings.scope_threshold,
	}

	if not scope.in_scope:
		return {
			"query": query,
			"direct_answer": (
				"The query appears to be outside the primary-care domain. "
				"Closest specialty: "
				f"{scope.best_specialty or 'unknown'} (score={scope.score:.3f})."
			),
			"evidence_summary": "",
			"limitations": "- Out-of-domain query; pipeline refused to answer.",
			"debug": {"scope": scope_debug},
		}

	# 2) Retrieve + verify
	verified_docs = verify_and_rank_final_flat(
		query,
		settings=settings,
		embedder=assets.embedder,
		faiss_assets=assets.faiss_assets,
		s2_cache=assets.s2_cache,
		semantic_n=settings.semantic_top_n,
		final_k=k or settings.default_top_k,
	)

	if not verified_docs:
		return {
			"query": query,
			"direct_answer": "No supporting documents were retrieved for this query.",
			"evidence_summary": "",
			"limitations": "- Retrieval returned no results.",
			"debug": {"scope": scope_debug},
		}

	# 3) Instructor LLM
	instruction_obj = run_instructor_llm(query, verified_docs, assets.lm_client)

	# 4) Answer LLM with semantic fallback
	answer_sections = run_answer_llm(
		query=query,
		verified_docs=verified_docs,
		instruction_obj=instruction_obj,
		client=assets.lm_client,
		embedder=assets.embedder,
		enforce_traceability=True,
		semantic_fallback_threshold=0.60,
	)

	return {
		"query": query,
		"direct_answer": answer_sections["direct_answer"],
		"evidence_summary": answer_sections["evidence_summary"],
		"limitations": answer_sections["limitations"],
		"debug": {"scope": scope_debug},
	}


def run_detailed_pipeline(*, assets: PipelineAssets, query: str, semantic_n: int = 10, final_k: int = 5) -> Dict[str, Any]:
	"""Run the end-to-end pipeline with detailed intermediate results for demo."""
	
	settings = assets.settings
	
	# === STAGE 1: Scope Gate ===
	scope = check_specialty_scope(
		query,
		embedder=assets.embedder,
		specialty_centroids=assets.specialty_centroids,
		threshold=settings.scope_threshold,
	)
	
	scope_gate_info = {
		"in_scope": scope.in_scope,
		"best_specialty": scope.best_specialty,
		"score": scope.score,
		"threshold": settings.scope_threshold,
	}
	
	if not scope.in_scope:
		return {
			"query": query,
			"scope_gate": scope_gate_info,
			"retrieval": [],
			"phase1_validation": [],
			"phase2_validation": [],
			"verification_level": 1,
			"instructor_prompt": {
				"verification_level": 1,
				"answer_mode": "refuse",
				"required_sections": [],
				"constraints": [],
				"context_plan": [],
				"rendered_prompt": "",
			},
			"final_answer": {
				"direct_answer": f"The query appears to be outside the primary-care domain. Closest specialty: {scope.best_specialty or 'unknown'} (score={scope.score:.3f}).",
				"evidence_summary": "",
				"limitations": "- Out-of-domain query; pipeline refused to answer.",
				"rendered_prompt": "",
			},
		}
	
	# === STAGE 2: Retrieval ===
	retrieved = retrieve_top_n_unique(
		query,
		n=semantic_n,
		oversample=3,
		embedder=assets.embedder,
		faiss_assets=assets.faiss_assets,
	)
	
	retrieval_docs = [
		{
			"qa_id": str(doc.get("qa_id")),
			"paper_id": str(doc.get("paper_id")),
			"title": doc.get("paper_title", ""),
			"semantic_score": round(float(doc.get("semantic_score", 0.0) or 0.0), 3),
		}
		for doc in retrieved
	]
	
	# === STAGE 3: Phase-1 Support Validation (S2 Enrichment) ===
	# Output: Raw candidate docs + semantic_score + title_similarity + answerability_score
	phase1_docs = []
	for doc in tqdm(retrieved, disable=True):
		s2_meta = fetch_s2_metadata(doc, settings=settings, cache=assets.s2_cache)
		verification = verify_document(doc, s2_meta, query=query, embedder=assets.embedder)
		
		semantic_score = round(float(doc.get("semantic_score", 0.0) or 0.0), 3)
		title_similarity = float(verification.get("title_similarity", 0.0) or 0.0)
		
		# Compute combined answerability score
		# Weight: 70% passage-level, 30% title-level
		answerability_score = (0.7 * semantic_score) + (0.3 * title_similarity)
		
		phase1_docs.append({
			"paper_id": str(doc.get("paper_id")),
			"title": doc.get("paper_title", ""),
			"semantic_score": semantic_score,
			"title_similarity": title_similarity,
			"answerability_score": round(answerability_score, 3),
		})
	
	# Re-rank by combined answerability score
	phase1_docs.sort(key=lambda x: x['answerability_score'], reverse=True)
	
	# === STAGE 4: Phase-2 Authority Validation (Final Scoring) ===
	verified_docs = verify_and_rank_final_flat(
		query,
		settings=settings,
		embedder=assets.embedder,
		faiss_assets=assets.faiss_assets,
		s2_cache=assets.s2_cache,
		semantic_n=semantic_n,
		final_k=final_k,
	)
	
	phase2_docs = [
		{
			# FROM PHASE-1
			"paper_id": str(doc.paper_id),
			"title": doc.title,
			"semantic_score": doc.semantic_score,
			"title_similarity": doc.title_similarity,
			"answerability_score": doc.answerability_score,
			
			# VERIFIED METADATA
			"citation_count": doc.citation_count,
			"influential_citation_count": doc.influential_citation_count,
			"year": doc.year,
			"publication_types": doc.publication_types,
			"title_match": doc.title_match,
			"external_id_match": doc.external_id_match,
			"authority_level": doc.authority_level,
			"freshness": doc.freshness,
			"influential": doc.influential,
			
			# QUALITY SCORES
			"evidence_score": doc.evidence_score,
			"final_score": doc.final_score,
			"quality_tier": doc.quality_tier,
			
			# RISK FLAGS
			"risk_flags": doc.risk_flags,
		}
		for doc in verified_docs
	]
	
	if not verified_docs:
		return {
			"query": query,
			"scope_gate": scope_gate_info,
			"retrieval": retrieval_docs,
			"phase1_validation": phase1_docs,
			"phase2_validation": [],
			"verification_level": 1,
			"instructor_prompt": {
				"verification_level": 1,
				"answer_mode": "refuse",
				"required_sections": [],
				"constraints": [],
				"context_plan": [],
				"rendered_prompt": "",
			},
			"final_answer": {
				"direct_answer": "No supporting documents were retrieved for this query.",
				"evidence_summary": "",
				"limitations": "- Retrieval returned no results.",
				"rendered_prompt": "",
			},
		}
	
	# === STAGE 5: Verification Level ===
	verification_level = compute_verification_level(verified_docs)
	
	# === STAGE 6: Instructor Prompt ===
	instruction_obj = run_instructor_llm(query, verified_docs, assets.lm_client)
	
	# For demo: use simplified prompt (show headers only, hide full passages)
	simplified_prompt = _make_simplified_prompt_for_demo(query, verified_docs, instruction_obj)
	
	instructor_info = {
		"verification_level": instruction_obj.verification_level,
		"answer_mode": instruction_obj.answer_mode,
		"required_sections": instruction_obj.required_sections,
		"constraints": instruction_obj.constraints,
		"context_plan": instruction_obj.context_plan,
		"rendered_prompt": simplified_prompt,
	}
	
	# === STAGE 7: Final Grounded Answer ===
	answer_sections = run_answer_llm(
		query=query,
		verified_docs=verified_docs,
		instruction_obj=instruction_obj,
		client=assets.lm_client,
		embedder=assets.embedder,
		enforce_traceability=True,
		semantic_fallback_threshold=0.60,
	)
	
	final_answer_info = {
		"direct_answer": answer_sections["direct_answer"],
		"evidence_summary": answer_sections["evidence_summary"],
		"limitations": answer_sections["limitations"],
		"rendered_prompt": answer_sections.get("rendered_prompt", ""),
	}
	
	return {
		"query": query,
		"scope_gate": scope_gate_info,
		"retrieval": retrieval_docs,
		"phase1_validation": phase1_docs,
		"phase2_validation": phase2_docs,
		"verification_level": verification_level,
		"instructor_prompt": instructor_info,
		"final_answer": final_answer_info,
	}

