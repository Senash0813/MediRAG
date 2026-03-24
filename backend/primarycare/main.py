from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medirag.api.schemas import (
	QueryRequest,
	QueryResponse,
	RetrieveDocsRequest,
	RetrieveDocsResponse,
	RetrievedDocument,
	DetailedPipelineResponse,
	BatchQueryRequest,
	BatchQueryResponse,
	BatchQueryItem,
)
from medirag.config import load_settings
from medirag.pipeline.orchestrator import init_assets, run_pipeline, run_detailed_pipeline
from medirag.retrieval.verifier import verify_and_rank_final_flat


app = FastAPI(title="MediRAG Backend")


app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		"http://localhost:3000",
		"http://127.0.0.1:3000",
		"null",  # Allow file:// protocol for animated_demo.html
	],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


logger = logging.getLogger(__name__)

_assets = None


@app.on_event("startup")
def _startup() -> None:
	global _assets
	settings = load_settings()
	_assets = init_assets(settings)


@app.post("/query4", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
	try:
		result = run_pipeline(assets=_assets, query=req.query, k=req.top_k)
		return QueryResponse(**result)
	except Exception:  # pragma: no cover - defensive logging
		logger.exception("/query4 failed")
		return QueryResponse(
			query=req.query,
			direct_answer=(
				"The problem is out of scope right now (or the LLM request was rejected). "
				"Please try again, or select a different question."
			),
			evidence_summary="",
			limitations="",
		)


@app.post("/retrieve", response_model=RetrieveDocsResponse)
def retrieve_documents(req: RetrieveDocsRequest) -> RetrieveDocsResponse:
	"""Retrieve and return top documents for a query without generating an answer."""
	try:
		verified_docs = verify_and_rank_final_flat(
			query=req.query,
			settings=_assets.settings,
			embedder=_assets.embedder,
			faiss_assets=_assets.faiss_assets,
			s2_cache=_assets.s2_cache,
			semantic_n=req.semantic_n,
			final_k=req.final_k,
		)

		documents = [
			RetrievedDocument(
				qa_id=str(doc.qa_id),
				paper_id=str(doc.paper_id),
				title=doc.title,
				passage_text=doc.passage_text,
				semantic_score=doc.semantic_score,
				final_score=doc.final_score,
				quality_tier=doc.quality_tier,
				citation_count=doc.citation_count,
				influential_citation_count=doc.influential_citation_count,
				year=doc.year,
				publication_types=doc.publication_types,
				evidence_score=doc.evidence_score,
				risk_flags=doc.risk_flags,
				title_match=doc.title_match,
				title_similarity=doc.title_similarity,
				external_id_match=doc.external_id_match,
				authority_level=doc.authority_level,
				freshness=doc.freshness,
				influential=doc.influential,
			)
			for doc in verified_docs
		]

		return RetrieveDocsResponse(
			query=req.query,
			documents=documents,
			count=len(documents),
		)
	except Exception:  # pragma: no cover - defensive logging
		logger.exception("/retrieve failed")
		return RetrieveDocsResponse(
			query=req.query,
			documents=[],
			count=0,
		)


@app.post("/retrieve/detailed", response_model=DetailedPipelineResponse)
def retrieve_detailed(req: RetrieveDocsRequest) -> DetailedPipelineResponse:
	"""Retrieve with full pipeline details for demo purposes."""
	try:
		result = run_detailed_pipeline(
			assets=_assets,
			query=req.query,
			semantic_n=req.semantic_n,
			final_k=req.final_k,
		)
		return DetailedPipelineResponse(**result)
	except Exception:  # pragma: no cover - defensive logging
		logger.exception("/retrieve/detailed failed")
		# Return minimal error response
		return DetailedPipelineResponse(
			query=req.query,
			scope_gate={
				"in_scope": False,
				"best_specialty": None,
				"score": 0.0,
				"threshold": 0.0,
			},
			retrieval=[],
			phase1_validation=[],
			phase2_validation=[],
			verification_level=1,
			instructor_prompt={
				"verification_level": 1,
				"answer_mode": "refuse",
				"required_sections": [],
				"constraints": [],
				"context_plan": [],
				"rendered_prompt": "",
			},
			final_answer={
				"direct_answer": "An error occurred during pipeline execution.",
				"evidence_summary": "",
				"limitations": "- System error occurred.",
			},
		)


@app.post("/batch-query", response_model=BatchQueryResponse)
def batch_query(req: BatchQueryRequest) -> BatchQueryResponse:
	"""Process multiple queries at once and return question-answer pairs with context."""
	results = []
	successful_count = 0
	
	for query_text in req.queries:
		try:
			# Get the answer from pipeline
			result = run_pipeline(assets=_assets, query=query_text, k=req.top_k)
			
			# Get context passages
			verified_docs = verify_and_rank_final_flat(
				query=query_text,
				settings=_assets.settings,
				embedder=_assets.embedder,
				faiss_assets=_assets.faiss_assets,
				s2_cache=_assets.s2_cache,
				semantic_n=_assets.settings.semantic_top_n,
				final_k=req.top_k,
			)
			
			# Extract passage texts as context
			context_passages = [doc.passage_text for doc in verified_docs if doc.passage_text]
			
			results.append(
				BatchQueryItem(
					question=query_text,
					answer=result["direct_answer"],
					context=context_passages,
				)
			)
			successful_count += 1
		except Exception:
			logger.exception(f"Failed to process query: {query_text}")
			results.append(
				BatchQueryItem(
					question=query_text,
					answer="Error: Unable to process this query.",
					context=[],
				)
			)
	
	return BatchQueryResponse(
		results=results,
		total_queries=len(req.queries),
		successful=successful_count,
	)



