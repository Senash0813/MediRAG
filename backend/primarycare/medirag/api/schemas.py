from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
	"""Incoming query payload for /query4."""

	query: str = Field(..., description="Natural language question from the user.")
	top_k: int = Field(5, description="Desired number of final context documents.")


class QueryResponse(BaseModel):
	"""Structured response returned by /query4.

	The three main fields correspond to the post-processed answer sections
	produced by the pipeline.
	"""

	query: str = Field(..., description="Echo of the original query.")
	direct_answer: str = Field(..., description="Primary answer text.")
	evidence_summary: str = Field(..., description="Bullet-style evidence summary.")
	limitations: str = Field(..., description="Limitations and caveats.")
	verification_level: Optional[int] = Field(
		default=None,
		description="Verification level (1-4) reflecting overall evidence strength.",
	)

	# Optional: could later include verified_docs or debug fields if desired.
	debug: Optional[dict] = Field(
		default=None,
		description="Optional debug payload (not populated in normal operation).",
	)


class RetrieveDocsRequest(BaseModel):
	"""Request payload for /retrieve endpoint."""

	query: str = Field(..., description="Natural language question from the user.")
	semantic_n: int = Field(10, description="Number of semantic candidates to retrieve.")
	final_k: int = Field(5, description="Number of final verified documents to return.")


class RetrievedDocument(BaseModel):
	"""Individual document from retrieval results."""

	qa_id: str = Field(..., description="Unique QA document identifier.")
	paper_id: str = Field(..., description="Paper identifier.")
	title: str = Field(..., description="Document title.")
	passage_text: str = Field(..., description="Document passage text.")
	semantic_score: float = Field(..., description="Initial semantic similarity score.")
	final_score: float = Field(..., description="Final ranking score after verification.")
	quality_tier: str = Field(..., description="Quality tier classification.")
	citation_count: int = Field(..., description="Total citation count.")
	influential_citation_count: int = Field(..., description="Influential citation count.")
	year: Optional[int] = Field(None, description="Publication year.")
	publication_types: List[str] = Field(default_factory=list, description="Publication types.")
	evidence_score: float = Field(..., description="Evidence quality score.")
	risk_flags: List[str] = Field(default_factory=list, description="Risk flags identified during verification.")
	title_match: bool = Field(..., description="Whether title matched query.")
	title_similarity: float = Field(..., description="Title similarity score.")
	external_id_match: bool = Field(..., description="Whether external ID matched.")
	authority_level: str = Field(..., description="Authority level classification.")
	freshness: str = Field(..., description="Freshness classification.")
	influential: bool = Field(..., description="Whether paper is influential.")


class RetrieveDocsResponse(BaseModel):
	"""Response containing retrieved and verified documents."""

	query: str = Field(..., description="Echo of the original query.")
	documents: List[RetrievedDocument] = Field(..., description="List of retrieved and verified documents.")
	count: int = Field(..., description="Number of documents returned.")


# === Enhanced Pipeline Debug Response ===

class ScopeGateInfo(BaseModel):
	"""Scope gate stage information."""
	
	in_scope: bool = Field(..., description="Whether query is in scope.")
	best_specialty: Optional[str] = Field(None, description="Best matching specialty.")
	score: float = Field(..., description="Similarity score to best specialty centroid.")
	threshold: float = Field(..., description="Threshold for in-scope decision.")


class RetrievalDocument(BaseModel):
	"""Document from initial retrieval stage."""
	
	qa_id: str
	paper_id: str
	title: str
	semantic_score: float


class Phase1ValidationDoc(BaseModel):
	"""Phase-1: Semantic Candidate Retrieval
	
	Output: Raw candidate docs + semantic_score + title_similarity + answerability_score
	- semantic_score: Passage-level semantic similarity (from FAISS)
	- title_similarity: Query-to-title semantic similarity (shows answerability)
	- answerability_score: Combined answerability score (70% passage + 30% title)
	"""
	
	paper_id: str = Field(..., description="Paper identifier")
	title: str = Field(..., description="Document title")
	semantic_score: float = Field(..., description="Passage-level cosine similarity from FAISS search")
	title_similarity: float = Field(..., description="Query-to-title semantic similarity (indicates how well title suggests paper can answer query)")
	answerability_score: float = Field(..., description="Combined answerability score (70% passage + 30% title)")


class Phase2ValidationDoc(BaseModel):
	"""Phase-2: Metadata Verification + Quality Scoring
	
	Output: Phase-1 data + verified metadata + quality scores + risk flags
	(Ready to be fed into Instructor + Generator)
	"""
	
	# FROM PHASE-1: Raw retrieval data
	paper_id: str = Field(..., description="Paper identifier")
	title: str = Field(..., description="Document title (verified from S2)")
	semantic_score: float = Field(..., description="Cosine similarity from FAISS search")
	title_similarity: float = Field(..., description="Title verification score (Jaccard similarity)")
	answerability_score: float = Field(..., description="Combined answerability score (70% passage + 30% title)")
	
	# VERIFIED METADATA: From Semantic Scholar
	citation_count: int = Field(..., description="Total citations from S2")
	influential_citation_count: int = Field(..., description="Influential citations from S2")
	year: Optional[int] = Field(None, description="Publication year from S2")
	publication_types: List[str] = Field(default_factory=list, description="Publication types from S2")
	title_match: bool = Field(..., description="Title verification (Jaccard >= 0.55)")
	external_id_match: bool = Field(..., description="DOI/PMID match found")
	authority_level: str = Field(..., description="High/Medium/Low based on citations")
	freshness: str = Field(..., description="Current/Acceptable/Outdated based on year")
	influential: bool = Field(..., description="Has >= 10 influential citations")
	
	# QUALITY SCORES: Computed in Phase-2
	evidence_score: float = Field(..., description="Evidence type score (RCT=1.0, Meta=0.95, etc.)")
	final_score: float = Field(..., description="Combined quality score for ranking")
	quality_tier: str = Field(..., description="VERY HIGH / HIGH / MEDIUM / LOW")
	
	# RISK FLAGS: Quality warnings for answer generation
	risk_flags: List[str] = Field(default_factory=list, description="OUTDATED_EVIDENCE, LOW_AUTHORITY, WEAK_EVIDENCE, etc.")


class InstructorPromptInfo(BaseModel):
	"""Instructor prompt stage information."""
	
	verification_level: int
	answer_mode: str
	required_sections: List[str]
	constraints: List[str]
	context_plan: List[dict]
	rendered_prompt: str


class FinalAnswerInfo(BaseModel):
	"""Final grounded answer information."""
	
	direct_answer: str
	evidence_summary: str
	limitations: str
	rendered_prompt: str


class DetailedPipelineResponse(BaseModel):
	"""Detailed response showing all pipeline stages."""
	
	query: str = Field(..., description="User query.")
	scope_gate: ScopeGateInfo = Field(..., description="Scope gate stage results.")
	retrieval: List[RetrievalDocument] = Field(..., description="Initial retrieval results.")
	phase1_validation: List[Phase1ValidationDoc] = Field(..., description="Phase-1 support validation results.")
	phase2_validation: List[Phase2ValidationDoc] = Field(..., description="Phase-2 authority validation results.")
	verification_level: int = Field(..., description="Computed verification level (1-4).")
	instructor_prompt: InstructorPromptInfo = Field(..., description="Instructor prompt details.")
	final_answer: FinalAnswerInfo = Field(..., description="Final grounded answer.")


# === Batch Query Schemas ===

class BatchQueryRequest(BaseModel):
	"""Request for processing multiple queries at once."""
	
	queries: List[str] = Field(..., description="List of questions to process.")
	top_k: int = Field(5, description="Number of context documents per query.")


class BatchQueryItem(BaseModel):
	"""Single question-answer pair in batch response."""
	
	question: str = Field(..., description="Original question.")
	answer: str = Field(..., description="Direct answer to the question.")
	context: List[str] = Field(default_factory=list, description="Context passages used for answering.")


class BatchQueryResponse(BaseModel):
	"""Response containing multiple question-answer pairs."""
	
	results: List[BatchQueryItem] = Field(..., description="List of question-answer pairs.")
	total_queries: int = Field(..., description="Total number of queries processed.")
	successful: int = Field(..., description="Number of successfully processed queries.")

