from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScopeDecision:
	in_scope: bool
	best_specialty: Optional[str]
	score: float


@dataclass
class VerifiedDoc:
	"""Flattened verified document used throughout the pipeline."""

	qa_id: Any
	paper_id: Any
	title: str
	passage_text: str
	semantic_score: float
	final_score: float
	quality_tier: str
	citation_count: int
	influential_citation_count: int
	year: Optional[int]
	publication_types: List[str] = field(default_factory=list)
	evidence_score: float = 0.0
	risk_flags: List[str] = field(default_factory=list)
	# Verification metadata
	title_match: bool = False
	title_similarity: float = 0.0
	answerability_score: float = 0.0  # Combined passage (70%) + title (30%) similarity
	external_id_match: bool = False
	authority_level: str = "low"
	freshness: str = "unknown"
	influential: bool = False


@dataclass
class InstructionObject:
	"""Instruction JSON structure produced by the instructor LLM."""

	verification_level: int
	answer_mode: str
	required_sections: List[str]
	constraints: List[str]
	context_plan: List[Dict[str, Any]]

