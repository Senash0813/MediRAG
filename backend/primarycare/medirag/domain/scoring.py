from __future__ import annotations

from datetime import datetime
from typing import Dict, List


EVIDENCE_HIERARCHY: Dict[str, int] = {
	"Meta-Analysis": 5,
	"Systematic Review": 4,
	"Randomized Controlled Trial": 3,
	"Clinical Trial": 3,
	"Observational Study": 2,
	"Case Report": 1,
	"Case Study": 1,
	"JournalArticle": 2,
	"Review": 3,
	"RCT": 5,
	"RandomizedControlledTrial": 5,
	"SystematicReview": 4,
	"MetaAnalysis": 4,
}


def normalize_similarity(score: float, min_s: float = 0.2, max_s: float = 0.9) -> float:
	return max(0.0, min(1.0, (score - min_s) / (max_s - min_s)))


def authority_score(citations: int) -> float:
	if citations >= 200:
		return 1.0
	if citations >= 50:
		return 0.7
	if citations >= 10:
		return 0.4
	return 0.1


def freshness_score(year: int | None, current_year: int | None = None) -> float:
	if not year:
		return 0.3
	if current_year is None:
		current_year = datetime.utcnow().year
	age = current_year - year
	if age <= 5:
		return 1.0
	if age <= 10:
		return 0.6
	return 0.2


def compute_evidence_score(publication_types: List[str]) -> int:
	if not publication_types:
		return 0
	scores = [EVIDENCE_HIERARCHY.get(pt, 0) for pt in publication_types]
	return max(scores) if scores else 0


def get_quality_tier(final_score: float) -> str:
	if final_score >= 0.6:
		return "HIGH"
	if final_score >= 0.4:
		return "MEDIUM"
	return "LOW"


def assign_risk_flags(doc: Dict) -> list[str]:
	flags: list[str] = []

	freshness = doc.get("freshness")
	if freshness == "outdated":
		flags.append("OUTDATED_EVIDENCE")

	citation_count = int(doc.get("citation_count", 0) or 0)
	if citation_count < 10:
		flags.append("LOW_AUTHORITY")

	evidence_score = int(doc.get("evidence_score", 0) or 0)
	if evidence_score < 2:
		flags.append("WEAK_EVIDENCE")

	quality_tier = str(doc.get("quality_tier", "")).upper()
	if quality_tier == "LOW":
		flags.append("LOW_QUALITY")

	return flags


def final_doc_score(
	answerability_score: float,
	*,
	citation_count: int,
	year: int | None,
	influential_citation_count: int,
	evidence_score: int,
	title_match: bool,
	external_id_match: bool,
) -> float:
	"""Combine answerability (passage + title), authority, freshness, influence, and evidence.
	
	Args:
		answerability_score: Combined score (0.7 × passage_semantic + 0.3 × title_similarity)
		                     from Phase-1 dual-level answerability assessment
	"""

	sim = normalize_similarity(answerability_score)

	mismatch_penalty = 0.15 if not (title_match or external_id_match) else 0.0

	score = (
		0.55 * sim
		+ 0.15 * authority_score(citation_count)
		+ 0.10 * freshness_score(year)
		+ 0.10 * (1.0 if influential_citation_count >= 10 else 0.3)
		+ 0.10 * (evidence_score / 5.0)
	) - mismatch_penalty

	if score < 0.0:
		score = 0.0
	if score > 1.0:
		score = 1.0

	return round(score, 3)

