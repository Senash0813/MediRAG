from __future__ import annotations

from typing import Any, Dict, List


EVIDENCE_HIERARCHY = {
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


def normalize_similarity(score, min_s=0.2, max_s=0.9):
    return max(0.0, min(1.0, (score - min_s) / (max_s - min_s)))


def authority_score(citations):
    if citations >= 200:
        return 1.0
    if citations >= 50:
        return 0.7
    if citations >= 10:
        return 0.4
    return 0.1


def freshness_score(year, current_year=2025):
    if not year:
        return 0.3
    age = current_year - year
    if age <= 5:
        return 1.0
    if age <= 10:
        return 0.6
    return 0.2


def compute_evidence_score(publication_types):
    if not publication_types:
        return 0
    scores = [EVIDENCE_HIERARCHY.get(pt, 0) for pt in publication_types]
    return max(scores)


def get_quality_tier(final_score):
    if final_score >= 0.6:
        return "HIGH"
    elif final_score >= 0.4:
        return "MEDIUM"
    return "LOW"


def verify_document(local_doc, s2_doc):
    verification = {}
    verification["title_match"] = (
        local_doc.get("paper_title", "").lower()[:60] in (s2_doc.get("title") or "").lower()
    )

    citations = s2_doc.get("citationCount", 0)
    if citations >= 100:
        verification["authority_level"] = "high"
    elif citations >= 20:
        verification["authority_level"] = "medium"
    else:
        verification["authority_level"] = "low"

    year = s2_doc.get("year")
    if year:
        age = 2025 - year
        if age <= 5:
            verification["freshness"] = "current"
        elif age <= 10:
            verification["freshness"] = "acceptable"
        else:
            verification["freshness"] = "outdated"
    else:
        verification["freshness"] = "unknown"

    verification["influential"] = s2_doc.get("influentialCitationCount", 0) >= 10
    return verification


def final_doc_score(semantic_score, s2_meta, evidence_score):
    sim = normalize_similarity(semantic_score)
    return round(
        0.55 * sim
        + 0.15 * authority_score(s2_meta.get("citationCount", 0))
        + 0.10 * freshness_score(s2_meta.get("year"))
        + 0.10 * (1.0 if s2_meta.get("influentialCitationCount", 0) >= 10 else 0.3)
        + 0.10 * (evidence_score / 5),
        3,
    )


def assign_risk_flags(doc: Dict[str, Any]):
    flags = []

    if doc["freshness"] == "outdated":
        flags.append("OUTDATED_EVIDENCE")

    if doc["citation_count"] < 10:
        flags.append("LOW_AUTHORITY")

    if doc["evidence_score"] < 2:
        flags.append("WEAK_EVIDENCE")

    if doc["quality_tier"] == "LOW":
        flags.append("LOW_QUALITY")

    return flags
