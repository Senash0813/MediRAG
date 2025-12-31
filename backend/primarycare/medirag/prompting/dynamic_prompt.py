from __future__ import annotations

from typing import Any, Dict, List


def _doc_id(d: Dict[str, Any]) -> str:
    return str(d.get("paper_id") or d.get("qa_id") or d.get("corpus_id") or d.get("paper_url") or d.get("title"))


def compute_verification_level(verified_docs: List[Dict[str, Any]]) -> int:
    if not verified_docs:
        return 1

    top = verified_docs[:5]
    avg_score = sum(float(d.get("final_score", 0.0) or 0.0) for d in top) / max(len(top), 1)

    tiers = [str(d.get("quality_tier", "Low")) for d in top]
    high_tier_count = sum(t in ("High", "Very High") for t in tiers)

    risk_terms = ("conflict", "low", "missing", "unknown", "outdated", "mismatch")
    any_risk = any(
        any(term in str(flag).lower() for term in risk_terms) for d in top for flag in (d.get("risk_flags") or [])
    )

    any_missing_year = any(d.get("year") in (None, 0, "", "None") for d in top)

    if avg_score >= 0.75 and high_tier_count >= 3 and (not any_risk) and (not any_missing_year):
        return 4
    if avg_score >= 0.60 and high_tier_count >= 2 and (not any_risk):
        return 3
    if avg_score >= 0.45:
        return 2
    return 1


def build_compact_evidence(verified_docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    compact = []
    for d in verified_docs[:top_k]:
        compact.append(
            {
                "doc_id": _doc_id(d),
                "title": d.get("title"),
                "year": d.get("year"),
                "final_score": d.get("final_score"),
                "quality_tier": d.get("quality_tier"),
                "citation_count": d.get("citation_count"),
                "influential_citation_count": d.get("influential_citation_count"),
                "publication_types": d.get("publication_types"),
                "evidence_score": d.get("evidence_score"),
                "risk_flags": d.get("risk_flags", []),
            }
        )
    return compact


def select_prompt_template(instruction_obj: Dict[str, Any]) -> str:
    v = int(instruction_obj["verification_level"])
    mode = instruction_obj["answer_mode"]

    if mode == "refuse":
        return (
            "You are a careful assistant.\n"
            "If you cannot answer reliably from the provided CONTEXT, refuse with a brief explanation.\n"
            "Do NOT ask clarifying questions.\n\n"
            "REQUIRED_SECTIONS: {required_sections}\n"
            "CONSTRAINTS: {constraints}\n"
            "CONTEXT_PLAN: {context_plan}\n\n"
            "USER_QUESTION: {query}\n\n"
            "CONTEXT:\n{context_block}\n"
        )

    if v >= 4:
        return (
            "You are a domain expert.\n"
            "Follow REQUIRED_SECTIONS exactly and in order.\n"
            "Follow CONSTRAINTS strictly.\n"
            "Do NOT ask clarifying questions.\n\n"
            "REQUIRED_SECTIONS: {required_sections}\n"
            "CONSTRAINTS: {constraints}\n"
            "CONTEXT_PLAN: {context_plan}\n\n"
            "USER_QUESTION: {query}\n\n"
            "CONTEXT:\n{context_block}\n"
        )

    if v == 3:
        return (
            "You are an evidence-based assistant.\n"
            "Follow REQUIRED_SECTIONS exactly and in order.\n"
            "Include a Limitations section.\n"
            "Use citations as instructed.\n"
            "Do NOT ask clarifying questions.\n\n"
            "REQUIRED_SECTIONS: {required_sections}\n"
            "CONSTRAINTS: {constraints}\n"
            "CONTEXT_PLAN: {context_plan}\n\n"
            "USER_QUESTION: {query}\n\n"
            "CONTEXT:\n{context_block}\n"
        )

    return (
        "You are a cautious assistant.\n"
        "Do not overclaim. Prefer hedged language.\n"
        "If evidence is weak, say so and proceed with a best-effort answer OR refuse.\n"
        "Do NOT ask clarifying questions.\n"
        "Follow REQUIRED_SECTIONS exactly.\n\n"
        "REQUIRED_SECTIONS: {required_sections}\n"
        "CONSTRAINTS: {constraints}\n"
        "CONTEXT_PLAN: {context_plan}\n\n"
        "USER_QUESTION: {query}\n\n"
        "CONTEXT:\n{context_block}\n"
    )


def make_context_block(verified_docs: List[Dict[str, Any]], instruction_obj: Dict[str, Any], fallback_k: int = 3) -> str:
    planned_ids = set()
    for item in instruction_obj.get("context_plan", []):
        if isinstance(item, dict) and "doc_id" in item:
            planned_ids.add(str(item["doc_id"]))

    blocks = []
    for d in verified_docs:
        did = _doc_id(d)
        if planned_ids and did not in planned_ids:
            continue

        blocks.append(
            f"[DOC_ID: {did}]\n"
            f"TITLE: {d.get('title')}\n"
            f"YEAR: {d.get('year')}\n"
            f"FINAL_SCORE: {d.get('final_score')}  QUALITY_TIER: {d.get('quality_tier')}\n"
            f"CITATIONS: {d.get('citation_count')}  INFLUENTIAL: {d.get('influential_citation_count')}\n"
            f"PUB_TYPES: {d.get('publication_types')}\n"
            f"EVIDENCE_SCORE: {d.get('evidence_score')}\n"
            f"RISK_FLAGS: {d.get('risk_flags')}\n"
            f"PASSAGE_TEXT: {d.get('passage_text')}\n"
        )

    if not blocks:
        for d in verified_docs[:fallback_k]:
            did = _doc_id(d)
            blocks.append(
                f"[DOC_ID: {did}]\n"
                f"TITLE: {d.get('title')}\n"
                f"YEAR: {d.get('year')}\n"
                f"FINAL_SCORE: {d.get('final_score')}  QUALITY_TIER: {d.get('quality_tier')}\n"
                f"CITATIONS: {d.get('citation_count')}  INFLUENTIAL: {d.get('influential_citation_count')}\n"
                f"PUB_TYPES: {d.get('publication_types')}\n"
                f"EVIDENCE_SCORE: {d.get('evidence_score')}\n"
                f"RISK_FLAGS: {d.get('risk_flags')}\n"
                f"PASSAGE_TEXT: {d.get('passage_text')}\n"
            )

    return "\n---\n".join(blocks)


def render_final_prompt(query: str, verified_docs: List[Dict[str, Any]], instruction_obj: Dict[str, Any]) -> str:
    template = select_prompt_template(instruction_obj)
    context_block = make_context_block(verified_docs, instruction_obj)
    return template.format(
        query=query,
        required_sections=instruction_obj.get("required_sections", []),
        constraints=instruction_obj.get("constraints", []),
        context_plan=instruction_obj.get("context_plan", []),
        context_block=context_block,
    )
