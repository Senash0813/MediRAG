from __future__ import annotations

from typing import Any, Dict, List

from medirag.prompting.dynamic_prompt import _doc_id, make_context_block


def _is_definition_query(query: str) -> bool:
    if not query:
        return False
    normalized = query.lower().strip()
    triggers = (
        "what is ",
        "what are ",
        "what's ",
        "define ",
        "definition of",
        "meaning of",
        "explain what ",
    )
    return any(trigger in normalized for trigger in triggers)


def _pick_high_quality_docs(
    verified_docs: List[Dict[str, Any]], threshold: float = 0.55, limit: int = 2
) -> List[Dict[str, Any]]:
    high_quality = []
    for doc in verified_docs:
        try:
            score = float(doc.get("final_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score >= threshold:
            high_quality.append(doc)
        if len(high_quality) >= limit:
            break
    return high_quality


def build_direct_answer_directive(query: str, verified_docs: List[Dict[str, Any]], quality_threshold: float = 0.55) -> str:
    if not _is_definition_query(query):
        return ""
    high_quality_docs = _pick_high_quality_docs(verified_docs, threshold=quality_threshold)
    if high_quality_docs:
        doc_refs = ", ".join(f"[DOC_ID: {_doc_id(doc)}]" for doc in high_quality_docs)
        return (
            "For this definition-style query, open the Direct answer with a concise definition grounded in the "
            "strongest available evidence (" + doc_refs + ") before moving to supporting context or nuances."
        )
    return (
        "The user asked for a definition, but no document met the quality threshold; explicitly state in the Direct "
        "answer that the provided CONTEXT lacks a reliable definition before mentioning any peripheral points."
    )


def select_prompt_template(instruction_obj: Dict[str, Any]) -> str:
    v = int(instruction_obj["verification_level"])
    mode = instruction_obj["answer_mode"]

    if mode == "refuse":
        return (
            "You are a careful assistant.\n"
            "If you cannot answer reliably from the provided CONTEXT, refuse with a brief explanation.\n"
            "Do NOT ask clarifying questions.\n"
            "{definition_directive}\n\n"
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
            "Do NOT ask clarifying questions.\n"
            "{definition_directive}\n\n"
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
            "Do NOT ask clarifying questions.\n"
            "{definition_directive}\n\n"
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
        "Follow REQUIRED_SECTIONS exactly.\n"
        "{definition_directive}\n\n"
        "REQUIRED_SECTIONS: {required_sections}\n"
        "CONSTRAINTS: {constraints}\n"
        "CONTEXT_PLAN: {context_plan}\n\n"
        "USER_QUESTION: {query}\n\n"
        "CONTEXT:\n{context_block}\n"
    )


def render_final_prompt(query: str, verified_docs: List[Dict[str, Any]], instruction_obj: Dict[str, Any]) -> str:
    template = select_prompt_template(instruction_obj)
    context_block = make_context_block(verified_docs, instruction_obj)
    definition_directive = ""
    if instruction_obj.get("answer_mode") != "refuse":
        definition_directive = build_direct_answer_directive(query, verified_docs)

    return template.format(
        query=query,
        required_sections=instruction_obj.get("required_sections", []),
        constraints=instruction_obj.get("constraints", []),
        context_plan=instruction_obj.get("context_plan", []),
        context_block=context_block,
        definition_directive=definition_directive or "",
    )
