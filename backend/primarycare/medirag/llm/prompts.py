from __future__ import annotations

import json
from typing import Any, Dict, List

from medirag.domain.models import InstructionObject, VerifiedDoc


INSTRUCTOR_SYSTEM = (
	"You are an Instruction Builder for a RAG pipeline.\n\n"
	"You MUST output ONLY one valid JSON object.\n"
	"No markdown. No code fences. No extra text before/after.\n"
	"Do NOT output a list. Do NOT output the evidence again.\n\n"
	"The JSON object MUST have EXACTLY these keys:\n"
	"verification_level, answer_mode, required_sections, constraints, context_plan\n\n"
	"Allowed values:\n"
	"- verification_level: integer 1..4\n"
	"- answer_mode: 'abstractive' | 'extractive' | 'refuse'\n"
	"- required_sections: list of strings\n"
	"- constraints: list of strings\n"
	"- context_plan: list of objects, each: {doc_id, use_for, priority}\n"
)


def _doc_id(d: VerifiedDoc) -> str:
	pid = d.paper_id if d.paper_id is not None else d.qa_id or d.title or "UNKNOWN"
	return str(pid)


def build_compact_evidence(verified_docs: List[VerifiedDoc], top_k: int = 5) -> List[Dict[str, Any]]:
	compact: List[Dict[str, Any]] = []
	for d in verified_docs[:top_k]:
		compact.append(
			{
				"doc_id": _doc_id(d),
				"title": d.title,
				"year": d.year,
				"final_score": d.final_score,
				"quality_tier": d.quality_tier,
				"citation_count": d.citation_count,
				"influential_citation_count": d.influential_citation_count,
				"publication_types": d.publication_types,
				"evidence_score": d.evidence_score,
				"risk_flags": d.risk_flags,
			}
		)
	return compact


def compute_verification_level(verified_docs: List[VerifiedDoc]) -> int:
	if not verified_docs:
		return 1

	top = verified_docs[:5]
	avg_score = sum(d.final_score for d in top) / max(len(top), 1)

	tiers = [d.quality_tier.upper() for d in top]
	high_tier_count = sum(t in ("HIGH", "VERY HIGH") for t in tiers)

	any_risk = any(d.risk_flags for d in top)
	any_missing_year = any(d.year in (None, 0, "", "None") for d in top)

	if avg_score >= 0.75 and high_tier_count >= 3 and (not any_risk) and (not any_missing_year):
		return 4
	if avg_score >= 0.60 and high_tier_count >= 2 and (not any_risk):
		return 3
	if avg_score >= 0.45:
		return 2
	return 1


def make_instructor_user_prompt(
	query: str,
	verification_level: int,
	compact_evidence: List[Dict[str, Any]],
) -> str:
	return (
		f"User query: {query}\n\n"
		f"Computed verification_level: {verification_level}\n\n"
		f"Evidence (compact JSON):\n{json.dumps(compact_evidence, ensure_ascii=False)}\n\n"
		"Now return ONLY the instruction JSON object in the required format.\n"
		"Do NOT return the evidence.\n"
		"Do NOT return a list.\n"
	)


def select_prompt_template(instruction: InstructionObject) -> str:
	v = int(instruction.verification_level)
	mode = instruction.answer_mode

	if mode == "refuse":
		base = (
			"You are a careful assistant.\n"
			"If you cannot answer reliably from the provided CONTEXT, refuse with a brief explanation.\n"
			"Do NOT ask clarifying questions.\n\n"
		)
	elif v >= 4:
		base = (
			"You are a domain expert.\n"
			"Follow REQUIRED_SECTIONS exactly and in order.\n"
			"Follow CONSTRAINTS strictly.\n"
			"Do NOT ask clarifying questions.\n\n"
		)
	elif v == 3:
		base = (
			"You are an evidence-based assistant.\n"
			"Follow REQUIRED_SECTIONS exactly and in order.\n"
			"Include a Limitations section.\n"
			"Use citations as instructed.\n"
			"Do NOT ask clarifying questions.\n\n"
		)
	else:
		base = (
			"You are a cautious assistant.\n"
			"Do not overclaim. Prefer hedged language.\n"
			"If evidence is weak, say so and proceed with a best-effort answer OR refuse.\n"
			"Do NOT ask clarifying questions.\n"
			"Follow REQUIRED_SECTIONS exactly.\n\n"
		)

	return (
		base
		+ "REQUIRED_SECTIONS: {required_sections}\n"
		+ "CONSTRAINTS: {constraints}\n"
		+ "CONTEXT_PLAN: {context_plan}\n\n"
		+ "USER_QUESTION: {query}\n\n"
		+ "CONTEXT:\n{context_block}\n"
	)


def make_context_block(verified_docs: List[VerifiedDoc], instruction: InstructionObject, max_docs: int = 5) -> str:
	# Prioritize docs from context_plan by doc_id, then fill with top-ranked
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
		selected = verified_docs[: max_docs or 3]

	blocks: List[str] = []
	for d in selected:
		did = _doc_id(d)
		blocks.append(
			f"[DOC_ID: {did}]\n"
			f"TITLE: {d.title}\n"
			f"YEAR: {d.year}\n"
			f"FINAL_SCORE: {d.final_score}  QUALITY_TIER: {d.quality_tier}\n"
			f"CITATIONS: {d.citation_count}  INFLUENTIAL: {d.influential_citation_count}\n"
			f"PUB_TYPES: {d.publication_types}\n"
			f"EVIDENCE_SCORE: {d.evidence_score}\n"
			f"RISK_FLAGS: {d.risk_flags}\n"
			f"PASSAGE_TEXT: {d.passage_text}\n"
		)

	return "\n---\n".join(blocks)


def render_final_prompt(query: str, verified_docs: List[VerifiedDoc], instruction: InstructionObject) -> str:
	template = select_prompt_template(instruction)
	context_block = make_context_block(verified_docs, instruction)
	return template.format(
		query=query,
		required_sections=instruction.required_sections,
		constraints=instruction.constraints,
		context_plan=instruction.context_plan,
		context_block=context_block,
	)

