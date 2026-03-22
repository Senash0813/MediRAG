from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from medirag.domain.models import InstructionObject, VerifiedDoc
from medirag.llm.client import LMStudioClient
from medirag.llm.prompts import (
	INSTRUCTOR_SYSTEM,
	build_compact_evidence,
	compute_verification_level,
	make_instructor_user_prompt,
)


REQUIRED_KEYS = {
	"verification_level",
	"answer_mode",
	"required_sections",
	"constraints",
	"context_plan",
}


def _extract_first_json_object(text: str) -> str:
	start = text.find("{")
	if start == -1:
		return text.strip()

	in_string = False
	escape = False
	depth = 0

	for i in range(start, len(text)):
		ch = text[i]
		if in_string:
			if escape:
				escape = False
			elif ch == "\\":
				escape = True
			elif ch == '"':
				in_string = False
		else:
			if ch == '"':
				in_string = True
			elif ch == "{":
				depth += 1
			elif ch == "}":
				depth -= 1
				if depth == 0:
					return text[start : i + 1].strip()

	end = text.rfind("}")
	if end > start:
		return text[start : end + 1].strip()
	return text[start:].strip()


def _sanitize_instructor_json(text: str) -> str:
	s = text.strip()
	first = s.find("{")
	if first != -1:
		s = s[first:]
	last = s.rfind("}")
	if last != -1:
		s = s[: last + 1]
	s = re.sub(r"\]\s*\]\s*,", "],", s)
	s = re.sub(r",\s*,", ",", s)
	return s


def _normalize_instruction_obj(obj: Dict[str, Any]) -> InstructionObject:
	out: Dict[str, Any] = {k: obj.get(k) for k in REQUIRED_KEYS}

	if out.get("verification_level") is None:
		out["verification_level"] = 1
	if out.get("answer_mode") is None:
		out["answer_mode"] = "abstractive"
	if out.get("required_sections") is None:
		out["required_sections"] = [
			"Direct answer",
			"Evidence summary",
			"Limitations",
		]
	if out.get("constraints") is None:
		out["constraints"] = [
			"Avoid absolute claims",
			"State uncertainty if evidence is weak",
		]
	if out.get("context_plan") is None:
		out["context_plan"] = []

	return InstructionObject(**out)


def _validate_instruction_obj(obj: InstructionObject) -> None:
	if obj.answer_mode not in ("abstractive", "extractive", "refuse"):
		raise ValueError("answer_mode must be one of: abstractive, extractive, refuse")


def run_instructor_llm(
	query: str,
	verified_docs: List[VerifiedDoc],
	client: LMStudioClient,
) -> InstructionObject:
	"""Call LM Studio to construct the instruction JSON object.

	Includes a simple repair pass and a deterministic fallback.
	"""

	vlevel = compute_verification_level(verified_docs)
	compact = build_compact_evidence(verified_docs, top_k=5)
	user_prompt = make_instructor_user_prompt(query, vlevel, compact)

	try:
		# Primary path: ask the LLM to build the instruction JSON
		raw = client.generate_chat(INSTRUCTOR_SYSTEM, user_prompt)
		candidate = _extract_first_json_object(raw)
		candidate = _sanitize_instructor_json(candidate)

		parsed = json.loads(candidate)
		obj = _normalize_instruction_obj(parsed)
		_validate_instruction_obj(obj)
		return obj
	except Exception:
		# Fallback: minimal deterministic instruction if the LLM call fails
		# (including HTTP 5xx from the backend) or returns malformed JSON.
		context_plan = [
			{"doc_id": str(d.paper_id), "use_for": "supporting evidence", "priority": i + 1}
			for i, d in enumerate(verified_docs[:2])
		]
		return InstructionObject(
			verification_level=int(vlevel),
			answer_mode="abstractive" if vlevel >= 2 else "refuse",
			required_sections=["Direct answer", "Evidence summary", "Limitations"],
			constraints=[
				"Avoid absolute claims",
				"State uncertainty if evidence is weak",
				"Do not ask clarifying questions; make reasonable assumptions if needed",
			],
			context_plan=context_plan,
		)

