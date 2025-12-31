from __future__ import annotations

from typing import Any, Dict, List

from medirag.llm.lmstudio_client import LMStudioClient
from medirag.prompting.dynamic_prompt import build_compact_evidence, compute_verification_level
from medirag.prompting.instructor_json import (
    extract_first_json_object,
    normalize_instruction_obj,
    sanitize_instructor_json,
    validate_instruction_obj,
)
from medirag.prompting.instructor_prompts import INSTRUCTOR_SYSTEM, make_instructor_user_prompt


def run_instructor_llm(
    *,
    client: LMStudioClient,
    query: str,
    verified_docs: List[Dict[str, Any]],
    debug: bool = False,
) -> Dict[str, Any]:  # noqa: C901
    vlevel = compute_verification_level(verified_docs)
    compact = build_compact_evidence(verified_docs, top_k=5)
    user_prompt = make_instructor_user_prompt(query, vlevel, compact)

    messages = [
        {"role": "system", "content": INSTRUCTOR_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    raw = client.chat(messages=messages, max_tokens=700, temperature=0.0).strip()

    candidate = extract_first_json_object(raw)
    candidate = sanitize_instructor_json(candidate)

    try:
        import json

        parsed = json.loads(candidate)
        obj = normalize_instruction_obj(parsed)
        validate_instruction_obj(obj)
        return obj
    except Exception as e1:
        if debug:
            print("\n===== RAW (attempt 1) =====\n", raw[:2500])
            print("\n===== SANITIZED (attempt 1) =====\n", candidate[:2500])
            print("\n===== ERROR (attempt 1) =====\n", repr(e1))

        repair_user = (
            "Return ONLY ONE valid JSON object with EXACTLY these keys:\n"
            "verification_level, answer_mode, required_sections, constraints, context_plan.\n"
            "Rules:\n"
            "- Use double quotes for ALL keys and ALL string values.\n"
            "- Do not include any extra keys.\n"
            "- Do not include any extra braces.\n"
            "- Do not include trailing commas.\n"
            "- doc_id must be a STRING.\n"
            "Now fix the following into valid JSON ONLY:\n\n" + candidate
        )

        repair_messages = [
            {"role": "system", "content": INSTRUCTOR_SYSTEM},
            {"role": "user", "content": repair_user},
        ]

        repair_raw = client.chat(messages=repair_messages, max_tokens=700, temperature=0.0).strip()

        repair_candidate = extract_first_json_object(repair_raw)
        repair_candidate = sanitize_instructor_json(repair_candidate)

        try:
            import json

            parsed = json.loads(repair_candidate)
            obj = normalize_instruction_obj(parsed)
            validate_instruction_obj(obj)
            return obj
        except Exception as e2:
            if debug:
                print("\n===== RAW (repair) =====\n", repair_raw[:2500])
                print("\n===== SANITIZED (repair) =====\n", repair_candidate[:2500])
                print("\n===== ERROR (repair) =====\n", repr(e2))

            fallback = {
                "verification_level": int(vlevel),
                "answer_mode": "refuse" if vlevel <= 1 else "abstractive",
                "required_sections": (
                    ["Direct answer", "Evidence summary", "Limitations"]
                    if vlevel <= 3
                    else ["Direct answer", "Evidence summary"]
                ),
                "constraints": [
                    "Avoid absolute claims",
                    "State uncertainty if evidence is weak",
                    "Do not ask clarifying questions; make reasonable assumptions if needed",
                ],
                "context_plan": [
                    {
                        "doc_id": str(compact[i]["doc_id"]),
                        "use_for": "supporting evidence",
                        "priority": i + 1,
                    }
                    for i in range(min(2, len(compact)))
                ],
            }
            if debug:
                print("\n⚠️ Returning FALLBACK instruction_obj due to JSON repair failure.")
            return fallback
