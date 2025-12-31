from __future__ import annotations

import json
from typing import Any, Dict, List


INSTRUCTOR_SYSTEM = """
You are an Instruction Builder for a RAG pipeline.

You MUST output ONLY one valid JSON object.
No markdown. No code fences. No extra text before/after.
Do NOT output a list. Do NOT output the evidence again.

The JSON object MUST have EXACTLY these keys:
verification_level, answer_mode, required_sections, constraints, context_plan

Allowed values:
- verification_level: integer 1..4
- answer_mode: \"abstractive\" | \"extractive\" | \"refuse\"
- required_sections: list of strings
- constraints: list of strings
- context_plan: list of objects, each: {doc_id, use_for, priority}

Example (FORMAT ONLY):
{
  \"verification_level\": 3,
  \"answer_mode\": \"abstractive\",
  \"required_sections\": [\"Direct answer\", \"Evidence summary\", \"Limitations\"],
  \"constraints\": [\"Avoid absolute claims\", \"State uncertainty when needed\"],
  \"context_plan\": [
    {\"doc_id\":\"DOC1\", \"use_for\":\"definition\", \"priority\":1},
    {\"doc_id\":\"DOC2\", \"use_for\":\"supporting evidence\", \"priority\":2}
  ]
}
""".strip()


def make_instructor_user_prompt(query: str, verification_level: int, compact_evidence: List[Dict[str, Any]]) -> str:
    return (
        f"User query: {query}\n\n"
        f"Computed verification_level: {verification_level}\n\n"
        f"Evidence (compact JSON):\n{json.dumps(compact_evidence, ensure_ascii=False)}\n\n"
        "Now return ONLY the instruction JSON object in the required format.\n"
        "Do NOT return the evidence.\n"
        "Do NOT return a list.\n"
    )
