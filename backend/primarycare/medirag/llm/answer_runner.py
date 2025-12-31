from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple

from medirag.llm.lmstudio_client import LMStudioClient


def run_answer_llm(
    *,
    client: LMStudioClient,
    final_prompt: str,
    max_new_tokens: int = 700,
    temperature: float = 0.0,
) -> str:  # noqa: C901
    """Structural port of Stage-4 Answer generation.

    Prompt logic and post-processing preserved; only model access is via LM Studio API.
    """

    def _parse_flag_list(raw_value: str) -> list:
        try:
            parsed = ast.literal_eval(raw_value)
            if isinstance(parsed, (list, tuple)):
                return [str(flag) for flag in parsed if str(flag).strip()]
        except Exception:
            pass
        cleaned = (
            raw_value.strip()[1:-1]
            if raw_value.strip().startswith("[") and raw_value.strip().endswith("]")
            else raw_value
        )
        candidates = [c.strip().strip("'\"") for c in cleaned.split(",") if c.strip()]
        return [c for c in candidates if c]

    def _extract_doc_risk_flags(prompt_text: str):
        doc_flags: List[Tuple[str, List[str]]] = []
        current_doc = None
        for raw_line in prompt_text.splitlines():
            line = raw_line.strip()
            if line.startswith("[DOC_ID:"):
                current_doc = line.split("[DOC_ID:", 1)[1].split("]", 1)[0].strip()
            elif line.startswith("RISK_FLAGS:") and current_doc:
                flags = _parse_flag_list(line.split(":", 1)[1].strip())
                if flags:
                    doc_flags.append((current_doc, flags))
        return doc_flags

    def _summarize_risk_flags(doc_flags):
        if not doc_flags:
            return ""
        sentences = []
        for doc_id, flags in doc_flags:
            humanized = ", ".join(flag.replace("_", " ").lower() for flag in flags)
            sentences.append(f"[DOC_ID: {doc_id}] flagged for {humanized}.")
        return " ".join(sentences)

    doc_risk_flags = _extract_doc_risk_flags(final_prompt)
    risk_summary = _summarize_risk_flags(doc_risk_flags)

    system_instruction = (
        "You are a retrieval-grounded medical assistant.\n"
        "Always answer with EXACTLY three sections named: Direct answer, Evidence summary, Limitations.\n"
        "Each section must be on its own block, start with the label followed by a colon, and contain 1-4 concise sentences.\n"
        "Direct answer should synthesize the best-supported conclusion. Evidence summary must cite doc ids like [DOC_ID: 123].\n"
        "Limitations must ONLY describe risk flags, uncertainties, or cautionary notes explicitly present in the retrieved CONTEXT, referencing doc ids when possible.\n"
        "Do NOT invent or repeat generic limitations or unrelated commentary."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": final_prompt},
    ]

    answer_text = client.chat(messages=messages, max_tokens=max_new_tokens, temperature=temperature).strip()

    section_headers = ["Direct answer", "Evidence summary", "Limitations"]
    sections = {header: [] for header in section_headers}
    current_header = None
    for line in answer_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = stripped.split(":", 1)
        candidate = normalized[0].strip().lower()
        matched = next((header for header in section_headers if candidate == header.lower()), None)
        if matched:
            current_header = matched
            if len(normalized) > 1 and normalized[1].strip():
                sections[matched].append(normalized[1].strip())
            continue
        if current_header:
            sections[current_header].append(stripped)

    fallback_messages = {
        "Direct answer": "Insufficient grounded evidence in retrieved context.",
        "Evidence summary": "Unable to map the provided documents to supporting evidence.",
        "Limitations": "Retrieved documents did not expose explicit risk flags; no additional cautionary notes were surfaced.",
    }

    normalized_blocks = []
    for header in section_headers:
        content = " ".join(sections[header]).strip()
        if header == "Limitations":
            if risk_summary:
                content = risk_summary
            elif not content:
                content = fallback_messages[header]
        elif not content:
            content = fallback_messages[header]
        normalized_blocks.append(f"{header}:\n{content}")

    return "\n\n".join(normalized_blocks)
