from __future__ import annotations

import re
from typing import Any, Dict, List

from medirag.retrieval.retriever import retrieve_top_n_unique


SECTION_HEADERS = ("Direct answer", "Evidence summary", "Limitations")
DIRECT_ANSWER_HEADER = SECTION_HEADERS[0]
EVIDENCE_SUMMARY_HEADER = SECTION_HEADERS[1]
LIMITATIONS_HEADER = SECTION_HEADERS[2]


def _build_context_block(retrieved_docs: List[Dict[str, Any]], max_docs: int) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(retrieved_docs[:max_docs], start=1):
        blocks.append(
            f"[DOC {i}]\n"
            f"TITLE: {doc.get('paper_title', '')}\n"
            f"PAPER_ID: {doc.get('paper_id', '')}\n"
            f"SEMANTIC_SCORE: {float(doc.get('semantic_score', 0.0) or 0.0):.4f}\n"
            f"PASSAGE: {doc.get('passage_text', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def _split_sections(answer_text: str) -> Dict[str, str]:
    text = (answer_text or "").strip()
    out = dict.fromkeys(SECTION_HEADERS, "")
    if not text:
        return out

    pattern = r"(?im)^(Direct answer|Evidence summary|Limitations)\s*:\s*"
    parts = re.split(pattern, text)

    if len(parts) < 3:
        out[DIRECT_ANSWER_HEADER] = text
        return out

    i = 1
    while i + 1 < len(parts):
        header = parts[i].strip()
        body = parts[i + 1].strip()
        if header in out:
            out[header] = body
        i += 2

    return out


def run_standard_rag_ablation(*, assets: Any, query: str, k: int) -> Dict[str, Any]:
    """Ablation pipeline: standard RAG only.

    Characteristics:
    - No validation engine
    - No dynamic instructor prompt
    - Uses direct FAISS retrieval + fixed prompt generation
    """
    top_k = max(1, int(k or assets.settings.default_top_k))

    retrieved = retrieve_top_n_unique(
        query,
        n=top_k,
        oversample=3,
        embedder=assets.embedder,
        faiss_assets=assets.faiss_assets,
    )

    if not retrieved:
        return {
            "direct_answer": "No supporting documents were retrieved for this query.",
        }

    context_block = _build_context_block(retrieved, max_docs=top_k)

    system_prompt = (
        "You are a careful medical RAG assistant. "
        "Answer only using the provided context. "
        "If context is insufficient, say so clearly in Limitations."
    )

    user_prompt = (
        "Use the context to answer the query. Return exactly three sections with these headers:\n"
        "Direct answer:\n"
        "Evidence summary:\n"
        "Limitations:\n\n"
        f"QUERY:\n{query}\n\n"
        f"CONTEXT:\n{context_block}\n"
    )

    answer_text = assets.lm_client.generate_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=assets.settings.lm_max_new_tokens,
        temperature=assets.settings.lm_temperature,
    )

    sections = _split_sections(answer_text)

    if not sections[DIRECT_ANSWER_HEADER]:
        sections[DIRECT_ANSWER_HEADER] = answer_text.strip()

    return {
        "direct_answer": sections[DIRECT_ANSWER_HEADER],
    }
