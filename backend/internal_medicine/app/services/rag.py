from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .embeddings_store import faiss_search
from .ollama_client import OllamaClient
from .risk_terms import detect_high_risk_terms
from .verification import nli_claim_verification


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def retrieve(
    *,
    query: str,
    embedder: SentenceTransformer,
    index,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = faiss_search(index=index, query_emb=q_emb.astype("float32"), top_k=top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        results.append(
            {
                "score": float(score),
                "id": str(entry.get("id")),
                "text": str(entry.get("text")),
                "meta": entry.get("meta", {}) or {},
            }
        )
    return results


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t and t not in STOPWORDS]


def _keyword_overlap_ratio(query: str, evidence_text: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    evidence_tokens = set(_tokenize(evidence_text))
    overlap = query_tokens.intersection(evidence_tokens)
    return float(len(overlap) / len(query_tokens))


def _distinct_doc_count(retrieved: List[Dict[str, Any]]) -> int:
    keys = set()
    for item in retrieved:
        meta = item.get("meta") or {}
        key = meta.get("paper_id") or meta.get("doc_id") or item.get("id")
        if key is not None:
            keys.add(str(key))
    return len(keys)


def retrieval_confidence_check(
    *,
    query: str,
    retrieved: List[Dict[str, Any]],
    min_top1: float,
    min_avg_topk: float,
    top_k: int,
    min_margin_top2: float,
    avg_topk_override: float,
    keyword_overlap_min: float,
    require_distinct_docs: bool,
    min_distinct_docs: int,
    nli_pipeline=None,
) -> Tuple[bool, Dict[str, Any]]:
    if not retrieved:
        return False, {
            "reason": "no_retrieval_results",
            "top1": 0.0,
            "top2": 0.0,
            "margin_top2": 0.0,
            "avg_topk": 0.0,
            "retrieved_count": 0,
            "keyword_overlap": 0.0,
            "nli_label": "NEUTRAL",
            "distinct_docs": 0,
            "layer2_checks": {
                "retrieval_strength": False,
                "evidence_confidence": False,
                "evidence_support": False,
                "distinct_docs": (not require_distinct_docs),
            },
            "scores_top5": [],
        }

    scores = [float(r.get("score", 0.0)) for r in retrieved]
    used_scores = scores[: max(1, top_k)]
    top1 = used_scores[0] if used_scores else 0.0
    top2 = used_scores[1] if len(used_scores) > 1 else 0.0
    margin = top1 - top2 if len(used_scores) > 1 else top1
    avg_topk = float(sum(used_scores) / len(used_scores)) if used_scores else 0.0
    evidence_text = "\n\n".join(str(r.get("text", "")) for r in retrieved[: max(1, top_k)])
    keyword_overlap = _keyword_overlap_ratio(query, evidence_text)
    nli_res = nli_claim_verification(claim=query, evidence=evidence_text, nli_pipeline=nli_pipeline)
    nli_label = str(nli_res.get("label", "NEUTRAL")).upper()
    distinct_docs = _distinct_doc_count(retrieved)

    retrieval_strength_pass = (top1 >= min_top1) and (avg_topk >= min_avg_topk)
    evidence_confidence_pass = (margin >= min_margin_top2) or (avg_topk >= avg_topk_override)
    evidence_support_pass = (keyword_overlap >= keyword_overlap_min) or (nli_label == "ENTAILMENT")
    distinct_docs_pass = (not require_distinct_docs) or (distinct_docs >= max(1, int(min_distinct_docs)))

    checks = {
        "retrieval_strength": retrieval_strength_pass,
        "evidence_confidence": evidence_confidence_pass,
        "evidence_support": evidence_support_pass,
        "distinct_docs": distinct_docs_pass,
    }

    if not all(checks.values()):
        return False, {
            "reason": "insufficient_answerability",
            "top1": top1,
            "top2": top2,
            "margin_top2": margin,
            "avg_topk": avg_topk,
            "retrieved_count": len(retrieved),
            "keyword_overlap": keyword_overlap,
            "nli_label": nli_label,
            "distinct_docs": distinct_docs,
            "layer2_checks": checks,
            "scores_top5": scores[:5],
        }

    return True, {
        "reason": "answerable",
        "top1": top1,
        "top2": top2,
        "margin_top2": margin,
        "avg_topk": avg_topk,
        "retrieved_count": len(retrieved),
        "keyword_overlap": keyword_overlap,
        "nli_label": nli_label,
        "distinct_docs": distinct_docs,
        "layer2_checks": checks,
        "scores_top5": scores[:5],
    }


def construct_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, r in enumerate(retrieved, start=1):
        context_parts.append(f"Source {i} (id:{r['id']}): {r['text']}")

    context_block = "\n\n".join(context_parts)

    risk = detect_high_risk_terms(query)
    risk_block = ""
    if risk.flags:
        terms = ", ".join(risk.matches) if risk.matches else ", ".join(risk.flags)
        risk_block = (
            "\n\nHIGH-RISK TERMS DETECTED IN QUESTION: "
            + terms
            + "\n"
            "Safety rules:\n"
            "- Do not provide specific dosing/titration instructions unless explicitly present in the sources.\n"
            "- If sources are insufficient for safe advice, say you don't know and recommend consulting a clinician.\n"
            "- Prefer surveillance/monitoring guidance over treatment directives when possible.\n"
        )

    prompt = (
        "You are a helpful medical assistant. Answer the user's question using ONLY the information in the provided sources. "
        "If the answer is not contained in the sources, say you don't know. "
        "Return only the direct answer. Do not include suggestions or additional prompts."
        f"{risk_block}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )
    return prompt


def strip_followup_questions(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"\n\s*follow[-\s]?up questions\s*:\s*",
        r"\n\s*next questions\s*:\s*",
        r"\n\s*would you like me to\b",
        r"\n\s*do you want me to\b",
    ]
    lowered = text.lower()
    cut_idx = len(text)
    for pattern in patterns:
        m = re.search(pattern, lowered, flags=re.IGNORECASE)
        if m:
            cut_idx = min(cut_idx, m.start())

    return text[:cut_idx].strip()


def answer_question(
    *,
    query: str,
    embedder: SentenceTransformer,
    index,
    metadata: List[Dict[str, Any]],
    top_k: int,
    min_retrieval_top1: float,
    min_retrieval_avg_topk: float,
    retrieval_score_topk: int,
    min_retrieval_margin_top2: float,
    retrieval_avg_topk_override: float,
    retrieval_keyword_overlap_min: float,
    retrieval_require_distinct_docs: bool,
    retrieval_min_distinct_docs: int,
    nli_pipeline=None,
    ollama: OllamaClient,
    generator_model: str,
    gen_max_tokens: int,
    temperature: float,
) -> Tuple[str, List[Dict[str, Any]], str, Dict[str, Any], bool, Dict[str, Any]]:
    retrieved = retrieve(
        query=query,
        embedder=embedder,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    if not retrieved:
        return "No relevant documents found in KB.", [], "", {"raw": None}, True, {
            "reason": "no_retrieval_results",
            "top1": 0.0,
            "avg_topk": 0.0,
            "retrieved_count": 0,
            "scores_top5": [],
        }

    retrieval_ok, retrieval_info = retrieval_confidence_check(
        query=query,
        retrieved=retrieved,
        min_top1=min_retrieval_top1,
        min_avg_topk=min_retrieval_avg_topk,
        top_k=retrieval_score_topk,
        min_margin_top2=min_retrieval_margin_top2,
        avg_topk_override=retrieval_avg_topk_override,
        keyword_overlap_min=retrieval_keyword_overlap_min,
        require_distinct_docs=retrieval_require_distinct_docs,
        min_distinct_docs=retrieval_min_distinct_docs,
        nli_pipeline=nli_pipeline,
    )

    if not retrieval_ok:
        return (
            "❌ Insufficient evidence in the knowledge base to answer this question reliably.",
            retrieved,
            "",
            {"raw": None},
            True,
            retrieval_info,
        )

    prompt = construct_prompt(query, retrieved)
    generated_text = ollama.generate(
        model=generator_model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=gen_max_tokens,
    ).strip()
    generated_text = strip_followup_questions(generated_text)

    return generated_text, retrieved, prompt, {"raw": None}, False, retrieval_info
