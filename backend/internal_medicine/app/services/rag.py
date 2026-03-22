from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .embeddings_store import faiss_search
from .ollama_client import OllamaClient
from .risk_terms import detect_high_risk_terms


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
        "If the answer is not contained in the sources, say you don't know and suggest possible next steps."
        f"{risk_block}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )
    return prompt


def answer_question(
    *,
    query: str,
    embedder: SentenceTransformer,
    index,
    metadata: List[Dict[str, Any]],
    top_k: int,
    ollama: OllamaClient,
    generator_model: str,
    gen_max_tokens: int,
    temperature: float,
) -> Tuple[str, List[Dict[str, Any]], str, Dict[str, Any]]:
    retrieved = retrieve(
        query=query,
        embedder=embedder,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    if not retrieved:
        return "No relevant documents found in KB.", [], "", {"raw": None}

    prompt = construct_prompt(query, retrieved)
    generated_text = ollama.generate(
        model=generator_model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=gen_max_tokens,
    ).strip()

    return generated_text, retrieved, prompt, {"raw": None}
