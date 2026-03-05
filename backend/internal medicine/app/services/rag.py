from __future__ import annotations

from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer


def retrieve(
    query: str,
    *,
    top_k: int,
    embedder: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # search
    D, I = index.search(q_emb.astype("float32"), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        results.append(
            {
                "score": float(score),
                "id": entry["id"],
                "text": entry["text"],
                "meta": entry.get("meta", {}),
            }
        )
    return results


def construct_prompt(query: str, retrieved: List[Dict[str, Any]]):
    """
    Build a simple prompt for the generator: instructions + concatenated contexts.
    You can change format as you like (more system-like, or use templates).
    """
    context_parts = []
    for i, r in enumerate(retrieved, start=1):
        # include id and text; keep it short
        context_parts.append(f"Source {i} (id:{r['id']}): {r['text']}")
    context_block = "\n\n".join(context_parts)
    prompt = (
        "You are a helpful medical assistant. Answer the user's question using ONLY the information in the provided sources. "
        "If the answer is not contained in the sources, say you don't know and suggest possible next steps.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )
    return prompt


def answer_question(
    query: str,
    *,
    top_k: int,
    gen_max_length: int,
    temperature: float,
    embedder: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    rag_generator,
):
    retrieved = retrieve(query, top_k=top_k, embedder=embedder, index=index, metadata=metadata)
    if not retrieved:
        return {"answer": "No relevant documents found in KB.", "retrieved": []}
    prompt = construct_prompt(query, retrieved)
    # generate
    outputs = rag_generator(
        prompt,
        max_length=gen_max_length,
        do_sample=(temperature > 0.0),
        temperature=temperature,
    )
    # pipeline returns list of dicts: [{'generated_text': '...'}]
    generated_text = outputs[0]["generated_text"].strip()
    return {
        "answer": generated_text,
        "retrieved": retrieved,
        "prompt": prompt,
        "raw_generation": outputs,
    }
