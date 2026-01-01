from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from app.services.ner import split_sentences


def semantic_similarity_check(
    claim_sentence: str,
    source_texts: List[str],
    *,
    top_k: int = 1,
    embedder,
    nlp_sci,
    sim_threshold: float,
) -> Dict[str, Any]:
    """
    Compare a high-risk claim sentence against a list of source documents.
    Returns the best matching source sentence and similarity score.
    """
    _ = top_k
    # Step 1: Split all source documents into sentences
    all_source_sents = []
    for doc in source_texts:
        sents = split_sentences(doc, nlp_sci=nlp_sci)
        all_source_sents.extend(sents)

    if not all_source_sents:
        return {
            "claim": claim_sentence,
            "best_evidence": None,
            "similarity_score": 0.0,
            "passed": False,
        }

    # Step 2: Compute embeddings
    claim_emb = embedder.encode([claim_sentence], convert_to_numpy=True, normalize_embeddings=True)
    source_embs = embedder.encode(all_source_sents, convert_to_numpy=True, normalize_embeddings=True)

    # Step 3: Compute cosine similarity (dot product since embeddings are normalized)
    sims = np.dot(source_embs, claim_emb.T).squeeze()  # shape: (num_source_sents,)

    # Step 4: Find best match
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_sentence = all_source_sents[best_idx]

    # Step 5: Pass/fail based on threshold
    passed = best_score >= sim_threshold

    return {
        "claim": claim_sentence,
        "best_evidence": best_sentence,
        "similarity_score": best_score,
        "passed": passed,
    }


def nli_claim_verification(claim: str, evidence: str, *, nli_pipeline):
    """
    Verifies whether evidence supports the claim using NLI.
    """
    if evidence is None:
        return {"claim": claim, "evidence": None, "label": "NEUTRAL", "score": 0.0}

    # NLI format: premise + hypothesis
    input_text = f"Premise: {evidence}\nHypothesis: {claim}"

    result = nli_pipeline(input_text, truncation=True)[0]

    return {
        "claim": claim,
        "evidence": evidence,
        "label": result["label"],  # ENTAILMENT / CONTRADICTION / NEUTRAL
        "score": result["score"],
    }


def build_regeneration_prompt(claim: str, evidence: str):
    return (
        "You are a medical expert assistant.\n"
        "Rewrite the claim so that it is fully supported by the given evidence.\n"
        "Do NOT add new information.\n"
        "If the claim is incorrect, correct it using the evidence.\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"CLAIM:\n{claim}\n\n"
        "REVISED CLAIM:"
    )


def regenerate_claim(claim: str, evidence: str, *, rag_generator, max_length: int = 128):
    prompt = build_regeneration_prompt(claim, evidence)
    outputs = rag_generator(prompt, max_length=max_length, do_sample=False)
    return outputs[0]["generated_text"].strip()


def post_process_answer(original_answer: str, nli_results: List[Dict[str, Any]], *, nlp_sci, rag_generator):
    # Split original answer into sentences
    original_sents = split_sentences(original_answer, nlp_sci=nlp_sci)

    # Map claims to NLI outcomes
    nli_map = {r["claim"]: r for r in nli_results}

    final_sentences = []

    for sent in original_sents:
        if sent not in nli_map:
            # Sentence was not high-risk → keep as-is
            final_sentences.append(sent)
            continue

        nli_label = nli_map[sent]["nli_label"]
        evidence = nli_map[sent]["evidence"]

        if nli_label == "ENTAILMENT":
            final_sentences.append(sent)

        elif nli_label in ["NEUTRAL", "CONTRADICTION"]:
            regenerated = regenerate_claim(sent, evidence, rag_generator=rag_generator)
            final_sentences.append(regenerated)

    return " ".join(final_sentences)


def verify_and_postprocess(
    *,
    rag_answer: str,
    retrieved_texts: List[str],
    hallucination_candidates: List[Dict[str, Any]],
    embedder,
    nlp_sci,
    nli_pipeline,
    rag_generator,
    sim_threshold: float,
    unverified_message: str,
):
    verification_failed = False
    nli_results = []

    for item in hallucination_candidates:
        claim_sent = item["sentence"]

        sim_result = semantic_similarity_check(
            claim_sent,
            retrieved_texts,
            embedder=embedder,
            nlp_sci=nlp_sci,
            sim_threshold=sim_threshold,
        )

        # ❌ HARD STOP if any high-risk claim is unsupported
        if not sim_result["passed"]:
            verification_failed = True
            break

        # ✅ Continue with NLI only if similarity passes
        nli_result = nli_claim_verification(
            claim=sim_result["claim"],
            evidence=sim_result["best_evidence"],
            nli_pipeline=nli_pipeline,
        )

        nli_results.append(
            {
                "claim": claim_sent,
                "similarity_score": sim_result["similarity_score"],
                "evidence": sim_result["best_evidence"],
                "nli_label": nli_result["label"],
                "nli_score": nli_result["score"],
            }
        )

    if verification_failed:
        final_answer = unverified_message
    else:
        final_answer = post_process_answer(
            original_answer=rag_answer,
            nli_results=nli_results,
            nlp_sci=nlp_sci,
            rag_generator=rag_generator,
        )

    return {"final_answer": final_answer, "verification_failed": verification_failed, "nli_results": nli_results}
