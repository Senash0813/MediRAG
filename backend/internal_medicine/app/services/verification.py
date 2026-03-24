from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .embeddings_store import split_sentences_regex
from .ollama_client import OllamaClient
from .risk_terms import DOSE_REGEX, FREQ_REGEX, SAFETY_REGEX


RISK_MAP: Dict[str, str] = {
    "DISEASE": "HIGH",
    "CHEMICAL": "HIGH",
    "DRUG": "HIGH",
    "MEDICATION": "HIGH",
    "TREATMENT": "HIGH",
    "PROCEDURE": "MEDIUM",
    "SYMPTOM": "MEDIUM",
    "TEST": "MEDIUM",
    "ANATOMY": "LOW",
    "GENE": "LOW",
    "PROTEIN": "LOW",
    "CELL_LINE": "LOW",
    "UNKNOWN": "LOW",
}

DISCLAIMER = (
    "⚠️ Disclaimer: Some high-risk medical statements were automatically verified and, "
    "where necessary, revised for safety. This output is for informational purposes only "
    "and is not medical advice. Please consult a qualified healthcare professional before "
    "making clinical decisions.\n\n"
)

ANSWER_LEVEL_DISCLAIMER = (
    "⚠️ Disclaimer: No high-risk entities were detected, but a full-response verification "
    "step found that the overall answer may not be fully supported by the retrieved evidence. "
    "Please verify with reliable medical sources or consult a qualified healthcare professional.\n\n"
)


@dataclass(frozen=True)
class VerificationConfig:
    risk_threshold: int
    max_evidence_chars: int
    answer_supported_th: float
    answer_unsupported_th: float
    fast_verified_th: float
    fast_hallucinated_th: float
    full_verified_th: float
    full_hallucinated_th: float
    sim_for_regen: float
    regen_max_tokens: int
    hf_judge_max_new_tokens: int = 32


@dataclass(frozen=True)
class VerificationModels:
    nli_pipeline: Any = None
    nlp_sci: Any = None
    nlp_bc5cdr: Any = None
    biomed_ner: Any = None
    judge_t2t: Any = None


def _parse_judge_verdict(text: str) -> str:
    out_norm = (text or "").strip().upper()
    if "UNSUPPORTED" in out_norm:
        return "UNSUPPORTED"
    if "SUPPORTED" in out_norm:
        return "SUPPORTED"
    token = out_norm.split()[0] if out_norm else "UNSUPPORTED"
    return "SUPPORTED" if token == "SUPPORTED" else "UNSUPPORTED"


def split_sentences_with_ids(text: str, *, nlp_sci=None) -> List[Dict[str, Any]]:
    sentences: List[str] = []
    try:
        if nlp_sci is not None:
            doc = nlp_sci(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        sentences = []

    if not sentences:
        sentences = split_sentences_regex(text)

    return [{"sent_id": i, "text": s} for i, s in enumerate(sentences)]


def extract_entities_multi_ner(
    text: str,
    *,
    nlp_bc5cdr=None,
    biomed_ner=None,
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []

    if nlp_bc5cdr is not None:
        try:
            doc = nlp_bc5cdr(text)
            for ent in doc.ents:
                entities.append({"text": ent.text, "label": ent.label_})
        except Exception:
            pass

    if biomed_ner is not None:
        try:
            for ent in biomed_ner(text):
                entities.append({"text": ent.get("word"), "label": ent.get("entity_group")})
        except Exception:
            pass

    return [e for e in entities if e.get("text")]


def merge_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for e in entities:
        key = str(e.get("text", "")).lower()
        if key and key not in seen:
            seen[key] = e
    return list(seen.values())


def score_sentence_risk(sentence: str, entities: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []

    for e in entities:
        label = e.get("label") or "UNKNOWN"
        risk = RISK_MAP.get(label, "LOW")

        if risk == "HIGH":
            score += 30
            reasons.append(f"HIGH label: {label} ({e.get('text')})")
        elif risk == "MEDIUM":
            score += 15
            reasons.append(f"MEDIUM label: {label} ({e.get('text')})")

    if DOSE_REGEX.search(sentence or ""):
        score += 40
        reasons.append("Dose pattern (mg/mcg/ml/IU/%)")

    if FREQ_REGEX.search(sentence or ""):
        score += 20
        reasons.append("Frequency pattern (qd/bid/tid/q12h/once daily)")

    if SAFETY_REGEX.search(sentence or ""):
        score += 40
        reasons.append("Safety keyword (contraindicated/adverse/toxicity/interaction/...)")

    score = min(score, 100)
    return int(score), reasons


def semantic_similarity_check(
    *,
    claim_sentence: str,
    source_texts: List[str],
    embedder: SentenceTransformer,
) -> Dict[str, Any]:
    all_source_sents: List[str] = []
    for doc in source_texts:
        sents = [s["text"] for s in split_sentences_with_ids(doc)]
        all_source_sents.extend(sents)

    if not all_source_sents:
        return {"claim": claim_sentence, "best_evidence": None, "similarity_score": 0.0}

    claim_emb = embedder.encode([claim_sentence], convert_to_numpy=True, normalize_embeddings=True)
    source_embs = embedder.encode(all_source_sents, convert_to_numpy=True, normalize_embeddings=True)

    sims = np.dot(source_embs, claim_emb.T).squeeze()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_sentence = all_source_sents[best_idx]

    return {"claim": claim_sentence, "best_evidence": best_sentence, "similarity_score": best_score}


def normalize_nli_label(label: str) -> str:
    lab = (label or "").upper()
    if lab in ("ENTAILMENT", "NEUTRAL", "CONTRADICTION"):
        return lab

    mapping = {"LABEL_0": "ENTAILMENT", "LABEL_1": "NEUTRAL", "LABEL_2": "CONTRADICTION"}
    return mapping.get(lab, lab)


def nli_to_entail_prob(label: str, score: float) -> float:
    lab = (label or "").upper()
    s = float(score or 0.0)

    if lab == "ENTAILMENT":
        return s
    if lab == "NEUTRAL":
        return 0.5 * s
    if lab == "CONTRADICTION":
        return 0.0
    return 0.0


def nli_claim_verification(
    *,
    claim: str,
    evidence: str,
    nli_pipeline=None,
    max_len: int = 512,
) -> Dict[str, Any]:
    if evidence is None or str(evidence).strip() == "" or nli_pipeline is None:
        return {"claim": claim, "evidence": evidence, "label": "NEUTRAL", "score": 0.0}

    input_text = f"Premise: {evidence}\nHypothesis: {claim}"
    result = nli_pipeline(input_text, truncation=True, max_length=max_len)[0]

    return {
        "claim": claim,
        "evidence": evidence,
        "label": normalize_nli_label(str(result.get("label"))),
        "score": float(result.get("score", 0.0)),
    }


def build_evidence_pack(retrieved_texts: List[str], *, max_chars: int) -> str:
    if not retrieved_texts:
        return ""
    joined = "\n\n".join([f"[Source {i+1}] {t}" for i, t in enumerate(retrieved_texts)])
    return joined[:max_chars]


def llm_as_judge(
    *,
    claim: str,
    evidence: str,
    ollama: OllamaClient,
    judge_model: str,
    max_tokens: int = 32,
) -> Dict[str, Any]:
    if evidence is None or str(evidence).strip() == "":
        return {"claim": claim, "evidence": evidence, "label": "UNSUPPORTED", "raw_output": "NO_EVIDENCE"}

    prompt = (
        "You are a strict medical fact-checking judge.\n"
        "Decide if the CLAIM is fully supported by the EVIDENCE.\n"
        "Rules:\n"
        "1) If the evidence does not explicitly support the claim, answer UNSUPPORTED.\n"
        "2) Do NOT use outside knowledge.\n"
        "3) Answer with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"CLAIM:\n{claim}\n\n"
        "VERDICT:"
    )

    out = ollama.generate(model=judge_model, prompt=prompt, temperature=0.0, max_tokens=max_tokens)
    out_norm = (out or "").strip().upper()
    label = _parse_judge_verdict(out_norm)

    return {
        "claim": claim,
        "evidence": evidence,
        "label": label,
        "raw_output": out_norm,
        "backend": "ollama",
        "model": judge_model,
    }


def hf_as_judge(
    *,
    claim: str,
    evidence: str,
    judge_t2t,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    if evidence is None or str(evidence).strip() == "" or judge_t2t is None:
        return {
            "claim": claim,
            "evidence": evidence,
            "label": "UNSUPPORTED",
            "raw_output": "NO_EVIDENCE_OR_NO_PIPELINE",
            "backend": "hf",
            "model": None,
        }

    prompt = (
        "You are a strict medical fact-checking judge.\n"
        "Decide if the CLAIM is fully supported by the EVIDENCE.\n"
        "Rules:\n"
        "1) If the evidence does not explicitly support the claim, answer UNSUPPORTED.\n"
        "2) Do NOT use outside knowledge.\n"
        "3) Answer with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"CLAIM:\n{claim}\n\n"
        "VERDICT:"
    )

    try:
        res = judge_t2t(prompt, max_new_tokens=int(max_new_tokens), do_sample=False)
        if isinstance(res, list) and res:
            generated = res[0].get("generated_text") or res[0].get("text") or ""
        else:
            generated = str(res)
    except Exception as e:
        generated = f"ERROR: {e}"

    verdict = _parse_judge_verdict(generated)
    model_name = getattr(getattr(judge_t2t, "model", None), "name_or_path", None)
    return {
        "claim": claim,
        "evidence": evidence,
        "label": verdict,
        "raw_output": (generated or "").strip(),
        "backend": "hf",
        "model": model_name,
    }


def judge_claim(
    *,
    claim: str,
    evidence: str,
    ollama: OllamaClient,
    judge_model: str,
    judge_t2t=None,
    hf_max_new_tokens: int = 32,
) -> Dict[str, Any]:
    if judge_t2t is not None:
        return hf_as_judge(
            claim=claim,
            evidence=evidence,
            judge_t2t=judge_t2t,
            max_new_tokens=hf_max_new_tokens,
        )

    return llm_as_judge(
        claim=claim,
        evidence=evidence,
        ollama=ollama,
        judge_model=judge_model,
    )


def answer_level_nli_check(
    *,
    full_answer: str,
    retrieved_texts: List[str],
    nli_pipeline=None,
    max_evidence_chars: int,
    supported_th: float,
    unsupported_th: float,
) -> Dict[str, Any]:
    evidence_pack = build_evidence_pack(retrieved_texts, max_chars=max_evidence_chars)
    res = nli_claim_verification(claim=full_answer, evidence=evidence_pack, nli_pipeline=nli_pipeline)
    entail_prob = nli_to_entail_prob(res["label"], res["score"])

    if entail_prob >= supported_th:
        status = "SUPPORTED"
    elif entail_prob < unsupported_th:
        status = "UNSUPPORTED"
    else:
        status = "UNCERTAIN"

    return {
        "evidence_pack": evidence_pack,
        "nli_label": res["label"],
        "nli_score": float(res["score"]),
        "entail_prob": float(entail_prob),
        "status": status,
    }


def answer_level_verify(
    *,
    full_answer: str,
    retrieved_texts: List[str],
    nli_pipeline=None,
    ollama: OllamaClient,
    judge_model: str,
    judge_t2t=None,
    hf_judge_max_new_tokens: int = 32,
    max_evidence_chars: int,
    supported_th: float,
    unsupported_th: float,
) -> Dict[str, Any]:
    nli_res = answer_level_nli_check(
        full_answer=full_answer,
        retrieved_texts=retrieved_texts,
        nli_pipeline=nli_pipeline,
        max_evidence_chars=max_evidence_chars,
        supported_th=supported_th,
        unsupported_th=unsupported_th,
    )

    judge_res = None
    final_status = nli_res["status"]
    if nli_res["status"] == "UNCERTAIN":
        judge_res = judge_claim(
            claim=full_answer,
            evidence=nli_res["evidence_pack"],
            ollama=ollama,
            judge_model=judge_model,
            judge_t2t=judge_t2t,
            hf_max_new_tokens=hf_judge_max_new_tokens,
        )
        final_status = "SUPPORTED" if judge_res["label"] == "SUPPORTED" else "UNSUPPORTED"

    return {"nli": nli_res, "judge": judge_res, "final_status": final_status}


def judge_to_score(judge_label: str) -> float:
    return 1.0 if (judge_label or "").upper() == "SUPPORTED" else 0.0


def confidence_fast(sim: float, nli_prob: float) -> float:
    return max(0.0, min(1.0, 0.55 * float(sim) + 0.45 * float(nli_prob)))


def confidence_full(sim: float, nli_prob: float, judge_score: float) -> float:
    return max(0.0, min(1.0, 0.40 * float(sim) + 0.40 * float(nli_prob) + 0.20 * float(judge_score)))


def classify_fast(conf: float, verified_th: float, hallucinated_th: float) -> str:
    if conf >= verified_th:
        return "VERIFIED"
    if conf < hallucinated_th:
        return "HALLUCINATED"
    return "UNCERTAIN"


def classify_full(conf: float, verified_th: float, hallucinated_th: float) -> str:
    if conf >= verified_th:
        return "VERIFIED"
    if conf < hallucinated_th:
        return "HALLUCINATED"
    return "UNCERTAIN"


def has_very_high_risk_patterns(sentence: str) -> bool:
    if sentence is None:
        return False
    return bool(DOSE_REGEX.search(sentence) or SAFETY_REGEX.search(sentence))


def build_regeneration_prompt(claim: str, evidence: str) -> str:
    return (
        "You are a medical expert assistant.\n"
        "Rewrite the claim so that it is fully supported by the given evidence.\n"
        "Do NOT add new information.\n"
        "If the claim is incorrect, correct it using the evidence.\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"CLAIM:\n{claim}\n\n"
        "REVISED CLAIM:"
    )


def regenerate_claim(
    *,
    claim: str,
    evidence: str,
    ollama: OllamaClient,
    generator_model: str,
    max_tokens: int,
) -> str:
    prompt = build_regeneration_prompt(claim, evidence)
    out = ollama.generate(model=generator_model, prompt=prompt, temperature=0.0, max_tokens=max_tokens)
    return (out or "").strip()


def rebuild_final_answer_by_id(original_answer: str, results: List[Dict[str, Any]], *, nlp_sci=None) -> str:
    original_sents = split_sentences_with_ids(original_answer, nlp_sci=nlp_sci)

    replace_map = {
        int(r["sent_id"]): r["final_claim"]
        for r in results
        if r.get("final_claim") is not None and r.get("final_claim") != r.get("original_claim")
    }

    final_sents: List[str] = []
    for s in original_sents:
        sent_id = int(s["sent_id"])
        final_sents.append(str(replace_map.get(sent_id, s["text"])))

    return " ".join(final_sents)


def run_full_verification(
    *,
    original_answer: str,
    retrieved_texts: List[str],
    embedder: SentenceTransformer,
    ollama: OllamaClient,
    generator_model: str,
    judge_model: str,
    config: VerificationConfig,
    models: VerificationModels,
) -> Dict[str, Any]:
    claims_with_ids = split_sentences_with_ids(original_answer, nlp_sci=models.nlp_sci)

    final_results: List[Dict[str, Any]] = []
    for item in claims_with_ids:
        claim = item["text"]
        sent_id = item["sent_id"]

        raw_entities = extract_entities_multi_ner(
            claim,
            nlp_bc5cdr=models.nlp_bc5cdr,
            biomed_ner=models.biomed_ner,
        )
        merged_entities = merge_entities(raw_entities)
        for e in merged_entities:
            e["risk"] = RISK_MAP.get(e.get("label") or "UNKNOWN", "LOW")

        final_results.append({"sent_id": sent_id, "claim": claim, "entities": merged_entities})

    hallucination_candidates: List[Dict[str, Any]] = []
    all_scored: List[Dict[str, Any]] = []

    for item in final_results:
        sent_id = item["sent_id"]
        sentence = item["claim"]
        entities = item["entities"]

        risk_score, reasons = score_sentence_risk(sentence, entities)
        scored_item = {
            "sent_id": sent_id,
            "sentence": sentence,
            "entities": entities,
            "risk_score": risk_score,
            "reasons": reasons,
        }
        all_scored.append(scored_item)

        if risk_score >= config.risk_threshold:
            hallucination_candidates.append(scored_item)

    sentence_level_results: List[Dict[str, Any]] = []
    answer_level_result = None

    if len(hallucination_candidates) == 0:
        answer_level_result = answer_level_verify(
            full_answer=original_answer,
            retrieved_texts=retrieved_texts,
            nli_pipeline=models.nli_pipeline,
            ollama=ollama,
            judge_model=judge_model,
            judge_t2t=models.judge_t2t,
            hf_judge_max_new_tokens=config.hf_judge_max_new_tokens,
            max_evidence_chars=config.max_evidence_chars,
            supported_th=config.answer_supported_th,
            unsupported_th=config.answer_unsupported_th,
        )
    else:
        for item in hallucination_candidates:
            sent_id = item["sent_id"]
            claim_sent = item["sentence"]

            sim_result = semantic_similarity_check(
                claim_sentence=claim_sent, source_texts=retrieved_texts, embedder=embedder
            )
            evidence = sim_result["best_evidence"]
            sim_score = float(sim_result["similarity_score"])

            nli_res = nli_claim_verification(
                claim=claim_sent, evidence=evidence, nli_pipeline=models.nli_pipeline
            )
            nli_prob = nli_to_entail_prob(nli_res["label"], nli_res["score"])

            conf_fast = confidence_fast(sim_score, nli_prob)
            status_fast = classify_fast(
                conf_fast,
                verified_th=config.fast_verified_th,
                hallucinated_th=config.fast_hallucinated_th,
            )

            judge_res = None
            conf_full = None
            status_full = None

            final_claim = claim_sent
            final_conf = conf_fast
            regen_status = None

            if status_fast == "HALLUCINATED":
                if evidence is not None and sim_score >= config.sim_for_regen:
                    regenerated = regenerate_claim(
                        claim=claim_sent,
                        evidence=evidence,
                        ollama=ollama,
                        generator_model=generator_model,
                        max_tokens=config.regen_max_tokens,
                    )

                    nli2 = nli_claim_verification(
                        claim=regenerated,
                        evidence=evidence,
                        nli_pipeline=models.nli_pipeline,
                    )
                    nli2_prob = nli_to_entail_prob(nli2["label"], nli2["score"])
                    conf2_fast = confidence_fast(sim_score, nli2_prob)
                    status2_fast = classify_fast(
                        conf2_fast,
                        verified_th=config.fast_verified_th,
                        hallucinated_th=config.fast_hallucinated_th,
                    )

                    if status2_fast == "UNCERTAIN" and has_very_high_risk_patterns(regenerated):
                        judge2 = judge_claim(
                            claim=regenerated,
                            evidence=evidence,
                            ollama=ollama,
                            judge_model=judge_model,
                            judge_t2t=models.judge_t2t,
                            hf_max_new_tokens=config.hf_judge_max_new_tokens,
                        )
                        conf2_full = confidence_full(sim_score, nli2_prob, judge_to_score(judge2["label"]))
                        status2_full = classify_full(
                            conf2_full,
                            verified_th=config.full_verified_th,
                            hallucinated_th=config.full_hallucinated_th,
                        )
                    else:
                        judge2 = None
                        conf2_full = None
                        status2_full = None

                    if status2_fast == "VERIFIED" or status2_full == "VERIFIED":
                        final_claim = regenerated
                        final_status = "REGEN_VERIFIED"
                        final_conf = conf2_full if (status2_full == "VERIFIED") else conf2_fast
                        regen_status = "REGEN_ACCEPTED"
                    else:
                        final_status = "HALLUCINATED_UNFIXED"
                        regen_status = "REGEN_REJECTED"
                else:
                    final_status = "HALLUCINATED_NO_EVIDENCE"
                    regen_status = "REGEN_SKIPPED"
            else:
                if has_very_high_risk_patterns(claim_sent):
                    judge_res = judge_claim(
                        claim=claim_sent,
                        evidence=evidence,
                        ollama=ollama,
                        judge_model=judge_model,
                        judge_t2t=models.judge_t2t,
                        hf_max_new_tokens=config.hf_judge_max_new_tokens,
                    )
                    conf_full = confidence_full(sim_score, nli_prob, judge_to_score(judge_res["label"]))
                    status_full = classify_full(
                        conf_full,
                        verified_th=config.full_verified_th,
                        hallucinated_th=config.full_hallucinated_th,
                    )

                    final_conf = conf_full
                    final_status = status_full

                    if status_full == "HALLUCINATED":
                        if evidence is not None and sim_score >= config.sim_for_regen:
                            regenerated = regenerate_claim(
                                claim=claim_sent,
                                evidence=evidence,
                                ollama=ollama,
                                generator_model=generator_model,
                                max_tokens=config.regen_max_tokens,
                            )

                            nli2 = nli_claim_verification(
                                claim=regenerated,
                                evidence=evidence,
                                nli_pipeline=models.nli_pipeline,
                            )
                            nli2_prob = nli_to_entail_prob(nli2["label"], nli2["score"])
                            conf2_fast = confidence_fast(sim_score, nli2_prob)
                            status2_fast = classify_fast(
                                conf2_fast,
                                verified_th=config.fast_verified_th,
                                hallucinated_th=config.fast_hallucinated_th,
                            )

                            if status2_fast == "UNCERTAIN" and has_very_high_risk_patterns(regenerated):
                                judge2 = judge_claim(
                                    claim=regenerated,
                                    evidence=evidence,
                                    ollama=ollama,
                                    judge_model=judge_model,
                                    judge_t2t=models.judge_t2t,
                                    hf_max_new_tokens=config.hf_judge_max_new_tokens,
                                )
                                conf2_full = confidence_full(sim_score, nli2_prob, judge_to_score(judge2["label"]))
                                status2_full = classify_full(
                                    conf2_full,
                                    verified_th=config.full_verified_th,
                                    hallucinated_th=config.full_hallucinated_th,
                                )
                            else:
                                judge2 = None
                                conf2_full = None
                                status2_full = None

                            if status2_fast == "VERIFIED" or status2_full == "VERIFIED":
                                final_claim = regenerated
                                final_status = "REGEN_VERIFIED"
                                final_conf = conf2_full if (status2_full == "VERIFIED") else conf2_fast
                                regen_status = "REGEN_ACCEPTED"
                            else:
                                final_status = "HALLUCINATED_UNFIXED"
                                regen_status = "REGEN_REJECTED"
                        else:
                            final_status = "HALLUCINATED_NO_EVIDENCE"
                            regen_status = "REGEN_SKIPPED"
                else:
                    # If the sentence isn't very-high-risk, don't spend judge tokens.
                    # Keep FAST result, but downgrade UNCERTAIN to a low-priority flag.
                    final_status = "UNCERTAIN_LOW_PRIORITY" if (status_fast == "UNCERTAIN") else status_fast

            sentence_level_results.append(
                {
                    "sent_id": sent_id,
                    "original_claim": claim_sent,
                    "final_claim": final_claim,
                    "evidence": evidence,
                    "similarity_score": sim_score,
                    "nli_label": nli_res["label"],
                    "nli_score": float(nli_res["score"]),
                    "fast_confidence": conf_fast,
                    "fast_status": status_fast,
                    "judge_label": (judge_res["label"] if judge_res else None),
                    "final_confidence": final_conf,
                    "final_status": final_status,
                    "regen_status": regen_status,
                }
            )

    final_answer = rebuild_final_answer_by_id(
        original_answer,
        sentence_level_results,
        nlp_sci=models.nlp_sci,
    )

    intervention_detected = any(
        r.get("final_status")
        in [
            "REGEN_VERIFIED",
            "HALLUCINATED_UNFIXED",
            "UNCERTAIN",
            "UNCERTAIN_LOW_PRIORITY",
        ]
        for r in sentence_level_results
    )

    answer_level_intervention = False
    if answer_level_result is not None and answer_level_result.get("final_status") == "UNSUPPORTED":
        answer_level_intervention = True

    if intervention_detected:
        final_answer = DISCLAIMER + final_answer
    if answer_level_intervention:
        final_answer = ANSWER_LEVEL_DISCLAIMER + final_answer

    return {
        "final_answer": final_answer,
        "original_answer": original_answer,
        "answer_level_result": answer_level_result,
        "sentence_level_results": sentence_level_results,
        "hallucination_candidates": hallucination_candidates,
        "all_scored": all_scored,
    }
