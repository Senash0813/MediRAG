from __future__ import annotations

from typing import Any, Dict, List


def split_sentences(text, *, nlp_sci):
    doc = nlp_sci(text)
    return [sent.text.strip() for sent in doc.sents]


def extract_entities_multi_ner(text, *, nlp_bc5cdr, biomed_ner):
    entities = []

    # 1️⃣ SciSpacy BC5CDR (diseases, chemicals)
    doc = nlp_bc5cdr(text)
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    # 2️⃣ Transformer Biomedical NER (more general coverage)
    for ent in biomed_ner(text):
        entities.append({"text": ent["word"], "label": ent["entity_group"]})

    return entities


def merge_entities(entities):
    seen = {}
    for e in entities:
        key = e["text"].lower()  # case-insensitive merge
        if key not in seen:
            seen[key] = e
    return list(seen.values())


RISK_MAP = {
    "DISEASE": "HIGH",
    "CHEMICAL": "HIGH",
    "DRUG": "HIGH",
    "PROCEDURE": "MEDIUM",
    "ANATOMY": "LOW",
    "SYMPTOM": "MEDIUM",
}


def build_final_results(claims: List[str], *, nlp_bc5cdr, biomed_ner):
    # Suppose you already have your RAG output split into claims
    # e.g., claims = sentences from RAG output

    final_results = []

    for claim in claims:
        # 1. Extract entities from all NER models
        raw_entities = extract_entities_multi_ner(claim, nlp_bc5cdr=nlp_bc5cdr, biomed_ner=biomed_ner)

        # 2. Merge duplicates
        merged_entities = merge_entities(raw_entities)

        # 3. Map risk
        for e in merged_entities:
            e["risk"] = RISK_MAP.get(e["label"], "LOW")

        # 4. Append final claim with entities
        final_results.append({"claim": claim, "entities": merged_entities})

    # ✅ Final results ready
    return final_results


def filter_high_risk_sentences(final_results):
    filtered = []

    for item in final_results:
        sentence = item["claim"]
        entities = item["entities"]

        high_risk_entities = [e for e in entities if e.get("risk") == "HIGH"]

        if not high_risk_entities:
            continue

        filtered.append({"sentence": sentence, "entities": high_risk_entities})

    return filtered
