from __future__ import annotations

from typing import Any, Dict, List

from tqdm import tqdm

from medirag.retrieval.retrieve import retrieve_top_k_unique
from medirag.verification.scoring import (
    assign_risk_flags,
    compute_evidence_score,
    final_doc_score,
    get_quality_tier,
    verify_document,
)
from medirag.verification.s2_client import SemanticScholarClient


def verify_and_rank_final_flat(
    *,
    query: str,
    data: List[Dict[str, Any]],
    embedder,
    index,
    s2_client: SemanticScholarClient,
    k: int,
):
    retrieved = retrieve_top_k_unique(query=query, data=data, embedder=embedder, index=index, k=k)
    print("Retrieved unique papers:", len(retrieved))

    verified_docs = []

    for doc in tqdm(retrieved):
        s2_meta = s2_client.fetch_metadata(doc)

        pub_types = s2_meta.get("publicationTypes") or []
        evidence_score = compute_evidence_score(pub_types)

        verification = verify_document(doc, s2_meta)
        verification["evidence_score"] = evidence_score

        final_score = final_doc_score(doc["semantic_score"], s2_meta, evidence_score)

        flattened_doc = {
            "qa_id": doc.get("qa_id"),
            "paper_id": doc.get("paper_id"),
            "title": s2_meta.get("title") or doc.get("paper_title"),
            "passage_text": doc.get("passage_text"),
            "semantic_score": round(doc["semantic_score"], 3),
            "final_score": round(final_score, 3),
            "quality_tier": get_quality_tier(final_score),
            "citation_count": s2_meta.get("citationCount", 0),
            "influential_citation_count": s2_meta.get("influentialCitationCount", 0),
            "year": s2_meta.get("year"),
            "publication_types": pub_types,
            "evidence_score": evidence_score,
        }

        flattened_doc["risk_flags"] = assign_risk_flags({**flattened_doc, "freshness": verification["freshness"]})

        flattened_doc.update(verification)

        verified_docs.append(flattened_doc)

    verified_docs.sort(key=lambda x: x["final_score"], reverse=True)
    print("Number of verified docs:", len(verified_docs))
    return verified_docs
