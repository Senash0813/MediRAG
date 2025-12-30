from collections import defaultdict

def rrf_fusion(faiss_results, bm25_results, k=60):
    scores = defaultdict(float)
    records = {}

    for r in faiss_results:
        doc_id = r["qa_id"]
        scores[doc_id] += 1 / (k + r["faiss_rank"])
        records[doc_id] = r

    for r in bm25_results:
        doc_id = r["qa_id"]
        scores[doc_id] += 1 / (k + r["bm25_rank"])
        records[doc_id] = r

    fused = []
    for doc_id, score in scores.items():
        rec = records[doc_id].copy()
        rec["rrf_score"] = score
        fused.append(rec)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused
