import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple

from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
from main import generate_final_answer, generate_out_of_domain_answer, l2_normalize


# ============================================================
# CONFIG
# ============================================================
TEST_DATA_PATH = "qa_pairs.json"   # your current file
OUTPUT_DIR = "ablation_outputs"
TOP_K = 5
ALPHA = 0.5

DEFAULT_DOMAIN_MAX_DISTANCE_TEXT = float(
    os.getenv("CARDIOLOGY_DOMAIN_MAX_DISTANCE_TEXT", "0.22")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TEST DATA
# Uses your format:
# {
#   "data": [
#       {"question": "...", "answer": "..."}
#   ]
# }
# ============================================================
def load_test_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = raw.get("data", [])

    formatted = []
    for i, row in enumerate(data):
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()

        if not question or not answer:
            continue

        formatted.append({
            "id": i + 1,
            "question": question,
            "ground_truth": answer
        })

    return formatted


# ============================================================
# DOMAIN GATE
# ============================================================
def domain_gate_text(vectorstore, query: str, max_distance: float = DEFAULT_DOMAIN_MAX_DISTANCE_TEXT) -> Tuple[bool, float]:
    try:
        res = vectorstore.similarity_search_with_score(query=query, k=1)
    except Exception:
        return True, None

    if not res:
        return False, None

    best_score = float(res[0][1])
    return best_score <= max_distance, best_score


# ============================================================
# RETRIEVAL HELPERS
# ============================================================
def retrieve_by_vector(vectorstore, query_embedding: np.ndarray, k: int = TOP_K):
    query_embedding = l2_normalize(np.array(query_embedding)).astype(np.float32)

    return vectorstore.similarity_search_with_score_by_vector(
        embedding=query_embedding.tolist(),
        k=k
    )


def retrieve_baseline_text(vectorstore, query: str, k: int = TOP_K):
    return vectorstore.similarity_search_with_score(query=query, k=k)


def docs_only(docs_and_scores):
    return [doc for doc, _ in docs_and_scores]


def docs_text_only(docs_and_scores):
    return [doc.page_content for doc, _ in docs_and_scores]


# ============================================================
# PIPELINE VARIANTS
# ============================================================
def run_baseline_rag(vectorstore, query: str, k: int = TOP_K) -> Dict[str, Any]:
    docs_and_scores = retrieve_baseline_text(vectorstore, query, k=k)
    retrieved_docs = docs_only(docs_and_scores)
    answer = generate_final_answer(query, retrieved_docs)

    return {
        "variant": "baseline_rag",
        "query": query,
        "answer": answer,
        "retrieved_docs": docs_text_only(docs_and_scores),
        "scores": [float(score) for _, score in docs_and_scores],
    }


def run_hyde_only(vectorstore, query: str, k: int = TOP_K) -> Dict[str, Any]:
    hyde_emb, hyde_docs = generate_hypothetical_docs(query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    docs_and_scores = retrieve_by_vector(vectorstore, hyde_emb, k=k)
    retrieved_docs = docs_only(docs_and_scores)
    answer = generate_final_answer(query, retrieved_docs)

    return {
        "variant": "hyde_only",
        "query": query,
        "answer": answer,
        "retrieved_docs": docs_text_only(docs_and_scores),
        "scores": [float(score) for _, score in docs_and_scores],
        "hyde_docs": hyde_docs,
    }


def run_projection_only(vectorstore, query: str, k: int = TOP_K) -> Dict[str, Any]:
    proj_emb = embed_and_project(query)
    proj_emb = l2_normalize(np.array(proj_emb))

    docs_and_scores = retrieve_by_vector(vectorstore, proj_emb, k=k)
    retrieved_docs = docs_only(docs_and_scores)
    answer = generate_final_answer(query, retrieved_docs)

    return {
        "variant": "projection_only",
        "query": query,
        "answer": answer,
        "retrieved_docs": docs_text_only(docs_and_scores),
        "scores": [float(score) for _, score in docs_and_scores],
    }


def run_fusion_no_domain_gate(vectorstore, query: str, k: int = TOP_K, alpha: float = ALPHA) -> Dict[str, Any]:
    proj_emb = embed_and_project(query)
    proj_emb = l2_normalize(np.array(proj_emb))

    hyde_emb, hyde_docs = generate_hypothetical_docs(query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    fused_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=alpha)
    fused_emb = l2_normalize(np.array(fused_emb))

    docs_and_scores = retrieve_by_vector(vectorstore, fused_emb, k=k)
    retrieved_docs = docs_only(docs_and_scores)
    answer = generate_final_answer(query, retrieved_docs)

    return {
        "variant": "fusion_no_domain_gate",
        "query": query,
        "answer": answer,
        "retrieved_docs": docs_text_only(docs_and_scores),
        "scores": [float(score) for _, score in docs_and_scores],
        "hyde_docs": hyde_docs,
    }


def run_full_cardiorag(vectorstore, query: str, k: int = TOP_K, alpha: float = ALPHA) -> Dict[str, Any]:
    in_domain, gate_score = domain_gate_text(vectorstore, query)

    if not in_domain:
        answer = generate_out_of_domain_answer(query)
        return {
            "variant": "full_cardiorag",
            "query": query,
            "answer": answer,
            "retrieved_docs": [],
            "scores": [],
            "domain_gate_passed": False,
            "domain_gate_score": gate_score,
        }

    proj_emb = embed_and_project(query)
    proj_emb = l2_normalize(np.array(proj_emb))

    hyde_emb, hyde_docs = generate_hypothetical_docs(query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    fused_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=alpha)
    fused_emb = l2_normalize(np.array(fused_emb))

    docs_and_scores = retrieve_by_vector(vectorstore, fused_emb, k=k)
    retrieved_docs = docs_only(docs_and_scores)
    answer = generate_final_answer(query, retrieved_docs)

    return {
        "variant": "full_cardiorag",
        "query": query,
        "answer": answer,
        "retrieved_docs": docs_text_only(docs_and_scores),
        "scores": [float(score) for _, score in docs_and_scores],
        "hyde_docs": hyde_docs,
        "domain_gate_passed": True,
        "domain_gate_score": gate_score,
    }


# ============================================================
# RUN ALL VARIANTS
# ============================================================
def run_ablation(limit: int = None):
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()

    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_data = load_test_data(TEST_DATA_PATH)

    if limit is not None:
        test_data = test_data[:limit]

    print(f"Total evaluation samples: {len(test_data)}")

    variants = {
        "baseline_rag": run_baseline_rag,
        "hyde_only": run_hyde_only,
        "projection_only": run_projection_only,
        "fusion_no_domain_gate": run_fusion_no_domain_gate,
        "full_cardiorag": run_full_cardiorag,
    }

    for variant_name, fn in variants.items():
        print("\n" + "=" * 60)
        print(f"Running variant: {variant_name}")
        print("=" * 60)

        outputs = []
        start = time.time()

        for i, row in enumerate(test_data, start=1):
            query = row["question"]
            ground_truth = row["ground_truth"]

            try:
                result = fn(vectorstore, query)
                result["id"] = row["id"]
                result["ground_truth"] = ground_truth
                outputs.append(result)

                print(f"[{variant_name}] {i}/{len(test_data)} done")

            except Exception as e:
                outputs.append({
                    "variant": variant_name,
                    "id": row["id"],
                    "query": query,
                    "ground_truth": ground_truth,
                    "error": str(e),
                    "answer": "",
                    "retrieved_docs": [],
                    "scores": []
                })
                print(f"[{variant_name}] {i}/{len(test_data)} ERROR: {e}")

        elapsed = time.time() - start
        print(f"Finished {variant_name} in {elapsed:.2f}s")

        output_path = os.path.join(OUTPUT_DIR, f"{variant_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)

        print(f"Saved: {output_path}")

    print("\nAll ablation variants completed successfully.")


if __name__ == "__main__":
    # Change limit if you want quick test first
    run_ablation(limit=20)