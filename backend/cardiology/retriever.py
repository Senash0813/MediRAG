import numpy as np
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# PATHS
# -----------------------------
INDEX_PATH = "vectorstore/faiss_index"
INSTRUCTOR_PATH = "models/instructor-large"

# -----------------------------
# UTILS
# -----------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Safely L2-normalize a vector"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# -----------------------------
# LOAD VECTORSTORE
# -----------------------------
def load_vectorstore() -> FAISS:
    """
    Load FAISS index using the SAME embedding model
    that was used during indexing (Instructor-large).
    """

    embeddings_model = HuggingFaceEmbeddings(
        model_name=INSTRUCTOR_PATH
    )

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore

# -----------------------------
# RETRIEVE (VECTOR-LEVEL)
# -----------------------------
def retrieve(
    vectorstore: FAISS,
    query_embedding: np.ndarray,
    k: int = 5,
    return_scores: bool = False,
):
    """
    Perform similarity search using a PRE-COMPUTED embedding.

    IMPORTANT:
    - query_embedding must already be in the SAME vector space
      as the indexed documents (768-d projected Instructor space).
    - No re-embedding happens here.
    """

    # Ensure correct shape + normalization
    query_embedding = l2_normalize(query_embedding).astype(np.float32)

    if return_scores:
        # ✅ Correct modern LangChain API
        results: List[Tuple] = (
            vectorstore.similarity_search_with_score_by_vector(
                embedding=query_embedding.tolist(),
                k=k
            )
        )
        return results

    else:
        results = vectorstore.similarity_search_by_vector(
            embedding=query_embedding.tolist(),
            k=k
        )
        return results

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    from embedder import embed_and_project

    vectorstore = load_vectorstore()

    query = "How does sevoflurane postconditioning reduce myocardial injury?"
    query_vec = embed_and_project(query)

    results = retrieve(vectorstore, query_vec, k=5, return_scores=True)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Result {i} | Score: {score:.4f} ---")
        print(doc.page_content[:300])
