import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# PATHS
# -----------------------------
INDEX_PATH = "vectorstore/faiss_index"
INSTRUCTOR_PATH = "models/instructor-large"

# -----------------------------
# UTILS
# -----------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Ensure vector is L2-normalized before FAISS search"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# -----------------------------
# LOAD VECTORSTORE
# -----------------------------
def load_vectorstore():
    """
    Loads FAISS index with the SAME embedding model
    used during index construction (Instructor-large).
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
    k: int = 5
):
    """
    Perform similarity search using a PRE-COMPUTED embedding.

    IMPORTANT:
    - query_embedding must already be in the SAME vector space
      as the indexed documents (Instructor / projected / HyDE).
    - LangChain will NOT re-embed anything here.
    """

    # Safety: normalize before FAISS search
    query_embedding = l2_normalize(query_embedding)

    docs = vectorstore.similarity_search_by_vector(
        embedding=query_embedding.tolist(),
        k=k
    )

    return docs

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    from embedder import embed_and_project  # your InBEDDER module

    vectorstore = load_vectorstore()

    query = "How does sevoflurane postconditioning reduce myocardial injury?"
    query_vec = embed_and_project(query)

    results = retrieve(vectorstore, query_vec, k=5)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:300])
