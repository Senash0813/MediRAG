from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
import numpy as np
import requests
from typing import List
from langchain_community.docstore.document import Document

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector safely."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def generate_final_answer(query: str, retrieved_docs: List[Document]) -> str:
    """Call locally running Ollama API to generate final answer."""
    if not retrieved_docs:
        return "No relevant documents found."

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are a cardiology assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer (concise, medical, factual):
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()

        # Ollama may return different shapes; try a few common keys
        candidate = None
        if isinstance(data, dict):
            candidate = data.get("response") or data.get("text")
            # older/newer formats might nest choices
            if not candidate and "choices" in data and isinstance(data["choices"], list):
                first = data["choices"][0]
                if isinstance(first, dict):
                    candidate = first.get("text") or first.get("message") or first.get("response")

        # fallback: if API returned a top-level string
        if candidate is None and isinstance(data, str):
            candidate = data

        if not candidate or not str(candidate).strip():
            return "⚠️ Ollama returned empty output or unexpected JSON format: " + str(data)[:400]

        return str(candidate).strip()

    except requests.exceptions.ConnectionError:
        return "❌ Ollama server not running. Start it with: ollama serve"
    except Exception as e:
        return f"❌ Ollama error: {e}"

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    print("\n🚀 MediRAG Cardiology Assistant\n")

    # -----------------------------
    # USER QUERY
    # -----------------------------
    query = input("Enter your cardiology question: ").strip()
    if not query:
        print("❌ Empty query provided. Exiting.")
        return

    # -----------------------------
    # STEP 1: EMBEDDING + PROJECTION
    # -----------------------------
    print("\n[1] Embedding and projecting user query...")
    proj_emb = embed_and_project(query)
    proj_emb = l2_normalize(np.array(proj_emb))

    # -----------------------------
    # STEP 2: HyDE Generation
    # -----------------------------
    print("[2] Generating hypothetical documents (HyDE)...")
    hyde_emb, hyde_docs = generate_hypothetical_docs(query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    # -----------------------------
    # STEP 3: Fusion
    # -----------------------------
    print("[3] Fusing embeddings...")
    final_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=0.5)
    final_emb = l2_normalize(np.array(final_emb))

    # -----------------------------
    # STEP 4: Retrieval
    # -----------------------------
    print("[4] Loading FAISS index and retrieving top documents...")
    vectorstore = load_vectorstore()

    # Debug: inspect FAISS index and embedding shape
    try:
        faiss_index = getattr(vectorstore, "index", None)
        if faiss_index is not None:
            try:
                print(f"FAISS index total vectors: {faiss_index.ntotal}, dim: {faiss_index.d}")
            except Exception:
                print("FAISS index present but couldn't read ntotal/d")
        else:
            print("Loaded vectorstore has no attribute 'index'")
    except Exception as e:
        print("Error inspecting vectorstore.index:", e)

    print(f"Final embedding shape: {getattr(final_emb, 'shape', None)}, dtype: {getattr(final_emb, 'dtype', None)}")

    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding=final_emb.tolist(),
            k=5
        )
    except Exception as e:
        print("Error during FAISS similarity search:", e)
        docs_and_scores = []

    if not docs_and_scores:
        print("❌ No documents retrieved from FAISS.")
        return

    # -----------------------------
    # PRINT RETRIEVAL RESULTS
    # -----------------------------
    print("\n📄 RETRIEVED DOCUMENTS (with similarity scores)")
    print("=" * 80)
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        print(f"\n--- Rank {rank} ---")
        print(f"FAISS distance score: {score:.4f}")
        print("-" * 80)
        print(doc.page_content[:400])
        print("-" * 80)

    # -----------------------------
    # PRINT HyDE OUTPUTS
    # -----------------------------
    print("\n🧠 GENERATED HYPOTHETICAL ANSWERS (HyDE)")
    print("=" * 80)
    for i, doc in enumerate(hyde_docs, 1):
        print(f"\n--- Hypothesis {i} ---")
        print(doc.replace("Represent the cardiology document for retrieval:\n", "").strip())

    # -----------------------------
    # STEP 5: FINAL ANSWER WITH OLLAMA
    # -----------------------------
    print("\n[5] Generating final answer with LLaMA 3.2 (Ollama)...")
    retrieved_docs = [doc for doc, _ in docs_and_scores]
    final_answer = generate_final_answer(query, retrieved_docs)

    # -----------------------------
    # PRINT FINAL ANSWER
    # -----------------------------
    print("\n====================== FINAL ANSWER ======================")
    print(final_answer if final_answer else "❌ Empty answer returned.")
    print("=========================================================")

    print("\n✅ Pipeline execution completed successfully.\n")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()
