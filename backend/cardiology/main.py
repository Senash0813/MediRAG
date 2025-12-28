from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
import numpy as np

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    # -----------------------------
    # USER QUERY
    # -----------------------------
    query = input("Enter your cardiology question: ").strip()

    # -----------------------------
    # STEP 1: InBEDDER + Projection
    # -----------------------------
    print("\n[1] Embedding and projecting user query...")
    proj_emb = embed_and_project(query)

    # -----------------------------
    # STEP 2: HyDE Generation
    # -----------------------------
    print("[2] Generating hypothetical documents (HyDE)...")
    hyde_emb, hyde_docs = generate_hypothetical_docs(query)

    # -----------------------------
    # STEP 3: Fusion
    # -----------------------------
    print("[3] Fusing embeddings...")
    final_emb = fuse_embeddings(
        proj_query=proj_emb,
        hyde_query=hyde_emb,
        alpha=0.5
    )

    # -----------------------------
    # STEP 4: Retrieval
    # -----------------------------
    print("[4] Loading FAISS index and retrieving top documents...")
    vectorstore = load_vectorstore()

    # Use FAISS directly to also get similarity scores
    docs_and_scores = vectorstore.similarity_search_by_vector_with_score(
        embedding=final_emb.tolist(),
        k=5
    )

    # -----------------------------
    # PRINT RETRIEVAL RESULTS
    # -----------------------------
    print("\nRETRIEVED DOCUMENTS (with similarity scores)")
    print("=" * 70)

    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        print(f"\n--- Rank {rank} ---")
        print(f"FAISS distance score: {score:.4f}")
        print("-" * 70)
        print(doc.page_content[:300])
        print("-" * 70)

    # -----------------------------
    # PRINT HyDE OUTPUTS
    # -----------------------------
    print("\nGENERATED HYPOTHETICAL ANSWERS (HyDE)")
    print("=" * 70)

    for i, doc in enumerate(hyde_docs, 1):
        print(f"\n--- Hypothesis {i} ---")
        print(
            doc.replace(
                "Represent the cardiology document for retrieval:\n",
                ""
            ).strip()
        )

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()
