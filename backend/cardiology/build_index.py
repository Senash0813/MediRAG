import os
import json
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document as LC_Document

# ================================
# CONFIG
# ================================
INSTRUCTOR_MODEL_PATH = "models/instructor-large"

DATA_PATH = "../../data/cardiology/miriad_cardiology.json"
VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ================================
# BUILD INDEX
# ================================
def main():
    print("🔹 Loading cardiology data...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = raw_data[:10000]  # limit for testing
    print(f"🔹 Loaded {len(raw_data)} records")

    # Prepare documents
    instruction = "Represent the cardiology answer for retrieval:"
    documents = []

    for row in raw_data:
        answer = row.get("answer")
        if isinstance(answer, str) and answer.strip():
            documents.append(
                LC_Document(
                    page_content=f"{instruction}\n{answer}",
                    metadata=row
                )
            )
    print(f"🔹 Prepared {len(documents)} documents for embedding")

    # Load embedding model
    print("🔹 Loading Instructor embedding model...")
    embeddings_model = HuggingFaceEmbeddings(model_name=INSTRUCTOR_MODEL_PATH)

    # OPTIONAL: check embedding of first document
    first_embedding = embeddings_model.embed_documents([documents[0].page_content])
    print(f"🔹 Sample embedding shape (first document): {np.array(first_embedding).shape}")

    # OPTIONAL: check embedding shapes for all docs
    all_embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])
    print(f"🔹 All embeddings shape: {np.array(all_embeddings).shape}")  # (num_docs, embedding_dim)

    # Build FAISS index
    print("🔹 Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings_model)

    # Save index
    vectorstore.save_local(INDEX_PATH)
    print(f"\n✅ FAISS index successfully saved to: {INDEX_PATH}")

# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    main()
