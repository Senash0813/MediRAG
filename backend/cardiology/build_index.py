import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document as LC_Document

# ================================
# CONFIG
# ================================

# Local Instructor model path
INSTRUCTOR_MODEL_PATH = "models/instructor-large"

# Data and output paths
DATA_PATH = "../../data/cardiology/miriad_cardiology.json"
VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")

# Ensure output directory exists
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ================================
# BUILD INDEX
# ================================

def main():
    print("🔹 Loading cardiology data...")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Limit to 100 rows for now (as you requested earlier)
    raw_data = raw_data[:100]

    print(f"🔹 Loaded {len(raw_data)} records")

    # Prepare documents (FULL ANSWER → ONE VECTOR)
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

    # Load Instructor embedding model
    print("🔹 Loading Instructor embedding model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=INSTRUCTOR_MODEL_PATH
    )

    # Build FAISS index (NO SPLITTING)
    print("🔹 Building FAISS index (full-answer embeddings)...")
    vectorstore = FAISS.from_documents(
        documents,
        embeddings_model
    )

    # Save FAISS index
    vectorstore.save_local(INDEX_PATH)

    print(f"\n✅ FAISS index successfully saved to:")
    print(f"   {INDEX_PATH}")

# ================================
# ENTRY POINT
# ================================

if __name__ == "__main__":
    main()
