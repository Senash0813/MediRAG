from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
from query_classifier import classify_query
import numpy as np
import requests
from typing import List
from langchain_community.docstore.document import Document

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi:2.7b"

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector safely."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def generate_final_answer(
    query: str, 
    retrieved_docs: List[Document],
    intent: str = "CARDIOLOGY",
    confidence: float = 1.0
) -> str:
    """
    Call locally running Ollama API to generate final answer.
    
    Args:
        query: User query
        retrieved_docs: Retrieved documents from vectorstore
        intent: Query intent classification (CARDIOLOGY, GENERAL_GREETING, etc.)
        confidence: Confidence score of classification
    """
    
    # ======================
    # GREETING RESPONSE
    # ======================
    if intent == "GENERAL_GREETING":
        greeting_responses = [
            "Hello! I'm a medical AI assistant specialized in cardiology. How can I help you today? Feel free to ask any cardiology questions!",
            "Hi there! 👋 I'm here to help with cardiology questions. What would you like to know?",
            "Greetings! I'm a cardiology assistant. Ask me anything about heart health, cardiovascular diseases, or cardiology procedures!"
        ]
        import random
        return random.choice(greeting_responses)
    
    # ======================
    # UNRELATED / OFF-TOPIC RESPONSE (NEW)
    # ======================
    if intent == "UNRELATED":
        unrelated_responses = [
            "I'm specialized in cardiology and medical topics. Your question appears to be outside my area of expertise. Could you ask something about heart health, cardiovascular diseases, or cardiology procedures?",
            "That question is outside my scope as a cardiology specialist. I'm here to help with heart health and cardiovascular topics. What cardiology questions do you have?",
            "I'm focused on cardiology assistance. I can't help with that particular topic, but I'd be happy to answer any questions about the heart, heart disease, or cardiac treatment!",
            "I can best assist with cardiology-related questions. Please feel free to ask me about heart conditions, cardiovascular procedures, or cardiac health!"
        ]
        import random
        return random.choice(unrelated_responses)
    
    # ======================
    # OTHER MEDICAL DOMAIN
    # ======================
    if intent == "OTHER_MEDICAL":
        return "I'm specialized in cardiology and therefore cannot provide reliable information about this medical domain. I recommend consulting with a specialist in that field. However, if you have any cardiology-related questions, I'd be happy to help!"
    
    # ======================
    # UNCLEAR / LOW CONFIDENCE
    # ======================
    if intent == "UNCLEAR":
        if not retrieved_docs:
            return "I couldn't find relevant information in my cardiology database for this query. Could you rephrase your question or ask something specifically about cardiology?"
        # Fall through to cardiology processing with retrieved docs
    
    # ======================
    # CARDIOLOGY RESPONSE (DEFAULT)
    # ======================
    if not retrieved_docs:
        return "No relevant documents found in the cardiology database."

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""You are an expert cardiology assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.
Provide accurate, medical, and factual information.

Context:
{context}

Question:
{query}

Answer:
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

        candidate = None
        if isinstance(data, dict):
            candidate = data.get("response") or data.get("text")
            if not candidate and "choices" in data and isinstance(data["choices"], list):
                first = data["choices"][0]
                if isinstance(first, dict):
                    candidate = first.get("text") or first.get("message") or first.get("response")

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
# MAIN PIPELINE (UPDATED WITH INTENT CLASSIFICATION)
# --------------------------------------------------
def main():
    print("\n🚀 MediRAG Cardiology Assistant (Intent-Aware)\n")

    # ===================================
    # USER QUERY
    # ===================================
    query = input("Enter your question: ").strip()
    if not query:
        print("❌ Empty query provided. Exiting.")
        return

    # ===================================
    # STEP 0: QUERY CLASSIFICATION
    # ===================================
    print("\n[0] Classifying query intent...")
    vectorstore = load_vectorstore()
    intent, confidence, metadata = classify_query(query, vectorstore=vectorstore)
    
    print(f"📊 Query Intent: {intent}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Similarity Score: {metadata['similarity_score']:.4f}")
    if metadata.get("top_k_similarity"):
        print(f"   Top-K Similarities: {[round(x, 4) for x in metadata['top_k_similarity']]}")
    print()

    # ===================================
    # HANDLE NON-CARDIOLOGY QUERIES
    # ===================================
    if intent in ["GENERAL_GREETING", "UNRELATED", "OTHER_MEDICAL"]:
        print("[1] Generating context-aware response...")
        final_answer = generate_final_answer(query, [], intent=intent, confidence=confidence)
        
        print("\n====================== RESPONSE ======================")
        print(final_answer if final_answer else "❌ Empty response returned.")
        print("====================================================\n")
        return

    # ===================================
    # CARDIOLOGY QUERY PIPELINE
    # ===================================
    print("[1] Embedding and projecting user query...")
    proj_emb = embed_and_project(query)
    proj_emb = l2_normalize(np.array(proj_emb))

    # ===================================
    # STEP 2: HYDE GENERATION (ONLY FOR CARDIOLOGY)
    # ===================================
    print("[2] Generating hypothetical documents (HyDE)...")
    print("   ⚡ HyDE only generated for cardiology queries!")
    hyde_emb, hyde_docs = generate_hypothetical_docs(query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    # ===================================
    # STEP 3: FUSION
    # ===================================
    print("[3] Fusing embeddings (Projection + HyDE)...")
    final_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=0.5)
    final_emb = l2_normalize(np.array(final_emb))

    # ===================================
    # STEP 4: RETRIEVAL
    # ===================================
    print("[4] Retrieving relevant cardiology documents...")

    # Debug: inspect FAISS index
    try:
        faiss_index = getattr(vectorstore, "index", None)
        if faiss_index is not None:
            try:
                print(f"   FAISS Index: {faiss_index.ntotal} vectors, dimension {faiss_index.d}")
            except Exception:
                pass
    except Exception:
        pass

    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding=final_emb.tolist(),
            k=5
        )
    except Exception as e:
        print("❌ Error during FAISS retrieval:", e)
        docs_and_scores = []

    if not docs_and_scores:
        print("❌ No documents retrieved from the cardiology database.")
        return

    # ===================================
    # DISPLAY RETRIEVAL RESULTS
    # ===================================
    print("\n📄 TOP RETRIEVED CARDIOLOGY DOCUMENTS:")
    print("=" * 80)
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        print(f"\n[Document {rank}] Score: {score:.4f}")
        print("-" * 80)
        print(doc.page_content[:400])
        print("-" * 80)

    # ===================================
    # DISPLAY HYDE OUTPUTS
    # ===================================
    print("\n🧠 GENERATED HYPOTHETICAL ANSWERS (HyDE):")
    print("=" * 80)
    for i, doc in enumerate(hyde_docs, 1):
        print(f"\n[Hypothesis {i}]")
        clean_doc = doc.replace("Represent the cardiology document for retrieval:\n", "").strip()
        print(clean_doc[:300] + ("..." if len(clean_doc) > 300 else ""))

    # ===================================
    # STEP 5: FINAL ANSWER GENERATION
    # ===================================
    print("\n[5] Generating final answer using LLM...")
    retrieved_docs = [doc for doc, _ in docs_and_scores]
    final_answer = generate_final_answer(query, retrieved_docs, intent=intent, confidence=confidence)

    # ===================================
    # DISPLAY FINAL ANSWER
    # ===================================
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(final_answer if final_answer else "❌ Empty answer returned.")
    print("=" * 80 + "\n")

    print("✅ Pipeline execution completed successfully.\n")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()
