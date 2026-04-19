from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
import numpy as np
import requests
import os
from typing import List
import re
from langchain_community.docstore.document import Document

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi:2.7b"


_GREETING_THANKS_PATTERN = re.compile(
    r"^\s*(?:"
    r"hi|hello|hey|hiya|howdy|greetings"
    r"|good\s+(?:morning|afternoon|evening)"
    r"|thanks|thank\s+you|thx|ty"
    r"|how\s+are\s+you|how\s+r\s+you|hru"
    r"|what'?s\s+up|whats\s+up"
    r")"
    r"(?:\s+(?:there|assistant|medirag|cardiorag|doc|doctor))?"
    r"\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_GOODBYE_PATTERN = re.compile(
    r"^\s*(?:bye|goodbye|see\s+you|see\s+ya|later|cya|take\s+care)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_IDENTITY_PATTERN = re.compile(
    r"^\s*(?:"
    r"who\s+are\s+you|what\s+are\s+you|are\s+you\s+(?:a\s+doctor|real|human)"
    r"|what\s+is\s+medirag|what\s+is\s+cardiorag"
    r")\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_HELP_PATTERN = re.compile(
    r"^\s*(?:help|what\s+can\s+you\s+do|how\s+do\s+i\s+use\s+this|how\s+does\s+this\s+work)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def is_greeting_intent(user_text: str) -> bool:
    """Backward-compatible greeting detector.

    True if `user_text` is *only* a greeting/thanks/smalltalk.
    """
    if not user_text:
        return False

    text = user_text.strip()
    if not text:
        return False

    # Avoid greeting classification for longer, contentful messages.
    if len(text) > 80:
        return False

    return _GREETING_THANKS_PATTERN.match(text) is not None


def detect_general_intent(user_text: str) -> str | None:
    """Detect short general intents that should bypass RAG.

    Returns one of: greeting, goodbye, identity, help; or None.
    Kept conservative to avoid stealing real medical queries.
    """
    if not user_text:
        return None

    text = user_text.strip()
    if not text:
        return None

    # Keep conservative: general intents are typically short.
    if len(text) > 120:
        return None

    if _GREETING_THANKS_PATTERN.match(text):
        return "greeting"
    if _GOODBYE_PATTERN.match(text):
        return "goodbye"
    if _IDENTITY_PATTERN.match(text):
        return "identity"
    if _HELP_PATTERN.match(text):
        return "help"

    return None


def generate_greeting_answer() -> str:
    """Friendly greeting shown when the user just says hi/thanks."""
    return (
        "Hi — I’m MediRAG Cardiology. Ask me a cardiology question and I’ll answer using the cardiology knowledge base. "
        "If you share key details (age/sex, symptoms, PMH, meds, vitals, ECG/echo/troponin), I can be more specific. "
        "Examples: ‘How is atrial fibrillation managed?’ • ‘Workup for chest pain?’ • ‘When to anticoagulate for AF?’"
    )


def generate_general_intent_answer(intent: str) -> str:
    """Canned response for general intents."""
    if intent == "greeting":
        return generate_greeting_answer()
    if intent == "goodbye":
        return "Glad to help. If you have another cardiology question, just send it."
    if intent == "identity":
        return (
            "I’m MediRAG Cardiology — a cardiology-focused Q&A assistant. "
            "I answer by retrieving relevant cardiology reference text and generating a concise response from that context."
        )
    if intent == "help":
        return (
            "Ask a cardiology question in one message. Helpful details: symptoms + timeline, age/sex, PMH, meds, vitals, and any ECG/echo/labs. "
            "Examples: ‘Approach to new atrial fibrillation?’ • ‘Differential for syncope?’ • ‘Initial management of STEMI?’"
        )

    # Fallback
    return generate_greeting_answer()


def _ollama_generate(prompt: str, timeout_s: int = 120) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
        response.raise_for_status()

        data = response.json()

        candidate = None
        if isinstance(data, dict):
            candidate = data.get("response") or data.get("text")
            if not candidate and "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict):
                    candidate = first.get("text") or first.get("message") or first.get("response")

        if candidate is None and isinstance(data, str):
            candidate = data

        if not candidate or not str(candidate).strip():
            return "⚠️ Ollama returned empty output or unexpected JSON format."

        return str(candidate).strip()

    except requests.exceptions.ConnectionError:
        return "❌ Ollama server not running. Start it with: ollama serve"
    except Exception as e:
        return f"❌ Ollama error: {e}"

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector safely."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _default_domain_max_distance_text() -> float:
    return float(os.getenv("CARDIOLOGY_DOMAIN_MAX_DISTANCE_TEXT", "0.22"))


def _domain_gate_text(vectorstore, query: str, max_distance: float) -> tuple[bool, float | None]:
    """Return (in_domain, best_score) using the vectorstore's embedding model.

    This matches the FastAPI behavior: we check domain-ness using the same
    embedding model that built the FAISS index (Instructor-large).
    """
    try:
        res = vectorstore.similarity_search_with_score(query=query, k=1)
    except Exception:
        # If the gate fails for any reason, don't block the user.
        return True, None

    if not res:
        return False, None

    best_score = res[0][1]
    try:
        best_score_f = float(best_score)
    except Exception:
        return True, None

    return (best_score_f <= max_distance), best_score_f

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

    return _ollama_generate(prompt)


def generate_out_of_domain_answer(query: str) -> str:
    """User-friendly response for non-cardiology questions.

    This intentionally skips retrieval so the cardiology RAG system doesn't
    hallucinate from irrelevant cardiology context.
    """

    # Default: deterministic message so we never accidentally answer the user's
    # non-cardiology question (e.g., Batman) from the cardiology service.
    base = (
        "I'm specialized in cardiology, and your question looks outside my scope. "
        "I can help with heart-related topics like atrial fibrillation treatment, heart failure management, or chest pain/MI evaluation. "
        "If you meant something cardiology-related, can you rephrase with symptoms, diagnosis, medications, or ECG/echo findings?"
    )

    # Optional: allow Ollama to rephrase/guide, but still keep the fixed scope disclaimer.
    use_ollama = os.getenv("CARDIOLOGY_OOD_USE_OLLAMA", "0").strip().lower() in {"1", "true", "yes"}
    if not use_ollama:
        return base

    prompt = f"""
You are MediRAG, a cardiology-focused medical assistant.

Write a short, friendly message that:
1) clearly says you are specialized in cardiology and the user's question is outside your domain,
2) gives 2-3 examples of cardiology questions you can answer,
3) asks one follow-up question to help the user rephrase into a cardiology question.

Do NOT answer the user's original question.

User question:
{query}
"""

    result = _ollama_generate(prompt)
    if result.startswith("❌ Ollama") or result.startswith("⚠️ Ollama"):
        return base

    # Ensure the scope disclaimer is always present even if the model drifts.
    return base + "\n\n" + result

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

    intent = detect_general_intent(query)
    if intent is not None:
        print("\n" + generate_general_intent_answer(intent) + "\n")
        return

    # -----------------------------
    # STEP 0: IN-DOMAIN GATE (FAST)
    # -----------------------------
    # IMPORTANT: This must run BEFORE HyDE/fusion. HyDE can pull nonsense queries
    # closer to random cardiology passages, which defeats domain detection.
    vectorstore = load_vectorstore()
    max_dist_text = _default_domain_max_distance_text()
    in_domain, best_score = _domain_gate_text(vectorstore, query, max_distance=max_dist_text)
    if not in_domain:
        print("\n[0] Domain check: OUTSIDE cardiology scope")
        if best_score is not None:
            print(f"    best_distance={best_score:.4f} (threshold={max_dist_text})")
        print("\n" + generate_out_of_domain_answer(query) + "\n")
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
    print("\n[5] Generating final answer with phi:2.7b (Ollama)...")
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
