from embedder import embed_and_project
from hyde import generate_hypothetical_docs
from fusion import fuse_embeddings
from retriever import load_vectorstore
import numpy as np
import os
import re
import torch
import requests
import json
from transformers import pipeline as hf_pipeline
from typing import List, Optional
from langchain_community.docstore.document import Document

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
MODEL_NAME = "microsoft/phi-2"  # Fallback HuggingFace model
OLLAMA_MODEL = "phi:2.7b"  # Quantized model name in Ollama
# For unified container: localhost; for docker-compose: http://ollama:11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))
_phi_pipeline = None
_use_ollama = False  # Will be set during initialization
_ollama_available = False  # Will be checked at startup


def _check_ollama_available() -> bool:
    """Check if Ollama is running and has the phi model available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m.get("name", "") for m in models_data.get("models", [])]
            print(f"[Ollama] Available models: {available_models}")
            
            # Check if our target model is available
            for model in available_models:
                if "phi" in model.lower() and "2.7" in model:
                    print(f"[Ollama] Found quantized Phi model: {model}")
                    return True
            
            print(f"[Ollama] ⚠️ Phi 2.7b not found in available models")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[Ollama] ❌ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        return False
    except Exception as e:
        print(f"[Ollama] ❌ Error checking Ollama: {e}")
        return False


def _get_phi_pipeline():
    """Initialize Phi model using Ollama if available, else fallback to HuggingFace."""
    global _phi_pipeline, _use_ollama, _ollama_available
    
    # Check Ollama availability
    _ollama_available = _check_ollama_available()
    
    if _ollama_available:
        print("[Model] Using Ollama quantized Phi model")
        _use_ollama = True
        return True  # Return True to indicate success without loading a pipeline
    
    # Fallback to HuggingFace
    print("[Model] Falling back to HuggingFace Phi-2...")
    if _phi_pipeline is None:
        print("Loading Phi model from HuggingFace...")
        if torch.cuda.is_available():
            print("GPU detected — using CUDA")
            device = 0
            dtype = torch.float16
        else:
            print("No GPU detected — using CPU")
            device = -1
            dtype = torch.float32

        _phi_pipeline = hf_pipeline(
            "text-generation",
            model=MODEL_NAME,
            dtype=dtype,
            device=device,
            trust_remote_code=True,
        )
    _use_ollama = False
    return _phi_pipeline


def _ollama_generate(prompt: str, timeout_s: int = OLLAMA_TIMEOUT_S) -> str:
    """Generate using Ollama if available, else HuggingFace transformers."""
    global _use_ollama
    
    if _use_ollama and _ollama_available:
        return _ollama_generate_request(prompt, timeout_s)
    else:
        return _hf_generate(prompt)


def _ollama_generate_request(prompt: str, timeout_s: int = 120) -> str:
    """Generate using Ollama API."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 300,
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout_s,
        )
        
        if response.status_code == 200:
            result = response.json()
            generated = result.get("response", "").strip()
            
            # Remove the prompt from the response if it's included
            if generated.startswith(prompt):
                answer = generated[len(prompt):].strip()
            else:
                answer = generated
            
            return answer if answer else "⚠️ Ollama returned empty output."
        else:
            print(f"[Ollama] API error: {response.status_code}")
            return f"❌ Ollama API error: {response.status_code}"
    
    except requests.exceptions.Timeout:
        return f"❌ Ollama request timed out after {timeout_s}s"
    except Exception as e:
        return f"❌ Ollama inference error: {e}"


def _hf_generate(prompt: str) -> str:
    """Generate using HuggingFace transformers."""
    try:
        pipe = _get_phi_pipeline()
        if isinstance(pipe, bool):
            return "❌ Model not initialized"
        
        result = pipe(
            prompt,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=True,
        )

        generated = result[0]["generated_text"]

        if generated.startswith(prompt):
            answer = generated[len(prompt):].strip()
        else:
            answer = generated.strip()

        return answer if answer else "⚠️ Model returned empty output."

    except Exception as e:
        return f"❌ HuggingFace inference error: {e}"


# --------------------------------------------------
# INTENT DETECTION (unchanged)
# --------------------------------------------------
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
    if not user_text:
        return False
    text = user_text.strip()
    if not text:
        return False
    if len(text) > 80:
        return False
    return _GREETING_THANKS_PATTERN.match(text) is not None


def detect_general_intent(user_text: str) -> str | None:
    if not user_text:
        return None
    text = user_text.strip()
    if not text:
        return None
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
    return (
        "Hi — I'm MediRAG Cardiology. Ask me a cardiology question and I'll answer using the cardiology knowledge base. "
        "If you share key details (age/sex, symptoms, PMH, meds, vitals, ECG/echo/troponin), I can be more specific. "
        "Examples: 'How is atrial fibrillation managed?' • 'Workup for chest pain?' • 'When to anticoagulate for AF?'"
    )


def generate_general_intent_answer(intent: str) -> str:
    if intent == "greeting":
        return generate_greeting_answer()
    if intent == "goodbye":
        return "Glad to help. If you have another cardiology question, just send it."
    if intent == "identity":
        return (
            "I'm MediRAG Cardiology — a cardiology-focused Q&A assistant. "
            "I answer by retrieving relevant cardiology reference text and generating a concise response from that context."
        )
    if intent == "help":
        return (
            "Ask a cardiology question in one message. Helpful details: symptoms + timeline, age/sex, PMH, meds, vitals, and any ECG/echo/labs. "
            "Examples: 'Approach to new atrial fibrillation?' • 'Differential for syncope?' • 'Initial management of STEMI?'"
        )
    return generate_greeting_answer()


# --------------------------------------------------
# UTILS
# --------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _default_domain_max_distance_text() -> float:
    return float(os.getenv("CARDIOLOGY_DOMAIN_MAX_DISTANCE_TEXT", "0.22"))


def _domain_gate_text(vectorstore, query: str, max_distance: float) -> tuple[bool, float | None]:
    try:
        res = vectorstore.similarity_search_with_score(query=query, k=1)
    except Exception:
        return True, None
    if not res:
        return False, None
    best_score = res[0][1]
    try:
        best_score_f = float(best_score)
    except Exception:
        return True, None
    return (best_score_f <= max_distance), best_score_f


def _strip_followup_content(text: str) -> str:
    """Remove follow-up questions, exercises, discussion sections, and extra metadata."""
    if not text:
        return text

    result = text.replace("\\n", "\n").strip()

    # Remove starting prefixes
    result = re.sub(
        r"^\s*(Answer|Final answer|Response|MediRAG Cardiology)\s*:?\s*",
        "",
        result,
        flags=re.IGNORECASE,
    ).strip()

    stop_patterns = [
        r"(?:\n|\s{2,}|^)\s*Follow[- ]?up\s+(?:Question|Questions|Exercise|Exercises|Discussion|Thought)s?\s*\d*\s*:?",
        r"(?:\n|\s{2,}|^)\s*Discussion\s*:?",
        r"(?:\n|\s{2,}|^)\s*Explanation\s*:?",
        r"(?:\n|\s{2,}|^)\s*Note\s*:?",
        r"(?:\n|\s{2,}|^)\s*Clinical\s+Pearls?\s*:?",
        r"(?:\n|\s{2,}|^)\s*Key\s+Points?\s*:?",
        r"(?:\n|\s{2,}|^)\s*References?\s*:?",
        r"(?:\n|\s{2,}|^)\s*Answers?\s*:?",
        r"(?:\n|\s{2,}|^)\s*Examples?\s*:?",
        r"(?:\n|\s{2,}|^)\s*Positive\s+impact\s*:?",
        r"(?:\n|\s{2,}|^)\s*Negative\s+impact\s*:?",
        r"(?:\n|\s{2,}|^)\s*Overall\s*:?",
    ]

    cut_positions = []

    for pattern in stop_patterns:
        match = re.search(pattern, result, flags=re.IGNORECASE | re.DOTALL)
        if match:
            cut_positions.append(match.start())

    if cut_positions:
        result = result[:min(cut_positions)].strip()

    # Keep only first paragraph if model still creates multiple paragraphs
    result = result.split("\n\n")[0].strip()

    # Clean extra spaces/newlines
    result = re.sub(r"\n+", " ", result)
    result = re.sub(r"\s+", " ", result).strip()

    return result if result else "I don't know based on the provided context."


def generate_final_answer(query: str, retrieved_docs: List[Document]) -> str:
    if not retrieved_docs:
        return "No relevant documents found."
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""You are MediRAG Cardiology, a cardiology question-answering assistant.

Your task is to answer the user's question using ONLY the provided context.

STRICT OUTPUT RULES:
- Write ONLY one clear answer paragraph (maximum 3-5 sentences).
- Do NOT write follow-up questions.
- Do NOT write follow-up exercises.
- Do NOT write discussion sections.
- Do NOT write "Answer:" or any other header/prefix.
- Do NOT write examples unless directly asked.
- Do NOT include bullet points, numbered lists, or extra sections.
- If the answer is not found in the context, write exactly: "I don't know based on the provided context."

Context:
{context}

Question:
{query}

Respond with ONLY the answer paragraph (no prefix, no follow-up):
"""
    raw_answer = _ollama_generate(prompt)
    # Clean up any remaining follow-up content or formatting issues
    clean_answer = _strip_followup_content(raw_answer)
    return clean_answer if clean_answer else "I don't know based on the provided context."


def generate_out_of_domain_answer(query: str) -> str:
    base = (
        "I'm specialized in cardiology, and your question looks outside my scope. "
        "I can help with heart-related topics like atrial fibrillation treatment, heart failure management, or chest pain/MI evaluation. "
        "If you meant something cardiology-related, can you rephrase with symptoms, diagnosis, medications, or ECG/echo findings?"
    )
    use_ollama = os.getenv("CARDIOLOGY_OOD_USE_OLLAMA", "0").strip().lower() in {"1", "true", "yes"}
    if not use_ollama:
        return base
    prompt = f"""You are MediRAG, a cardiology-focused medical assistant.

Write a short, friendly message that:
1) clearly says you are specialized in cardiology and the user's question is outside your domain,
2) gives 2-3 examples of cardiology questions you can answer,
3) asks one follow-up question to help the user rephrase into a cardiology question.

Do NOT answer the user's original question.

User question:
{query}
"""
    result = _ollama_generate(prompt)
    if result.startswith("❌") or result.startswith("⚠️"):
        return base
    return base + "\n\n" + result


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    print("\nMediRAG Cardiology Assistant\n")
    query = input("Enter your cardiology question: ").strip()
    if not query:
        print("❌ Empty query provided. Exiting.")
        return

    intent = detect_general_intent(query)
    if intent is not None:
        print("\n" + generate_general_intent_answer(intent) + "\n")
        return

    vectorstore = load_vectorstore()
    max_dist_text = _default_domain_max_distance_text()
    in_domain, best_score = _domain_gate_text(vectorstore, query, max_distance=max_dist_text)
    if not in_domain:
        print("\n[0] Domain check: OUTSIDE cardiology scope")
        if best_score is not None:
            print(f"    best_distance={best_score:.4f} (threshold={max_dist_text})")
        print("\n" + generate_out_of_domain_answer(query) + "\n")
        return

    print("\n[1] Embedding and projecting user query...")
    proj_emb = embed_and_project(query)
    proj_emb = l2_normalize(np.array(proj_emb))

    print("[2] Generating hypothetical documents (HyDE)...")
    hyde_emb, hyde_docs = generate_hypothetical_docs(query)
    hyde_emb = l2_normalize(np.array(hyde_emb))

    print("[3] Fusing embeddings...")
    final_emb = fuse_embeddings(proj_query=proj_emb, hyde_query=hyde_emb, alpha=0.5)
    final_emb = l2_normalize(np.array(final_emb))

    print("[4] Loading FAISS index and retrieving top documents...")
    try:
        faiss_index = getattr(vectorstore, "index", None)
        if faiss_index is not None:
            print(f"FAISS index total vectors: {faiss_index.ntotal}, dim: {faiss_index.d}")
    except Exception as e:
        print("Error inspecting vectorstore.index:", e)

    print(f"Final embedding shape: {getattr(final_emb, 'shape', None)}, dtype: {getattr(final_emb, 'dtype', None)}")

    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding=final_emb.tolist(), k=5
        )
    except Exception as e:
        print("Error during FAISS similarity search:", e)
        docs_and_scores = []

    if not docs_and_scores:
        print("❌ No documents retrieved from FAISS.")
        return

    print("\nRETRIEVED DOCUMENTS (with similarity scores)")
    print("=" * 80)
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        print(f"\n--- Rank {rank} ---")
        print(f"FAISS distance score: {score:.4f}")
        print("-" * 80)
        print(doc.page_content[:400])
        print("-" * 80)

    print("\nGENERATED HYPOTHETICAL ANSWERS (HyDE)")
    print("=" * 80)
    for i, doc in enumerate(hyde_docs, 1):
        print(f"\n--- Hypothesis {i} ---")
        print(doc.replace("Represent the cardiology document for retrieval:\n", "").strip())

    print("\n[5] Generating final answer with Phi (HuggingFace)...")
    retrieved_docs = [doc for doc, _ in docs_and_scores]
    final_answer = generate_final_answer(query, retrieved_docs)

    print("\n====================== FINAL ANSWER ======================")
    print(final_answer if final_answer else "❌ Empty answer returned.")
    print("=========================================================")
    print("\nPipeline execution completed successfully.\n")


if __name__ == "__main__":
    main()