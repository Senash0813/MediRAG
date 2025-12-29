import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# MODEL PATHS
# -----------------------------
HYDE_PATH = "models/hyde-sciFive-cardiology-generator"
INSTRUCTOR_PATH = "models/instructor-large"

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD HYDE GENERATOR
# -----------------------------
hyde_tokenizer = AutoTokenizer.from_pretrained(HYDE_PATH)
hyde_model = AutoModelForSeq2SeqLM.from_pretrained(HYDE_PATH).to(device)
hyde_model.eval()

# -----------------------------
# LOAD INSTRUCTOR EMBEDDINGS (768-d)
# -----------------------------
embeddings_model = HuggingFaceEmbeddings(
    model_name=INSTRUCTOR_PATH,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": False}
)

# -----------------------------
# UTILS
# -----------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

# -----------------------------
# HYDE PIPELINE
# -----------------------------
def generate_hypothetical_docs(
    query: str,
    num_return_sequences: int = 4,
    max_new_tokens: int = 300
):
    """
    HyDE:
    Query → Hypothetical answers → Instructor embeddings
    → mean pool → L2 normalize (768-d)
    """

    instruction = "Represent the cardiology document for retrieval:"
    prompt = f"Question: {query}\nParagraph:"

    # Tokenize
    inputs = hyde_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(device)

    # Generate hypothetical answers
    with torch.no_grad():
        outputs = hyde_model.generate(
            **inputs,
            do_sample=True,
            num_beams=4,
            num_return_sequences=num_return_sequences,
            temperature=0.7,
            max_new_tokens=max_new_tokens,
        )

    replies = [
        hyde_tokenizer.decode(out, skip_special_tokens=True)
        for out in outputs
    ]

    docs = [f"{instruction}\n{reply}" for reply in replies]

    # Instructor embeddings (768-d)
    embeddings = np.array(
        embeddings_model.embed_documents(docs),
        dtype=np.float32
    )

    # Row-wise normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Mean pooling + normalize
    avg_embedding = l2_normalize(np.mean(embeddings, axis=0))

    return avg_embedding, docs

# -----------------------------
# DEBUG
# -----------------------------
if __name__ == "__main__":
    emb, docs = generate_hypothetical_docs("What is cardiology?")
    print("Embedding shape:", emb.shape)  # MUST be (768,)
