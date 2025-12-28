import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings

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
hyde_model.eval()  # inference mode

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
embeddings_model = HuggingFaceEmbeddings(
    model_name=INSTRUCTOR_PATH,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": False}  # we normalize manually
)

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
# HYDE PIPELINE
# -----------------------------
def generate_hypothetical_docs(
    query: str,
    num_return_sequences: int = 4,
    max_new_tokens: int = 300
):
    """
    Generates HyDE hypothetical documents, embeds them,
    and returns a single averaged normalized embedding.
    """

    instruction = "Represent the cardiology document for retrieval:"

    prompt = f"Question: {query}\nParagraph:"

    # Tokenize
    inputs = hyde_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(device)

    # Generate hypothetical documents
    with torch.no_grad():
        outputs = hyde_model.generate(
            **inputs,
            do_sample=True,
            num_beams=4,
            num_return_sequences=num_return_sequences,
            temperature=0.7,
            max_new_tokens=max_new_tokens,
        )

    # Decode outputs
    replies = [
        hyde_tokenizer.decode(out, skip_special_tokens=True)
        for out in outputs
    ]

    # Add Instructor-style instruction
    docs = [f"{instruction}\n{reply}" for reply in replies]

    # Embed documents
    embeddings = embeddings_model.embed_documents(docs)
    embeddings = np.array(embeddings)

    # Mean pooling + L2 normalize (HyDE standard)
    avg_embedding = l2_normalize(np.mean(embeddings, axis=0))

    return avg_embedding, docs

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    query = "How does sevoflurane postconditioning protect against myocardial ischemia-reperfusion injury?"

    embedding, hyde_docs = generate_hypothetical_docs(query)

    print("Generated HyDE Documents:\n")
    for i, doc in enumerate(hyde_docs, 1):
        print(f"[{i}] {doc}\n")

    print("Embedding shape:", embedding.shape)
