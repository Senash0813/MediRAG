from sentence_transformers import SentenceTransformer
import os

# Use a repo-relative models path so the script saves into
# backend/cardiology/models regardless of the current working dir.
MODEL_NAME = "hkunlp/instructor-large"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_DIR = os.path.join(BASE_DIR, "models", "instructor-large")

os.makedirs(LOCAL_DIR, exist_ok=True)

print("Loading Instructor model using SentenceTransformer...")
model = SentenceTransformer(MODEL_NAME)

print("Saving model in SentenceTransformer format...")
model.save(LOCAL_DIR)

print(f"✅ Instructor model saved correctly at: {LOCAL_DIR}")
print("Embedding dimension:", model.get_sentence_embedding_dimension())
