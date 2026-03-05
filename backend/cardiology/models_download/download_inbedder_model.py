from sentence_transformers import SentenceTransformer
import os

model_name = "KomeijiForce/inbedder-roberta-large"

# Make local_dir repo-relative (backend/cardiology/models/...)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
local_dir = os.path.join(BASE_DIR, "models", "inbedder-roberta-large")

os.makedirs(local_dir, exist_ok=True)

print(f"Loading model {model_name} as a SentenceTransformer...")
model = SentenceTransformer(model_name)

print(f"Saving Sentence-Transformers model to {local_dir}...")
model.save(local_dir)

print(f"✅ Sentence-Transformers model saved to {local_dir}")
