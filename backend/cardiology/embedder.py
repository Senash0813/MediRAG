import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# PATHS
# -----------------------------
INBEDDER_PATH = "models/inbedder-roberta-large"
PROJECTOR_PATH = "models/projector.pt"

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# UTILS
# -----------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector safely"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# -----------------------------
# PROJECTION LAYER
# -----------------------------
class ProjectionLayer(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

# -----------------------------
# LOAD MODELS (ONCE)
# -----------------------------
# InBEDDER encoder
inbed_model = SentenceTransformer(INBEDDER_PATH, device=device)

# Projection head
projector = ProjectionLayer()
projector.load_state_dict(torch.load(PROJECTOR_PATH, map_location=device))
projector.to(device)
projector.eval()  # inference mode

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def embed_and_project(query: str) -> np.ndarray:
    """
    Encodes a cardiology question using InBEDDER,
    applies projection head, and returns a 768-d vector.
    """

    instruction = "Cluster this cardiology question semantically:"
    text = f"{instruction}\n{query}"

    # SentenceTransformer → numpy
    emb = inbed_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype(np.float32)

    # Normalize before projection (as per your design)
    emb = l2_normalize(emb)

    # Project to target space
    with torch.no_grad():
        x = torch.from_numpy(emb).unsqueeze(0).to(device)
        projected = projector(x).squeeze(0)

    return projected.cpu().numpy()

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    query = "How does sevoflurane postconditioning protect the myocardium?"

    vec = embed_and_project(query)
    print("Projected embedding shape:", vec.shape)
