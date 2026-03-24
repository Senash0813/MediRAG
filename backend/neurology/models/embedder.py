import os
from sentence_transformers import SentenceTransformer

class Embedder:
    # def __init__(self, model_name: str):
    #     self.model = SentenceTransformer(model_name)
    def __init__(self, model_name: str, hf_token: str = None):
        # Get token from parameter or environment variable
        token = hf_token or os.getenv("HF_TOKEN")
        
        self.model = SentenceTransformer(
            model_name,
            token=token  # Pass authentication token
        )

    def encode(self, text: str):
        return self.model.encode(
            [text],
            normalize_embeddings=True
        )
