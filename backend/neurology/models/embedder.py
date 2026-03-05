from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str):
        return self.model.encode(
            [text],
            normalize_embeddings=True
        )
