import faiss
import pickle
import numpy as np

class FaissRetriever:
    def __init__(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, embedding, top_k: int):
        embedding = np.array(embedding).astype("float32")
        scores, indices = self.index.search(embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            item = self.metadata[idx].copy()
            item["faiss_rank"] = rank
            item["faiss_score"] = float(score)
            results.append(item)

        return results
