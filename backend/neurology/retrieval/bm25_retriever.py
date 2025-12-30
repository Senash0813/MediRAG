import pickle
from nltk.tokenize import word_tokenize

class BM25Retriever:
    def __init__(self, bm25_path, metadata_path):
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, query: str, top_k: int):
        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked, start=1):
            item = self.metadata[idx].copy()
            item["bm25_rank"] = rank
            item["bm25_score"] = float(scores[idx])
            results.append(item)

        return results
