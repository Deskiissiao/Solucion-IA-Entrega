import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(self, index_path: str, embedding_model: str):
        self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))
        with open(os.path.join(index_path, "metadatas.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        self.model = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, k=4):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            if idx < len(self.meta["texts"]):
                results.append({
                    "text": self.meta["texts"][idx],
                    "meta": self.meta["metadatas"][idx]
                })
        return results
