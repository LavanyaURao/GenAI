"""
vector_store.py
---------------
Handles document chunking, embedding generation using Sentence Transformers,
and FAISS-based vector indexing for semantic retrieval.

Model Used: sentence-transformers/all-MiniLM-L6-v2
  - Lightweight 22M parameter model
  - Produces 384-dimensional embeddings
  - Fast inference, suitable for local semantic search
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[VectorStore] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def load_and_index(self, filepath: str, chunk_size: int = 5):
        print(f"[VectorStore] Loading knowledge base from: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk = " ".join(lines[i: i + chunk_size])
            self.chunks.append(chunk)

        print(f"[VectorStore] Created {len(self.chunks)} chunks.")

        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"[VectorStore] FAISS index built with dimension={dimension}.")

    def search(self, query: str, top_k: int = 3) -> list:
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results
