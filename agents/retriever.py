"""
retriever.py
------------
Retriever Agent: Fetches the most relevant knowledge chunks
from the FAISS vector store given a user query.
"""

from database.vector_store import VectorStore


class RetrieverAgent:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> str:
        print(f"[RetrieverAgent] Retrieving context for: '{query}'")
        chunks = self.vector_store.search(query, top_k=top_k)
        context = "\n\n".join([f"[Chunk {i+1}]: {c}" for i, c in enumerate(chunks)])
        print(f"[RetrieverAgent] Retrieved {len(chunks)} chunks.")
        return context
