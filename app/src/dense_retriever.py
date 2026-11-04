"""Dense (embedding) retrieval with all-MiniLM-L6-v2.

Two backends return the same `[(chunk, score), ...]` shape:
- `retrieve_dense_cosmos`: production. Cosmos DB `VectorDistance` over a flat
  (exact) cosine index, so results match in-memory search.
- `InMemoryDenseIndex`: offline eval and local runs, no Azure needed.
"""

from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts):
    return get_model().encode(texts, convert_to_numpy=True)


def retrieve_dense_cosmos(query, k=10):
    from src.database import get_container

    query_embedding = embed(query).tolist()
    items = get_container("chunks").query_items(
        query=f"""
            SELECT TOP {int(k)} c.source, c.chunk_id, c.text,
                   VectorDistance(c.embedding, @query_embedding) AS score
            FROM c
            ORDER BY VectorDistance(c.embedding, @query_embedding)
        """,
        parameters=[{"name": "@query_embedding", "value": query_embedding}],
        enable_cross_partition_query=True,
    )
    return [
        ({"source": i["source"], "chunk_id": i["chunk_id"], "text": i["text"]}, i["score"])
        for i in items
    ]


class InMemoryDenseIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        vectors = embed([c["text"] for c in chunks])
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, query, k=10):
        q = embed(query)
        scores = self.vectors @ (q / np.linalg.norm(q))
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
