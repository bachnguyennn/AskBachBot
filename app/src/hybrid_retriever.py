"""Hybrid retrieval: dense + BM25 candidates merged with Reciprocal Rank Fusion."""

from functools import lru_cache

from src.bm25_retriever import build_bm25, retrieve_bm25


def reciprocal_rank_fusion(*rankings, rrf_k=60):
    """Merge ranked lists by summing 1 / (rrf_k + rank). Raw scores are ignored,
    so cosine similarities and BM25 scores never need to be put on one scale."""
    rrf_scores = {}
    chunk_by_key = {}
    for results in rankings:
        for rank, (chunk, _score) in enumerate(results, start=1):
            key = (chunk["source"], chunk["chunk_id"])
            chunk_by_key[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rrf_k + rank)

    ranked_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
    return [(chunk_by_key[key], rrf_scores[key]) for key in ranked_keys]


class HybridRetriever:
    def __init__(self, chunks, dense_search):
        """`dense_search(query, k)` returns [(chunk, score), ...]: Cosmos DB in
        production, an in-memory index in the offline eval."""
        self.chunks = chunks
        self.bm25 = build_bm25(chunks)
        self.dense_search = dense_search

    def dense(self, query, k=10):
        return self.dense_search(query, k)

    def lexical(self, query, k=10):
        return retrieve_bm25(query, self.chunks, self.bm25, k=k)

    def retrieve(self, query, candidate_k=10, final_k=3, rrf_k=60):
        fused = reciprocal_rank_fusion(
            self.dense(query, candidate_k), self.lexical(query, candidate_k), rrf_k=rrf_k
        )
        return fused[:final_k] if final_k else fused


@lru_cache(maxsize=1)
def get_retriever():
    """Production retriever. Built once per process: one Cosmos DB documents query."""
    from src.dense_retriever import retrieve_dense_cosmos
    from src.load_documents import load_chunks

    return HybridRetriever(load_chunks(source="cosmos"), retrieve_dense_cosmos)


def retrieve_hybrid(query, candidate_k=10, final_k=3, rrf_k=60):
    return get_retriever().retrieve(query, candidate_k=candidate_k, final_k=final_k, rrf_k=rrf_k)
