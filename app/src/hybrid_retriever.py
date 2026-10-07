from src.cosmos_retriever import retrieve_top_k_cosmos
from src.bm25_retriever import bm25, retrieve_bm25
from src.load_documents import load_chunks

chunks = load_chunks()

def reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=60):
    rrf_scores = {}
    chunk_by_key = {}

    for results in (dense_results, bm25_results):
        for rank, (chunk, _score) in enumerate(results, start=1):
            key = (chunk["source"], chunk["chunk_id"])
            chunk_by_key[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rrf_k + rank)

    ranked_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
    return [(chunk_by_key[key], rrf_scores[key]) for key in ranked_keys]


def retrieve_hybrid(query, candidate_k=10, final_k=3, rrf_k=60):
    dense_results = retrieve_top_k_cosmos(
    query,
    k=candidate_k
    )
    bm25_results = retrieve_bm25(query, chunks, bm25, k=candidate_k)
    return reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=rrf_k)[:final_k]


if __name__ == "__main__":
    candidate_k = 10
    final_k = 3
    rrf_k = 60

    questions = [
        "What is Bach researching?",
        "What programming languages does Bach know?",
    ]

    for question in questions:
        dense_results = retrieve_top_k_cosmos(question, k=candidate_k)
        bm25_results = retrieve_bm25(question, chunks, bm25, k=candidate_k)
        hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=rrf_k)

        print(f"\n===== {question}")
        for name, results, label in [
            ("DENSE", dense_results, "cosine"),
            ("BM25", bm25_results, "BM25 score"),
            ("HYBRID RRF", hybrid_results, "RRF score"),
        ]:
            print(f"\n{name} TOP {final_k}")
            print(f"{'rank':<5}| {'source':<14}| {'chunk_id':<9}| {label}")
            for rank, (chunk, score) in enumerate(results[:final_k], start=1):
                precision = 6 if name == "HYBRID RRF" else 4
                print(f"{rank:<5}| {chunk['source']:<14}| {chunk['chunk_id']:<9}| {score:.{precision}f}")
