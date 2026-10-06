def reciprocal_rank(results, relevant_chunks, k=3):
    """1/rank of the first relevant chunk within the top k, or 0.0 if none."""
    for rank, (chunk, _score) in enumerate(results[:k], start=1):
        if (chunk["source"], chunk["chunk_id"]) in relevant_chunks:
            return 1 / rank
    return 0.0
