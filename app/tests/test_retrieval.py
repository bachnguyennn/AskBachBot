from src.bm25_retriever import build_bm25, retrieve_bm25, tokenize
from src.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion


def chunk(source, chunk_id, text=""):
    return {"source": source, "chunk_id": chunk_id, "text": text}


A, B, C, D = chunk("a.txt", 0), chunk("b.txt", 0), chunk("c.txt", 0), chunk("d.txt", 0)


def test_tokenize_lowercases_and_drops_stopwords_but_keeps_names():
    assert tokenize("What did Bach do at CMHA?") == ["bach", "at", "cmha"]


def test_bm25_ranks_exact_term_match_first():
    chunks = [
        chunk("x.txt", 0, "Bach studies mathematics"),
        chunk("y.txt", 0, "Bach interned at CMHA building Power Apps"),
        chunk("z.txt", 0, "Bach likes football"),
    ]
    results = retrieve_bm25("CMHA internship", chunks, build_bm25(chunks), k=1)
    assert results[0][0]["source"] == "y.txt"


def test_rrf_rewards_agreement_between_retrievers():
    dense = [(A, 0.9), (B, 0.8), (C, 0.7)]
    lexical = [(B, 12.0), (D, 11.0)]
    fused = reciprocal_rank_fusion(dense, lexical)
    order = [c["source"] for c, _ in fused]
    # B is found by both retrievers, so it beats A, which only dense ranked first.
    assert order[:2] == ["b.txt", "a.txt"]
    assert fused[0][1] == 1 / 62 + 1 / 61
    assert set(order) == {"a.txt", "b.txt", "c.txt", "d.txt"}


def test_rrf_ignores_raw_score_scale():
    # A huge BM25 score must not dominate: only ranks matter.
    fused = reciprocal_rank_fusion([(A, 0.01)], [(B, 1e9)])
    scores = [s for _, s in fused]
    assert scores[0] == scores[1]


def test_rrf_deduplicates_by_source_and_chunk_id():
    fused = reciprocal_rank_fusion([(A, 1.0)], [(dict(A), 5.0)])
    assert len(fused) == 1
    assert fused[0][1] == 2 / 61


def test_hybrid_retriever_truncates_to_final_k():
    chunks = [chunk(f"{i}.txt", 0, f"document number {i} about bach") for i in range(6)]
    retriever = HybridRetriever(chunks, dense_search=lambda q, k: [(c, 1.0) for c in chunks[:k]])
    assert len(retriever.retrieve("bach", candidate_k=5, final_k=3)) == 3
    assert len(retriever.retrieve("bach", candidate_k=5, final_k=None)) == 5
