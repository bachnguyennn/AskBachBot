import json
from pathlib import Path

from embed_documents import embeddings, chunks, retrieve_top_k_documents
from bm25_retriever import bm25, retrieve_bm25
from hybrid_retriever import retrieve_hybrid
from metrics import reciprocal_rank

RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "retrieval_results.json"

# Frozen configuration -- do not tune against this eval set.
CANDIDATE_K = 10
FINAL_K = 3
RRF_K = 60

# Relevant = the chunk alone supports a correct, specific answer
# (names the role / concrete work / the fact), not just a passing mention.
CMHA = {("cmha.txt", 0), ("cmha.txt", 1), ("cmha.txt", 2), ("cmha.txt", 3), ("about.txt", 1)}
AZURE = {("azure.txt", 0), ("azure.txt", 1), ("azure.txt", 2), ("about.txt", 0), ("about.txt", 1)}
RESEARCH = {
    ("research.txt", 0), ("research.txt", 1), ("research.txt", 5), ("research.txt", 6),
    ("about.txt", 0), ("about.txt", 1), ("education.txt", 1),
}
GRADUATION = {("education.txt", 0), ("education.txt", 3), ("about.txt", 0)}
LANGUAGES = {("about.txt", 2)}

# Each paraphrase shares its original's label set (same information need).
ANSWERABLE = [
    {"question": "What did Bach do at CMHA?", "variant": "original", "relevant": CMHA},
    {"question": "What Azure experience does Bach have?", "variant": "original", "relevant": AZURE},
    {"question": "What is Bach researching?", "variant": "original", "relevant": RESEARCH},
    {"question": "When does Bach graduate?", "variant": "original", "relevant": GRADUATION},
    {"question": "What programming languages does Bach know?", "variant": "original", "relevant": LANGUAGES},
    {"question": "Describe Bach's work at the Canadian Mental Health Association.", "variant": "paraphrase", "relevant": CMHA},
    {"question": "What cloud computing work has Bach done?", "variant": "paraphrase", "relevant": AZURE},
    {"question": "What topic is Bach's undergraduate thesis focused on?", "variant": "paraphrase", "relevant": RESEARCH},
    {"question": "When is Bach expected to finish university?", "variant": "paraphrase", "relevant": GRADUATION},
    {"question": "Which coding languages can Bach use?", "variant": "paraphrase", "relevant": LANGUAGES},
]

# Evaluated later for refusal / hallucination, not retrieval.
UNANSWERABLE = [
    "Did Bach work at Microsoft?",
    "What is Bach's favorite programming language?",
    "How many years did Bach work at Google?",
]

RETRIEVERS = {
    "Dense": lambda q: retrieve_top_k_documents(q, embeddings, chunks, k=CANDIDATE_K),
    "BM25": lambda q: retrieve_bm25(q, chunks, bm25, k=CANDIDATE_K),
    "Hybrid": lambda q: retrieve_hybrid(q, candidate_k=CANDIDATE_K, final_k=None, rrf_k=RRF_K),
}


def key(chunk):
    return (chunk["source"], chunk["chunk_id"])


def first_relevant_rank(ranking, relevant):
    for rank, (chunk, _score) in enumerate(ranking, start=1):
        if key(chunk) in relevant:
            return rank
    return None


def evaluate():
    records = []
    for item in ANSWERABLE:
        record = {"question": item["question"], "variant": item["variant"],
                  "relevant": sorted(item["relevant"]), "systems": {}}
        for name, retrieve in RETRIEVERS.items():
            ranking = retrieve(item["question"])
            top = ranking[:FINAL_K]
            record["systems"][name] = {
                "hit": any(key(c) in item["relevant"] for c, _ in top),
                "relevant_in_top_k": sum(key(c) in item["relevant"] for c, _ in top),
                "reciprocal_rank": reciprocal_rank(ranking, item["relevant"], k=FINAL_K),
                "first_relevant_rank": first_relevant_rank(ranking, item["relevant"]),
                "top_k": [
                    {"source": c["source"], "chunk_id": c["chunk_id"], "score": float(s),
                     "relevant": key(c) in item["relevant"]}
                    for c, s in top
                ],
            }
        records.append(record)
    return records


def print_report(records):
    names = list(RETRIEVERS)
    width = 64
    print(f"\n{'Question':<{width}}" + "".join(f"{n:<10}" for n in names))
    for r in records:
        row = "".join(f"{'✅' if r['systems'][n]['hit'] else '❌':<9}" for n in names)
        print(f"{r['question']:<{width}}{row}")

    groups = [("Original queries", "original"), ("Paraphrases", "paraphrase"), ("Overall", None)]
    for metric, field in [(f"Recall@{FINAL_K}", "hit"), (f"Context Precision@{FINAL_K}", "relevant_in_top_k")]:
        print(f"\n{metric:<20}" + "".join(f"{n:<12}" for n in names))
        for label, variant in groups:
            subset = [r for r in records if variant is None or r["variant"] == variant]
            denom = len(subset) * (FINAL_K if field == "relevant_in_top_k" else 1)
            cells = "".join(f"{str(sum(r['systems'][n][field] for r in subset)) + '/' + str(denom):<12}" for n in names)
            print(f"{label:<20}{cells}")

    # MRR@k added post-hoc after the paraphrase run; reported for every run from now on.
    print(f"\n{'MRR@' + str(FINAL_K):<20}" + "".join(f"{n:<12}" for n in names))
    for label, variant in groups:
        subset = [r for r in records if variant is None or r["variant"] == variant]
        cells = "".join(f"{sum(r['systems'][n]['reciprocal_rank'] for r in subset) / len(subset):<12.3f}" for n in names)
        print(f"{label:<20}{cells}")

    for r in records:
        print(f"\n### [{r['variant']}] {r['question']}")
        for n in names:
            s = r["systems"][n]
            first = s["first_relevant_rank"]
            print(f"  {n:<7} {'HIT ' if s['hit'] else 'MISS'} | first relevant: "
                  f"{'#' + str(first) if first else f'not in top {CANDIDATE_K}'}")
            for rank, t in enumerate(s["top_k"], start=1):
                mark = "✓" if t["relevant"] else " "
                print(f"     {rank}. {mark} {t['source']:<14} c{t['chunk_id']:<3} {t['score']:.4f}")


if __name__ == "__main__":
    records = evaluate()
    print_report(records)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "config": {"candidate_k": CANDIDATE_K, "final_k": FINAL_K, "rrf_k": RRF_K,
                   "dense": "all-MiniLM-L6-v2 + cosine",
                   "bm25": "regex tokenizer + stopwords, no stemming, 'bach' kept"},
        "results": records,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {RESULTS_PATH}")
