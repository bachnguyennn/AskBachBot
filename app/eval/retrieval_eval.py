"""Retrieval evaluation: Dense vs BM25 vs Hybrid (RRF) on the frozen benchmark.

    python -m eval.retrieval_eval                    # offline: local .txt files, in-memory dense
    python -m eval.retrieval_eval --backend cosmos   # live: Cosmos DB documents + VectorDistance
    python -m eval.retrieval_eval --check            # exit 1 if Hybrid drops below the floor (CI)

The offline backend needs no Azure credentials, so it runs in CI on every push.
Cosmos DB uses a flat (exact) vector index, so both backends should rank identically
as long as the `chunks` container has been re-indexed from the current documents.
"""

import argparse
import json
import sys
from pathlib import Path

from eval.benchmark import ANSWERABLE
from src.hybrid_retriever import HybridRetriever
from src.load_documents import load_chunks

# Frozen configuration: do not tune against this eval set.
CANDIDATE_K = 10
FINAL_K = 3
RRF_K = 60

# Regression floor for CI: the Hybrid score at the time the floor was set.
# Raise it when retrieval improves; never lower it to make a change pass.
HYBRID_FLOOR = {"recall": 1.0, "precision": 0.60, "mrr": 0.88}

RESULTS_PATH = Path(__file__).resolve().parent / "retrieval_results.json"


def key(chunk):
    return (chunk["source"], chunk["chunk_id"])


def reciprocal_rank(ranking, relevant, k=FINAL_K):
    for rank, (chunk, _score) in enumerate(ranking[:k], start=1):
        if key(chunk) in relevant:
            return 1 / rank
    return 0.0


def build_retriever(backend):
    if backend == "cosmos":
        from src.dense_retriever import retrieve_dense_cosmos

        return HybridRetriever(load_chunks(source="cosmos"), retrieve_dense_cosmos)

    from src.dense_retriever import InMemoryDenseIndex

    chunks = load_chunks(source="disk")
    return HybridRetriever(chunks, InMemoryDenseIndex(chunks).search)


def evaluate(retriever):
    systems = {
        "Dense": lambda q: retriever.dense(q, CANDIDATE_K),
        "BM25": lambda q: retriever.lexical(q, CANDIDATE_K),
        "Hybrid": lambda q: retriever.retrieve(q, CANDIDATE_K, final_k=None, rrf_k=RRF_K),
    }
    records = []
    for item in ANSWERABLE:
        record = {"question": item["question"], "variant": item["variant"], "systems": {}}
        for name, retrieve in systems.items():
            ranking = retrieve(item["question"])
            top = ranking[:FINAL_K]
            record["systems"][name] = {
                "hit": any(key(c) in item["relevant"] for c, _ in top),
                "relevant_in_top_k": sum(key(c) in item["relevant"] for c, _ in top),
                "reciprocal_rank": reciprocal_rank(ranking, item["relevant"]),
                "top_k": [
                    {"source": c["source"], "chunk_id": c["chunk_id"], "score": round(float(s), 4),
                     "relevant": key(c) in item["relevant"]}
                    for c, s in top
                ],
            }
        records.append(record)
    return records


def summarize(records):
    n = len(records)
    return {
        name: {
            "recall": sum(r["systems"][name]["hit"] for r in records) / n,
            "precision": sum(r["systems"][name]["relevant_in_top_k"] for r in records) / (n * FINAL_K),
            "mrr": sum(r["systems"][name]["reciprocal_rank"] for r in records) / n,
        }
        for name in records[0]["systems"]
    }


def print_report(records, summary):
    n = len(records)
    print(f"\n{'Metric (n=' + str(n) + ')':<16}" + "".join(f"{name:<10}" for name in summary))
    for metric, label in [("recall", f"Recall@{FINAL_K}"), ("precision", f"Precision@{FINAL_K}"),
                          ("mrr", f"MRR@{FINAL_K}")]:
        print(f"{label:<16}" + "".join(f"{s[metric]:<10.3f}" for s in summary.values()))
    for r in records:
        hybrid = r["systems"]["Hybrid"]
        marks = " ".join(
            f"{'✓' if t['relevant'] else '✗'}{t['source'].removesuffix('.txt')}#{t['chunk_id']}"
            for t in hybrid["top_k"]
        )
        print(f"  [{r['variant'][:4]}] {r['question']:<62} {marks}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=["disk", "cosmos"], default="disk")
    parser.add_argument("--check", action="store_true", help="fail if Hybrid is below the floor")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    records = evaluate(build_retriever(args.backend))
    summary = summarize(records)
    print_report(records, summary)

    if not args.no_save:
        RESULTS_PATH.write_text(json.dumps({
            "backend": args.backend,
            "config": {"candidate_k": CANDIDATE_K, "final_k": FINAL_K, "rrf_k": RRF_K,
                       "dense": "all-MiniLM-L6-v2 + cosine",
                       "bm25": "regex tokenizer + stopwords, no stemming"},
            "summary": summary,
            "results": records,
        }, indent=2, ensure_ascii=False) + "\n")
        print(f"\nSaved to {RESULTS_PATH}")

    if args.check:
        hybrid = summary["Hybrid"]
        failed = {m: hybrid[m] for m, floor in HYBRID_FLOOR.items() if hybrid[m] < floor}
        if failed:
            print(f"\nFAIL: Hybrid below floor {HYBRID_FLOOR}: {failed}")
            return 1
        print(f"\nPASS: Hybrid meets floor {HYBRID_FLOOR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
