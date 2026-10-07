import json
import time
from pathlib import Path

from src.embed_documents import embeddings, chunks, retrieve_top_k_documents, build_context, build_prompt
from src.generate import generate_answer, MODEL

RESULTS_PATH = Path(__file__).resolve().parents[2] / "eval" / "results.json"

# should_answer: True if the documents contain the evidence, False if the model should refuse
EVAL_SET = [
    {"question": "What did Bach do at CMHA?", "should_answer": True},
    {"question": "What Azure experience does Bach have?", "should_answer": True},
    {"question": "What is Bach researching?", "should_answer": True},
    {"question": "When does Bach graduate?", "should_answer": True},
    {"question": "What programming languages does Bach know?", "should_answer": True},
    {"question": "Did Bach work at Microsoft?", "should_answer": False},
    {"question": "What is Bach's favorite programming language?", "should_answer": False},
    {"question": "How many years did Bach work at Google?", "should_answer": False},
]


def run_eval(k=3):
    records = []
    for i, item in enumerate(EVAL_SET, start=1):
        question = item["question"]

        start = time.perf_counter()
        results = retrieve_top_k_documents(question, embeddings, chunks, k=k)
        answer = generate_answer(build_prompt(question, build_context(results)))
        latency = time.perf_counter() - start

        top_k = [
            {"source": c["source"], "chunk_id": c["chunk_id"], "score": round(float(s), 4)}
            for c, s in results
        ]
        records.append({**item, "top_k": top_k, "answer": answer, "latency_s": round(latency, 2)})

        print(f"\n{i}. Question: {question}")
        print("Top 3:")
        for rank, r in enumerate(top_k, start=1):
            print(f"   {rank}. {r['source']} | chunk {r['chunk_id']} | {r['score']:.4f}")
        print(f"Generated answer: {answer}")
    return records


if __name__ == "__main__":
    records = run_eval()
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"model": MODEL, "k": 3, "results": records}, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(records)} results to {RESULTS_PATH}")
