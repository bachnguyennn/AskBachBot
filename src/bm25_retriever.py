import re

from rank_bm25 import BM25Okapi
from load_documents import load_chunks


STOPWORDS = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "what", "which", "who", "where", "when",
    "do", "does", "did",
    "has", "have", "had",
    "of", "to", "in", "on", "for", "and", "or",
}


def tokenize(text):
    tokens = re.findall(r"\w+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


chunks = load_chunks()

tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]

bm25 = BM25Okapi(tokenized_chunks)


def retrieve_bm25(query, chunks, bm25, k=3):
    scores = bm25.get_scores(tokenize(query))
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], scores[i]) for i in top_k_indices]


if __name__ == "__main__":
    questions = [
        "What is Bach researching?",
        "What programming languages does Bach know?",
    ]

    for question in questions:
        print(f"\n{question}")
        print(f"{'rank':<5}| {'source':<14}| {'chunk_id':<9}| BM25 score")
        for rank, (chunk, score) in enumerate(retrieve_bm25(question, chunks, bm25, k=10), start=1):
            print(f"{rank:<5}| {chunk['source']:<14}| {chunk['chunk_id']:<9}| {score:.4f}")
