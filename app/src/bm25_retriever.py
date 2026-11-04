"""Lexical retrieval with BM25 (rank_bm25), built in memory from the chunks."""

import re

from rank_bm25 import BM25Okapi

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


def build_bm25(chunks):
    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])


def retrieve_bm25(query, chunks, bm25, k=3):
    scores = bm25.get_scores(tokenize(query))
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], float(scores[i])) for i in top_k_indices]
