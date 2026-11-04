"""Retrieval benchmark: questions and hand-labelled relevant chunks.

Relevance rule (fixed before any results were seen): a chunk is relevant if, on its
own, it supports a correct and specific answer, i.e. it names the role, the concrete
work, or the fact. A passing mention or a bare skill keyword does not count.

Labels v2 (2026-11-04). The v1 labels (2026-10-06) were written for a 6-document,
33-chunk corpus. Since then the corpus has grown to 21 documents and 115 chunks,
and about.txt, education.txt and research.txt were rewritten, so several v1 chunk
IDs no longer existed (research.txt now has chunks 0-3, while v1 labelled chunks 5
and 6). The v2 labels apply the same rule to the current chunks and were written
before the v2 eval was run. The questions are unchanged.

Labels are tied to chunk IDs from `chunk_document(max_chars=1200)`.
tests/test_benchmark.py fails if a label points at a chunk that no longer exists.
"""

CMHA = {
    ("cmha.txt", 0), ("cmha.txt", 1), ("cmha.txt", 2), ("cmha.txt", 3),
    ("about.txt", 1),
}
CLOUD = {
    ("azure.txt", 0), ("azure.txt", 1), ("azure.txt", 2),
    ("about.txt", 0), ("about.txt", 1),
    # added in v2: documents that did not exist in v1 and describe concrete Azure work
    ("ask_bach.txt", 6), ("ask_bach.txt", 7), ("project_market_guard.txt", 4),
}
RESEARCH = {
    ("research.txt", 0), ("research.txt", 1), ("research.txt", 2), ("research.txt", 3),
    ("about.txt", 0), ("about.txt", 1), ("education.txt", 1),
    ("project_cpg_cascadedvit.txt", 0),
}
GRADUATION = {("education.txt", 0), ("education.txt", 3), ("about.txt", 0)}
LANGUAGES = {("about.txt", 2)}

# Each paraphrase shares its original's label set (same information need).
# Caveat: all were written by the corpus author, and several reuse document wording.
ANSWERABLE = [
    {"question": "What did Bach do at CMHA?", "variant": "original", "relevant": CMHA},
    {"question": "What Azure experience does Bach have?", "variant": "original", "relevant": CLOUD},
    {"question": "What is Bach researching?", "variant": "original", "relevant": RESEARCH},
    {"question": "When does Bach graduate?", "variant": "original", "relevant": GRADUATION},
    {"question": "What programming languages does Bach know?", "variant": "original", "relevant": LANGUAGES},
    {"question": "Describe Bach's work at the Canadian Mental Health Association.", "variant": "paraphrase", "relevant": CMHA},
    {"question": "What cloud computing work has Bach done?", "variant": "paraphrase", "relevant": CLOUD},
    {"question": "What topic is Bach's undergraduate thesis focused on?", "variant": "paraphrase", "relevant": RESEARCH},
    {"question": "When is Bach expected to finish university?", "variant": "paraphrase", "relevant": GRADUATION},
    {"question": "Which coding languages can Bach use?", "variant": "paraphrase", "relevant": LANGUAGES},
]

# For refusal / hallucination testing of the full pipeline, not scored for retrieval.
UNANSWERABLE = [
    "Did Bach work at Microsoft?",
    "What is Bach's favorite programming language?",
    "How many years did Bach work at Google?",
]
