# Ask Bach — a RAG system over Bach Nguyen's profile documents

Ask Bach answers natural-language questions about Bach Nguyen (experience, research, education, skills, projects) using **retrieval-augmented generation**. It retrieves the most relevant passages from a small set of hand-written profile documents, has an LLM answer **only from those passages**, and returns the answer together with **source metadata controlled by the application, not by the model**.

```json
{
  "answer": "Bach was an IT intern at the Canadian Mental Health Association (CMHA) Durham where ...",
  "sources": [
    {"source": "cmha.txt", "chunk_id": 3},
    {"source": "cmha.txt", "chunk_id": 2}
  ]
}
```

When the documents don't contain the answer, the system declines instead of guessing, and cites nothing:

```json
{
  "answer": "I don't have enough information to answer.",
  "sources": []
}
```

The project is also a deliberate exercise in **ML-engineering method**: every retrieval change was made one variable at a time and measured against a frozen evaluation set (see [Evaluation](#evaluation) and [Experiment log](#experiment-log-what-we-learned)).

> **Status (2026-10-06):** retrieval, structured generation and source resolution work end to end from the command line. There is no HTTP API, UI, or automated test suite yet — see [Roadmap](#roadmap).

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Components](#components)
  - [1. Documents and chunking](#1-documents-and-chunking)
  - [2. Dense retrieval](#2-dense-retrieval)
  - [3. Sparse retrieval (BM25)](#3-sparse-retrieval-bm25)
  - [4. Hybrid retrieval (Reciprocal Rank Fusion)](#4-hybrid-retrieval-reciprocal-rank-fusion)
  - [5. Context and source map](#5-context-and-source-map)
  - [6. Structured generation](#6-structured-generation)
  - [7. Source resolution](#7-source-resolution)
- [Response contract](#response-contract)
- [Evaluation](#evaluation)
- [Experiment log: what we learned](#experiment-log-what-we-learned)
- [Design decisions](#design-decisions)
- [Known limitations and backlog](#known-limitations-and-backlog)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

---

## Quick start

**Requirements:** Python 3.12, a free [Groq](https://console.groq.com) API key. The embedding model (`all-MiniLM-L6-v2`, ~90 MB) downloads from Hugging Face on first run.

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install sentence-transformers rank-bm25 groq python-dotenv numpy

# 3. Add your Groq key to .env in the project root (it is git-ignored)
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

Ask the two built-in demo questions (one answerable, one not):

```bash
python src/rag.py
```

Use it from Python:

```python
# run from src/, or add src/ to sys.path
from rag import answer_question

response = answer_question("What Azure experience does Bach have?")
print(response["answer"])
print(response["sources"])
```

Scripts can be run from the project root or from `src/`; all file paths are resolved relative to the source files, not the working directory.

> There is no `requirements.txt` yet. Versions this was developed with: `sentence-transformers 6.1.0`, `torch 2.14.1`, `rank-bm25 0.2.2`, `groq 1.7.0`, `python-dotenv 1.2.4`, `numpy 2.5.3`.

---

## How it works

```
                    question
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Dense retrieval            BM25 retrieval            (each returns top 10 candidates)
   MiniLM embeddings          keyword matching
   + cosine similarity        + stopword removal
          └────────────┬────────────┘
                       ▼
          Reciprocal Rank Fusion (k = 60)               → final top 3 chunks
                       │
                       ▼
     Context [1] [2] [3]   +   source_map {1: (file, chunk), …}
     (sent to the LLM)         (kept in Python, never sent)
                       │
                       ▼
     Groq LLM, JSON-schema structured output
     → {"answer": "...", "used_context": [1, 2]}
                       │
                       ▼
     resolve_sources(used_context, source_map)
     → {"answer": "...", "sources": [{"source": ..., "chunk_id": ...}]}
```

Who is responsible for what:

| Layer | Decides |
|---|---|
| **Retriever** | Which 3 chunks are worth showing the model |
| **LLM** | What to say, and which evidence IDs it used |
| **Python** | Whether those IDs are valid, and which real document/chunk each one is |
| **UI** (future) | How sources are displayed |

---

## Project layout

```
ML Project/
├── data/documents/          # Knowledge base: 6 plain-text profile documents
├── eval/                    # Saved evaluation outputs (JSON)
│   ├── results.json             # end-to-end answers (legacy dense pipeline)
│   └── retrieval_results.json   # retrieval-only eval, 3 retrievers × 10 questions
├── src/
│   ├── load_documents.py    # load .txt files, split into titled chunks
│   ├── similarity.py        # cosine similarity
│   ├── embed_documents.py   # dense retriever (embeds all chunks at import)
│   ├── bm25_retriever.py    # tokenizer + BM25 index + BM25 retriever
│   ├── hybrid_retriever.py  # Reciprocal Rank Fusion of dense + BM25
│   ├── generate.py          # Groq client, free-text and structured generation
│   ├── rag.py               # ★ full pipeline: answer_question()
│   ├── metrics.py           # reciprocal_rank() for MRR@k
│   ├── eval_retrieval.py    # retrieval evaluation (Recall@3, Precision@3, MRR@3)
│   └── run_eval.py          # end-to-end answer evaluation (legacy dense pipeline)
├── tests/                   # empty — no automated tests yet
├── .env                     # GROQ_API_KEY (git-ignored, you create it)
└── README.md
```

**Where to start reading:** `src/rag.py` — `answer_question()` calls every other stage in order.

Every module with a `__main__` block doubles as a runnable demo:

| Command | What it shows |
|---|---|
| `python src/load_documents.py` | Number of chunks created |
| `python src/bm25_retriever.py` | BM25 top 10 for two test questions |
| `python src/hybrid_retriever.py` | Dense vs BM25 vs hybrid top 3 side by side |
| `python src/rag.py` | `resolve_sources` self-checks, then two full structured answers |
| `python src/eval_retrieval.py` | Retrieval evaluation table (no LLM calls) |
| `python src/run_eval.py` | 8-question end-to-end answer evaluation (makes LLM calls) |
| `python src/embed_documents.py` | Legacy dense demo (prints rankings, makes 2 LLM calls) |

---

## Components

### 1. Documents and chunking

**File:** `src/load_documents.py`

The knowledge base is six `.txt` files in `data/documents/`:

| File | Covers |
|---|---|
| `about.txt` | Profile, bio, experience summary, technical skills, contact |
| `azure.txt` | Statistics Without Borders Azure work, Hack Hive Azure challenge, cloud/data-engineering skills |
| `cmha.txt` | IT internship at CMHA Durham |
| `education.txt` | Degree, coursework, certifications, TA role, campus leadership |
| `projects.txt` | Personal and academic ML/data projects |
| `research.txt` | Undergraduate thesis research (continual learning, vision transformers) |

**Document format conventions** (the chunker relies on these):

- The **first line is the document title**, e.g. `IT INTERN — CANADIAN MENTAL HEALTH ASSOCIATION (CMHA DURHAM)`.
- Lines made only of `=` characters are treated as dividers and removed.
- Paragraphs are separated by a blank line.
- Many sections end in a `QUICK FACTS (FOR COMMON QUESTIONS)` list written as Q&A lines. These strongly affect retrieval — see the [experiment log](#experiment-log-what-we-learned).

**Chunking** (`chunk_document`): paragraphs are packed greedily into chunks of at most `max_chars=1200` characters, and **the document title is prepended to every chunk**, so a chunk from the middle of `cmha.txt` still says what it is about. A single paragraph longer than 1200 characters becomes its own (oversized) chunk; it is never split.

Each chunk is a dict:

```python
{"source": "cmha.txt", "chunk_id": 3, "text": "IT INTERN — CANADIAN MENTAL HEALTH ...\n\n- Where did Bach intern? ..."}
```

`(source, chunk_id)` is the chunk's **stable identity** throughout the system (retrieval fusion, evaluation labels, source map). The current corpus produces **33 chunks**.

> ⚠️ Chunk IDs depend on document content and `max_chars`. Editing a document or changing the chunk size can renumber chunks, which silently invalidates the relevance labels in `src/eval_retrieval.py`. Re-check labels after any such change.

### 2. Dense retrieval

**File:** `src/embed_documents.py` (name is historical — this is the dense retriever)

- Model: `sentence-transformers/all-MiniLM-L6-v2` → one **384-dimensional** vector per chunk.
- Similarity: cosine (`src/similarity.py`), computed against every chunk (brute force — fine for 33 chunks).
- **Side effect on import:** loading this module loads the model and embeds all chunks. Every module that imports it (hybrid, RAG, eval) pays that cost once.

```python
retrieve_top_k_documents(query, embeddings, chunks, k=3) -> [(chunk, cosine_score), ...]
```

The module also contains the earlier `build_context()` / `build_prompt()` free-text prompt used by `run_eval.py`. The current pipeline uses the versions in `rag.py` instead.

**Strength:** matches meaning, not exact words. **Weakness in this corpus:** queries containing "Bach" are pulled toward general profile chunks (see the experiment log).

### 3. Sparse retrieval (BM25)

**File:** `src/bm25_retriever.py` — uses `rank_bm25.BM25Okapi`.

Tokenizer (the final, frozen version):

```python
tokenize("What programming languages does Bach know?")
# → ['programming', 'languages', 'bach', 'know']
```

1. Lowercase, then `re.findall(r"\w+", ...)` — keeps runs of letters/digits, drops punctuation (`"languages:"` → `languages`, `C++` → `c`, `Bach's` → `bach`, `s`).
2. Remove a small stopword list (`what`, `is`, `does`, `the`, …; see `STOPWORDS`).
3. **No stemming** (`researching` does not match `research`), and **`bach` is deliberately kept**.

```python
retrieve_bm25(query, chunks, bm25, k=3) -> [(chunk, bm25_score), ...]
```

It returns the same `(chunk, score)` shape as the dense retriever, which is what makes fusion simple.

**Strength:** exact rare-term matches (`programming languages`). **Weakness:** pure string matching — `"graduate"` matches the "new-graduate opportunities" line in the contact chunk, and `"finish"` matches an infrastructure project "expected to finish by 2028".

### 4. Hybrid retrieval (Reciprocal Rank Fusion)

**File:** `src/hybrid_retriever.py`

Each retriever produces its top `candidate_k = 10`; RRF merges them by **rank only**:

$$\text{RRF}(c) = \frac{1}{k + \text{rank}_\text{dense}(c)} + \frac{1}{k + \text{rank}_\text{BM25}(c)}, \quad k = 60$$

A chunk missing from one list gets no contribution from it. Chunks are matched across lists by `(source, chunk_id)`, never by text. The top `final_k = 3` are returned.

```python
retrieve_hybrid(query, candidate_k=10, final_k=3, rrf_k=60) -> [(chunk, rrf_score), ...]
# final_k=None returns the full fused ranking (used by the evaluation)
```

**Why retrieve 10 and keep 3?** Correct evidence often sits at rank 4–5 in one retriever. Fusing only each retriever's top 3 would drop it before fusion could rescue it. *Candidate depth does not have to equal the amount of context sent to the LLM.*

**Known property:** RRF ignores score magnitude. A confident BM25 ranking (top score 2× the next) and a near-tie (1.40 vs 1.11) contribute identically. Fusion also cannot help when both retrievers rank the same wrong chunks highly.

### 5. Context and source map

**File:** `src/rag.py` — `build_context_and_source_map(results)`

Returns two things:

```
context (sent to the LLM)                 source_map (kept in Python)
──────────────────────────                ──────────────────────────────────────────
[1]                                       {1: {"source": "cmha.txt",  "chunk_id": 3},
IT INTERN — CANADIAN MENTAL HEALTH ...     2: {"source": "cmha.txt",  "chunk_id": 2},
...                                        3: {"source": "about.txt", "chunk_id": 0}}
---
[2]
...
```

The LLM sees only **evidence IDs and text** — no filenames, chunk IDs, or scores. See [Design decisions](#design-decisions) for why.

### 6. Structured generation

**File:** `src/generate.py` — `generate_structured(prompt)`

- Provider: **Groq** (free tier), model `openai/gpt-oss-120b` (constant `MODEL`).
- The API key is read from the `GROQ_API_KEY` environment variable, loaded from `.env`. It never appears in code.
- Uses Groq's **JSON-schema structured outputs in strict mode**, so the response is guaranteed to parse into exactly this shape:

```python
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "used_context": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "used_context"],
    "additionalProperties": False,
}
```

The prompt (`build_structured_prompt` in `rag.py`) tells the model to answer only from the context, to say it doesn't have enough information otherwise, to list the evidence IDs it actually used, and **not** to put citation markers in the answer text.

> **Schema validation checks structure, not truth.** `{"answer": "Bach worked at Google.", "used_context": [999]}` is schema-valid. That is why the next step exists.

`generate_answer(prompt)` (free text, no schema) is kept only for the legacy `run_eval.py`.

### 7. Source resolution

**File:** `src/rag.py` — `resolve_sources(used_context, source_map)`

Turns the model's evidence IDs into real source metadata. Current (development) policy:

| Model returns | Result | Why |
|---|---|---|
| `[1, 3]` | sources for 1 and 3, in that order | normal case |
| `[1, 999]` | source for 1 only; 999 ignored | invalid ID — ignored for now so model behaviour can be observed; may become a hard failure later |
| `[1, 1]` | source for 1 once | duplicate |
| `[]` | `[]` | model used no evidence (e.g. a refusal) |

Because sources come only from `source_map`, **every displayed source is guaranteed to be a chunk that was actually retrieved and shown to the model.** It is *not* guaranteed to support every sentence of the answer (see [limitations](#known-limitations-and-backlog)).

---

## Response contract

`answer_question(question, candidate_k=10, final_k=3, verbose=False)` returns:

| Field | Type | Meaning |
|---|---|---|
| `answer` | `str` | The model's answer, with no inline citation markers. A refusal is a normal answer string. |
| `sources` | `list[{"source": str, "chunk_id": int}]` | Chunks the model says it used, validated against what was retrieved. Empty for refusals. |

Side effects: one Groq API call per question (network; counts against the free-tier rate limit). `verbose=True` also prints the raw model JSON, source map and resolved sources.

This `{answer, sources}` object is intended to be the response body of the future HTTP endpoint.

---

## Evaluation

There are two separate evaluations, because retrieval and generation fail in different ways.

### Retrieval evaluation — `src/eval_retrieval.py`

Measures **question → retriever → top-3 evidence**, with no LLM involved. Output is saved to `eval/retrieval_results.json`.

**Question set:** 5 answerable questions plus a paraphrase of each (10 total). Paraphrases share their original's relevance labels because they ask for the same information.

**Relevance rule** (decided before any results were seen): *a chunk is relevant if, on its own, it supports a correct and specific answer — it names the role, the concrete work, or the fact. A passing mention or a bare skill keyword does not count.* Relevance is a **set** of chunks per question, so a valid summary source (e.g. `about.txt` chunk 0) is not penalised.

**Metrics** (all reported on every run):

| Metric | Question it answers | Definition |
|---|---|---|
| **Recall@3** | Did we retrieve at least one useful piece of evidence? | 1 if any relevant chunk is in the top 3, else 0; averaged |
| **Context Precision@3** | How much of the context is actually relevant? | relevant chunks in top 3 ÷ 3; summed over questions |
| **MRR@3** | How early does the first relevant evidence appear? | mean of 1/rank of first relevant chunk in the top 3 (0 if none) |

MRR@3 was added *post hoc*, after inspecting the paraphrase run. It is truncated at 3 because chunks ranked lower are never sent to the LLM.

**Frozen configuration** (do not tune against this set — that would overfit 10 questions):

| Component | Setting |
|---|---|
| Dense | `all-MiniLM-L6-v2` + cosine |
| BM25 | regex tokenizer + stopword removal, no stemming, `bach` kept |
| Hybrid | `candidate_k=10` per retriever, `rrf_k=60`, `final_k=3` |

**Results** (snapshot, 2026-10-06):

| Metric | Dense | BM25 | Hybrid |
|---|---|---|---|
| Recall@3 — originals | 4/5 | 5/5 | 5/5 |
| Recall@3 — paraphrases | 4/5 | 5/5 | 5/5 |
| **Recall@3 — overall** | **8/10** | **10/10** | **10/10** |
| **Context Precision@3** | 13/30 | **22/30** | 21/30 |
| **MRR@3** *(post hoc)* | 0.700 | 0.833 | **0.950** |

**Reading these honestly:**
- Hybrid beats dense clearly. Hybrid vs BM25 is **not settled**: they tie on recall, BM25 is slightly ahead on precision, and hybrid's MRR lead comes from just two questions where BM25's #1 was a false keyword match.
- Recall@3 can read 100% while hiding a quality problem: for *"What is Bach researching?"*, every system "passes" only via a one-sentence summary in `about.txt`; no `research.txt` chunk reaches the top 3.
- The paraphrases were written by someone who knows the documents, so several reuse document wording ("undergraduate thesis", "Canadian Mental Health Association") and did not really stress-test BM25.
- 10 questions is enough to compare systems on *this* set, not to claim production accuracy.

### End-to-end answer evaluation — `src/run_eval.py` (legacy)

Runs 8 questions (5 answerable, 3 that should be refused) through the **earlier** pipeline — dense retrieval, top 3, free-text answer with inline citations — and saves the question, top 3, answer and latency to `eval/results.json`.

Snapshot (2026-10-06, `openai/gpt-oss-120b`): 4/5 answerable questions answered correctly (the miss was a retrieval failure — the evidence was ranked 5th), 3/3 unanswerable questions correctly refused, 0.4–1.3 s per question.

> This script has not been migrated to the hybrid + structured pipeline yet.

---

## Experiment log: what we learned

Each step changed **one variable** and was measured before moving on.

| # | Experiment | Finding |
|---|---|---|
| 1 | Dense top 3 on 8 questions | 4/5 answerable, 3/3 refusals. "Programming languages" failed: evidence ranked **#5**. |
| 2 | Dense top 10 | Correct evidence for both failing questions sat at **#5**. The same three general chunks (`about` c0, `education` c3, `about` c4) topped *every* query. |
| 3 | Query wording (remove "Bach" / keyword-style) | Languages: removing "Bach" moved evidence **#5 → #1** — the name was pulling queries toward general chunks. Research: removing "Bach" did *not* help; the query has no topical words to match ("researching" vs "continual learning"). Two different failure causes. |
| 4 | BM25 with `.split()` | Evidence only #4. `"languages:"` ≠ `"languages"` (punctuation), and Q&A-style chunks scored on `what`/`does`/`is`. |
| 5 | BM25 + regex tokenizer | Languages → **#1**, but narrowly (7.45 vs 6.93); a chunk containing "Does Bach **know** Airflow…?" came second. |
| 6 | BM25 + stopwords | Languages lead widened to ~2× (7.08 vs 3.60). Research got *worse*: with only `bach` left to match, the ranking became near-random (scores 1.40 → 1.11). |
| 7 | Hybrid RRF | Languages fixed (#1) by a margin of 0.000024. Research unchanged: both retrievers agreed on the same wrong chunks, and **fusion cannot fix shared failures**. |
| 8 | Frozen 3-way eval + paraphrases | See [results](#retrieval-evaluation--srcevalretrievalpy). |
| 9 | Structured generation | Model chose cmha [1][2] and skipped the weaker summary [3]; the Google question returned `used_context: []`. Inline-citation errors (`【1†L7-L9】`) disappeared. |

---

## Design decisions

**The LLM sees evidence IDs, not filenames.** Earlier, with metadata in the prompt, the model produced inconsistent and sometimes invented citations (`【2†source】`, `【1†L7-L9】`, citing line numbers that don't exist). Integer IDs are trivially checkable (only `1..3` are valid), the model cannot invent a filename, storage details can change without touching the prompt, and filenames can't leak into answers. *Rule: the model's `used_context` is a claim; Python decides what it maps to.*

**Structured output instead of parsing prose.** Strict JSON-schema mode guarantees the response shape, so there's no fragile regex over the answer text. Validation of the *content* (`resolve_sources`) still happens in Python.

**Hybrid retrieval kept, with an open question.** Dense alone fails when the query is dominated by the subject's name; BM25 alone fails on coincidental keyword matches. RRF was never worse than BM25 on any question, but evidence that it is *better* is still thin (see results).

**Candidate depth ≠ context size.** Searching 10 deep per retriever but sending 3 chunks lets fusion rescue evidence ranked 4–5 without inflating the prompt.

**Evaluation frozen before measuring.** Tuning `k`, stopwords or `rrf_k` against 10 questions would overfit the benchmark. Adding a question *because* a system fails or wins on it would too.

**Groq, `openai/gpt-oss-120b`.** Free tier. The originally chosen `llama-3.3-70b-versatile` has been retired on Groq; available models can be listed with `client.models.list()`.

---

## Known limitations and backlog

**Answer quality**
- **Groundedness is not checked.** Valid sources ≠ supported sentences. Observed example: *"saved more than five staff hours per week **for over 100 staff**"* merges two separate facts, and *"accessibility **and reliability**"* adds a word not in the source. Needs its own evaluation.
- The documents never state Bach's pronouns, so the model guesses ("he"). Adding them to `about.txt` would remove the guess.
- Refusal wording varies ("I don't know." vs "I don't have enough information to answer.").

**Retrieval**
- Research questions still don't retrieve `research.txt` in the top 3 (query has no topical overlap; no stemming).
- Q&A-style "QUICK FACTS" chunks act as hubs that rank highly for many queries.
- **Chunking artifact:** a section heading is its own paragraph, so it ends up at the *end* of the previous chunk (e.g. a chunk ending in `QUICK FACTS (FOR COMMON QUESTIONS)`).
- Planned metric: *preferred-source Recall@3* (authoritative chunk vs any sufficient chunk) to expose the research weakness.

**Evaluation**
- Small (10 retrieval questions, 8 end-to-end); paraphrases share document vocabulary.
- `run_eval.py` still uses the legacy dense/free-text pipeline.
- Relevance labels are tied to current chunk IDs (see the warning in [chunking](#1-documents-and-chunking)).

**Engineering**
- No `requirements.txt`, no automated tests (`tests/` is empty).
- Importing `embed_documents` / `bm25_retriever` builds indexes at import time; both call `load_chunks()`, so "Loaded 6 documents" prints twice.
- `src/__pycache__/` is not git-ignored.

---

## Roadmap

1. Expose `answer_question` through a **FastAPI** endpoint returning `{answer, sources}`.
2. Call it from the Astro portfolio site ("Ask Bach").
3. Grow the eval set to 50–100 questions across categories — direct, paraphrased (ideally written without looking at chunk text), multi-fact, vague, unanswerable, adversarial/misleading — and add groundedness, citation-correctness, refusal-rate and latency metrics.
4. Revisit retrieval only with the larger eval set: stemming, re-ranking, query rewriting, chunking fixes, stronger embedding model.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `groq.AuthenticationError: 401 invalid_api_key` | The value in `.env` isn't a Groq key. Groq keys start with `gsk_`. Create one at console.groq.com/keys and copy the full key from the creation popup (it's shown once). |
| `groq.NotFoundError: 404 model_not_found` | The model was retired. List available models with `from generate import client; [m.id for m in client.models.list().data]` and update `MODEL` in `src/generate.py`. |
| `GroqError: The api_key client option must be set` | No `GROQ_API_KEY` found. Make sure `.env` is in the **project root**, not `src/`. |
| `Warning: You are sending unauthenticated requests to the HF Hub` | Harmless. Set `HF_TOKEN` only if model downloads get rate-limited. |
| `ValueError: ... unrecognized modality keys: ['chunk_id', 'source']` | Chunk dicts were passed to `model.encode`; pass the list of `chunk["text"]` strings instead. |
| Eval relevance looks wrong after editing a document | Chunk IDs shifted. Re-check the labels in `src/eval_retrieval.py`. |
