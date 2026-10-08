from src.hybrid_retriever import retrieve_hybrid

def chunk_key(chunk):
    return (chunk["source"], chunk["chunk_id"])


def recall_at_k(results, relevant):
    retrieved = {
        chunk_key(chunk)
        for chunk, _score in results
    }

    return 1 if retrieved & relevant else 0


def precision_at_k(results, relevant):
    relevant_count = sum(
        1
        for chunk, _score in results
        if chunk_key(chunk) in relevant
    )

    return relevant_count / len(results)


def reciprocal_rank(results, relevant):
    for rank, (chunk, _score) in enumerate(results, start=1):
        if chunk_key(chunk) in relevant:
            return 1 / rank

    return 0

TEST_CASES = [
    {
        "query": "What did Bach do at CMHA?",
        "relevant": {
            ("cmha.txt", 0),
            ("cmha.txt", 1),
            ("cmha.txt", 2),
            ("cmha.txt", 3),
            ("about.txt", 1),
        },
    },

    {
        "query": "What cloud work has Bach done with Azure?",
        "relevant": {
            ("azure.txt", 0),
            ("azure.txt", 1),
            ("azure.txt", 2),
            ("about.txt", 0),
            ("about.txt", 1),
        },
    },

    {
        "query": "What is Bach researching?",
        "relevant": {
            ("research.txt", 0),
            ("research.txt", 1),
            ("research.txt", 5),
            ("research.txt", 6),
            ("about.txt", 0),
            ("about.txt", 1),
            ("education.txt", 1),
        },
    },

    {
        "query": "When does Bach graduate?",
        "relevant": {
            ("education.txt", 0),
            ("education.txt", 3),
            ("about.txt", 0),
        },
    },

    {
        "query": "What programming languages does Bach know?",
        "relevant": {
            ("about.txt", 2),
        },
    },

    {
        "query": "Describe Bach's work at the Canadian Mental Health Association.",
        "relevant": {
            ("cmha.txt", 0),
            ("cmha.txt", 1),
            ("cmha.txt", 2),
            ("cmha.txt", 3),
            ("about.txt", 1),
        },
    },

    {
        "query": "What cloud computing work has Bach done?",
        "relevant": {
            ("azure.txt", 0),
            ("azure.txt", 1),
            ("azure.txt", 2),
            ("about.txt", 0),
            ("about.txt", 1),
        },
    },

    {
        "query": "What topic is Bach's undergraduate thesis focused on?",
        "relevant": {
            ("research.txt", 0),
            ("research.txt", 1),
            ("research.txt", 5),
            ("research.txt", 6),
            ("about.txt", 0),
            ("about.txt", 1),
            ("education.txt", 1),
        },
    },

    {
        "query": "When is Bach expected to finish university?",
        "relevant": {
            ("education.txt", 0),
            ("education.txt", 3),
            ("about.txt", 0),
        },
    },

    {
        "query": "Which coding languages can Bach use?",
        "relevant": {
            ("about.txt", 2),
        },
    },
]

recalls = []
precisions = []
mrrs = []


for case in TEST_CASES:
    query = case["query"]
    relevant = case["relevant"]

    results = retrieve_hybrid(
    query,
    candidate_k=10,
    final_k=3
    )

    recall = recall_at_k(results, relevant)
    precision = precision_at_k(results, relevant)
    mrr = reciprocal_rank(results, relevant)

    recalls.append(recall)
    precisions.append(precision)
    mrrs.append(mrr)

    print("\n" + "=" * 70)
    print("QUERY:", query)

    for rank, (chunk, score) in enumerate(results, start=1):
        marker = "✓" if chunk_key(chunk) in relevant else "✗"

        print(
            f"{rank}. {marker} "
            f"{chunk['source']} "
            f"chunk {chunk['chunk_id']} "
            f"score={score:.4f}"
        )

    print(
        f"Recall@3={recall} | "
        f"Precision@3={precision:.3f} | "
        f"MRR@3={mrr:.3f}"
    )


n = len(TEST_CASES)

print("\n" + "=" * 70)
print("COSMOS HYBRID RESULTS")

print(f"Recall@3:    {sum(recalls)}/{n} = {sum(recalls) / n:.3f}")
print(f"Precision@3: {sum(precisions) / n:.3f}")
print(f"MRR@3:       {sum(mrrs) / n:.3f}")