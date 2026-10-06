from sentence_transformers import SentenceTransformer
from load_documents import load_chunks
from similarity import cosine_similarity
from generate import generate_answer


model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = load_chunks()
texts = [c["text"] for c in chunks]

embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


assert len(chunks) == len(embeddings)

queries = [
    "What Azure experience does Bach have?",
    "What computer vision research has Bach done?",
    "What did Bach do at CMHA?"
]


def retrieve_top_k_documents(query, embeddings, chunks, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    similarities = [cosine_similarity(query_embedding, embedding) for embedding in embeddings]
    top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
    return [(chunks[i], similarities[i]) for i in top_k_indices]


for query in queries:
    print(f"\nQuery: {query}")

    results = retrieve_top_k_documents(
        query,
        embeddings,
        chunks,
        k=3
    )

    for rank, (chunk, score) in enumerate(results, start=1):
        print(
            f"{rank}. {chunk['source']} | "
            f"chunk {chunk['chunk_id']} | "
            f"{score:.4f}"
        )

def build_context(results):
    blocks = []
    for rank, (chunk, score) in enumerate(results, start=1):
        blocks.append(
            f"[{rank}] (source: {chunk['source']}, chunk {chunk['chunk_id']})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt(question, context):
    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know. "
        "Cite the sources you use, like [1] or [2].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


test_questions = [
    "What did Bach do at CMHA?",
    "What position did Bach have at Google?",
]

for question in test_questions:
    results = retrieve_top_k_documents(question, embeddings, chunks, k=3)
    context = build_context(results)
    prompt = build_prompt(question, context)
    answer = generate_answer(prompt)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
