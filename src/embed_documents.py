from sentence_transformers import SentenceTransformer
from load_documents import load_chunks
from similarity import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = load_chunks()
texts = [c["text"] for c in chunks]

embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
question = "What cloud technologies has Bach worked with?"


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
