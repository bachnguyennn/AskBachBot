from sentence_transformers import SentenceTransformer

from src.database import client
from src.load_documents import load_chunks


model = SentenceTransformer("all-MiniLM-L6-v2")

database = client.get_database_client("ask-bach")
chunks_container = database.get_container_client("chunks")


chunks = load_chunks()

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
)

for chunk, embedding in zip(chunks, embeddings):
    item = {
        "id": f"{chunk['source'].replace('.txt', '')}-{chunk['chunk_id']}",
        "source": chunk["source"],
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],

        # NumPy arrays cannot be stored directly as JSON,
        # so convert it to a normal Python list.
        "embedding": embedding.tolist(),
    }

    chunks_container.upsert_item(item)

    print(
        f"Indexed {chunk['source']} "
        f"chunk {chunk['chunk_id']}"
    )

print(f"Indexed {len(chunks)} chunks.")