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

# Remove chunks left over from older versions of a document
# (e.g. a document that now splits into fewer chunks).
current_ids = {
    f"{chunk['source'].replace('.txt', '')}-{chunk['chunk_id']}"
    for chunk in chunks
}

indexed = chunks_container.query_items(
    query="SELECT c.id, c.source FROM c",
    enable_cross_partition_query=True,
)

for item in list(indexed):
    if item["id"] not in current_ids:
        chunks_container.delete_item(item["id"], partition_key=item["source"])
        print(f"Deleted stale chunk {item['id']}")
