"""Chunk the documents in Cosmos DB, embed them, and upsert into the `chunks`
container. Removes chunks left over from older document versions.

Run from app/:  python -m scripts.index_chunks
"""

from src.database import get_container
from src.dense_retriever import embed
from src.load_documents import load_chunks

chunks_container = get_container("chunks")

chunks = load_chunks(source="cosmos")
embeddings = embed([chunk["text"] for chunk in chunks])

for chunk, embedding in zip(chunks, embeddings, strict=True):
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
