from sentence_transformers import SentenceTransformer

from src.database import client


model = SentenceTransformer("all-MiniLM-L6-v2")

database = client.get_database_client("ask-bach")
chunks_container = database.get_container_client("chunks")


query = "What did Bach do at CMHA?"

query_embedding = model.encode(
    query,
    convert_to_numpy=True,
).tolist()


results = list(
    chunks_container.query_items(
        query="""
            SELECT TOP 3
                c.source,
                c.chunk_id,
                c.text,
                VectorDistance(c.embedding, @query_embedding) AS distance
            FROM c
            ORDER BY VectorDistance(c.embedding, @query_embedding)
        """,
        parameters=[
            {
                "name": "@query_embedding",
                "value": query_embedding,
            }
        ],
        enable_cross_partition_query=True,
    )
)


for i, result in enumerate(results, start=1):
    print(f"\nRank {i}")
    print("Source:", result["source"])
    print("Chunk:", result["chunk_id"])
    print("Distance:", result["distance"])
    print(result["text"][:300])