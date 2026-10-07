from sentence_transformers import SentenceTransformer

from src.database import client


model = SentenceTransformer("all-MiniLM-L6-v2")

database = client.get_database_client("ask-bach")
chunks_container = database.get_container_client("chunks")


def retrieve_top_k_cosmos(query, k=10):
    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

    results = list(
        chunks_container.query_items(
            query=f"""
                SELECT TOP {k}
                    c.source,
                    c.chunk_id,
                    c.text,
                    VectorDistance(
                        c.embedding,
                        @query_embedding
                    ) AS score
                FROM c
                ORDER BY VectorDistance(
                    c.embedding,
                    @query_embedding
                )
            """,
            parameters=[
                {
                    "name": "@query_embedding",
                    "value": query_embedding
                }
            ],
            enable_cross_partition_query=True,
        )
    )

    return [
        (
            {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
            },
            item["score"],
        )
        for item in results
    ]