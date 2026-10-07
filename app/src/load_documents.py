from pathlib import Path

from src.database import container

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def load_documents():
    items = list(
        container.query_items(
            query="""
                SELECT c.source, c.text
                FROM c
                WHERE c.type = 'document'
            """,
            enable_cross_partition_query=True,
        )
    )

    docs = [
        {
            "source": item["source"],
            "text": item["text"],
        }
        for item in items
        if item.get("text", "").strip()
    ]

    docs.sort(key=lambda d: d["source"])

    print(f"Loaded {len(docs)} documents from Cosmos DB")

    return docs


def chunk_document(doc, max_chars=1200):
    # drop "=====" divider lines, keep the first line as the document title
    lines = [l for l in doc["text"].splitlines() if not l.strip() or set(l.strip()) != {"="}]
    title = lines[0].strip()
    paragraphs = [p.strip() for p in "\n".join(lines[1:]).split("\n\n") if p.strip()]

    # pack paragraphs into chunks up to max_chars
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len("\n\n") + len(p) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)

    return [
        {"source": doc["source"], "chunk_id": i, "text": f"{title}\n\n{c}"}
        for i, c in enumerate(chunks)
    ]


def load_chunks(max_chars=1200):
    return [
        c
        for d in load_documents()
        for c in chunk_document(d, max_chars)
    ]

if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Created {len(chunks)} chunks")
