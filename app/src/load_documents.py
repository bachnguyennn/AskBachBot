"""Load knowledge-base documents and split them into chunks.

Production reads documents from Cosmos DB (`content` container). The `.txt` files in
`data/documents/` are the authoring source that `scripts/migrate_documents.py`
uploads; the offline eval and the tests read them directly from disk.
"""

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def load_documents_from_cosmos():
    from src.database import get_container

    items = get_container("content").query_items(
        query="SELECT c.source, c.text FROM c WHERE c.type = 'document'",
        enable_cross_partition_query=True,
    )
    docs = [{"source": i["source"], "text": i["text"]} for i in items if i.get("text", "").strip()]
    return sorted(docs, key=lambda d: d["source"])


def load_documents_from_disk(docs_dir=DOCS_DIR):
    docs = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": path.name, "text": text})
    return docs


def chunk_document(doc, max_chars=1200):
    # drop "=====" divider lines, keep the first line as the document title
    lines = [
        line
        for line in doc["text"].splitlines()
        if not line.strip() or set(line.strip()) != {"="}
    ]
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


def chunk_documents(docs, max_chars=1200):
    return [c for d in docs for c in chunk_document(d, max_chars)]


def load_chunks(source="cosmos", max_chars=1200):
    docs = load_documents_from_cosmos() if source == "cosmos" else load_documents_from_disk()
    return chunk_documents(docs, max_chars)


if __name__ == "__main__":
    chunks = load_chunks(source="disk")
    print(f"Created {len(chunks)} chunks from {DOCS_DIR}")
