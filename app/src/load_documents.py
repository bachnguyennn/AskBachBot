from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def load_documents(docs_dir=DOCS_DIR):
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": path.name, "text": text})
    print(f"Loaded {len(docs)} documents from {docs_dir}")
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


def load_chunks(docs_dir=DOCS_DIR, max_chars=1200):
    return [c for d in load_documents(docs_dir) for c in chunk_document(d, max_chars)]


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Created {len(chunks)} chunks")
