from src.load_documents import chunk_document, chunk_documents, load_documents_from_disk


def test_chunk_document_drops_dividers_and_prefixes_title():
    doc = {"source": "x.txt", "text": "MY TITLE\n=====\n\nFirst paragraph.\n\nSecond paragraph."}
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "MY TITLE\n\nFirst paragraph.\n\nSecond paragraph."
    assert "=====" not in chunks[0]["text"]


def test_chunk_document_packs_paragraphs_up_to_max_chars():
    paragraphs = ["p" * 400 for _ in range(5)]
    doc = {"source": "x.txt", "text": "TITLE\n\n" + "\n\n".join(paragraphs)}
    chunks = chunk_document(doc, max_chars=1000)
    # 400 + 2 + 400 = 802 fits; adding a third (1204) would not.
    assert [c["chunk_id"] for c in chunks] == [0, 1, 2]
    assert all(c["text"].startswith("TITLE\n\n") for c in chunks)
    assert all(len(c["text"]) - len("TITLE\n\n") <= 1000 for c in chunks)


def test_real_corpus_chunks_cleanly():
    """Content lint for the knowledge base: every document loads and chunks, chunk
    identities are unique, and no chunk is empty."""
    docs = load_documents_from_disk()
    assert docs, "no documents found in data/documents"
    chunks = chunk_documents(docs)
    keys = [(c["source"], c["chunk_id"]) for c in chunks]
    assert len(keys) == len(set(keys))
    assert all(c["text"].split("\n\n", 1)[-1].strip() for c in chunks)
