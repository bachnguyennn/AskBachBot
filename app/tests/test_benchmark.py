from eval.benchmark import ANSWERABLE
from src.load_documents import load_chunks


def test_every_label_points_at_an_existing_chunk():
    """Labels are tied to chunk IDs. If a document edit renumbers its chunks, this
    fails, so the benchmark gets re-labelled instead of silently scoring against
    chunks that no longer exist."""
    existing = {(c["source"], c["chunk_id"]) for c in load_chunks(source="disk")}
    for item in ANSWERABLE:
        missing = item["relevant"] - existing
        assert not missing, f"{item['question']!r} labels missing chunks: {sorted(missing)}"
