from src import rag

SOURCE_MAP = {
    1: {"source": "cmha.txt", "chunk_id": 3},
    2: {"source": "cmha.txt", "chunk_id": 2},
    3: {"source": "about.txt", "chunk_id": 0},
}


def test_resolve_sources_keeps_valid_ids_in_order():
    assert rag.resolve_sources([1, 3], SOURCE_MAP) == [SOURCE_MAP[1], SOURCE_MAP[3]]


def test_resolve_sources_drops_ids_the_model_invented():
    assert rag.resolve_sources([1, 999, -1, 0], SOURCE_MAP) == [SOURCE_MAP[1]]


def test_resolve_sources_deduplicates_and_handles_empty():
    assert rag.resolve_sources([2, 2, 2], SOURCE_MAP) == [SOURCE_MAP[2]]
    assert rag.resolve_sources([], SOURCE_MAP) == []


def test_context_uses_numbered_evidence_without_leaking_filenames():
    results = [
        ({"source": "cmha.txt", "chunk_id": 3, "text": "CMHA text"}, 0.5),
        ({"source": "about.txt", "chunk_id": 0, "text": "About text"}, 0.4),
    ]
    context, source_map = rag.build_context_and_source_map(results)
    assert context.startswith("[1]\nCMHA text")
    assert "[2]\nAbout text" in context
    assert "cmha.txt" not in context  # the model only ever sees integer IDs
    assert source_map == {1: {"source": "cmha.txt", "chunk_id": 3},
                          2: {"source": "about.txt", "chunk_id": 0}}


def test_answer_question_returns_only_real_sources(monkeypatch):
    retrieved = [({"source": "cmha.txt", "chunk_id": 3, "text": "CMHA text"}, 0.5)]
    monkeypatch.setattr(rag, "retrieve_hybrid", lambda q, candidate_k, final_k: retrieved)
    monkeypatch.setattr(
        rag, "generate_structured",
        lambda prompt: {"answer": "Bach built booking workflows.", "used_context": [1, 7]},
    )
    response = rag.answer_question("What did Bach do at CMHA?")
    assert response == {
        "answer": "Bach built booking workflows.",
        "sources": [{"source": "cmha.txt", "chunk_id": 3}],
    }
