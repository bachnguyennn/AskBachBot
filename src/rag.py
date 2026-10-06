import json

from src.hybrid_retriever import retrieve_hybrid
from src.generate import generate_structured


def build_context_and_source_map(results):
    blocks = []
    source_map = {}
    for evidence_id, (chunk, _score) in enumerate(results, start=1):
        blocks.append(f"[{evidence_id}]\n{chunk['text']}")
        source_map[evidence_id] = {"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
    return "\n\n---\n\n".join(blocks), source_map


def build_structured_prompt(question, context):
    return (
        "Answer using only the provided context.\n\n"
        "If the context does not contain enough information, say that you "
        "don't have enough information to answer.\n\n"
        "Return the evidence IDs you actually used in used_context.\n\n"
        "Do not include citation markers in the answer itself.\n\n"
        f"Context:\n\n{context}\n\n"
        f"Question:\n{question}"
    )


def resolve_sources(used_context, source_map):
    # invalid ID -> ignored, duplicate -> included once, empty -> []
    sources = []
    seen = set()
    for evidence_id in used_context:
        if evidence_id in source_map and evidence_id not in seen:
            seen.add(evidence_id)
            sources.append(source_map[evidence_id])
    return sources


def answer_question(question, candidate_k=10, final_k=3, verbose=False):
    results = retrieve_hybrid(question, candidate_k=candidate_k, final_k=final_k)
    context, source_map = build_context_and_source_map(results)
    data = generate_structured(build_structured_prompt(question, context))
    sources = resolve_sources(data["used_context"], source_map)
    response = {"answer": data["answer"], "sources": sources}

    if verbose:
        print(f"\n===== {question}")
        print("\nRAW MODEL DATA:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("\nSOURCE MAP:")
        print(json.dumps(source_map, indent=2))
        print("\nRESOLVED SOURCES:")
        print(json.dumps(sources, indent=2))
        print("\nFINAL RESPONSE:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    return response


if __name__ == "__main__":
    source_map = {
        1: {"source": "cmha.txt", "chunk_id": 3},
        2: {"source": "cmha.txt", "chunk_id": 2},
        3: {"source": "about.txt", "chunk_id": 0},
    }
    assert resolve_sources([1, 3], source_map) == [source_map[1], source_map[3]]
    assert resolve_sources([1, 999], source_map) == [source_map[1]]
    assert resolve_sources([1, 1], source_map) == [source_map[1]]
    assert resolve_sources([], source_map) == []
    print("resolve_sources checks passed")

    for question in ["What did Bach do at CMHA?", "What position did Bach have at Google?"]:
        answer_question(question, verbose=True)
