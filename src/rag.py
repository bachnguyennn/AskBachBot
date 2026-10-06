from hybrid_retriever import retrieve_hybrid


def build_context_and_source_map(results):
    blocks = []
    source_map = {}
    for evidence_id, (chunk, _score) in enumerate(results, start=1):
        blocks.append(f"[{evidence_id}]\n{chunk['text']}")
        source_map[evidence_id] = {"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
    return "\n\n---\n\n".join(blocks), source_map


if __name__ == "__main__":
    question = "What did Bach do at CMHA?"
    results = retrieve_hybrid(question, candidate_k=10, final_k=3)

    context, source_map = build_context_and_source_map(results)

    print(context)
    print()
    print(source_map)
