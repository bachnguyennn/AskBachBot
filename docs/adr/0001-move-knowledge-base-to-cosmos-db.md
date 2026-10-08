# ADR 0001: Move the knowledge base to Azure Cosmos DB for NoSQL

- **Status:** Accepted, migration in progress (phase 1 of 3 done, see [Migration status](#migration-status))
- **Date:** 2026-10-07
- **Deciders:** Bach Nguyen
- **Relates to:** README sections [Data pipeline](../../README.md#data-pipeline) and [Design decisions](../../README.md#design-decisions)

## Context

Until this change, the six profile documents in `app/data/documents/*.txt` were **baked into the Docker image** and read from disk at startup. Both indexes (dense MiniLM and BM25) were rebuilt in memory on every container start. The README's earlier position was that a vector database "would add a service and a sync problem without improving quality" at 33 chunks.

That still holds for retrieval *quality*. The reasons to move anyway are operational:

1. **Content updates require a full redeploy.** Fixing one sentence in `cmha.txt` meant rebuilding a multi-GB image, pushing it to ACR, and rolling a new Container Apps revision.
2. **Cold start embeds the whole corpus.** With scale-to-zero, every cold start loads the model *and* re-encodes all 33 chunks before serving the first request.
3. **Learning goal.** The project exists to build each layer of a production RAG system. A managed store with native vector search is the next layer.

Constraints:

- **Cost:** must fit the Azure for Students subscription. Cosmos DB's free tier (first 1,000 RU/s and 25 GB per account) covers this.
- **Auth:** no keys in code or in the image. The existing deployment already uses a managed identity for ACR, so the same pattern should apply to data.
- **Evaluation:** `(source, chunk_id)` is the stable chunk identity used by the eval labels in `src/eval_retrieval.py`. The migration must not renumber chunks.

## Decision

We store the knowledge base in **Azure Cosmos DB for NoSQL**, with its **built-in vector search**. Cosmos DB becomes the source of truth at runtime, and the `.txt` files become the authoring format that gets uploaded to it.

### Resources

| Resource | Value | Notes |
|---|---|---|
| Account | `bach-rag-cosmos-central` (resource group `ask-bach-rg`) | Region **Central US**, free tier on, capability `EnableNoSQLVectorSearch`, Session consistency |
| Database | `ask-bach` | |
| Container `content` | partition key `/type`, 400 RU/s | Holds whole documents (`type = "document"`) |
| Container `chunks` | partition key `/source`, 400 RU/s, vector policy below | Holds pre-chunked text plus its embedding |

The endpoint is set in `app/src/database.py`. The vector and indexing policies for `chunks` are checked in as [`app/vector-policy.json`](../../app/vector-policy.json) and [`app/index-policy.json`](../../app/index-policy.json):

- `/embedding`: `float32`, **384** dimensions (matches `all-MiniLM-L6-v2`), **cosine** distance.
- Vector index type **`flat`**, which is exact brute-force search, the same behaviour as the in-memory index. It is the right choice at 33 vectors. Revisit `quantizedFlat` or `diskANN` only at thousands of chunks.
- `/embedding/*` is **excluded** from the regular range index. Range-indexing 384 floats per item wastes RUs and storage, and vectors are only ever queried through `VectorDistance`.

### Item shapes

`content` container (one item per `.txt` file, written by `scripts/migrate_documents.py`):

```json
{
  "id": "cmha",
  "type": "document",
  "source": "cmha.txt",
  "text": "IT INTERN — CANADIAN MENTAL HEALTH ASSOCIATION (CMHA DURHAM)\n=====\n..."
}
```

`chunks` container (one item per chunk, written by `scripts/index_chunks.py`):

```json
{
  "id": "cmha-3",
  "source": "cmha.txt",
  "chunk_id": 3,
  "text": "IT INTERN — CANADIAN MENTAL HEALTH ...\n\n- Where did Bach intern? ...",
  "embedding": [0.0123, -0.0456, "... 384 floats"]
}
```

`id` is deterministic (`<file stem>` or `<file stem>-<chunk_id>`), so both scripts are **idempotent**: re-running them upserts over the same items instead of creating duplicates.

### Rules the rest of the system must follow

- **Chunking still happens in Python** (`chunk_document`), not in the database. `load_documents()` sorts by `source` after querying, because Cosmos DB does not guarantee result order. Without that sort, chunk numbering and the eval labels would drift.
- **`chunks` must be re-indexed whenever `content` changes.** The two containers are not linked. Editing a document and running only `migrate_documents.py` leaves stale vectors in `chunks`.
- **Authenticate with Microsoft Entra ID (`DefaultAzureCredential`), never account keys.** Locally this resolves to your `az login` session. In Azure it must resolve to the Container App's managed identity.

## Consequences

**Gains**

- Document text can change without rebuilding the image (once phase 3 lands, see below).
- Vectors can be computed once, offline, so the corpus is no longer re-embedded on every cold start.
- Entra ID auth with no secrets in code. The account uses the built-in **Cosmos DB Built-in Data Contributor** data-plane role.

**Costs and risks**

- **A new runtime dependency.** If Cosmos DB is unreachable or auth fails, the app can't build its indexes and won't start. Previously startup had no external dependencies apart from the Hugging Face model download.
- **Cross-region hop.** The account is in **Central US** while the Container App runs in **Canada Central**. That's fine for a few startup queries, but per-request vector queries would add cross-region latency to every `/ask`. Consider co-locating before phase 2.
- **Two copies of the content.** The `.txt` files remain in the repo (and the Dockerfile still copies `app/data/`). Which copy is authoritative depends on the workflow, and that has to be stated clearly (see [Updating content](#updating-content)).
- **RU budget.** The two containers use 400 RU/s each (800 of the 1,000 free RU/s). A third container at 400 RU/s would go over the free tier. Use database-level shared throughput if more containers are added.
- `load_chunks()` is still called once per retriever, so startup runs the `content` query twice. That's harmless, but it now costs RUs.

## Alternatives considered

| Option | Why not |
|---|---|
| **Keep files in the image** (status quo) | Correct and simple, but every content edit is a full image rebuild and redeploy, and every cold start re-embeds the whole corpus. |
| **Azure AI Search** | Purpose-built hybrid (BM25 + vector) search with RRF, which would replace both retrievers. But the free tier is small and limited to one per subscription, and it would replace the hand-built retrieval this project exists to study and evaluate. |
| **Azure Blob Storage for the `.txt` files** | Fixes "redeploy to edit content" but has no vector search, so cold-start embedding stays. |
| **PostgreSQL + pgvector** (Azure Flexible Server) | Solid choice, but an always-on server costs more than Cosmos DB's free tier and needs schema management. |
| **Dedicated vector DB** (Pinecone, Qdrant, etc.) | Another vendor and another secret, outside the Azure identity model used everywhere else. |

## Migration status

As of 2026-10-08 (verified against the live account and code):

| Phase | Scope | Status |
|---|---|---|
| **1. Documents in Cosmos DB** | `migrate_documents.py` uploads documents; `load_documents()` reads them from the `content` container | ✅ Done locally. 6 documents, which chunk into the same **33 chunks** as before, so eval labels remain valid. |
| **2. Vectors in Cosmos DB** | `index_chunks.py` stores embeddings; `test_vector_search.py` checks `VectorDistance` queries | ✅ Done locally (2026-10-08). `retrieve_hybrid` gets dense candidates from `src/cosmos_retriever.py`. The [parity check](../../README.md#cosmos-db-dense-retrieval-parity-check) shows the same top 3 and scores as in-memory search on all 10 benchmark questions. |
| **3. Production deployment** | The Container App reads from Cosmos DB | ❌ Not deployed. Production still runs `ask-bach-api:v4` (file-based). The blockers are listed below. |

### Before deploying to Container Apps

Each of these will stop the current code from starting in Azure:

1. **Add the SDKs to `app/requirements.txt`.** `azure-cosmos` and `azure-identity` are installed locally (4.17.1 / 1.26.0) but are **not** in `requirements.txt`, so the image will fail with `ModuleNotFoundError: azure`.
2. **Grant the managed identity data-plane access.** Right now only your user account has a Cosmos DB role assignment. `ask-bach-identity` (used today only for `AcrPull`) needs one too:

   ```bash
   RG=ask-bach-rg; ACCOUNT=bach-rag-cosmos-central
   PRINCIPAL_ID=$(az identity show -g $RG -n ask-bach-identity --query principalId -o tsv)
   az cosmosdb sql role assignment create -g $RG -a $ACCOUNT \
     --role-definition-id 00000000-0000-0000-0000-000000000002 \
     --principal-id $PRINCIPAL_ID --scope "/"
   ```

   You could also create a custom read-only role, since the API only reads data.
3. **Tell `DefaultAzureCredential` which identity to use.** With a *user-assigned* identity it needs the client ID:

   ```bash
   CLIENT_ID=$(az identity show -g $RG -n ask-bach-identity --query clientId -o tsv)
   az containerapp update -g $RG -n ask-bach-api --set-env-vars AZURE_CLIENT_ID=$CLIENT_ID
   ```

Recommended clean-up (not blocking):

- Move `COSMOS_ENDPOINT` from a hard-coded constant to an environment variable, as `GROQ_API_KEY` already is.
- Remove the now-unused `DOCS_DIR` from `src/load_documents.py`.
- Delete the leftover `type = "test"` item in `content`. The container has 7 items: 6 documents plus 1 test item.

## Updating content

Until a CI job automates this, `.txt` files remain the authoring format and Cosmos DB is the runtime copy. From `app/`, signed in with `az login`:

```bash
python -m scripts.migrate_documents   # upsert documents → content
python -m scripts.index_chunks        # re-chunk, re-embed, upsert → chunks
python -m scripts.test_vector_search  # sanity check: top 3 for "What did Bach do at CMHA?"
python -m src.eval_retrieval          # confirm chunk IDs / labels still line up
```

`index_chunks.py` reads chunks **from Cosmos DB** (through `load_chunks()`), not from disk, so it must run *after* `migrate_documents.py`.

> ⚠️ **Deleting or renaming a document is not propagated.** Both scripts only upsert. Items for removed files, or for chunks that disappear when a document gets shorter, stay in Cosmos DB until you delete them by hand. Because `load_documents()` returns every `type = "document"` item, a deleted file will keep being served. Orphaned `chunks` items don't affect the API yet, but they will once phase 2 queries `chunks` directly.

`app/test_cosmos.py` is a one-off connectivity check (it writes, reads and deletes a `type = "test"` item). Run it from `app/` with `python test_cosmos.py` when credentials or networking are in doubt.
