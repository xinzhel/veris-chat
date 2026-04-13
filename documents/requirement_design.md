# ✅ **Document-Grounded Conversational System**

We are developing an interactive system where, for each **conversation session**, the user supplies a **small set of task-relevant document URLs**. These documents are:

1. downloaded and parsed on-demand,
2. chunked and embedded on-the-fly,
3. stored in a **shared vector store** (indexed by URL, deduplicated across sessions), and
4. queried with **session-scoped retrieval** (each session only sees its own documents) + **conversation memory + citations**.

If no document URLs are provided, the system still responds using conversation memory and general LLM knowledge (no citations).

* Request: Each request includes `session_id`. `document_urls` is optional.
    ```json
    {
    "session_id": "a157",
    "document_urls": ["https://...pdf", "https://...pdf"],
    "message": "..."
    }
    ```
* The system maintains:
    * **Session index** (`session_id → Set[url]`) — tracks which URLs belong to each session
    * **URL cache** (`url → metadata`) — tracks ingestion state, avoids re-ingestion
    * **Session memory** (chat history / mem0-style, per-session Qdrant collection)
    * **Shared vector store** (documents indexed by URL, shared across sessions)

* Structure of the repository:
    ```
    /documents/
        requirement_design.md (this file)
        TODO.md
        async.md (sync vs async concurrency Q&A)
        bedrock_caveats.md

    /rag_core/ 
        chat/
            config.py       (config loader)
            retriever.py    (retrieval + memory + citation formatting)
            service.py      (chat orchestration: sync chat() + async async_chat())
        ingestion/
            main_client.py  (IngestionClient: download, parse, chunk, embed, store)
        utils/
            citation_query_engine.py  (customized CitationQueryEngine + NoOpRetriever)
            memory.py       (customized Mem0Memory)
            logger.py       (logging + timing utilities)

    /script/
        (test scripts, runnable in interactive notebook mode)

    /app/
        chat_api.py  (FastAPI endpoints)

    /deploy/
        ec2_launch.sh   (EC2 launch script)
        user_data.sh    (EC2 user-data setup)
        push_clean.sh   (git push helper)

    config.yaml
    environment.yaml
    ```

---

Below are the Core Capabilities.

## **1 Session-Attached, Multi-PDF Document Ingestion**

If `document_urls` are provided for a session:

* download PDFs (with local cache)
* parse into per-page text + page_number
* chunk into retrievable chunks
* embed chunks
* store vectors into Qdrant (indexed by URL, NOT session_id)

### **1.1 Session/URL Architecture (Separation of Concerns)**

```
Session Index:  session_id → Set[url]     (O(1) lookup)
URL Cache:      url → {local_path, ingestion_time, ingested_at}
Qdrant:         chunks indexed by url field (shared across sessions)
```

**Ingestion flow:**
```python
def store(url, session_id):
    session_index[session_id].add(url)     # Always add to session
    if url not in url_cache:               # Only ingest if never processed
        download → parse → chunk → embed → store to Qdrant
        url_cache[url] = metadata
```

**Benefits:**
- Same URL in 2 sessions → 1x Qdrant storage (shared chunks)
- Per-session deletion: remove from session_index only
- Clean separation: ingestion knows nothing about sessions

### **1.2 Qdrant Payload Metadata per Chunk**
* `url`
* `filename`
* `page_number`
* `chunk_id`
* `chunk_index`
* `section_header` (optional)
* `text`

Note: `session_id` is NOT stored in Qdrant payload. Session-URL mapping is tracked in `session_index.json`.

### **1.3 IngestionClient API**

```python
from veris_chat.ingestion.main_client import IngestionClient

client = IngestionClient(
    collection_name="veris_pdfs",
    embedding_model="cohere.embed-english-v3",
    embedding_dim=1024,
    chunk_size=512,
    chunk_overlap=50,
)
client.store(url, session_id="a157")

# Session index queries
urls = client.get_session_urls("a157")  # Set of URLs for session
client.reset_collection()               # Full reset (Qdrant + caches)
```

Note: The ingestion code is independent from the rest of the system. The rest is implemented directly under LlamaIndex, including:
* AWS Bedrock LLMs/embeddings
* Qdrant DB (via LlamaIndex's QdrantVectorStore)
* Conversation memory (source code copied into `utils/memory.py` for customization)
* Citation-grounded generation (source code copied into `utils/citation_query_engine.py` for customization)

---

## **2 Retrieval-Augmented Generation**

### **2.1 Retrieval**

Retrieval uses URL-based filtering via `MatchAny`:

```python
# Get URLs for session from session_index
urls = ingestion_client.get_session_urls(session_id)

# Build Qdrant filter
filter = MatchAny(key="url", any=list(urls))

# Retrieve via LlamaIndex
retriever = index.as_retriever(
    similarity_top_k=top_k,
    vector_store_kwargs={"qdrant_filters": filter},
)
```

**No-document sessions:** When `session_urls` is empty, a `NoOpRetriever` is used instead. The LLM can still respond using general knowledge and conversation memory, but without document citations.

### **2.2 LLM Generation**

Inject:
* conversation memory context (session-scoped): see Section 3
* retrieved chunks (session-scoped via URL filter)
* user query

Requiring **Citation-Grounded Generation**: Responses must include citations referencing the source documents.
* Citations use inline markdown links: `[filename (p.X)](url)`
* This format is clickable in frontend markdown renderers
* The system exposes the full set of source nodes used for each answer

### **2.3 AWS Bedrock Integration under LlamaIndex**

We connect to Bedrock supporting both `aws sso login` and Access Keys. If env vars are empty (set in `.env`), rely on SSO/Instance Profile.

#### **2.3.1 Embedding model via AWS Bedrock**

```python
from llama_index.embeddings.bedrock import BedrockEmbedding
embed_model = BedrockEmbedding(model_name="cohere.embed-english-v3")
```

#### **2.3.2 LLM Generation**

**Sync Generation** (using `Bedrock` class for `chat()` endpoint):

```python
from llama_index.llms.bedrock import Bedrock
llm = Bedrock(model="us.anthropic.claude-opus-4-5-20251101-v1:0", context_size=200000)
```

**Async Streaming** (using `BedrockConverse` for `/chat/stream/` endpoint):

```python
from llama_index.llms.bedrock_converse import BedrockConverse
llm = BedrockConverse(model="us.anthropic.claude-opus-4-5-20251101-v1:0")

stream = await llm.astream_chat(messages)
async for chunk in stream:
    print(chunk.delta, end="")
```

**Model-specific caveats:**
- `llama_index.llms.bedrock.Bedrock`: `astream_chat()` raises `NotImplementedError`. Use for sync only.
- `llama_index.llms.bedrock_converse.BedrockConverse`: Full async support via `aioboto3`.
- Cross-region inference profiles (`us.` prefix): Required for RMIT SCP. See `documents/bedrock_caveats.md`.
- Non-foundation models (e.g., Opus 4.5): Require `context_size=200000` for old `Bedrock` class.
- See `documents/async.md` for sync vs async concurrency patterns.

#### **2.3.3 Citation-Grounded Generation**

```python
engine = CitationQueryEngine(
    retriever=retriever,
    llm=llm,
    citation_chunk_size=512,
)
response = engine.query("What is the site status?")
```

For async streaming, `CitationQueryEngine.prepare_streaming_context()` replicates the retrieval + context packing workflow without the LLM call, then `BedrockConverse.astream_chat()` streams the generation.

---

## **3. Conversation Memory**

Uses Mem0 integration with **dual isolation architecture**:

### **Layer 1: Qdrant Collection** (Physical isolation)
Each session gets its own collection: `mem0_memory_{session_id}`

### **Layer 2: Mem0Context** (Logical filtering)
`context = {"user_id": session_id}` passed to Mem0Memory for filtering within collection.

### Behavior:
* Memory store is session-keyed
* Memory is persisted via Qdrant (cloud or local)
* Memory content is injected into the LLM prompt for each new query:
  ```
  prompt = memory_context + retrieved_chunks + user_query
  ```

### Known Limitation:
`CitationQueryEngine.query()` only accepts a query string, not a chat message list. Memory context is prepended to the query text, but full chat history structure is lost. See `documents/TODO.md` for options.

---

## **4. FastAPI**

### Request format:
```json
{
  "message": "Is the site a priority site?",
  "session_id": "a157",
  "document_urls": ["https://.../doc1.pdf"]
}
```

### Endpoints:
- `POST /chat/` — Sync endpoint, OpenAI-compatible response format
- `POST /chat/stream/` — Async streaming, OpenAI-compatible SSE format
- `GET /health` — Health check

### Response format (`/chat/`):
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "citations": ["[doc.pdf (p.2)](https://...)"],
  "sources": [{"file": "doc.pdf", "page": 2, "chunk_id": "c_1", "url": "https://..."}],
  "timing": {"ingestion": 1.2, "retrieval": 0.3, "generation": 2.1, "memory": 0.1, "total": 3.7}
}
```

### Streaming format (`/chat/stream/`):
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"The site"}}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" is classified"}}]}
...
data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop"}],"citations":[...],"sources":[...]}
data: [DONE]
```

See `documents/async.md` for details on sync vs async concurrency patterns.

---

## **5. Deployment**

EC2 instance in `ap-southeast-2` with:
- Instance Profile (`rmit-workload-veris`) for Bedrock access
- Elastic IP for stable endpoint
- systemd service for auto-restart
- Qdrant Cloud for vector storage

Launch: `bash deploy/ec2_launch.sh`

See `.kiro/specs/aws-deployment/tasks.md` for deployment details.


---

## **Q&A**

### Q1: What does "shared vector store indexed by URL" mean vs "session-scoped ephemeral vector memory"?

The original design stored `session_id` in each Qdrant chunk payload and filtered by `session_id` at retrieval time. Same PDF in 2 sessions = stored twice.

The current implementation stores chunks indexed by `url` only (no `session_id` in Qdrant). A separate `session_index` maps `session_id → Set[url]`. Retrieval uses `MatchAny(key="url", any=session_urls)`. Same PDF in 2 sessions = stored once, shared.

Retrieval is still session-scoped (each session only sees its own documents), but storage is shared and deduplicated.

### Q2: What is `citation_chunk_size`?

After retrieval, each retrieved chunk gets further split into smaller "citation chunks" by `CitationQueryEngine._create_citation_nodes()`. Each sub-chunk gets its own `Source: [filename (p.X)](url)` label.

Smaller `citation_chunk_size` = more precise citations (LLM can point to specific sub-chunk). Larger = fewer sources in prompt, less token usage.

With both ingestion `chunk_size` and `citation_chunk_size` at 512, retrieved chunks are typically not split further.

### Q3: What is lost when memory context is prepended as text?

`memory.get()` returns a full message list: `[SystemMessage(Mem0 facts), USER(turn1), ASSISTANT(turn1), USER(turn2)]`. But `service.py` only extracts the system message (Mem0 extracted facts) and discards the chat history turns:

```python
messages_with_context = memory.get(input=message)
memory_context = messages_with_context[0].content  # Only Mem0 facts
query_text = f"Context:\n{memory_context}\n\nQuestion: {message}"
engine.query(query_text)  # Chat history discarded here
```

The chat history IS available from `memory.get()`, but `CitationQueryEngine.query()` only accepts a string, so it can't be passed through. This means:
- **Turn-by-turn conversation flow** — The LLM doesn't see the back-and-forth. "Can you elaborate?" has no reference to what it previously said. It can't elaborate on its own prior answer because it never sees it.
- **Role distinction** — Everything becomes one flat string. The LLM can't distinguish between what the user said vs what it previously answered. This matters for instructions like "repeat what you said earlier."
- **Multi-turn coreference** — "What about the second document?" — the LLM doesn't know which documents were discussed in prior turns because it only sees Mem0's extracted facts, not the actual conversation.

Mem0's extracted facts (e.g., "User's name is Alice") survive. Single-turn document Q&A works fine. The limitation is in the `service.py` ↔ `CitationQueryEngine` integration, not in `memory.get()` itself.
