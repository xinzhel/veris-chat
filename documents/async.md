# Async/Sync Concurrency Q&A for RAG Pipeline

## Q1: How do sync retrieval and sync generation handle concurrent requests?

Each request occupies 1 thread for the entire retrieval→generation pipeline. A 2nd request runs in parallel on another thread if you explicitly use threading (`ThreadPoolExecutor`) or if a framework like FastAPI handles it for you.

| Mode | Threads Used | Behavior |
|------|--------------|----------|
| Sync retrieval + Sync generation | 1 thread per request | Thread blocked during all I/O waits |

## Q2: What's the difference between sync and async?

- **Sync**: Thread blocks while waiting for I/O (network call to Qdrant, Bedrock API)
- **Async**: Thread releases during I/O wait, event loop can handle other tasks

## Q3: How do different combinations compare for 2 concurrent requests?

| Mode | Threads Used | Concurrency |
|------|--------------|-------------|
| Sync retrieval + Sync generation | 2 threads | Parallel but wasteful - threads idle during I/O |
| Sync retrieval + Async generation | 1-2 threads | Partial benefit - generation can interleave |
| Async retrieval + Async generation | 1 thread handles many | Full concurrency - event loop multiplexes |

## Q4: Can sync code run in parallel across threads without async?

Yes, using `concurrent.futures.ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor

def sync_chat(session_id, message):
    nodes = retriever.retrieve(message)      # blocks ~100ms
    response = llm.complete(prompt)          # blocks ~2-5s
    return response

with ThreadPoolExecutor(max_workers=4) as executor:
    future_a = executor.submit(sync_chat, "session_a", "What is X?")
    future_b = executor.submit(sync_chat, "session_b", "What is Y?")
    result_a = future_a.result()
    result_b = future_b.result()
```

## Q5: Does plain Python run multi-threaded by default?

No. Without `threading`, `concurrent.futures`, or `asyncio`, Python runs single-threaded sequentially:

```python
result_a = sync_chat("session_a", "What is X?")  # waits 3s
result_b = sync_chat("session_b", "What is Y?")  # waits 3s
# Total: 6s (sequential)
```

FastAPI/Uvicorn handles this automatically - sync endpoints run in thread pool, async endpoints run on event loop.

## Q6: What is GIL?

**GIL = Global Interpreter Lock** - A mutex in CPython that allows only one thread to execute Python bytecode at a time.

- **CPU-bound tasks**: Threads don't speed up (use `multiprocessing`)
- **I/O-bound tasks**: Threads work fine because GIL releases during I/O wait

For RAG apps (Qdrant queries, Bedrock API calls), all operations are I/O-bound, so GIL isn't a bottleneck.

## Q7: Which approach for FastAPI streaming endpoint?

Use **async retrieval + async generation** for best scalability:

```python
# Async endpoint - runs on event loop, scales to thousands of connections
@app.post("/chat/stream/")
async def stream_endpoint(request: ChatRequest):
    async for chunk in async_chat(...):
        yield chunk
```

## Q8: Does LlamaIndex support async retrieval?

Yes. `VectorIndexRetriever` has `aretrieve()` for async retrieval with async embedding. `CitationQueryEngine` has `_aquery()` which uses `aretrieve()` + `asynthesize()`.

## Q9: Does Bedrock LLM support async streaming?

- `llama_index.llms.bedrock.Bedrock`: `astream_chat()` raises `NotImplementedError` ❌
- `llama_index.llms.bedrock_converse.BedrockConverse`: Native `astream_chat()` ✅ (uses `aioboto3`)
