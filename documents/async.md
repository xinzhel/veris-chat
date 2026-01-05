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


---

## LlamaIndex + Bedrock Async Implementation Q&A

## Q10: Why use `BedrockConverse` instead of `Bedrock`?

`llama_index.llms.bedrock.Bedrock` is deprecated. LlamaIndex recommends `BedrockConverse`:

```python
# Old (deprecated) - astream_chat raises NotImplementedError
from llama_index.llms.bedrock import Bedrock
llm = Bedrock(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

# New (recommended) - full async support
from llama_index.llms.bedrock_converse import BedrockConverse
llm = BedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
```

`BedrockConverse` uses AWS Converse API and `aioboto3` for native async.

## Q11: How to use `BedrockConverse.astream_chat()`?

```python
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.llms import ChatMessage

llm = BedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Explain RAG in one sentence"),
]

async def stream_response():
    stream = await llm.astream_chat(messages)
    async for chunk in stream:
        print(chunk.delta, end="", flush=True)

import asyncio
asyncio.run(stream_response())
```

## Q12: How does `CitationQueryEngine` support async?

`CitationQueryEngine` has built-in async methods:

```python
# Sync query
response = engine.query("What is the site status?")

# Async query (uses aretrieve + asynthesize internally)
response = await engine.aquery("What is the site status?")
```

Internally, `_aquery()` calls:
1. `aretrieve()` - async retrieval from Qdrant
2. `asynthesize()` - async LLM generation via response synthesizer

## Q13: How to combine async retrieval with streaming generation?

For streaming, you can't use `CitationQueryEngine` directly (it returns complete response). Instead:

```python
async def async_chat_stream(session_id: str, message: str):
    # 1. Async retrieval
    nodes = await retriever.aretrieve(message)
    
    # 2. Build prompt with retrieved context
    context = "\n".join([n.text for n in nodes])
    messages = [
        ChatMessage(role="system", content=f"Context:\n{context}"),
        ChatMessage(role="user", content=message),
    ]
    
    # 3. Stream generation
    async for chunk in llm.astream_chat(messages):
        yield {"type": "token", "content": chunk.delta}
    
    # 4. Return citations at end
    yield {"type": "done", "citations": extract_citations(nodes)}
```

## Q14: Does Qdrant support async queries?

Yes, via `qdrant_client.AsyncQdrantClient`:

```python
from qdrant_client import AsyncQdrantClient

async_client = AsyncQdrantClient(path="./qdrant_local")
results = await async_client.search(
    collection_name="veris_pdfs",
    query_vector=embedding,
    limit=5,
)
```

LlamaIndex's `QdrantVectorStore` uses this internally when you call `aretrieve()`.

## Q15: What's the recommended async pattern for FastAPI streaming?

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/chat/stream/")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in async_chat_stream(request.session_id, request.message):
            yield f"data: {json.dumps(chunk)}\n\n"  # SSE format
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Q16: Can I mix sync and async in the same function?

Yes, using `asyncio.to_thread()` for sync operations:

```python
import asyncio

async def hybrid_chat(session_id: str, message: str):
    # Run sync ingestion in thread pool
    await asyncio.to_thread(ingestion_client.store, url, session_id)
    
    # Async retrieval
    nodes = await retriever.aretrieve(message)
    
    # Async streaming generation
    async for chunk in llm.astream_chat(messages):
        yield chunk
```

This is useful when some components (like `IngestionClient`) don't have async versions.


---

## Python Async Fundamentals Q&A

## Q17: What does `async def` do?

`async def` declares a coroutine function. It doesn't run immediately when called - it returns a coroutine object:

```python
# Regular function - runs immediately
def regular_func():
    return "hello"
result = regular_func()  # Returns "hello"

# Async function - returns coroutine object, doesn't run yet
async def async_func():
    return "hello"
result = async_func()  # Returns <coroutine object>, NOT "hello"!
```

## Q18: What does `await` do?

`await` actually runs the coroutine. It does two things:
1. Runs the coroutine until it completes (or hits I/O)
2. Yields control back to event loop during I/O waits

```python
async def main():
    result = await async_func()  # NOW it runs, result = "hello"
```

## Q19: Why does `await` yield control?

When you `await` an I/O operation:
- Thread says "I'm waiting for network, event loop can do other work"
- Event loop runs other coroutines while waiting
- When I/O completes, event loop resumes this coroutine

```python
# Sync - blocks thread during network wait
response = requests.get(url)  # Thread stuck here

# Async - releases thread during network wait
response = await aiohttp.get(url)  # Thread free to do other work
```

## Q20: What is async really about?

Async is about **any I/O wait**, whether:
- Waiting to send a request
- Waiting to receive a response
- Waiting for each chunk of a streaming response
- Waiting for database query
- Waiting for file read

All are I/O waits where async lets the thread do other work instead of blocking.

## Q21: For `await` to work, must the function being awaited be async?

Yes. The entire I/O chain must be async:

```python
# Works - aiohttp.get is async
response = await aiohttp.get(url)

# Error - requests.get is sync, can't await it
response = await requests.get(url)  # TypeError!
```

That's why:
- `requests` (sync) → use in sync code or threads
- `aiohttp` (async) → use with `await`
- `boto3` (sync) → use in sync code
- `aioboto3` (async) → use with `await` (what BedrockConverse uses)

## Q22: Why use `async for` instead of regular `for`?

Both iterate over streams, but differ in how they wait for chunks:

```python
# Sync streaming - thread blocks while waiting for each chunk
for chunk in sync_stream:
    print(chunk)  # Between chunks, thread is BLOCKED

# Async streaming - thread released while waiting for each chunk
async for chunk in async_stream:
    print(chunk)  # Between chunks, thread is FREE for other tasks
```

## Q23: Does `async for` require `await`?

No explicit `await` needed - it's built into `async for` as syntactic sugar:

```python
# What you write:
async for chunk in stream:
    print(chunk)

# What Python actually does:
iterator = stream.__aiter__()
while True:
    try:
        chunk = await iterator.__anext__()  # <-- await is hidden here!
        print(chunk)
    except StopAsyncIteration:
        break
```

## Q24: Summary of async keywords

| Syntax | Hidden await? | What it does |
|--------|---------------|--------------|
| `await func()` | Explicit | Run coroutine |
| `async for x in gen` | Yes, in `__anext__()` | Iterate async generator |
| `async with ctx` | Yes, in `__aenter__()/__aexit__()` | Async context manager |

All three are ways to "await" - just different syntax for different patterns.

## Q25: Where can you use these keywords?

| Keyword | What it does | Where you can use it |
|---------|--------------|---------------------|
| `async def` | Declares coroutine function | Anywhere |
| `await` | Runs a coroutine | Only inside `async def` |
| `async for` | Iterates async generator | Only inside `async def` |
| `asyncio.run()` | Starts event loop, runs coroutine | Regular (sync) code |

## Q26: Why did `async for chunk in llm.astream_chat(messages)` fail?

`astream_chat()` is an `async def` function that **returns** an async generator. Two async operations:

1. Calling the function → returns coroutine (needs `await`)
2. Iterating the result → async generator (needs `async for`)

```python
# Wrong - astream_chat() returns coroutine, not generator directly
async for chunk in llm.astream_chat(messages):  # Error!

# Correct - await first, then iterate
stream = await llm.astream_chat(messages)  # Step 1: get the generator
async for chunk in stream:                  # Step 2: iterate it
    print(chunk)
```

## Q27: What is a coroutine?

A **coroutine = a pausable function**. Unlike regular functions that run start-to-finish, coroutines can pause at `await` points and release control.

```python
# Regular function - runs start to finish, can't pause
def regular():
    print("start")
    result = expensive_operation()  # blocks here
    print("end")
    return result

# Coroutine - can pause at await points
async def coroutine():
    print("start")
    result = await expensive_operation()  # pauses here, releases control
    print("end")
    return result
```

**Key difference:**
- **Function**: Calling it runs it immediately
- **Coroutine**: Calling it returns a coroutine object (doesn't run yet)

```python
async def my_coroutine():
    return "hello"

# This doesn't run the code!
coro = my_coroutine()  # <coroutine object my_coroutine at 0x...>

# You must explicitly run it:
result = await coro  # NOW it runs
```

## Q28: What are the 3 ways to run a coroutine?

### 1. `asyncio.run()` - Start from sync code (entry point)

```python
async def main():
    return "hello"

# From regular Python code:
result = asyncio.run(main())  # Starts event loop, runs coroutine
```

### 2. `await` - From inside another coroutine

```python
async def fetch_data():
    return "data"

async def main():
    result = await fetch_data()  # Run coroutine, wait for result
    print(result)

asyncio.run(main())
```

### 3. `asyncio.gather()` - Run multiple coroutines concurrently

```python
async def fetch(url):
    return f"data from {url}"

async def main():
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
    )
    # All 3 run concurrently, results = ["data from url1", ...]

asyncio.run(main())
```

## Q29: Summary - Coroutine terminology

| Term | What it is |
|------|------------|
| `async def` | Defines a coroutine function |
| Coroutine | The pausable function object created when you call `async def` |
| `await` | Runs a coroutine and waits for result |
| `asyncio.run()` | Entry point to start event loop from sync code |

## Q30: Visual of async event loop handling multiple users

```
Event Loop (single thread):

User A: [send req]--[wait chunk 1]--[wait chunk 2]--[wait chunk 3]--
User B: ----[send req]--[wait chunk 1]--[wait chunk 2]--[wait chunk 3]--

        ↑ Every "wait" is an opportunity for the thread to do other work
```

With async, a single thread can handle many concurrent connections by interleaving work during I/O waits.
