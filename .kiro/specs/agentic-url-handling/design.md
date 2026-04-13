# Agentic URL Handling

## Problem

When a user says "summarize this PDF: https://xxx" or "what's in https://xxx", the current system:
1. Ignores URLs in the user message (only uses KG-resolved or request-supplied `document_urls`)
2. Uses top-K chunk retrieval, which doesn't work for "summarize the whole document"

## Current Architecture (Pipeline)

```
User message → Ingest KG URLs → Retrieve top-K chunks → Generate with citations
```

Fixed pipeline: every message goes through the same steps regardless of intent.

## Proposed Architecture (ReAct Agent)

```
User message → LLM Agent decides → Tool calls → Response
```

The LLM (Opus 4.6 with strong tool use) decides what to do based on the user's intent:

| User Intent | Agent Action |
|-------------|-------------|
| "Is this a priority site?" | Use `retrieve_chunks` tool → answer with citations |
| "Summarize https://xxx.pdf" | Use `ingest_url` tool → `retrieve_all_chunks` tool → summarize |
| "What's in this link?" | Use `ingest_url` tool → `retrieve_chunks` tool → answer |
| "What is contamination?" | No tool needed → answer from general knowledge + parcel context |

### Tools Available to Agent

1. `ingest_url(url)` — Download, parse, chunk, embed, store a PDF URL
2. `retrieve_chunks(query, top_k)` — Semantic search over session documents
3. `retrieve_all_chunks(url)` — Get ALL chunks for a specific document (for summarization)
4. `get_parcel_context(parcel_id)` — Already resolved at app level, injected as system message

### How It Fits with Current Architecture

The agent approach wraps the existing pipeline components as tools:
- `ingest_url` → calls `IngestionClient.store()`
- `retrieve_chunks` → calls existing retriever
- `retrieve_all_chunks` → new: retrieves all chunks for a URL (no semantic search, just filter by URL)

`chat_api.py` still resolves parcel data from KG and passes `system_message` + `parcel_context`.
The difference is that `service.py` uses an agent loop instead of a fixed pipeline.

### LlamaIndex ReAct Agent

LlamaIndex has built-in ReAct agent support:

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

tools = [
    FunctionTool.from_defaults(fn=ingest_url, name="ingest_url", description="..."),
    FunctionTool.from_defaults(fn=retrieve_chunks, name="retrieve_chunks", description="..."),
]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
response = agent.chat("Summarize this PDF: https://xxx")
```

### Migration Path

1. Keep current `chat()` as fallback (non-agent mode)
2. Add `agent_chat()` that uses ReAct agent with tools
3. Toggle via config or request parameter
4. Gradually migrate once agent mode is validated

## Scope

This spec covers:
- URL extraction from user messages (agent decides when to ingest)
- Full-document summarization (agent uses `retrieve_all_chunks`)
- ReAct agent integration with existing tools

This spec does NOT cover:
- Multi-step retrieval / query rewriting (future Agentic RAG)
- Self-critique / reflection loops (future)
