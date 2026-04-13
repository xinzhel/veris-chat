# Agentic URL Handling

## Problem

When a user says "summarize this PDF: https://xxx" or "what's in https://xxx", the current system:
1. Ignores URLs in the user message (only uses KG-resolved or request-supplied `document_urls`)
2. Uses top-K chunk retrieval, which doesn't work for "summarize the whole document"

## Design Principle

**Don't touch existing RAG workflow.** The current `chat()` / `async_chat()` pipeline stays as-is. The agent is a new, separate subpackage that can call into existing components as tools.

## Package Structure

```
rag_core/       ← existing RAG pipeline (untouched)
  chat/
    service.py
    retriever.py
    config.py
  ingestion/
    main_client.py
  kg/
    client.py
    context.py
  utils/

agent/          ← NEW: top-level, not inside rag_core
  __init__.py
  tools.py      ← tool definitions (ingest_pdf, search_docs, get_all_chunks)
  loop.py       ← ReAct tool-use loop with BedrockConverse

app/            ← application layer (chat_api.py)
```

`agent/` is a top-level package that imports from `rag_core/` but `rag_core/` never imports from `agent/`.

## Architecture

### Current Pipeline (unchanged)

```
chat_api.py → service.chat() → fixed pipeline → response
```

### New Agent Mode (parallel path)

```
chat_api.py → agent.loop.agent_chat() → ReAct loop → response
```

`chat_api.py` decides which path based on config or request parameter.

## ReAct Loop (`agent/loop.py`)

Uses Bedrock's native tool use API (not text-parsing like some frameworks). Opus 4.6 has strong tool use support.

```python
async def agent_chat(session_id, message, system_message, parcel_context, ...):
    """ReAct tool-use loop with streaming final answer."""
    messages = build_initial_messages(system_message, parcel_context, message)
    
    while True:
        response = await llm.achat(messages, tools=tool_schemas)
        
        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call, session_id)
                messages.append(tool_result_message(result))
        else:
            # Final answer — stream it
            async for chunk in llm.astream_chat(messages):
                yield {"type": "token", "content": chunk.delta}
            break
```

## Tools (`agent/tools.py`)

Tool interface follows lits_llm's `BaseTool` pattern: `name`, `description`, `args_schema` (Pydantic), `_run()`.

Note: lits_llm has its own general LLM interface that includes BedrockConverse — not litellm.

| Tool | Description | Wraps existing code |
|------|-------------|---------------------|
| `ingest_pdf` | Download + ingest a PDF URL into session | `IngestionClient.store(url, session_id)` |
| `search_documents` | Semantic search over session documents | `retrieve_with_url_filter()` |
| `get_all_chunks` | Get ALL chunks for a URL (for summarization) | New: Qdrant filter by URL, return all |

Tools are thin wrappers around existing components. No duplication of ingestion/retrieval logic.

### Tool Schema Example (for Bedrock tool use API)

```python
tool_schemas = [
    {
        "name": "ingest_pdf",
        "description": "Download and ingest a PDF from URL into the session's document store",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "PDF URL to ingest"}
            },
            "required": ["url"]
        }
    },
    ...
]
```

## Integration with chat_api.py

```python
# chat_api.py
from agent import agent_chat  # new import

@app.post("/chat/stream/")
async def chat_stream_endpoint(request: ChatRequest):
    parcel_data = _resolve_parcel_data(request.session_id, logger)
    
    if request.use_agent:  # or config-based toggle
        generator = agent_chat(...)
    else:
        generator = async_chat(...)  # existing pipeline
    
    # Same SSE streaming for both
    async def generate():
        async for chunk in generator:
            yield formatter.format_sse(chunk)
```

## What We Reuse from lits_llm

1. **Tool interface pattern**: `BaseTool` with `name`, `description`, `args_schema`, `_run()`
2. **PDFQueryTool concept**: URL + query → retrieve relevant chunks
3. **Loop structure**: `while not answer: get_action → execute → append`

What we DON'T reuse:
- LiTS framework (policy, transition, reward, state) — overkill for chat
- Text-based action parsing — Bedrock has native tool use API
- Evaluation/benchmarking infrastructure

## Scope

Covers:
- `agent/` subpackage (tools + loop)
- Bedrock native tool use integration
- Streaming support for final answer
- Toggle between pipeline and agent mode

Does NOT cover:
- Multi-step retrieval / query rewriting
- Self-critique / reflection loops
- Modifying existing `chat/service.py`
