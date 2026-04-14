# Implementation Plan: Agentic URL Handling

## Dependency Graph

```
T1 ──→ T3 ──→ T5 ──→ T7 ──→ T8
T2 ──→ T3
T4 ──→ T5
T6 ──→ T7
```

T1 = AsyncBedrockChatModel (lits)
T2 = NativeToolUsePolicy (lits)
T3 = NativeReAct (lits)
T4 = Tool definitions (react/)
T5 = react/loop.py integration
T6 = react_app/ + main.py
T7 = End-to-end test
T8 = Citation follow-up test

## Tasks

- [ ] Task 1: Implement `AsyncBedrockChatModel` (`lits/lm/async_bedrock.py`)
  - [ ] Add `ToolCallOutput` and `ToolCall` to `lits/lm/base.py`
  - [ ] Async `__call__` with `tools` parameter using aioboto3 Converse API
  - [ ] `format_tool_result(tool_use_id, observation)` method
  - [ ] `astream()` for token streaming
  - [ ] Add `async-bedrock/` prefix to `get_lm()` factory

- [ ] Task 2: Implement `NativeToolUsePolicy` (`lits/components/policy/native_tool_use.py`)
  - [ ] Override `_build_messages()`: use `assistant_raw` from steps + `base_model.format_tool_result()`
  - [ ] Override `_get_actions()`: pass `tools` via `_call_model(**kwargs)`, handle `ToolCallOutput`
  - [ ] Add `assistant_raw: Optional[dict] = None` and `user_message: Optional[str] = None` to `ToolUseStep`

- [ ] Task 3: Implement `NativeReAct` (`lits/agents/chain/native_react.py`)
  - [ ] `from_tools(tools, model_name, system_message, ...)` factory method
  - [ ] `run()` sync execution (reuse ChainAgent checkpoint: `query_idx` as session_id)
  - [ ] `run_async()` async execution
  - [ ] `stream()` async generator: yields token chunks + done message
  - [ ] Multi-turn: append `ToolUseStep(user_message=...)` to state before each run

- [ ] Task 4: Implement tool definitions (`react/tools.py`)
  - [ ] `IngestPDFTool` — wraps `IngestionClient.store()`
  - [ ] `SearchDocumentsTool` — wraps `retrieve_with_url_filter()`
  - [ ] `GetAllChunksTool` — new: Qdrant `scroll` with URL payload filter (no embedding)
  - [ ] Add `get_all_chunks_by_url()` to `rag_core/chat/retriever.py`

- [ ] Task 5: Implement `react/loop.py`
  - [ ] `react_chat()` async generator: build tools → create agent → stream with checkpoint
  - [ ] Pass `query_idx=session_id`, `checkpoint_dir="data/chat_state/"` to lits

- [ ] Task 6: Implement `react_app/` and `main.py`
  - [ ] Rename `app/` → `rag_app/`, convert to `APIRouter`
  - [ ] Create `react_app/chat_api.py` with `APIRouter`
  - [ ] `POST /react/chat/stream/` endpoint
  - [ ] `DELETE /react/sessions/{session_id}` endpoint (archive state, clean session_index)
  - [ ] `main.py`: mount both routers (`/rag/*`, `/react/*`)

- [ ] Task 7: End-to-end test
  - [ ] Test: "summarize this PDF: <url>" → agent calls ingest_pdf + get_all_chunks + generates summary
  - [ ] Test: follow-up question uses conversation history (state persists)
  - [ ] Test: DELETE archives state file, same session_id starts fresh
  - [ ] Test: `/rag/chat/stream/` still works unchanged

- [ ] Task 8: Citation follow-up test
  - [ ] Test multi-turn citation reasoning:
    1. User asks: "Is this a priority site?" → agent answers with citations (e.g., OL000112921.pdf)
    2. User follows up: "summarize the last cited document" (no URL, vague reference)
    3. Agent reasons from conversation history to identify the cited file
    4. Agent calls `get_all_chunks(url=<resolved URL>)` and summarizes
  - [ ] Validates: conversation history in state + LLM reasoning over previous citations + tool use
