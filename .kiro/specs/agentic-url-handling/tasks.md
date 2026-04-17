# Implementation Plan: Agentic URL Handling

## Dependency Graph

```
T1 ──→ T2 ──→ T3 ──→ T5 ──→ T7 ──→ T8
              T4 ──→ T5
              T6 ──→ T7
              T3 ──→ T9
```

T1 = AsyncBedrockChatModel (lits/lm)
T2 = AsyncNativeToolUsePolicy (lits/components) — depends on T1 for format_tool_result()
T3 = AsyncNativeReAct (lits/agents)
T4 = Tool definitions (react/)
T5 = react/loop.py integration
T6 = rag_app/ + react_app/ + main.py
T7 = End-to-end test
T8 = Citation follow-up test
T9 = docs/agents/AsyncNativeReAct.md

## Tasks

- [x] Task 1: Implement `AsyncBedrockChatModel` (`lits/lm/async_bedrock.py`)
  - [x] Add `ToolCallOutput(Output)` and `ToolCall` dataclass to `lits/lm/base.py`
    - `ToolCallOutput.tool_calls: list[ToolCall]`, `stop_reason: str`, `raw_message: dict`
  - [x] Async `__call__` using aioboto3 `converse_stream()`
    - When `tools=None`: returns `Output(text)`
    - When `tools` provided: uses Converse API `toolConfig`, returns `ToolCallOutput`
    - Dispatch on `contentBlockStart`: `toolUse` → collect tool call, text → stream tokens
  - [x] `format_tool_result(tool_use_id, observation) -> dict` — Bedrock Converse format
  - [x] Add `async-bedrock/` prefix to `get_lm()` factory in `lits/lm/__init__.py`

- [x] Task 2: Implement `AsyncNativeToolUsePolicy` (`lits/components/policy/native_tool_use.py`)
  - [x] Extract `BaseToolUseStep` from `ToolUseStep` in `lits/structures/tool_use.py`
    - Shared fields: `action`, `observation`, `answer` + `get_action()`, `get_observation()`, `get_answer()`
    - `ToolUseStep` inherits `BaseToolUseStep`, keeps `think`, `assistant_message`, extractors (no change to existing code)
  - [x] Create `NativeToolUseStep(BaseToolUseStep)` in `lits/structures/tool_use.py`
    - Fields: `assistant_message_dict: Optional[dict]` (LLM raw response), `user_message: Optional[str]` (multi-turn)
    - Override `to_dict()`, `from_dict()`, `to_messages()`
  - [x] Update `ToolUseTransition.step()` assert: `isinstance(step, BaseToolUseStep)` instead of `ToolUseStep`
  - [x] Override `_build_messages(query, state)` in `AsyncNativeToolUsePolicy`:
    - Use `step.assistant_message_dict` directly for assistant messages
    - Use `self.base_model.format_tool_result()` for tool result messages (provider-agnostic)
    - Handle `step.user_message` for multi-turn conversation history
  - [x] Override `_get_actions()`: pass `tools=self.tool_schemas` via `_call_model(**kwargs)`, handle `ToolCallOutput` vs `Output`

- [x] Task 3: Implement `AsyncNativeReAct` (`lits/agents/chain/native_react.py`)
  - [x] `from_tools(tools, model_name, system_message, max_iter, ...)` factory method
  - [x] `run_async(query, query_idx, checkpoint_dir, ...)` async non-streaming — reuses `ChainAgent` checkpoint mechanism (`query_idx` as session_id)
  - [x] `stream(query, query_idx, checkpoint_dir, ...)` async generator:
    - Load state from checkpoint (if exists)
    - Append `NativeToolUseStep(user_message=query)` to state
    - ReAct loop: call `policy._get_actions_stream()` → if tool_call, yield `{"type": "status", ...}` via `STATUS_MAP`, execute tool → if text_delta, yield `{"type": "token", ...}`
    - Save state to checkpoint
    - Yield `{"type": "done", ...}` with full answer + timing
  - [x] `_get_actions_stream()` added to `AsyncNativeToolUsePolicy` — streaming version reusing `_build_messages()` + `set_system_prompt()`
  - [x] Multi-turn: state persists across calls via checkpoint, not reset between turns

- [x] Task 4: Implement tool definitions (`react/tools.py`)
  - [x] `SearchDocumentsTool(BaseTool)` — imports `retrieve_with_url_filter()` from `rag_core/chat/retriever.py` (react imports rag_core, not the other way)
  - [x] `GetAllChunksTool(BaseTool)` — implements Qdrant `scroll` with URL payload filter directly in `react/tools.py` (no changes to `rag_core/`)
  - [x] Define `STATUS_MAP` in `react/tools.py`
  - NOTE: `react/` can import from `rag_core/`, but `rag_core/` never imports from `react/`. No new code added to `rag_core/`.

- [x] Task 5: Implement `react/loop.py`
  - [x] `react_chat()` async generator: ingest documents → build tools → create agent → stream with checkpoint
  - [x] Ingestion happens before ReAct loop (same as RAG pipeline)
  - [x] Pass `query_idx=session_id`, `checkpoint_dir="data/chat_state/"` to lits

- [ ] Task 6: Implement `rag_app/`, `react_app/`, and `main.py`
  - [ ] Rename `app/` → `rag_app/`, convert `app` FastAPI instance to `APIRouter`
  - [ ] Create `react_app/__init__.py` and `react_app/chat_api.py` with `APIRouter`
  - [ ] `POST /react/chat/stream/` — resolve parcel data, call `react_chat()`, SSE stream with `formatter.format_sse()`
  - [ ] `DELETE /react/sessions/{session_id}` — clean session_index, archive state JSON with timestamp (`{session_id}__{timestamp}.json`), keep parcel_cache
  - [ ] `main.py`: `app = FastAPI()`, `app.include_router(rag_router, prefix="/rag")`, `app.include_router(react_router, prefix="/react")`
  - [ ] Update `uvicorn` start command: `uvicorn main:app --reload`

- [ ] Task 7: End-to-end test
  - [ ] Test: parcel session → agent uses `search_documents` to answer question with citations
  - [ ] Test: "summarize the document" → agent calls `get_all_chunks` + generates summary
  - [ ] Test: follow-up question uses conversation history (state persists via checkpoint)
  - [ ] Test: DELETE archives state file with timestamp, same session_id starts fresh
  - [ ] Test: `/rag/chat/stream/` still works unchanged
  - [ ] Test: status events appear during tool calls

- [ ] Task 8: Citation follow-up test
  - [ ] Test multi-turn citation reasoning:
    1. User asks: "Is this a priority site?" → agent answers with citations (e.g., OL000112921.pdf)
    2. User follows up: "summarize the last cited document" (no URL, vague reference)
    3. Agent reasons from conversation history to identify the cited file
    4. Agent calls `get_all_chunks(url=<resolved URL>)` and summarizes
  - [ ] Validates: conversation history in state + LLM reasoning over previous citations + tool use

- [ ] Task 9: Write `docs/agents/AsyncNativeReAct.md` in lits_llm
  - [ ] Explain "Native" = uses LLM's native tool use API (structured JSON tool calls), not text-based XML parsing
  - [ ] Highlight provider-agnostic abstraction: `ToolCall`, `ToolCallOutput`, `format_tool_result()`, `tool_use_id`
  - [ ] Architecture: AsyncBedrockChatModel → AsyncNativeToolUsePolicy → AsyncNativeReAct
  - [ ] Usage examples: `from_tools()` factory, `run()`, `stream()`
