Hi Ozzy and Lukesh,

 

As discussed with Ozzy today, an interim version of the backend API, addressing what we discussed last Wednesday, has just been pre-released. The agentic version will follow later next week given the amount of work involved.

 

Quick recap of what changed:

KG integration is live — the backend now resolves parcel data (document URLs + environmental context) from the Neo4j knowledge graph automatically. Session IDs use parcel_id::temp_id format (e.g., 433375739::test1). There's also a new DELETE /chat/sessions/{session_id} endpoint for session cleanup.

The SSE stream now includes a new event type chat.status (with "object": "chat.status") before token streaming begins. These are status updates like "Resolving parcel data...", "Ingesting documents...", "Retrieving memory...", "Generating response...". The existing chat.completion.chunk format is unchanged. Lukesh Bhusal — these could be useful for giving users visibility into what's happening while they wait for a response. Which ones to surface and how to present them is up to you.

The v2 is on a new EC2 instance at http://16.176.180.235:8000. The old endpoint (54.66.111.21) is still running the v1 and untouched. Once we confirm the frontend works with v2, I'll switch the IP over （consistently using the elastic IP: 54.66.111.21）.

Full API docs and test commands: https://github.com/AEA-MapTalk/veris-chat/blob/deploy-clean/README.md

Let me know if anything breaks or looks off.


---

## 2026-04-21 — v3: ReAct Agent Backend

Hi Ozzy and Lukesh,

The agentic version of the backend is now live on the same IP: **http://54.66.111.21:8000**

### What's new

The backend now has two modes running in parallel:

- `/rag/*` — the existing RAG pipeline (same as v2, unchanged)
- `/react/*` — new ReAct agent that uses LLM-driven tool use

The ReAct agent lets the LLM decide how to answer: it can search documents semantically, or read an entire document for summarization. It also maintains full conversation history, so follow-up questions like "what did you just say about X?" work naturally.

### New endpoints

| Endpoint | What it does |
|----------|-------------|
| `POST /react/chat/stream/` | Streaming ReAct chat (SSE) |
| `DELETE /react/sessions/{session_id}` | Clean up session |
| `GET /health` | Health check (shared) |

The existing `/rag/*` endpoints are unchanged and still work.

### SSE format

The ReAct stream uses a different event format from the RAG endpoints. RAG uses OpenAI-compatible format (`object`, `choices`, `delta`); ReAct uses a flat format. Let me know if you'd prefer the same OpenAI format for both — easy to add a wrapper.

```
data: {"type": "status", "content": "Searching documents..."}
data: {"type": "token", "content": "The licence number is "}
data: {"type": "token", "content": "**OL000112921**..."}
data: {"type": "done", "answer": "...", "timing": {"total": 11.5}}
data: [DONE]
```

Event types: `status` (tool execution), `token` (streamed answer), `done` (final result with full answer + timing), `error`.

@Lukesh — again, the `status` events are optional to display. They indicate the agent is working (e.g., "Searching documents...", "Reading the full document..."). You could show them as a spinner or loading indicator.

### Request format

Same as before — just change the endpoint path:

```bash
curl -X POST http://54.66.111.21:8000/react/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "433375739::test1", "message": "What is the licence number?"}' \
  --no-buffer
```

### GitHub repo

The deploy repo has moved to: https://github.com/AEA-MapTalk/veris-llm-agent

Full docs and test commands in the README.

Let me know if anything breaks or looks off.
