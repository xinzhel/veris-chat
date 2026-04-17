# VERIS Chat

Environmental assessment chat system with two modes: RAG pipeline and ReAct agent. Integrates with a Neo4j Knowledge Graph for parcel-level environmental data resolution.

## Architecture

Two parallel endpoints on the same FastAPI server:

| Mode | Prefix | How it works | Memory |
|------|--------|-------------|--------|
| RAG | `/rag/*` | Fixed pipeline: CitationQueryEngine + top-K retrieval | Mem0 (fact extraction) |
| ReAct | `/react/*` | LLM-driven tool use loop (search + summarize) | Conversation history in state |

Infrastructure:
- RAG EC2 (`54.66.111.21:8000`) — FastAPI + LlamaIndex + Qdrant Cloud
- KG EC2 (`54.253.127.203:7687`) — Neo4j with Victorian environmental data (3.7M parcels)
- Bedrock (us-east-1) — Claude Opus 4.6 via `us.*` inference profile (RMIT SCP)

## Run API Server
```bash
uvicorn main:app --reload
```

---

## VERIS ReAct

ReAct agent using LLM's native tool use API. The LLM decides when to search documents or read full documents, then streams the answer.

### Endpoints

**POST /react/chat/stream/** — Streaming ReAct chat
```bash
curl -X POST http://localhost:8000/react/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "433375739::test1", "message": "What is the licence number?"}' \
  --no-buffer
```

**SSE events:**
```
data: {"type": "status", "content": "Searching documents..."}
data: {"type": "token", "content": "The licence number is "}
data: {"type": "token", "content": "**OL000112921**..."}
data: {"type": "done", "answer": "...", "timing": {"total": 11.5}}
data: [DONE]
```

**DELETE /react/sessions/{session_id}** — Archive state, clean session
```bash
curl -X DELETE http://localhost:8000/react/sessions/433375739::test1
```

### Tools

| Tool | What it does |
|------|-------------|
| `search_documents` | Semantic search over session documents (top-K, needs embedding) |
| `get_all_chunks` | Get all chunks of a document by URL (payload filter, no embedding) |

### Conversation History

State persists across requests via checkpoint files (`data/chat_state/`). No Mem0 — raw conversation history in `ToolUseState`, supports follow-up questions like "What did you just say about X?"

---

## VERIS RAG

Fixed RAG pipeline using LlamaIndex CitationQueryEngine. Endpoints under `/rag/*`.

### Endpoints

**POST /rag/chat/** — Synchronous chat (OpenAI-compatible response)

**POST /rag/chat/stream/** — Streaming chat (OpenAI-compatible SSE)
```bash
curl -X POST http://localhost:8000/rag/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "433375739::test1", "message": "Is this a priority site?"}' \
  --no-buffer
```

**DELETE /rag/chat/sessions/{session_id}** — Clean session (memory + session_index)

### Memory

Uses Mem0 for conversation memory (fact extraction to Qdrant). Note: Mem0 adds ~10s per request and loses conversation ordering context.

---

## Deployment

### Auto-deploy (GitHub Actions)

`bash deploy/push_clean.sh` pushes to `deploy-clean` branch → `.github/workflows/deploy.yml` SSHs into RAG EC2, pulls code, restarts service.

Prerequisites: KG EC2 must be running:
```bash
aws ec2 start-instances --instance-ids i-018c87e156b4cbd8a --region ap-southeast-2
```

### Manual deploy
```bash
bash deploy/push_clean.sh
bash deploy/ec2_launch.sh --terminate
bash deploy/ec2_launch.sh
```

## Local Setup

See [documents/environment_setup.md](documents/environment_setup.md) for full instructions.

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r pyproject.toml
uv pip install -e prev_projects_repo/lits_llm
aws sso login
uvicorn main:app --reload
```

## Network

RMIT WiFi ❌ blocked | eduroam ❌ timeout — use SSH tunnel for Qdrant Cloud and Neo4j.

Local dev requires SSH tunnel through KG EC2 (must be running):
```bash
ssh -fN \
  -L 7687:localhost:7687 \
  -L 6333:629f296f-654f-4576-ab93-e335d1ab3641.ap-southeast-2-0.aws.cloud.qdrant.io:6333 \
  -i ~/.ssh/race_lits_server.pem ec2-user@54.253.127.203
```

This tunnels Neo4j (7687) and Qdrant Cloud (6333) through the KG EC2. Set `QDRANT_TUNNEL=true` in `.env`.
