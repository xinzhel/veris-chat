# VERIS RAG

Multi-PDF RAG system using LlamaIndex with AWS Bedrock and Qdrant. Integrates with a Neo4j Knowledge Graph for parcel-level environmental data resolution.

## Architecture

- RAG EC2 (`54.66.111.21:8000`) — FastAPI + LlamaIndex + Qdrant Cloud
- KG EC2 (`54.253.127.203:7687`) — Neo4j with Victorian environmental data (3.7M parcels)
- Bedrock (us-east-1) — Claude Opus 4.6 via `us.*` inference profile (RMIT SCP)

## Deployment

### Auto-deploy (GitHub Actions)

Running `bash deploy/push_clean.sh` pushes to `deploy-clean` branch, which triggers `.github/workflows/deploy.yml`. The workflow SSHs into the RAG EC2, pulls latest code, and restarts the service.

Prerequisites: KG EC2 must be running. Start it if stopped:
```bash
aws ec2 start-instances --instance-ids i-018c87e156b4cbd8a --region ap-southeast-2
```

GitHub repo secrets needed: `EC2_HOST` (`54.66.111.21`), `EC2_SSH_KEY` (contents of `~/.ssh/race_lits_server.pem`)

When auto-deploy won't work (need manual intervention):
- New Python dependencies added to `pyproject.toml`
- EC2 instance terminated/replaced (need fresh `user_data.sh` launch)
- System-level or `.env` changes
- KG EC2 stopped

See `.kiro/specs/aws-deployment/tasks-rag2.md` for full details.

### Manual deploy

```bash
# Push code to deploy-clean branch
bash deploy/push_clean.sh

# Terminate old + launch fresh EC2
bash deploy/ec2_launch.sh --terminate
bash deploy/ec2_launch.sh
```

## API Endpoints

### POST /chat/
Synchronous chat with OpenAI-compatible response format.

**Request:**
```json
{
  "session_id": "user123",
  "message": "What is the site status?",
  "document_urls": ["https://example.com/doc.pdf"],
  "top_k": 5,
  "use_memory": true,
  "citation_style": "markdown_link"
}
```

**Response (OpenAI-compatible):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704067200,
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The site is classified as priority..."},
    "finish_reason": "stop"
  }],
  "citations": ["[doc.pdf (p.2)](https://...)"],
  "sources": [{"file": "doc.pdf", "page": 2, "chunk_id": "c_1", "url": "https://..."}],
  "timing": {"ingestion": 1.2, "retrieval": 0.3, "generation": 2.1, "total": 3.6},
  "session_id": "user123"
}
```

### POST /chat/stream/
Async streaming chat with OpenAI-compatible SSE format.

**Request:** Same as `/chat/`

**Response (SSE):**
```
data: {"id":"chatcmpl-xxx","object":"chat.status","status":"Resolving parcel data..."}
data: {"id":"chatcmpl-xxx","object":"chat.status","status":"Ingesting documents..."}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"The"}}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" site"}}]}
...
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"finish_reason":"stop"}],"citations":[...],"sources":[...]}
data: [DONE]
```

SSE event types:
| `object` | When | Frontend handling |
|----------|------|-------------------|
| `chat.status` | KG resolution, document ingestion | Show as spinner/indicator, auto-dismiss when first `chat.completion.chunk` arrives |
| `chat.completion.chunk` | LLM token streaming | Append `delta.content` to response |

### DELETE /chat/sessions/{session_id}
Clean up a session: removes session index, Mem0 memory collection, and optionally cached KG data.

**Query params:** `clear_parcel_cache=true` (optional) — also clears cached KG data for the parcel

**Response:**
```json
{"status": "cleaned", "session_id": "433375739::test1", "cleaned": {"session_index": true, "memory": true, "parcel_cache": true}}
```

### GET /health
Health check endpoint.

## Test Commands
```bash
# Set API host (use localhost for local dev, or EC2 IP for deployed)
export API_HOST=localhost:8000
# export API_HOST=54.66.111.21:8000  # EC2 deployment

# Health check
curl http://${API_HOST}/health

# Parcel session — KG resolves document URLs and parcel context automatically
curl -X POST http://${API_HOST}/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "433375739::test1",
    "message": "Is this a priority site?"
  }' --no-buffer

# Follow-up (uses memory + cached KG data)
curl -X POST http://${API_HOST}/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "433375739::test1",
    "message": "What audits were done?"
  }' --no-buffer

# Session cleanup
curl -X DELETE "http://${API_HOST}/chat/sessions/433375739::test1?clear_parcel_cache=true"

# Backward compatible (no parcel, manual document_urls)
curl -X POST http://${API_HOST}/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "What is the purpose of this document?",
    "document_urls": ["https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf"]
  }' --no-buffer
```


Be Careful of Network firewall, e.g.,
- RMIT WiFi ❌ blocked by university firewall (shows "Website Blocked")
- eduroam ❌ timeout (likely also firewall)

## (Optional) Local Setup

See [documents/environment_setup.md](documents/environment_setup.md) for full setup instructions (Python version, uv, lits-llm dependency).

Quick start:
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r pyproject.toml
uv pip install -e prev_projects_repo/lits_llm
aws sso login  # if using SSO credentials
```

## Run API Server
```bash
uvicorn app.chat_api:app --reload
```

