# VERIS RAG

Multi-PDF RAG system using LlamaIndex with AWS Bedrock and Qdrant.

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
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"The"}}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" site"}}]}
...
data: {"id":"chatcmpl-xxx","choices":[{"finish_reason":"stop"}],"citations":[...],"sources":[...]}
data: [DONE]
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

# Sync chat with document ingestion
curl -X POST http://${API_HOST}/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "What is the purpose of this document?",
    "document_urls": ["https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf"]
  }'

# Streaming chat with document ingestion
curl -X POST http://${API_HOST}/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "seession2",
    "message": "What is my name?",
    "document_urls": ["https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf"]
  }' --no-buffer
```


Be Careful of Network firewall, e.g.,
- RMIT WiFi ❌ blocked by university firewall (shows "Website Blocked")
- eduroam ❌ timeout (likely also firewall)

## (Optional) Local Setup
```bash
conda env create -f environment.yaml
conda activate veris_vectordb
aws sso login  # if using SSO credentials
```

## Run API Server
```bash
uvicorn app.chat_api:app --reload
```

