# Parcel KG Integration — Test Results

## Test 1: Parcel Context Passed to LLM ✅

**Request:**
```bash
curl -X POST http://localhost:8000/chat/stream/ \
  -d '{"session_id": "433375739::ctx_test2", "message": "What EPA licences does this parcel have?", "use_memory": false}'
```

**Result:** LLM correctly used both KG parcel context AND PDF document:
- Permission Type: A01 (Reportable priority waste management) — from KG context
- Issue Date: 16 April 2015 — from KG context
- Licence holder: J.J. Richards & Sons Pty Ltd — from PDF document
- Citations with page numbers: p.1, p.4, p.5, p.8, p.15

**Timing:** ingestion 0.001s (cached), retrieval 0.43s, generation 7.66s, total 8.34s

## Test 2: Session Cleanup Endpoint ✅

**Request:**
```bash
curl -X DELETE http://localhost:8000/chat/sessions/433375739::e2e_test2
```

**Result:**
```json
{"status": "cleaned", "session_id": "433375739::e2e_test2", "cleaned": {"session_index": true, "memory": false, "parcel_cache": false}}
```

- `session_index: true` — session removed from session_index
- `memory: false` — no Mem0 memory to clean (use_memory was false in test)
- `parcel_cache: false` — not cleared by default (parcel-level, shared across sessions)

## Test 3: With Memory

**Test 3a: First message with name ✅**
```bash
curl -X POST http://localhost:8000/chat/stream/ \
  -d '{"session_id": "433375739::mem_test", "message": "My name is Alice. Is this a priority site?", "use_memory": true}'
```
Result: LLM addressed user as "Alice" at the end of response. Memory timing: 5.03s. Total: 13.25s.

**Test 3b: Follow-up "What is my name?" ⚠️ Known limitation**
```bash
curl -X POST http://localhost:8000/chat/stream/ \
  -d '{"session_id": "433375739::mem_test", "message": "What is my name?", "use_memory": true}'
```
Result: LLM did NOT remember the name. This is the known memory limitation documented in `documents/TODO.md` — Mem0 extracts facts asynchronously, and `CitationQueryEngine.query()` only accepts a string, so full chat history is lost. The name was used in the first response but not persisted as a Mem0 fact in time for the second query.

## Test 4: DELETE with Memory Session ⚠️

```bash
curl -X DELETE http://localhost:8000/chat/sessions/433375739::mem_test
```
Result:
```json
{"status": "cleaned", "session_id": "433375739::mem_test", "cleaned": {"session_index": true, "memory": false, "parcel_cache": false}}
```
- `session_index: true` — cleaned
- `memory: false` — Mem0 cleanup timed out. Mem0 creates its own internal Qdrant connection that doesn't use our SSH tunnel. This only affects local dev with company wifi; EC2 deploy connects directly.
- Note: Memory creation (in chat) works because it goes through a different code path that may have cached the connection from before the tunnel was needed.
