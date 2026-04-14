# Frontend Checklist — Backend API Changes

Items to confirm with frontend developer before switching RAG EC2 to v2.

## 1. New SSE event type: `chat.status`

Backend now sends `chat.status` events before token streaming begins:

```
data: {"id":"chatcmpl-xxx","object":"chat.status","created":1776145769,"status":"Resolving parcel data..."}
data: {"id":"chatcmpl-xxx","object":"chat.status","created":1776145769,"status":"Ingesting documents..."}
data: {"id":"chatcmpl-xxx","object":"chat.status","created":1776145769,"status":"Retrieving memory..."}
data: {"id":"chatcmpl-xxx","object":"chat.status","created":1776145769,"status":"Generating response..."}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"The"}}]}
...
```

Check with frontend:
- [ ] Does the SSE parser handle unknown `object` types gracefully (ignore/skip), or will it throw an error?

## 2. Session ID format change

Parcel sessions now use `parcel_id::temp_id` format (e.g., `433375739::test1`). The `::` is part of the session ID.

## 3. New DELETE endpoint

`DELETE /chat/sessions/{session_id}?clear_parcel_cache=true`

Cleans up session index, Mem0 memory collection, and optionally cached KG data.

## After confirmation

Once frontend confirms compatibility:
1. Reassociate Elastic IP `54.66.111.21` from old EC2 to new v2 EC2
2. Terminate old EC2 instance
3. Verify frontend works end-to-end
