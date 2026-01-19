# Implementation Plan

## Document-Grounded Conversational System

Based on the requirements in `documents/requirement_design.md`, this plan implements a session-scoped RAG system with citation-grounded generation.

**Design Principle**: Use LlamaIndex native components directly. Avoid unnecessary wrapper classes.

---

- [x] 1. Set up Core Configuration and Bedrock Integration
  - [x] 1.1 Create `veris_chat/chat/config.py` - simple config loader
    - Load config.yaml and .env using yaml/dotenv
    - Return dict with model names, Qdrant settings
    - Support AWS SSO fallback when env vars empty
    - _Requirements: 2.3_
  - [x] 1.2 Create `script/test_bedrock.py` to verify Bedrock connectivity
    - Directly use `BedrockEmbedding` and `Bedrock` LLM from LlamaIndex
    - Test embedding and chat completion
    - _Requirements: 2.3.1, 2.3.2_

- [x] 2. Verify and Extend Ingestion Pipeline with session_id
  - [x] 2.1 Update `veris_chat/ingestion/main_client.py` to support session_id
    - Add `session_id` parameter to `store()` method
    - **Note**: Task 7.4 redesigns this - session_id tracked in session_index, NOT in Qdrant payload
    - Qdrant payload: `url`, `filename`, `page_number`, `chunk_id`, `chunk_index`, `section_header`, `text`
    - _Requirements: 1_
  - [x] 2.2 Create `script/test_ingestion.py` to test IngestionClient
    - Test with sample URLs:
      - `https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf`
      - `https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/8b2790ea-4fb2-eb11-8236-00224814b9c3/OL000073004 - Statutory Document.pdf`
    - Verify download, parse, chunk, embed, store to Qdrant
    - Verify payload metadata includes `session_id`
    - _Requirements: 1_

- [x] 3. Implement Session-Scoped Retrieval (Direct LlamaIndex)
  - [x] 3.1 Create `veris_chat/chat/retriever.py` - thin utility functions
    - `get_vector_index()`: return VectorStoreIndex with QdrantVectorStore
    - `retrieve_with_session_filter(index, query, session_id, top_k)`: use Qdrant filter on session_id
    - **Note**: Task 7.4 changes filter from `session_id` to `url IN session_urls` (MatchAny)
    - `retrieve_nodes_metadata(nodes)`: extract citation metadata from nodes
    - `retrieve_for_session(query, session_id, top_k)`: high-level function coordinating all steps
    - No wrapper class, just functions using LlamaIndex directly
    - _Requirements: 2.1_
  - [x] 3.2 Create `script/test_session_retrieval.py`
    - Test retrieval with session_id filter
    - _Requirements: 2.1_

- [x] 4. Implement Citation-Grounded Generation (Direct LlamaIndex)
  - [x] 4.1 Update `utils/citation_query_engine.py` if needed
    - Use CitationQueryEngine.from_args() directly with session index
    - Extract source nodes metadata for citations
    - _Requirements: 2.2, 2.3.3_
  - [x] 4.2 Add citation formatting functions to `veris_chat/chat/retriever.py`
    - `format_citations(source_nodes, style="inline")`: format metadata as citations
    - Support inline/bracket/footnote styles
    - _Requirements: 2.2_
  - [x] 4.3 Create `script/test_citation.py`
    - Test CitationQueryEngine query and citation extraction
    - _Requirements: 2.2_

- [-] 5. Implement Conversation Memory (Direct Mem0)
  - [x] 5.1 Add memory functions
    - `get_session_memory(session_id)`: return Mem0Memory with session context
    - Use `utils/memory.py` Mem0Memory directly
    - _Requirements: 3_
  - [x] 5.2 Create `script/test_memory.py`
    - Test memory persistence and session isolation
    - _Requirements: 3_
  - [x] 5.3 **Dual Isolation Architecture** (Deliberate Design)
    - **Layer 1: Qdrant Collection** - Physical storage isolation
      - Each session gets its own collection: `mem0_memory_{session_id}`
      - Configured in `get_session_memory()` via `collection_name` in mem0_config
      - Deleting one session's collection has NO impact on other sessions
    - **Layer 2: Mem0Context** - Logical filtering within collection
      - `context = {"user_id": session_id}` passed to Mem0Memory
      - Used by Mem0 for `search()`, `add()`, `delete_all()` filtering
      - Redundant given Layer 1, but provides defense-in-depth
    - **Why both layers?**
      - Layer 1 ensures complete isolation even if Mem0 has bugs
      - Layer 2 follows Mem0's intended API design
      - Together they make accidental cross-session data access impossible
    - **Critical Mem0 Bug Discovery**: `Memory.delete_all(user_id=X)` has a bug:
      ```python
      # In mem0/memory/main.py delete_all():
      memories = self.vector_store.list(filters=filters)[0]  # Filter by user_id ✓
      for memory in memories:
          self._delete_memory(memory.id)  # Delete filtered memories ✓
      self.vector_store.reset()  # BUG: Wipes ENTIRE collection and recreates empty! ✗
      ```
      - After reset: collection still exists but with 0 points (all data gone)
      - This is safe in our design because each session has its own collection
      - If we shared one collection across sessions, this would delete ALL users' data

- [x] 6. Checkpoint - Verify Core Components
  - Ensure all tests pass, ask the user if questions arise.

- [-] 7. Build Chat Service (Single Orchestration Module)
  - [x] 7.1 Create `veris_chat/chat/service.py` - main chat function
    - `chat(session_id, message, document_urls=None)`: orchestrate full flow
    - Use IngestionClient for document ingestion (already exists)
    - Use retriever functions for session-scoped retrieval
    - Use CitationQueryEngine for generation
    - Return response dict with citations
    - _Requirements: 1, 2, 3_
  - [x] 7.2 Create `script/test_chat_service.py`
    - Test end-to-end chat flow
    - _Requirements: 1, 2, 3_
  - [x] 7.3 Add `async_chat()` function for streaming generation
    - Use `BedrockConverse` from `llama_index.llms.bedrock_converse` (has native `astream_chat()`)
    - **Model ID**: Use `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (cross-region inference profile)
    - **Critical**: Do NOT pass `region_name` for cross-region models - let boto3 handle routing
    - **Implementation (completed):**
      - `CitationQueryEngine.prepare_streaming_context()`: Replicates workflow without LLM call
        - Retrieves nodes, creates citation nodes with markdown links
        - Packs context using PromptHelper.repack() (CompactAndRefine logic)
        - Formats CITATION_QA_TEMPLATE, returns prompt + citation_nodes
      - `async_chat()`: Orchestrates streaming flow
        - Uses shared helpers: `_init_timing`, `_ingest_documents`, `_create_session_retriever`, etc.
        - Calls `prepare_streaming_context()` to get formatted prompt
        - Streams with `BedrockConverse.astream_chat()`
      - `_get_streaming_llm()`: Handles cross-region model initialization
        - Detects `us.`, `eu.`, `ap.` prefixes and skips `region_name` parameter
    - **Output format**: Clean internal format for flexibility
      - Token: `{"type": "token", "content": "..."}`
      - Done: `{"type": "done", "answer": "...", "citations": [...], "sources": [...], "timing": {...}}`
      - Error: `{"type": "error", "content": "..."}`
    - **OpenAI-compatible converters** (for API layer):
      - `OpenAIStreamFormatter`: Class to convert async_chat chunks to OpenAI streaming format
      - `format_chat_response_openai()`: Convert sync chat() response to OpenAI format
    - **Shared helpers extracted** (reduce duplication between chat/async_chat):
      - `_init_timing()`, `_ingest_documents()`, `_create_session_retriever()`
      - `_get_memory_context()`, `_augment_query_with_memory()`
      - `_format_citation_response()`, `_store_assistant_response()`
    - **Known limitation**: Context overflow uses first chunk only (no refinement loop in streaming)
    - _Requirements: 4_
  - [x] 7.4 Redesign session/URL cache architecture
    - **Current problem**: URL cache skips ingestion for new session_ids; chunks only get first session_id
    - **New architecture** (separation of concerns):
      - **Session Index**: `session_id → Set[url]` (tracks which URLs belong to each session)
        - O(1) check if URL in session: `url in session_index[session_id]`
        - O(1) get all URLs for session: `session_index[session_id]`
      - **URL Cache**: `url → metadata` (tracks ingestion state, unchanged)
        - O(1) check if URL ever ingested: `url in url_cache`
        - Stores: `{local_path, ingestion_time, ingested_at}`
      - **Qdrant**: Chunks indexed by URL only (remove `session_id` from payload)
        - Retrieval uses `MatchAny` filter on URLs from session index
    - **Ingestion flow**:
      ```python
      def store(url, session_id):
          session_index.setdefault(session_id, set()).add(url)  # Always add to session
          if url not in url_cache:  # Only ingest if never processed
              download → parse → chunk → embed → store to Qdrant
              url_cache[url] = metadata
      ```
    - **Retrieval flow**:
      ```python
      def retrieve(query, session_id, top_k):
          session_urls = session_index.get(session_id, set())
          filter = MatchAny(key="url", any=list(session_urls))
          return qdrant.query(..., filter=filter)
      ```
    - **NoOpRetriever for no-document sessions**:
      - When `session_urls` is empty, returns `NoOpRetriever` instead of failing
      - LLM can still respond using general knowledge and conversation memory
      - Consistent response format regardless of document availability
    - **Benefits**:
      - Same URL in 2 sessions → 1x Qdrant storage (shared chunks)
      - Per-session deletion: remove from session_index only
      - Clean separation: ingestion knows nothing about sessions
      - Graceful fallback when no documents provided
    - **Files modified**:
      - `veris_chat/ingestion/main_client.py`: Added session_index, updated store(), removed session_id from Qdrant payload
      - `veris_chat/chat/retriever.py`: Added `retrieve_with_url_filter()`, `retrieve_for_urls()`
      - `veris_chat/chat/service.py`: Updated `_create_session_retriever()` to use URL filter + NoOpRetriever
      - `veris_chat/utils/citation_query_engine.py`: Added `NoOpRetriever` class
    - _Requirements: 1, 2.1_

- [x] 8. Implement FastAPI Endpoints
  - [x] 8.1 Create `app/chat_api.py`
    - Pydantic models: `ChatRequest`, `ChatResponse`, `SourceMetadata`, `TimingInfo`
    - `/chat/` endpoint with OpenAI-compatible response format
    - `/chat/stream/` endpoint with OpenAI-compatible SSE streaming
    - Session-specific logging via `setup_logging` (logs to `logs/api_{session_id}.log`)
    - Datetime logging for each API call
    - _Requirements: 4_
  - [x] 8.2 Test via curl commands
    - `curl http://localhost:8000/health`
    - `curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" -d '{"session_id": "test", "message": "Hello"}'`
    - `curl -X POST http://localhost:8000/chat/stream/ -H "Content-Type: application/json" -d '{"session_id": "test", "message": "Hello"}' --no-buffer`
    - _Requirements: 4_

- [ ] 9. Final Checkpoint
  - Ensure all tests pass, ask the user if questions arise.
