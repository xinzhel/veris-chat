# Test Scripts

Test scripts for the VERIS RAG system components.

## Execution Order

Run tests in this order (some tests depend on data from previous tests):

1. `1.test_ingestion.py` - Populates test collection
2. `2.1.test_session_retrieval.py` - Requires ingestion data
3. `3.test_citation.py` - Requires ingestion data
4. `4.test_memory.py` - Independent (creates own collections)

## Collection Mapping

| Script | Collection(s) | Action | Notes |
|--------|---------------|--------|-------|
| `1.test_ingestion.py` | `veris_pdfs_test` | Creates/populates | Deletes and recreates collection each run |
| `2.1.test_session_retrieval.py` | `veris_pdfs_test` | Reads | Requires `1.test_ingestion.py` first |
| `3.test_citation.py` | `veris_pdfs_test` | Reads | Requires `1.test_ingestion.py` first |
| `4.test_memory.py` | `mem0_memory_test_memory_session_001`<br>`mem0_memory_test_memory_session_002` | Creates/deletes | Independent, cleans up after itself |
| `test_bedrock_llamaindex.py` | None | N/A | Tests Bedrock connectivity only |
| `test_mem0_fact_extraction.py` | None | N/A | Tests Mem0 LLM fact extraction |

## Session IDs

- `veris_pdfs_test` collection uses session_id: `test_session_001`
- Memory tests use session_ids: `test_memory_session_001`, `test_memory_session_002`

## Prerequisites

- AWS SSO login: `aws sso login`
- Qdrant Cloud access (configured in `.env`)
- Python environment with dependencies installed
