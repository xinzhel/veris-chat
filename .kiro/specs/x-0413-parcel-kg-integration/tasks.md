# Implementation Plan: Parcel KG Integration

## Overview

Replace frontend-supplied `document_urls` with backend-driven KG resolution. The parcel ID (from session ID `parcel_id::temp_id`) drives document URL retrieval and parcel context assembly from Neo4j. KG resolution happens at the app/ layer (`chat_api.py`); `service.py` stays application-agnostic, receiving `system_message` + `parcel_context` as parameters.

## Dependency Graph

```
T1 ──→ T2 ──→ T3 ──→ T5
                ↘      ↑
                T4 ──→ T5 ──→ T6 ──→ T7
```

T1 = Deploy Neo4j on EC2
T2 = KGClient module
T3 = format_parcel_context()
T4 = Modify service.py
T5 = Modify chat_api.py (KG resolution)
T6 = Session cleanup endpoint
T7 = Final checkpoint

## Tasks

- [x] Task 1: Connect to existing Neo4j on EC2 and verify
  - Neo4j deployed on EC2 instance `i-018c87e156b4cbd8a` (t3.medium, ap-southeast-2, EIP `54.253.127.203`)
  - [x] Neo4j Docker built and running with n10s plugin
  - [x] RDF data loaded: 8,717,108 nodes, 11,949,449 relationships, 3,690,107 parcels
  - [x] Data stored in S3: `s3://veris-kg-data-xinzhe/output/`
  - [x] SSH tunnel access: `ssh -fN -L 7687:localhost:7687 -L 7474:localhost:7474 -i ~/.ssh/race_lits_server.pem ec2-user@54.253.127.203`
  - [x] Add `neo4j` config block to `config.yaml` with `uri`, `user`, `password` fields
  - [x] Add `neo4j` Python driver to `environment.yaml` dependencies
  - _Requirements: 4.1, 4.2_

- [x] Task 2: Implement KGClient module (`rag_core/kg/client.py`)
  - [x] Create `rag_core/kg/__init__.py` and `rag_core/kg/client.py`
  - [x] Implement `KGClient.__init__(uri, user, password)` — wraps `neo4j.GraphDatabase.driver`
  - [x] Implement `get_document_urls(parcel_id) -> list[str]` — Cypher query: `MATCH (p:Parcel)-[:hasOnsiteAssessment|hasOffsiteAssessment]->(a:Resource)-[:hasAssessmentReport]->(r:AssessmentReport) WHERE parcel_id IN p.hasPFI RETURN r.hasLink[0] AS pdf_url`
  - [x] Implement `get_parcel_context(parcel_id) -> dict` — returns dict with 7 keys: `audits`, `licences`, `prsa`, `psr`, `vlr`, `overlays`, `business_listings`. Each value is a list of dicts with assessment-specific fields from the KG
  - [x] Implement `close()` to shut down the Neo4j driver
  - [x] Add `get_kg_client()` factory function that reads `neo4j` config from `config.yaml` and caches the client instance
  - Note: Query performance ~54s cold / ~35s warm on t3.medium. See `.kiro/specs/neo4j-optimization/` for optimization plan.
  - _Requirements: 2.1, 2.3, 3.1_

- [x] Task 3: Implement `format_parcel_context()` (`rag_core/kg/context.py`)
  - [x] Create `rag_core/kg/context.py`
  - [x] Implement `format_parcel_context(parcel_id, kg_context) -> str` — converts the dict from `get_parcel_context()` into a natural-language system message block with section headers for all 7 connection types
  - [x] For empty connection types, output "No data found" (confirmed absence, not missing info — see design Q&A)
  - [x] Implement `parse_session_id(session_id) -> tuple[str, str]` — splits `parcel_id::temp_id`, raises `ValueError` if `::` separator missing
  - _Requirements: 3.1, 1.1_

- [x] Task 4: Modify `service.py` to accept `system_message` and `parcel_context` parameters
  - [x] Add `system_message: Optional[str] = None` and `parcel_context: Optional[str] = None` parameters to `chat()` function signature
  - [x] Add same parameters to `async_chat()` function signature
  - [x] Integrate `system_message` + `parcel_context` into the prompt construction — prepend as context layers before memory context and retrieved chunks (Layer 1: system_message, Layer 2: parcel_context, Layer 3: memory, Layer 4: chunks)
  - [x] Ensure backward compatibility: when `system_message` and `parcel_context` are `None`, behavior is identical to current implementation
  - Note on system message placement:
    - `chat()`: system_message prepended to query text (CitationQueryEngine only accepts a string)
    - `async_chat()`: system_message placed in Bedrock API system message position (better LLM adherence)
  - [x] In `async_chat()`, use `ChatMessage(role=SYSTEM)` for system_message + parcel_context instead of prepending to query text
  - _Requirements: 3.1, 2.2_

- [x] Task 5: Modify `chat_api.py` — KG resolution at app level
  - [x] Import `KGClient`, `get_kg_client`, `format_parcel_context`, `parse_session_id` from `rag_core/kg/`
  - [x] Define `APP_SYSTEM_MESSAGE` constant (Layer 1 static system message for environmental assessment assistant)
  - [x] Implement parcel cache (`_parcel_cache: dict`) keyed by `parcel_id` to avoid re-querying KG per message
  - [x] In `chat_endpoint()`: parse session ID → extract `parcel_id` → check cache → if miss, query KG for `document_urls` and `parcel_context` → cache results → pass `document_urls`, `system_message`, `parcel_context` to `chat()`
  - [x] In `chat_stream_endpoint()`: same KG resolution logic as sync endpoint, pass extra params to `async_chat()`
  - [x] Keep `document_urls` in `ChatRequest` as optional for backward compatibility
  - _Requirements: 1.1, 2.1, 2.2, 3.1_

- [x] Task 6: Implement session cleanup endpoint
  - [x] Add `DELETE /chat/sessions/{session_id}` endpoint in `chat_api.py`
  - [x] Remove session from `session_index` (via `IngestionClient`)
  - [x] Delete Mem0 memory collection for the session
  - [x] Clear cached KG data for the parcel from `_parcel_cache`
  - [x] Return appropriate HTTP response (200 on success, 404 if session not found)
  - _Requirements: 1.2, 1.3_

- [x] Task 7: Checkpoint — Ensure all components work end-to-end
  - [x] First message triggers KG lookup + PDF ingestion + streaming generation (PFI 433375739)
  - [x] Parcel context correctly passed to LLM (cited EPA licence A01 details)
  - [x] Memory with `::` session IDs works (collection name uses `replace('::', '_')`)
  - [x] DELETE endpoint cleans up session_index + memory collection
  - [x] Parcel cache preserved by default on delete (correct behavior)
  - [x] Fixed: `memory_llm` updated from EOL `anthropic.claude-3-5-sonnet-20241022-v2:0` to `us.anthropic.claude-opus-4-6-v1`

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Task 1 (EC2 deployment) is a prerequisite for all subsequent tasks — KGClient needs a running Neo4j instance
- `service.py` stays application-agnostic: it never imports from `rag_core/kg/` or knows about parcels
- All KG resolution logic lives in `chat_api.py` (app/ layer) and `rag_core/kg/` (library layer)
