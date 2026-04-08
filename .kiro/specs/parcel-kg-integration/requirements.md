# Requirements Document

## Introduction

Sessions are parcel-wise — each session is scoped to a specific parcel. Since the parcel determines everything (which documents to retrieve, what environmental context to provide), the backend resolves all of this from the Neo4j knowledge graph (`neptune_deployment/vic_unearthed_kg/`). The frontend just sends `parcel_id::temp_id` + message — no `document_urls` needed.

## Glossary

- **Session**: A temporary, parcel-wise conversation session identified by `parcel_id::temp_id`
- **Parcel_ID**: A geospatial location identifier for a parcel
- **Temp_ID**: A temporary identifier to distinguish multiple users opening sessions for the same parcel
- **Knowledge_Graph**: The Neo4j knowledge graph deployed from `neptune_deployment/vic_unearthed_kg/`
- **Backend_Service**: The existing chat backend service (`veris_chat/chat/service.py`, `app/chat_api.py`)
- **System_Message**: The LLM system prompt that provides first-layer application context (e.g., role, behavior) and second-layer parcel-specific context from the knowledge graph

## Requirements

### Requirement 1: Parcel-Wise Session Management

**User Story:** As a developer, I want sessions to be parcel-wise and temporary, so that each session is scoped to a specific parcel and cleaned up after use.

#### Acceptance Criteria

1. THE Session SHALL use a session ID format of `parcel_id::temp_id`, where `parcel_id` is a geospatial location and `temp_id` distinguishes multiple users opening a session for the same parcel
2. WHEN an ending event is sent from the front end, THE Backend_Service SHALL remove all memory of the session
3. WHEN designing session cleanup, THE Backend_Service SHALL determine common practice for cleanup — e.g., whether a new request endpoint is needed for session cleanup, and if so, what the schema should be

### Requirement 2: Document URL Retrieval from Knowledge Graph

**User Story:** As a developer, I want the backend to retrieve document URLs from the Neo4j knowledge graph based on the parcel ID, so that document URLs are fixed per parcel and no longer need to be supplied in the front-end request.

#### Acceptance Criteria

1. WHEN a session is initiated, THE Backend_Service SHALL extract the parcel ID from the session ID and retrieve the corresponding document URLs from the Knowledge_Graph
2. THE Backend_Service SHALL reuse the existing ingestion pipeline — only the API entry point that currently accepts `document_urls` from the request needs to change (to resolve URLs from the Knowledge_Graph instead). The rest of the pipeline remains unchanged.
3. THE Backend_Service SHALL retrieve document URLs according to the parcel ID from the Neo4j knowledge graph deployment (see `neptune_deployment/vic_unearthed_kg/`)

### Requirement 3: Parcel Context in System Message

**User Story:** As a developer, I want the system message to include second-layer parcel context from the knowledge graph, so that the LLM has relevant geospatial and environmental information about the parcel.

#### Acceptance Criteria

1. THE Backend_Service SHALL provide second-layer context in the System_Message, retrieved from the Knowledge_Graph, covering the 7 connections specified in `neptune_deployment/vic_unearthed_kg/README.md`:
   - parcel overlap with audit
   - parcel overlap with licence
   - parcel overlap with prsa
   - parcel overlap with psr
   - parcel overlap with vlr (500m buffer)
   - parcel overlap with overlay (erosion and environmental)
   - parcel overlap with business list (adjacency = potential contamination)

### Requirement 4: Knowledge Graph Deployment

**User Story:** As a developer, I want the Neo4j knowledge graph deployed and accessible, so that requirements 2 and 3 can retrieve data from it.

#### Acceptance Criteria

1. THE Knowledge_Graph SHALL be deployed as a Neo4j Docker container on EC2, accessible via SSH tunnel (see `neptune_deployment/vic_unearthed_kg/`)
2. THE Knowledge_Graph SHALL be deployed before requirements 2 and 3 can be implemented
