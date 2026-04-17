"""
ReAct chat loop — thin wrapper around AsyncNativeReAct.

Handles:
1. Document ingestion (before ReAct loop, same as RAG pipeline)
2. Tool construction (session-scoped)
3. Agent creation via factory
4. Streaming via agent.stream() with checkpoint persistence

Usage (from react_app/chat_api.py):
    async for chunk in react_chat(
        session_id="433375739::test1",
        message="Is this a priority site?",
        system_message=APP_SYSTEM_MESSAGE,
        parcel_context=parcel_context_str,
        document_urls=["https://...pdf"],
    ):
        yield formatter.format_sse(chunk)
"""

import logging
from typing import AsyncGenerator, Dict, List, Optional, Set

from lits.agents.chain.native_react import AsyncNativeReAct

from react.tools import SearchDocumentsTool, GetAllChunksTool, STATUS_MAP
from rag_core.chat.config import load_config
from rag_core.ingestion.main_client import IngestionClient

logger = logging.getLogger(__name__)

# Module-level cache
_cached_ingestion_client: Optional[IngestionClient] = None

MODEL_NAME = "us.anthropic.claude-opus-4-6-v1"
CHECKPOINT_DIR = "data/chat_state"
MAX_ITER = 10


def _get_ingestion_client() -> IngestionClient:
    """Get or create cached IngestionClient."""
    global _cached_ingestion_client
    if _cached_ingestion_client is None:
        config = load_config()
        qdrant_cfg = config.get("qdrant", {})
        models_cfg = config.get("models", {})
        chunking_cfg = config.get("chunking", {})
        _cached_ingestion_client = IngestionClient(
            collection_name=qdrant_cfg.get("collection_name", "veris_pdfs"),
            embedding_model=models_cfg.get("embedding_model", "cohere.embed-english-v3"),
            embedding_dim=qdrant_cfg.get("vector_size", 1024),
            chunk_size=chunking_cfg.get("chunk_size", 500),
            chunk_overlap=chunking_cfg.get("overlap", 50),
        )
    return _cached_ingestion_client


async def react_chat(
    session_id: str,
    message: str,
    system_message: Optional[str] = None,
    parcel_context: Optional[str] = None,
    document_urls: Optional[List[str]] = None,
    model_name: str = MODEL_NAME,
    max_iter: int = MAX_ITER,
) -> AsyncGenerator[Dict, None]:
    """Multi-turn ReAct chat with streaming and state persistence.

    Flow:
    1. Ingest document_urls (if any) — before ReAct loop
    2. Build tools (SearchDocumentsTool, GetAllChunksTool) with session URLs
    3. Create AsyncNativeReAct agent via factory
    4. Stream via agent.stream() — lits handles checkpoint load/save

    Args:
        session_id: Session identifier (used as checkpoint filename).
        message: User's current message.
        system_message: Static system prompt (Layer 1).
        parcel_context: Dynamic parcel context from KG (Layer 2).
        document_urls: URLs to ingest before entering ReAct loop.
        model_name: Bedrock model ID.
        max_iter: Maximum ReAct iterations per turn.

    Yields:
        Dicts with ``type`` key: ``"token"``, ``"status"``, ``"done"``, ``"error"``.
    """
    config = load_config()
    collection_name = config.get("qdrant", {}).get("collection_name", "veris_pdfs")

    # 1. Ingest documents before ReAct loop (same as RAG pipeline)
    session_urls: Set[str] = set()
    if document_urls:
        ingestion_client = _get_ingestion_client()
        for url in document_urls:
            try:
                ingestion_client.store(url, session_id=session_id)
                session_urls.add(url)
                logger.info(f"[REACT] Ingested: {url}")
            except Exception as e:
                logger.error(f"[REACT] Failed to ingest {url}: {e}")
    else:
        # Load existing session URLs from session_index
        ingestion_client = _get_ingestion_client()
        session_urls = ingestion_client.get_session_urls(session_id)

    # 2. Build tools (session-scoped)
    tools = [
        SearchDocumentsTool(session_urls=session_urls, collection_name=collection_name),
        GetAllChunksTool(collection_name=collection_name),
    ]

    # 3. Build system prompt (Layer 1 + Layer 2)
    sys_parts = []
    if system_message:
        sys_parts.append(system_message)
    if parcel_context:
        sys_parts.append(parcel_context)
    full_system_message = "\n\n".join(sys_parts) if sys_parts else None

    # 4. Create agent
    agent = AsyncNativeReAct.from_tools(
        tools=tools,
        model_name=model_name,
        system_message=full_system_message,
        max_iter=max_iter,
        status_map=STATUS_MAP,
    )

    # 5. Stream — lits handles checkpoint load/save internally
    #    query_idx = session_id (checkpoint filename)
    async for chunk in agent.stream(
        query=message,
        query_idx=session_id,
        checkpoint_dir=CHECKPOINT_DIR,
    ):
        yield chunk
