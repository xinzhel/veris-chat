"""
FastAPI endpoints for the Veris Chat service.

Endpoints:
- POST /chat/ - Synchronous chat with OpenAI-compatible response
- POST /chat/stream/ - Async streaming chat with OpenAI-compatible SSE
- DELETE /chat/sessions/{session_id} - Clean up session (memory, cache)
- GET /health - Health check

Start server:
    uvicorn app.chat_api:app --reload

Test commands (copy-paste ready):

# Health check
curl http://localhost:8000/health

# Sync chat (parcel session — KG resolves URLs and context)
curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" -d '{"session_id": "433375739::test1", "message": "Is this a priority site?"}'

# Streaming chat (parcel session)
curl -X POST http://localhost:8000/chat/stream/ -H "Content-Type: application/json" -d '{"session_id": "433375739::test1", "message": "What audits were done?"}' --no-buffer

# Backward compatible (no parcel, manual document_urls)
curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" -d '{"session_id": "test", "message": "Hello"}'

# Session cleanup
curl -X DELETE http://localhost:8000/chat/sessions/433375739::test1

# Session cleanup with parcel cache clear
curl -X DELETE "http://localhost:8000/chat/sessions/433375739::test1?clear_parcel_cache=true"
"""

import os
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any

# Set AWS_REGION before other imports (us-east-1 required for Opus 4.5 with us.* prefix)
os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_core.chat.service import (
    chat,
    async_chat,
    OpenAIStreamFormatter,
    format_chat_response_openai,
)
from rag_core.kg import get_kg_client, format_parcel_context, parse_session_id
from rag_core.utils.logger import setup_logging

# =============================================================================
# APPLICATION SYSTEM MESSAGE (Layer 1 — static)
# =============================================================================

APP_SYSTEM_MESSAGE = """You are an environmental assessment assistant for Victorian land parcels.
You answer questions grounded in assessment reports and knowledge graph data.
When referencing information from source documents, cite them using the provided markdown link format.
If the parcel context below indicates "No data found" for a category, that means we confirmed there is no data — not that we failed to check."""

# Session-specific loggers cache
_session_loggers: Dict[str, any] = {}

# Parcel cache: parcel_id → {"document_urls": [...], "parcel_context": "...", "cached_at": ...}
_parcel_cache: Dict[str, Dict[str, Any]] = {}


def get_session_logger(session_id: str):
    """
    Get or create a logger for a specific session.
    
    Each session gets its own log file: logs/api_{session_id}.log
    """
    if session_id not in _session_loggers:
        _session_loggers[session_id] = setup_logging(
            run_id=f"api_{session_id}",
            result_dir="./logs",
            add_console_handler=True,
            verbose=False,
            allowed_namespaces=("rag_core", "app", "__main__"),
            override=False,  # Append to existing log
        )
    return _session_loggers[session_id]


router = APIRouter()


def _resolve_parcel_data(session_id: str, logger) -> Dict[str, Any]:
    """
    Resolve document URLs and parcel context from KG.
    
    Parses session_id as parcel_id::temp_id, queries KG, caches results.
    Falls back gracefully if session_id doesn't contain '::' (backward compat).
    
    Returns:
        Dict with keys: document_urls, system_message, parcel_context, parcel_id.
        All values are None if session_id is not parcel-format.
    """
    try:
        parcel_id, temp_id = parse_session_id(session_id)
    except ValueError:
        # Not a parcel session — backward compatible, no KG resolution
        return {"document_urls": None, "system_message": None, "parcel_context": None, "parcel_id": None}
    
    # Check cache
    if parcel_id in _parcel_cache:
        cached = _parcel_cache[parcel_id]
        logger.info(f"[API] Parcel cache hit for PFI {parcel_id}")
        return cached
    
    # Query KG
    logger.info(f"[API] Querying KG for PFI {parcel_id}...")
    kg_client = get_kg_client()
    
    document_urls = kg_client.get_document_urls(parcel_id)
    kg_context = kg_client.get_parcel_context(parcel_id)
    parcel_context_str = format_parcel_context(parcel_id, kg_context)
    
    result = {
        "document_urls": document_urls if document_urls else None,
        "system_message": APP_SYSTEM_MESSAGE,
        "parcel_context": parcel_context_str,
        "parcel_id": parcel_id,
    }
    
    # Cache
    _parcel_cache[parcel_id] = result
    logger.info(f"[API] Cached parcel data for PFI {parcel_id}: {len(document_urls)} URLs")
    
    return result


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ChatRequest(BaseModel):
    """Request model for chat endpoints."""
    
    session_id: str = Field(..., description="Unique session identifier for scoping documents and memory")
    message: str = Field(..., description="User's chat message/query")
    document_urls: Optional[List[str]] = Field(
        default=None,
        description="Optional list of PDF URLs to ingest before answering"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top chunks to retrieve")
    use_memory: bool = Field(default=True, description="Whether to use conversation memory")
    citation_style: str = Field(
        default="markdown_link",
        description="Citation format: markdown_link, inline, bracket, footnote"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "a157",
                    "message": "Is the site a priority site?",
                    "document_urls": ["https://example.com/doc.pdf"],
                    "top_k": 5,
                    "use_memory": True,
                    "citation_style": "markdown_link"
                }
            ]
        }
    }


class SourceMetadata(BaseModel):
    """Metadata for a single source citation."""
    
    file: str = Field(..., description="Source filename")
    page: int = Field(..., description="Page number in the document")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    url: str = Field(..., description="Original document URL")
    chunk_index: Optional[int] = Field(default=None, description="Index of chunk within document")
    section_header: Optional[str] = Field(default=None, description="Section header if available")


class TimingInfo(BaseModel):
    """Timing breakdown for the chat operation."""
    
    ingestion: float = Field(default=0.0, description="Time spent on document ingestion (seconds)")
    retrieval: float = Field(default=0.0, description="Time spent on retrieval (seconds)")
    generation: float = Field(default=0.0, description="Time spent on LLM generation (seconds)")
    memory: float = Field(default=0.0, description="Time spent on memory operations (seconds)")
    total: float = Field(default=0.0, description="Total processing time (seconds)")


class ChatResponse(BaseModel):
    """Response model for sync chat endpoint."""
    
    answer: str = Field(..., description="Generated response with inline citations")
    citations: List[str] = Field(default_factory=list, description="Formatted citation strings")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source metadata list")
    timing: TimingInfo = Field(default_factory=TimingInfo, description="Timing breakdown")
    session_id: str = Field(..., description="Echo of the session_id used")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "The site is classified as priority [doc.pdf (p.2)](https://...)...",
                    "citations": ["[doc.pdf (p.2)](https://example.com/doc.pdf)"],
                    "sources": [{"file": "doc.pdf", "page": 2, "chunk_id": "c_1", "url": "https://..."}],
                    "timing": {"ingestion": 1.2, "retrieval": 0.3, "generation": 2.1, "memory": 0.1, "total": 3.7},
                    "session_id": "a157"
                }
            ]
        }
    }


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/chat/")
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    Synchronous chat endpoint with OpenAI-compatible response format.
    
    Processes the user's message with optional document ingestion,
    retrieves relevant context from session-scoped documents,
    and generates a response with citations.
    
    Response follows OpenAI chat completion format with extended fields.
    """
    logger = get_session_logger(request.session_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] /chat/ request: session_id={request.session_id}, message={request.message[:50]}...")
    
    try:
        # Resolve parcel data from KG (if parcel session)
        parcel_data = _resolve_parcel_data(request.session_id, logger)
        
        # KG-resolved URLs override request document_urls
        document_urls = parcel_data["document_urls"] or request.document_urls
        
        result = chat(
            session_id=request.session_id,
            message=request.message,
            document_urls=document_urls,
            top_k=request.top_k,
            use_memory=request.use_memory,
            citation_style=request.citation_style,
            system_message=parcel_data["system_message"],
            parcel_context=parcel_data["parcel_context"],
        )
        
        timing = result.get("timing", {})
        logger.info(f"[{timestamp}] /chat/ completed: total={timing.get('total', 0):.2f}s")
        
        # Return OpenAI-compatible format
        return format_chat_response_openai(result)
        
    except Exception as e:
        logger.error(f"[{timestamp}] /chat/ error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream/")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Async streaming chat endpoint with OpenAI-compatible SSE format.
    
    Response format follows OpenAI's streaming API:
        data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}
        data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}
        ...
        data: [DONE]
    
    The final chunk before [DONE] includes extended fields:
        - citations: List of formatted citation strings
        - sources: List of source metadata
        - timing: Performance breakdown
    """
    logger = get_session_logger(request.session_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] /chat/stream/ request: session_id={request.session_id}, message={request.message[:50]}...")
    
    async def generate():
        """Generator for OpenAI-compatible SSE streaming."""
        import json
        
        formatter = OpenAIStreamFormatter()
        
        try:
            # Resolve parcel data from KG (if parcel session)
            yield formatter.format_sse({"type": "status", "content": "Resolving parcel data..."})
            parcel_data = _resolve_parcel_data(request.session_id, logger)
            document_urls = parcel_data["document_urls"] or request.document_urls
            
            async for chunk in async_chat(
                session_id=request.session_id,
                message=request.message,
                document_urls=document_urls,
                top_k=request.top_k,
                use_memory=request.use_memory,
                citation_style=request.citation_style,
                system_message=parcel_data["system_message"],
                parcel_context=parcel_data["parcel_context"],
            ):
                if chunk.get("type") == "done":
                    timing = chunk.get("timing", {})
                    logger.info(f"[{timestamp}] /chat/stream/ completed: total={timing.get('total', 0):.2f}s")
                yield formatter.format_sse(chunk)
            
            # OpenAI format ends with [DONE]
            yield "data: [DONE]\n\n"
                
        except Exception as e:
            logger.error(f"[{timestamp}] /chat/stream/ error: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            error_chunk = {
                "id": formatter.completion_id,
                "object": "chat.completion.chunk",
                "created": formatter.created,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                "error": {"message": str(e)},
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, clear_parcel_cache: bool = False):
    """
    Clean up a parcel session: remove session index, memory, and cached KG data.
    
    Cleanup steps:
    1. Remove session from IngestionClient's session_index
    2. Delete Mem0 memory collection for the session
    3. Clear cached KG data for the parcel from _parcel_cache
    """
    import logging
    logger = logging.getLogger("app")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] DELETE /chat/sessions/{session_id}")
    
    cleaned = {"session_index": False, "memory": False, "parcel_cache": False}
    qdrant_client_ref = None  # Share Qdrant connection between steps
    
    # 1. Remove from session_index
    try:
        from rag_core.chat.config import load_config
        from rag_core.ingestion.main_client import IngestionClient
        
        config = load_config()
        qdrant_cfg = config.get("qdrant", {})
        models_cfg = config.get("models", {})
        chunking_cfg = config.get("chunking", {})
        
        client = IngestionClient(
            collection_name=qdrant_cfg.get("collection_name", "veris_pdfs"),
            embedding_model=models_cfg.get("embedding_model", "cohere.embed-english-v3"),
            embedding_dim=qdrant_cfg.get("vector_size", 1024),
            chunk_size=chunking_cfg.get("chunk_size", 500),
            chunk_overlap=chunking_cfg.get("overlap", 50),
        )
        qdrant_client_ref = client.qdrant  # Save for step 2
        if session_id in client.session_index:
            del client.session_index[session_id]
            client._save_session_index()
            cleaned["session_index"] = True
            logger.info(f"[{timestamp}] Removed session from session_index")
    except Exception as e:
        logger.warning(f"[{timestamp}] Failed to clean session_index: {e}")
    
    # 2. Delete Mem0 memory collection directly via Qdrant
    # Reuse IngestionClient's Qdrant connection (already tunnel-aware)
    try:
        memory_collection = f"mem0_memory_{session_id.replace('::', '_')}"
        if qdrant_client_ref is None:
            from rag_core.chat.retriever import get_qdrant_client
            qdrant_client_ref = get_qdrant_client()
        collections = [c.name for c in qdrant_client_ref.get_collections().collections]
        if memory_collection in collections:
            qdrant_client_ref.delete_collection(memory_collection)
            cleaned["memory"] = True
            logger.info(f"[{timestamp}] Deleted memory collection: {memory_collection}")
        else:
            logger.info(f"[{timestamp}] Memory collection not found: {memory_collection}")
        logger.info(f"[{timestamp}] Deleted memory for session")
    except Exception as e:
        logger.warning(f"[{timestamp}] Failed to clean memory: {e}")
    
    # 3. Optionally clear parcel cache (default: keep it for other sessions on same parcel)
    if clear_parcel_cache:
        try:
            parcel_id, _ = parse_session_id(session_id)
            if parcel_id in _parcel_cache:
                del _parcel_cache[parcel_id]
                cleaned["parcel_cache"] = True
                logger.info(f"[{timestamp}] Cleared parcel cache for PFI {parcel_id}")
        except ValueError:
            pass  # Not a parcel session, no cache to clear
    
    if not any(cleaned.values()):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {"status": "cleaned", "session_id": session_id, "cleaned": cleaned}


