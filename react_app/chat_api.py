"""
FastAPI endpoints for the ReAct agent.

Endpoints:
- POST /chat/stream/ — Async streaming ReAct chat with SSE
- DELETE /sessions/{session_id} — Archive state, clean session_index

Start server:
    uvicorn main:app --reload

Test commands:
    # Streaming ReAct chat (parcel session)
    curl -X POST http://localhost:8000/react/chat/stream/ \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "433375739::test1", "message": "What is the licence number?"}' \\
      --no-buffer

    # Session cleanup (archives state, keeps parcel cache)
    curl -X DELETE http://localhost:8000/react/sessions/433375739::test1
"""

import os
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from react.loop import react_chat
from rag_core.kg import get_kg_client, format_parcel_context, parse_session_id
from rag_core.utils.logger import setup_logging

import logging
import json

logger = logging.getLogger(__name__)

# =============================================================================
# APPLICATION SYSTEM MESSAGE (same as rag_app — Layer 1, static)
# =============================================================================

APP_SYSTEM_MESSAGE = """You are an environmental assessment assistant for Victorian land parcels.
You answer questions grounded in assessment reports and knowledge graph data.
When referencing information from source documents, cite them using the provided markdown link format.
If the parcel context below indicates "No data found" for a category, that means we confirmed there is no data — not that we failed to check.
You only have access to internal assessment documents. If the user asks about external URLs or documents not in the system, let them know you can only work with the available assessment reports."""

# Parcel cache: parcel_id → {"document_urls": [...], "parcel_context": "..."}
_parcel_cache: Dict[str, Dict[str, Any]] = {}

# Session loggers cache
_session_loggers: Dict[str, Any] = {}

router = APIRouter()


def _get_session_logger(session_id: str):
    if session_id not in _session_loggers:
        _session_loggers[session_id] = setup_logging(
            run_id=f"react_{session_id}",
            result_dir="./logs",
            add_console_handler=True,
            verbose=False,
            allowed_namespaces=("rag_core", "react", "react_app", "lits", "__main__"),
            override=False,
        )
    return _session_loggers[session_id]


def _resolve_parcel_data(session_id: str, log) -> Dict[str, Any]:
    """Resolve document URLs and parcel context from KG (same logic as rag_app)."""
    try:
        parcel_id, temp_id = parse_session_id(session_id)
    except ValueError:
        return {"document_urls": None, "system_message": None, "parcel_context": None}

    if parcel_id in _parcel_cache:
        log.info(f"[REACT_API] Parcel cache hit for PFI {parcel_id}")
        return _parcel_cache[parcel_id]

    log.info(f"[REACT_API] Querying KG for PFI {parcel_id}...")
    kg_client = get_kg_client()
    document_urls = kg_client.get_document_urls(parcel_id)
    kg_context = kg_client.get_parcel_context(parcel_id)
    parcel_context_str = format_parcel_context(parcel_id, kg_context)

    result = {
        "document_urls": document_urls if document_urls else None,
        "system_message": APP_SYSTEM_MESSAGE,
        "parcel_context": parcel_context_str,
    }
    _parcel_cache[parcel_id] = result
    log.info(f"[REACT_API] Cached parcel data for PFI {parcel_id}: {len(document_urls)} URLs")
    return result


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ReactChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID in format parcel_id::temp_id")
    message: str = Field(..., description="User's chat message")
    document_urls: Optional[List[str]] = Field(default=None, description="Optional PDF URLs to ingest")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/chat/stream/")
async def react_stream_endpoint(request: ReactChatRequest):
    """Async streaming ReAct chat with SSE format.

    Response format:
        data: {"type": "status", "content": "Searching documents..."}
        data: {"type": "token", "content": "The licence number is..."}
        data: {"type": "done", "answer": "...", "timing": {...}}
        data: [DONE]
    """
    log = _get_session_logger(request.session_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"[{timestamp}] /react/chat/stream/ session={request.session_id} message={request.message[:50]}...")

    async def generate():
        try:
            yield f"data: {json.dumps({'type': 'status', 'content': 'Resolving parcel data...'})}\n\n"
            parcel_data = _resolve_parcel_data(request.session_id, log)
            kg_urls = parcel_data.get("document_urls")
            if request.document_urls:
                # Frontend explicitly specified URLs — use only those
                document_urls = request.document_urls
            else:
                # No frontend override — use all KG URLs
                document_urls = kg_urls

            yield f"data: {json.dumps({'type': 'status', 'content': 'Ingesting documents...'})}\n\n"
            async for chunk in react_chat(
                session_id=request.session_id,
                message=request.message,
                system_message=parcel_data.get("system_message"),
                parcel_context=parcel_data.get("parcel_context"),
                document_urls=document_urls,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            log.error(f"[{timestamp}] /react/chat/stream/ error: {e}")
            log.error(f"Traceback:\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.delete("/sessions/{session_id}")
async def delete_react_session(session_id: str):
    """Archive state and clean session_index.

    State JSON is renamed with timestamp for analysis.
    Parcel cache is kept (shared across sessions).
    """
    import shutil
    from pathlib import Path
    from rag_core.chat.config import load_config

    log = logging.getLogger("react_app")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned = {"session_index": False, "state_archived": False}

    # 1. Clean session_index
    try:
        from rag_core.chat.service import _get_ingestion_client
        config = load_config()
        client = _get_ingestion_client(config)
        if session_id in client.session_index:
            del client.session_index[session_id]
            client._save_session_index()
            cleaned["session_index"] = True
    except Exception as e:
        log.warning(f"Failed to clean session_index: {e}")

    # 2. Archive state JSON (rename with timestamp, don't delete)
    try:
        react_cfg = config.get("react", {})
        checkpoint_dir = react_cfg.get("checkpoint_dir", "data/chat_state")
        state_file = Path(checkpoint_dir) / f"{session_id.replace('::', '__')}.json"
        if state_file.exists():
            archive_name = f"{session_id.replace('::', '__')}__{timestamp}.json"
            archive_path = state_file.parent / archive_name
            state_file.rename(archive_path)
            cleaned["state_archived"] = True
            log.info(f"Archived state: {archive_path}")
    except Exception as e:
        log.warning(f"Failed to archive state: {e}")

    if not any(cleaned.values()):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {"status": "cleaned", "session_id": session_id, "cleaned": cleaned}
