"""
FastAPI endpoints for the Veris Chat service.

Endpoints:
- POST /chat/ - Synchronous chat with OpenAI-compatible response
- POST /chat/stream/ - Async streaming chat with OpenAI-compatible SSE
- GET /health - Health check

Start server:
    uvicorn app.chat_api:app --reload

Test commands (copy-paste ready):

# Health check
curl http://localhost:8000/health

# Sync chat
curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" -d '{"session_id": "test", "message": "Hello"}'

# Streaming chat
curl -X POST http://localhost:8000/chat/stream/ -H "Content-Type: application/json" -d '{"session_id": "test", "message": "Hello"}' --no-buffer
"""

import os
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any

# Set AWS_REGION before other imports (us-east-1 required for Opus 4.5 with us.* prefix)
os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from veris_chat.chat.service import (
    chat,
    async_chat,
    OpenAIStreamFormatter,
    format_chat_response_openai,
)
from veris_chat.utils.logger import setup_logging

# Session-specific loggers cache
_session_loggers: Dict[str, any] = {}


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
            allowed_namespaces=("veris_chat", "app", "__main__"),
            override=False,  # Append to existing log
        )
    return _session_loggers[session_id]


app = FastAPI(
    title="Veris Chat API",
    description="Document-grounded conversational system with citation support",
    version="1.0.0",
)


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


@app.post("/chat/")
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
        result = chat(
            session_id=request.session_id,
            message=request.message,
            document_urls=request.document_urls,
            top_k=request.top_k,
            use_memory=request.use_memory,
            citation_style=request.citation_style,
        )
        
        timing = result.get("timing", {})
        logger.info(f"[{timestamp}] /chat/ completed: total={timing.get('total', 0):.2f}s")
        
        # Return OpenAI-compatible format
        return format_chat_response_openai(result)
        
    except Exception as e:
        logger.error(f"[{timestamp}] /chat/ error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream/")
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
            async for chunk in async_chat(
                session_id=request.session_id,
                message=request.message,
                document_urls=request.document_urls,
                top_k=request.top_k,
                use_memory=request.use_memory,
                citation_style=request.citation_style,
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint - redirect to docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
