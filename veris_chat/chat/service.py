"""
Chat service orchestrating document ingestion, retrieval, and citation-grounded generation.

This module provides the main chat function that coordinates:
1. Document ingestion via IngestionClient (if document_urls provided)
2. Session-scoped retrieval via retriever functions
3. Citation-grounded generation via CitationQueryEngine
4. Conversation memory via Mem0Memory

Usage:
    from veris_chat.chat.service import chat
    
    # Chat with document ingestion
    response = chat(
        session_id="user_123",
        message="What is the site status?",
        document_urls=["https://example.com/doc.pdf"],
    )
    print(response["answer"])
    print(response["citations"])
    
    # Follow-up chat (no new documents)
    response = chat(
        session_id="user_123",
        message="Can you elaborate on that?",
    )
    
    # Async streaming chat (OpenAI-compatible format)
    async for chunk in async_chat(session_id="user_123", message="Summarize the document"):
        if chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "done":
            print(f"\\nCitations: {chunk['citations']}")
"""

import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.bedrock_converse import BedrockConverse

from qdrant_client.http import models as qdrant_models

from veris_chat.chat.config import load_config, get_bedrock_kwargs
from veris_chat.chat.retriever import (
    get_vector_index,
    retrieve_with_session_filter,
    format_citations_for_response,
    get_session_memory,
    retrieve_nodes_metadata,
)
from veris_chat.ingestion.main_client import IngestionClient
from veris_chat.utils.citation_query_engine import CitationQueryEngine
from veris_chat.utils.logger import print_timing_summary

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================


def _init_timing() -> Dict[str, float]:
    """Initialize timing dictionary for performance tracking."""
    return {
        "ingestion": 0.0,
        "retrieval": 0.0,
        "generation": 0.0,
        "memory": 0.0,
        "total": 0.0,
    }


def _ingest_documents(
    document_urls: List[str],
    session_id: str,
    config: Dict[str, Any],
) -> float:
    """
    Ingest documents and return elapsed time.
    
    Args:
        document_urls: List of PDF URLs to ingest.
        session_id: Session ID for document scoping.
        config: Application configuration.
        
    Returns:
        Elapsed time in seconds.
    """
    logger.info(f"[SERVICE] Ingesting {len(document_urls)} document(s)...")
    t_start = time.perf_counter()
    
    ingestion_client = _get_ingestion_client(config)
    for url in document_urls:
        try:
            ingestion_client.store(url, session_id=session_id)
            logger.info(f"[SERVICE] Ingested: {url}")
        except Exception as e:
            logger.error(f"[SERVICE] Failed to ingest {url}: {e}")
    
    elapsed = time.perf_counter() - t_start
    logger.info(f"[SERVICE] Ingestion completed in {elapsed:.2f}s")
    return elapsed


def _create_session_retriever(
    session_id: str,
    config: Dict[str, Any],
    embed_model: BedrockEmbedding,
    top_k: int,
):
    """
    Create a retriever with session-scoped filter.
    
    Args:
        session_id: Session ID for filtering.
        config: Application configuration.
        embed_model: Embedding model for vector index.
        top_k: Number of top results to retrieve.
        
    Returns:
        Retriever with session filter applied.
    """
    qdrant_cfg = config.get("qdrant", {})
    index = get_vector_index(
        collection_name=qdrant_cfg.get("collection_name", "veris_pdfs"),
        embed_model=embed_model,
    )
    
    qdrant_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="session_id",
                match=qdrant_models.MatchValue(value=session_id),
            )
        ]
    )
    
    return index.as_retriever(
        similarity_top_k=top_k,
        vector_store_kwargs={"qdrant_filters": qdrant_filter},
    )


def _get_memory_context(
    session_id: str,
    message: str,
    config: Dict[str, Any],
) -> tuple:
    """
    Initialize memory and retrieve context from previous conversations.
    
    Args:
        session_id: Session ID for memory scoping.
        message: User's current message.
        config: Application configuration.
        
    Returns:
        Tuple of (memory_instance, memory_context_str, elapsed_time).
        memory_instance is None if memory fails to initialize.
        memory_context_str is None if no context available.
    """
    t_start = time.perf_counter()
    
    try:
        # Set AWS_REGION for Mem0
        aws_region = config.get("aws", {}).get("region", "ap-southeast-2")
        os.environ["AWS_REGION"] = aws_region
        
        memory = get_session_memory(session_id)
        
        # Store user message in memory first
        user_msg = ChatMessage(role=MessageRole.USER, content=message)
        memory.put(user_msg)
        logger.info(f"[SERVICE] Stored user message in memory")
        
        # Retrieve historical context from memory
        messages_with_context = memory.get(input=message)
        
        # Extract memory context from system message (first message if SYSTEM role)
        memory_context = None
        if messages_with_context and messages_with_context[0].role == MessageRole.SYSTEM:
            memory_context = messages_with_context[0].content
            logger.info(f"[SERVICE] Retrieved memory context: {len(memory_context)} chars")
        else:
            logger.info(f"[SERVICE] No system message in memory response, messages: {len(messages_with_context)}")
        
        elapsed = time.perf_counter() - t_start
        return memory, memory_context, elapsed
        
    except Exception as e:
        logger.warning(f"[SERVICE] Memory initialization failed: {e}")
        elapsed = time.perf_counter() - t_start
        return None, None, elapsed


def _augment_query_with_memory(message: str, memory_context: Optional[str]) -> str:
    """
    Augment query with memory context if available.
    
    Args:
        message: Original user message.
        memory_context: Context from previous conversations (or None).
        
    Returns:
        Augmented query string.
    """
    if memory_context:
        query = f"Context from previous conversations:\n{memory_context}\n\nCurrent question: {message}"
        logger.info(f"[SERVICE] Augmented query with memory context")
        return query
    return message


def _format_citation_response(
    source_nodes: List[Any],
    citation_style: str,
) -> Dict[str, Any]:
    """
    Format citations from source nodes.
    
    Args:
        source_nodes: List of source nodes with metadata.
        citation_style: Citation format style.
        
    Returns:
        Dict with 'citations' and 'sources' keys.
    """
    if source_nodes:
        citations_metadata = retrieve_nodes_metadata(source_nodes)
        return format_citations_for_response(citations_metadata, style=citation_style)
    return {"citations": [], "sources": []}


def _store_assistant_response(memory: Any, response: str) -> None:
    """
    Store assistant response in memory.
    
    Args:
        memory: Memory instance (or None to skip).
        response: Assistant's response text.
    """
    if memory is not None:
        try:
            assistant_msg = ChatMessage(role=MessageRole.ASSISTANT, content=response)
            memory.put(assistant_msg)
            logger.info(f"[SERVICE] Stored assistant response in memory")
        except Exception as e:
            logger.warning(f"[SERVICE] Failed to store assistant response in memory: {e}")

# Module-level cache for expensive resources
_cached_resources: Dict[str, Any] = {}


def _get_models(config: Dict[str, Any]) -> tuple:
    """
    Get or create cached Bedrock embedding and LLM models.
    
    Returns:
        Tuple of (embed_model, llm)
    """
    if "embed_model" not in _cached_resources:
        bedrock_kwargs = get_bedrock_kwargs(config)
        models_cfg = config.get("models", {})
        
        embed_model = BedrockEmbedding(
            model_name=models_cfg.get("embedding_model", "cohere.embed-english-v3"),
            **bedrock_kwargs,
        )
        llm = Bedrock(
            model=models_cfg.get("generation_model", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            **bedrock_kwargs,
        )
        
        _cached_resources["embed_model"] = embed_model
        _cached_resources["llm"] = llm
        
        # Set global settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        logger.info(f"[SERVICE] Initialized models: embed={models_cfg.get('embedding_model')}, llm={models_cfg.get('generation_model')}")
    
    return _cached_resources["embed_model"], _cached_resources["llm"]


def _get_ingestion_client(config: Dict[str, Any]) -> IngestionClient:
    """
    Get or create cached IngestionClient.
    """
    if "ingestion_client" not in _cached_resources:
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
        _cached_resources["ingestion_client"] = client
        logger.info(f"[SERVICE] Initialized IngestionClient for collection: {qdrant_cfg.get('collection_name')}")
    
    return _cached_resources["ingestion_client"]


def chat(
    session_id: str,
    message: str,
    document_urls: Optional[List[str]] = None,
    top_k: int = 5,
    use_memory: bool = True,
    citation_style: str = "markdown_link",
) -> Dict[str, Any]:
    """
    Orchestrate the full chat flow: ingestion, retrieval, and generation.
    
    Args:
        session_id: Unique session identifier for scoping documents and memory.
        message: User's chat message/query.
        document_urls: Optional list of PDF URLs to ingest before answering.
        top_k: Number of top chunks to retrieve for context.
        use_memory: Whether to use conversation memory (Mem0).
        citation_style: Citation format - "markdown_link", "inline", "bracket", "footnote".
        
    Returns:
        Dict containing:
        - answer: Generated response text with inline citations
        - citations: List of formatted citation strings
        - sources: List of source metadata dicts (file, page, url, etc.)
        - timing: Dict with timing breakdown (ingestion, retrieval, generation)
        - session_id: Echo of the session_id used
        
    Example:
        response = chat(
            session_id="user_123",
            message="What is the site contamination status?",
            document_urls=["https://example.com/report.pdf"],
        )
        print(response["answer"])
        # "The site is classified as priority [OL000071228.pdf (p.2)](https://...)..."
        
        print(response["sources"])
        # [{"file": "OL000071228.pdf", "page": 2, "url": "https://...", ...}]
    """
    logger.info(f"[SERVICE] chat() called: session_id={session_id}, message={message[:50]}...")
    
    timing = _init_timing()
    t_total_start = time.perf_counter()
    
    # Load configuration and initialize models
    config = load_config()
    embed_model, llm = _get_models(config)
    
    # Step 1: Document Ingestion (if URLs provided)
    if document_urls:
        timing["ingestion"] = _ingest_documents(document_urls, session_id, config)
    
    # Step 2: Create retriever and query engine
    logger.info(f"[SERVICE] Creating index and query engine...")
    retriever = _create_session_retriever(session_id, config, embed_model, top_k)
    engine = CitationQueryEngine(
        retriever=retriever,
        llm=llm,
        citation_chunk_size=512,
    )
    
    # Step 3: Memory Context (optional)
    memory = None
    memory_context = None
    if use_memory:
        memory, memory_context, timing["memory"] = _get_memory_context(
            session_id, message, config
        )
    
    query_text = _augment_query_with_memory(message, memory_context)
    
    # Step 4: Execute query
    logger.info(f"[SERVICE] Executing query: {message[:50]}...")
    try:
        response = engine.query(query_text)
        
        # Get timing breakdown from engine
        engine_timing = engine.get_last_timing()
        timing["retrieval"] = engine_timing.get("retrieval_time", 0)
        timing["generation"] = engine_timing.get("generation_time", 0)
        
        answer = str(response)
        source_nodes = response.source_nodes
        logger.info(f"[SERVICE] Query completed, {len(source_nodes)} source nodes")
        
    except Exception as e:
        logger.error(f"[SERVICE] Query failed: {e}")
        answer = f"I apologize, but I encountered an error while processing your query: {str(e)}"
        source_nodes = []
    
    # Step 5: Format Citations
    citation_data = _format_citation_response(source_nodes, citation_style)
    
    # Step 6: Store Assistant Response in Memory
    _store_assistant_response(memory, answer)
    
    timing["total"] = time.perf_counter() - t_total_start
    
    result = {
        "answer": answer,
        "citations": citation_data["citations"],
        "sources": citation_data["sources"],
        "timing": timing,
        "session_id": session_id,
    }
    
    logger.info(f"[SERVICE] chat() completed - Ingestion: {timing['ingestion']:.2f}s, "
                f"Memory: {timing['memory']:.2f}s, Retrieval: {timing['retrieval']:.2f}s, "
                f"Generation: {timing['generation']:.2f}s, Total: {timing['total']:.2f}s")
    return result


def clear_cache():
    """
    Clear cached resources (models, clients).
    
    Useful for testing or when configuration changes.
    """
    global _cached_resources
    _cached_resources = {}
    logger.info("[SERVICE] Cleared cached resources")


# =============================================================================
# ASYNC STREAMING CHAT
# =============================================================================


def _get_streaming_llm(config: Dict[str, Any]) -> BedrockConverse:
    """
    Get BedrockConverse LLM for async streaming.
     
    Note 1: BedrockConverse requires cross-region inference profile for Claude 3.5 Sonnet v2
    in ap-southeast-2. The model ID is configured in config.yaml under models.streaming_model.
    
    Note 2: Cross-region inference profiles (model IDs starting with 'us.', 'eu.', etc.)
    require NOT passing region_name explicitly - let boto3 use env/default credentials
    to allow the cross-region routing to work.
    
    Returns:
        BedrockConverse instance configured for streaming.
    """
    if "streaming_llm" not in _cached_resources:
        models_cfg = config.get("models", {})
        
        # Get streaming model from config (defaults to cross-region inference profile)
        streaming_model = models_cfg.get(
            "streaming_model", 
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        
        # For cross-region inference profiles (us., eu., etc.), do NOT pass region_name
        # This allows boto3 to use the cross-region routing
        if streaming_model.startswith(("us.", "eu.", "ap.")):
            streaming_llm = BedrockConverse(model=streaming_model)
            logger.info(f"[SERVICE] Initialized BedrockConverse (cross-region): {streaming_model}")
        else:
            bedrock_kwargs = get_bedrock_kwargs(config)
            streaming_llm = BedrockConverse(model=streaming_model, **bedrock_kwargs)
            logger.info(f"[SERVICE] Initialized BedrockConverse: {streaming_model}")
        
        _cached_resources["streaming_llm"] = streaming_llm
    
    return _cached_resources["streaming_llm"]


async def async_chat(
    session_id: str,
    message: str,
    document_urls: Optional[List[str]] = None,
    top_k: int = 5,
    use_memory: bool = True,
    citation_style: str = "markdown_link",
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async streaming chat with citation-grounded generation.
    
    This function replicates the exact workflow of `chat()` but streams tokens
    as they are generated by the LLM.
    
    Workflow (same as CitationQueryEngine):
    1. Document ingestion (sync, if URLs provided)
    2. Session-scoped retrieval
    3. Create citation nodes with markdown links
    4. Pack context using CompactAndRefine logic
    5. Format CITATION_QA_TEMPLATE
    6. Stream generation with BedrockConverse.astream_chat()
    
    Args:
        session_id: Unique session identifier for scoping documents and memory.
        message: User's chat message/query.
        document_urls: Optional list of PDF URLs to ingest before answering.
        top_k: Number of top chunks to retrieve for context.
        use_memory: Whether to use conversation memory (Mem0).
        citation_style: Citation format - "markdown_link", "inline", "bracket", "footnote".
        
    Yields:
        Dict with one of:
        - {"type": "token", "content": "..."} for each streamed token
        - {"type": "error", "content": "..."} on error
        - {"type": "done", "answer": "...", "citations": [...], ...} at completion
        
    Example:
        async for chunk in async_chat(session_id="user_123", message="Summarize"):
            if chunk["type"] == "token":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "done":
                print(f"\\nSources: {len(chunk['sources'])}")
    """
    logger.info(f"[SERVICE] async_chat() called: session_id={session_id}, message={message[:50]}...")
    
    timing = _init_timing()
    t_total_start = time.perf_counter()
    
    # Load configuration and initialize models
    config = load_config()
    embed_model, _ = _get_models(config)  # Only need embed_model, not sync llm
    streaming_llm = _get_streaming_llm(config)
    
    # Step 1: Document Ingestion (sync - typically fast or cached)
    if document_urls:
        timing["ingestion"] = _ingest_documents(document_urls, session_id, config)
    
    # Step 2: Create retriever and query engine
    # Use streaming_llm for PromptHelper metadata (context window sizing)
    retriever = _create_session_retriever(session_id, config, embed_model, top_k)
    engine = CitationQueryEngine(
        retriever=retriever,
        llm=streaming_llm,
        citation_chunk_size=512,
    )
    
    # Step 3: Memory Context (optional)
    memory = None
    memory_context = None
    if use_memory:
        memory, memory_context, timing["memory"] = _get_memory_context(
            session_id, message, config
        )
    
    query_text = _augment_query_with_memory(message, memory_context)
    
    # Step 4: Prepare Streaming Context (replicate CitationQueryEngine workflow)
    logger.info(f"[SERVICE] Preparing streaming context...")
    t_retrieval_start = time.perf_counter()
    
    try:
        streaming_context = engine.prepare_streaming_context(query=query_text)
        prompt = streaming_context["prompt"]
        citation_nodes = streaming_context["citation_nodes"]
        
        timing["retrieval"] = time.perf_counter() - t_retrieval_start
        logger.info(f"[SERVICE] Context prepared, {len(citation_nodes)} citation nodes")
        
    except Exception as e:
        logger.error(f"[SERVICE] Failed to prepare streaming context: {e}")
        yield {"type": "error", "content": f"Failed to prepare context: {str(e)}"}
        return
    
    # Step 5: Stream Generation with BedrockConverse
    logger.info(f"[SERVICE] Starting streaming generation...")
    t_generation_start = time.perf_counter()
    
    messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
    full_response = ""
    token_count = 0
    
    try:
        stream = await streaming_llm.astream_chat(messages)
        async for chunk in stream:
            token_count += 1
            delta = chunk.delta
            if delta:
                full_response += delta
                yield {"type": "token", "content": delta}
        
        timing["generation"] = time.perf_counter() - t_generation_start
        logger.info(f"[SERVICE] Streaming completed, {token_count} tokens in {timing['generation']:.2f}s")
        
    except Exception as e:
        logger.error(f"[SERVICE] Streaming generation failed: {e}")
        yield {"type": "error", "content": f"Generation failed: {str(e)}"}
        return
    
    # Step 6: Format Citations
    citation_data = _format_citation_response(citation_nodes, citation_style)
    
    # Step 7: Store Assistant Response in Memory
    _store_assistant_response(memory, full_response)
    
    timing["total"] = time.perf_counter() - t_total_start
    
    # Final completion message with all metadata
    yield {
        "type": "done",
        "answer": full_response,
        "citations": citation_data["citations"],
        "sources": citation_data["sources"],
        "timing": timing,
        "session_id": session_id,
        "token_count": token_count,
    }
    
    logger.info(f"[SERVICE] async_chat() completed - Ingestion: {timing['ingestion']:.2f}s, "
                f"Memory: {timing['memory']:.2f}s, Retrieval: {timing['retrieval']:.2f}s, "
                f"Generation: {timing['generation']:.2f}s, Total: {timing['total']:.2f}s")


# =============================================================================
# OPENAI FORMAT CONVERTERS
# =============================================================================


class OpenAIStreamFormatter:
    """
    Convert async_chat() output to OpenAI-compatible streaming format.
    
    Usage in FastAPI:
        formatter = OpenAIStreamFormatter()
        
        async def stream_generator():
            async for chunk in async_chat(session_id, message):
                yield formatter.format_sse(chunk)
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    """
    
    def __init__(self):
        self.completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.created = int(time.time())
    
    def format_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single async_chat chunk to OpenAI format.
        
        Args:
            chunk: Output from async_chat() generator
            
        Returns:
            OpenAI-compatible chunk dict
        """
        chunk_type = chunk.get("type")
        
        if chunk_type == "token":
            return {
                "id": self.completion_id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk["content"]},
                    "finish_reason": None,
                }],
            }
        
        elif chunk_type == "error":
            return {
                "id": self.completion_id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }],
                "error": {"message": chunk["content"]},
            }
        
        elif chunk_type == "done":
            return {
                "id": self.completion_id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": chunk["answer"]},
                }],
                # Extended fields
                "citations": chunk.get("citations", []),
                "sources": chunk.get("sources", []),
                "timing": chunk.get("timing", {}),
                "session_id": chunk.get("session_id"),
                "usage": {"completion_tokens": chunk.get("token_count", 0)},
            }
        
        # Pass through unknown types
        return chunk
    
    def format_sse(self, chunk: Dict[str, Any]) -> str:
        """
        Format chunk as Server-Sent Event string.
        
        Args:
            chunk: Output from async_chat() generator
            
        Returns:
            SSE-formatted string: "data: {...}\\n\\n"
        """
        import json
        openai_chunk = self.format_chunk(chunk)
        return f"data: {json.dumps(openai_chunk)}\n\n"


def format_chat_response_openai(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert chat() response to OpenAI-compatible format.
    
    Args:
        response: Output from chat() function
        
    Returns:
        OpenAI-compatible completion dict
        
    Example:
        result = chat(session_id="123", message="Hello")
        openai_result = format_chat_response_openai(result)
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response["answer"]},
            "finish_reason": "stop",
        }],
        # Extended fields
        "citations": response.get("citations", []),
        "sources": response.get("sources", []),
        "timing": response.get("timing", {}),
        "session_id": response.get("session_id"),
    }