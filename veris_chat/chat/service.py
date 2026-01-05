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
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

from veris_chat.chat.config import load_config, get_bedrock_kwargs
from veris_chat.chat.retriever import (
    get_vector_index,
    retrieve_with_session_filter,
    format_citations_for_response,
    get_session_memory,
)
from veris_chat.ingestion.main_client import IngestionClient
from veris_chat.utils.citation_query_engine import CitationQueryEngine

logger = logging.getLogger(__name__)

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
    
    timing = {
        "ingestion": 0.0,
        "retrieval": 0.0,
        "generation": 0.0,
        "total": 0.0,
    }
    t_total_start = time.perf_counter()
    
    # Load configuration
    config = load_config()
    
    # Initialize models
    embed_model, llm = _get_models(config)
    
    # -------------------------------------------------------------------------
    # Step 1: Document Ingestion (if URLs provided)
    # -------------------------------------------------------------------------
    if document_urls:
        logger.info(f"[SERVICE] Ingesting {len(document_urls)} document(s)...")
        t_ingest_start = time.perf_counter()
        
        ingestion_client = _get_ingestion_client(config)
        for url in document_urls:
            try:
                ingestion_client.store(url, session_id=session_id)
                logger.info(f"[SERVICE] Ingested: {url}")
            except Exception as e:
                logger.error(f"[SERVICE] Failed to ingest {url}: {e}")
        
        timing["ingestion"] = time.perf_counter() - t_ingest_start
        logger.info(f"[SERVICE] Ingestion completed in {timing['ingestion']:.2f}s")
    
    # -------------------------------------------------------------------------
    # Step 2: Session-Scoped Retrieval + Citation-Grounded Generation
    # -------------------------------------------------------------------------
    logger.info(f"[SERVICE] Creating index and query engine...")
    
    qdrant_cfg = config.get("qdrant", {})
    index = get_vector_index(
        collection_name=qdrant_cfg.get("collection_name", "veris_pdfs"),
        embed_model=embed_model,
    )
    
    # Create CitationQueryEngine with session filter
    from qdrant_client.http import models as qdrant_models
    
    qdrant_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="session_id",
                match=qdrant_models.MatchValue(value=session_id),
            )
        ]
    )
    
    # Create retriever with session filter
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        vector_store_kwargs={"qdrant_filters": qdrant_filter},
    )
    
    engine = CitationQueryEngine(
        retriever=retriever,
        llm=llm,
        citation_chunk_size=512,
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Query with Memory Context (optional)
    # -------------------------------------------------------------------------
    query_text = message
    memory_context = None
    
    if use_memory:
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
            # memory.get() returns chat history augmented with long-term memory
            messages_with_context = memory.get(input=message)
            
            # Extract memory context from system message (first message if SYSTEM role)
            if messages_with_context and messages_with_context[0].role == MessageRole.SYSTEM:
                memory_context = messages_with_context[0].content
                logger.info(f"[SERVICE] Retrieved memory context: {len(memory_context)} chars")
                logger.debug(f"[SERVICE] Memory context content: {memory_context[:500]}...")
            else:
                logger.info(f"[SERVICE] No system message in memory response, messages: {len(messages_with_context)}")
                for i, msg in enumerate(messages_with_context[:3]):
                    logger.debug(f"[SERVICE] Message {i}: role={msg.role}, content={str(msg.content)[:100]}...")
            
        except Exception as e:
            logger.warning(f"[SERVICE] Memory initialization failed, continuing without memory: {e}")
            memory = None
    else:
        memory = None
    
    # Build query with memory context if available
    if memory_context:
        # Prepend memory context to help the LLM understand user preferences/history
        query_text = f"Context from previous conversations:\n{memory_context}\n\nCurrent question: {message}"
        logger.info(f"[SERVICE] Augmented query with memory context")
        logger.debug(f"[SERVICE] Augmented query (first 500 chars): {query_text[:500]}...")
    
    # Execute query
    logger.info(f"[SERVICE] Executing query: {message[:50]}...")
    t_query_start = time.perf_counter()
    
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
    
    # -------------------------------------------------------------------------
    # Step 4: Format Citations
    # -------------------------------------------------------------------------
    from veris_chat.chat.retriever import retrieve_nodes_metadata
    
    if source_nodes:
        citations_metadata = retrieve_nodes_metadata(source_nodes)
        citation_data = format_citations_for_response(citations_metadata, style=citation_style)
    else:
        citation_data = {"citations": [], "sources": []}
    
    # -------------------------------------------------------------------------
    # Step 5: Store Assistant Response in Memory
    # -------------------------------------------------------------------------
    if memory is not None:
        try:
            assistant_msg = ChatMessage(role=MessageRole.ASSISTANT, content=answer)
            memory.put(assistant_msg)
            logger.info(f"[SERVICE] Stored assistant response in memory")
        except Exception as e:
            logger.warning(f"[SERVICE] Failed to store assistant response in memory: {e}")
    
    timing["total"] = time.perf_counter() - t_total_start
    
    result = {
        "answer": answer,
        "citations": citation_data["citations"],
        "sources": citation_data["sources"],
        "timing": timing,
        "session_id": session_id,
    }
    
    logger.info(f"[SERVICE] chat() completed in {timing['total']:.2f}s")
    return result


def clear_cache():
    """
    Clear cached resources (models, clients).
    
    Useful for testing or when configuration changes.
    """
    global _cached_resources
    _cached_resources = {}
    logger.info("[SERVICE] Cleared cached resources")
