"""
Session-scoped retrieval utilities for VERIS RAG.

Thin utility functions using LlamaIndex directly for session-filtered
vector retrieval from Qdrant.

Usage (high-level):
    from veris_chat.chat.retriever import retrieve_for_session
    
    # Single function that coordinates everything
    results = retrieve_for_session(query="What is the site status?", session_id="a157", top_k=5)
    for r in results:
        print(f"{r['filename']} (p.{r['page_number']}): {r['text'][:100]}...")

Usage (low-level):
    from veris_chat.chat.retriever import get_vector_index, retrieve_with_session_filter, retrieve_nodes_metadata
    
    index = get_vector_index()  # 1. Create index (reusable)
    nodes = retrieve_with_session_filter(index, "query", "a157", top_k=5)  # 2. Retrieve
    results = retrieve_nodes_metadata(nodes)  # 3. Extract citation metadata
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

from llama_index.core import Settings
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore

try:
    import qdrant_client
    from qdrant_client.http import models as qdrant_models
except ImportError:
    raise ImportError(
        "qdrant-client is not installed. Install with: pip install qdrant-client"
    )

from veris_chat.chat.config import load_config, get_bedrock_kwargs


def get_qdrant_client(
    storage_path: str = "./qdrant_local",
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> qdrant_client.QdrantClient:
    """
    Build a Qdrant client instance.
    
    Priority: url param > QDRANT_URL env var > local storage_path
    
    Args:
        storage_path: Path for local embedded Qdrant storage (used only if no cloud URL).
        url: Qdrant cloud URL (optional, overrides QDRANT_URL env var).
        api_key: Qdrant API key (optional, overrides QDRANT_API_KEY env var).
        
    Returns:
        QdrantClient instance configured for cloud or local storage.
        
    Example:
        # Use Qdrant Cloud (from .env: QDRANT_URL, QDRANT_API_KEY)
        client = get_qdrant_client()
        
        # Use local storage (only works if QDRANT_URL is not set in .env)
        client = get_qdrant_client(storage_path="./qdrant_local_test")
        
        # Force local storage by passing empty url
        client = get_qdrant_client(url="", storage_path="./my_local_db")
        
        # Explicit cloud credentials
        client = get_qdrant_client(url="https://xyz.qdrant.io", api_key="my_key")
    """
    qdrant_url = url if url is not None else os.getenv("QDRANT_URL")
    qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")
    
    if qdrant_url:
        logger.info(f"[RETRIEVER] Connecting to Qdrant cloud at {qdrant_url}")
        return qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    logger.info(f"[RETRIEVER] Using local Qdrant storage at {storage_path}")
    return qdrant_client.QdrantClient(path=storage_path)


def get_vector_index(
    collection_name: Optional[str] = None,
    embed_model=None,
    storage_path: str = "./qdrant_local",
) -> VectorStoreIndex:
    """
    Return a VectorStoreIndex connected to Qdrant.
    
    Args:
        collection_name: Qdrant collection name. If None, loads from config.yaml.
        embed_model: LlamaIndex embedding model. If None, creates BedrockEmbedding from config.
        storage_path: Path for local Qdrant storage if not using cloud.
        
    Returns:
        VectorStoreIndex connected to the Qdrant vector store.
    """
    logger.info("[RETRIEVER] Creating VectorStoreIndex")
    config = load_config()
    
    if collection_name is None:
        collection_name = config["qdrant"].get("collection_name", "veris_pdfs")
    logger.info(f"[RETRIEVER] Using collection: {collection_name}")
    
    # Set up embedding model if not provided
    if embed_model is None:
        from llama_index.embeddings.bedrock import BedrockEmbedding
        bedrock_kwargs = get_bedrock_kwargs(config)
        embed_model = BedrockEmbedding(
            model_name=config["models"].get("embedding_model", "cohere.embed-english-v3"),
            **bedrock_kwargs,
        )
        logger.info(f"[RETRIEVER] Initialized BedrockEmbedding: {config['models'].get('embedding_model')}")
    
    Settings.embed_model = embed_model
    
    # Build Qdrant client
    client = get_qdrant_client(
        storage_path=storage_path,
        url=config["qdrant"].get("url"),
        api_key=config["qdrant"].get("api_key"),
    )
    
    # Create vector store and index
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    
    logger.info("[RETRIEVER] VectorStoreIndex created successfully")
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def retrieve_with_session_filter(
    index: VectorStoreIndex,
    query: str,
    session_id: str,
    top_k: int = 5,
) -> List[NodeWithScore]:
    """
    Retrieve relevant chunks filtered by session_id.
    
    Args:
        index: VectorStoreIndex connected to Qdrant.
        query: User query string.
        session_id: Session identifier to filter chunks.
        top_k: Number of top results to return.
        
    Returns:
        List of NodeWithScore objects containing retrieved chunks with scores.
    """
    logger.info(f"[RETRIEVER] Retrieving with session_id={session_id}, top_k={top_k}")
    logger.debug(f"[RETRIEVER] Query: {query[:100]}...")
    
    # Build Qdrant filter for session_id
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
    
    nodes = retriever.retrieve(query)
    logger.info(f"[RETRIEVER] Retrieved {len(nodes)} nodes for session_id={session_id}")
    return nodes


def retrieve_nodes_metadata(nodes: List[NodeWithScore]) -> List[dict]:
    """
    Extract metadata from retrieved nodes for citation purposes.
    
    Args:
        nodes: List of NodeWithScore from retrieval.
        
    Returns:
        List of dicts with metadata: filename, page_number, url, chunk_index, text, score.
    """
    logger.debug(f"[RETRIEVER] Extracting metadata from {len(nodes)} nodes")
    results = []
    for node in nodes:
        metadata = node.node.metadata or {}
        results.append({
            "filename": metadata.get("filename"),
            "page_number": metadata.get("page_number"),
            "url": metadata.get("url"),
            "chunk_index": metadata.get("chunk_index"),
            "chunk_id": metadata.get("chunk_id"),
            "section_header": metadata.get("section_header"),
            "text": node.node.get_content(),
            "score": node.score,
        })
    logger.debug(f"[RETRIEVER] Extracted metadata for {len(results)} nodes")
    return results


def retrieve_for_session(
    query: str,
    session_id: str,
    top_k: int = 5,
    collection_name: Optional[str] = None,
    embed_model=None,
    storage_path: str = "./qdrant_local",
) -> List[dict]:
    """
    High-level retrieval function that coordinates index creation, retrieval, and metadata extraction.
    
    This is the main entry point for session-scoped retrieval, combining:
    1. get_vector_index() - create index connected to Qdrant
    2. retrieve_with_session_filter() - retrieve nodes filtered by session_id
    3. retrieve_nodes_metadata() - extract citation metadata
    
    Args:
        query: User query string.
        session_id: Session identifier to filter chunks.
        top_k: Number of top results to return.
        collection_name: Qdrant collection name. If None, loads from config.yaml.
        embed_model: LlamaIndex embedding model. If None, creates from config.
        storage_path: Path for local Qdrant storage if not using cloud.
        
    Returns:
        List of dicts with citation metadata: filename, page_number, url, chunk_index, text, score.
        
    Example:
        results = retrieve_for_session(
            query="What is the site status?",
            session_id="a157",
            top_k=5,
        )
        for r in results:
            print(f"{r['filename']} (p.{r['page_number']}): {r['text'][:100]}...")
    """
    logger.info(f"[RETRIEVER] retrieve_for_session: session_id={session_id}, top_k={top_k}")
    
    # 1. Get vector index
    index = get_vector_index(
        collection_name=collection_name,
        embed_model=embed_model,
        storage_path=storage_path,
    )
    
    # 2. Retrieve with session filter
    nodes = retrieve_with_session_filter(
        index=index,
        query=query,
        session_id=session_id,
        top_k=top_k,
    )
    
    # 3. Extract metadata for citations
    return retrieve_nodes_metadata(nodes)


def format_citations(
    source_nodes: List[dict],
    style: str = "markdown_link",
) -> List[str]:
    """
    Format citation metadata into human-readable citation strings.
    
    Supports four citation styles:
    - markdown_link (default): "[filename (p.X)](url)" - clickable markdown links
    - inline: "as noted in filename.pdf (p. 12)"
    - bracket: "[filename.pdf, p. 12]"
    - footnote: "[1] filename.pdf, page 12, chunk 5"
    
    Args:
        source_nodes: List of dicts with citation metadata (from retrieve_nodes_metadata 
                      or extract_source_metadata). Expected keys: filename, page_number, 
                      url, chunk_index, chunk_id.
        style: Citation style - "markdown_link", "inline", "bracket", or "footnote".
        
    Returns:
        List of formatted citation strings.
        
    Example:
        results = retrieve_for_session(query="...", session_id="a157")
        
        # Markdown links (default) - clickable in frontend
        citations = format_citations(results, style="markdown_link")
        # ['[OL000071228.pdf (p.1)](https://...)', '[OL000073004.pdf (p.2)](https://...)']
        
        citations = format_citations(results, style="inline")
        # ['as noted in OL000071228.pdf (p. 1)', 'as noted in OL000073004.pdf (p. 2)']
        
        citations = format_citations(results, style="bracket")
        # ['[OL000071228.pdf, p. 1]', '[OL000073004.pdf, p. 2]']
        
        citations = format_citations(results, style="footnote")
        # ['[1] OL000071228.pdf, page 1, chunk 0', '[2] OL000073004.pdf, page 2, chunk 1']
    """
    logger.debug(f"[RETRIEVER] Formatting {len(source_nodes)} citations with style={style}")
    
    formatted = []
    for i, node in enumerate(source_nodes, start=1):
        filename = node.get("filename", "unknown")
        page = node.get("page_number", "?")
        url = node.get("url", "")
        chunk_idx = node.get("chunk_index", node.get("chunk_id", "?"))
        
        if style == "markdown_link":
            # Clickable markdown link: [filename (p.X)](url)
            if url:
                citation = f"[{filename} (p.{page})]({url})"
            else:
                citation = f"[{filename} (p.{page})]"
        elif style == "inline":
            citation = f"as noted in {filename} (p. {page})"
        elif style == "bracket":
            citation = f"[{filename}, p. {page}]"
        elif style == "footnote":
            citation = f"[{i}] {filename}, page {page}, chunk {chunk_idx}"
        else:
            logger.warning(f"[RETRIEVER] Unknown citation style '{style}', defaulting to markdown_link")
            if url:
                citation = f"[{filename} (p.{page})]({url})"
            else:
                citation = f"[{filename} (p.{page})]"
        
        formatted.append(citation)
    
    logger.debug(f"[RETRIEVER] Formatted {len(formatted)} citations")
    return formatted


def get_session_memory(
    session_id: str,
    search_msg_limit: int = 5,
) -> "Mem0Memory":
    """
    Return a Mem0Memory instance configured for the given session.
    
    Uses Mem0Memory.from_config() with Qdrant as the vector store backend,
    enabling session-scoped conversation memory that persists across requests.
    
    Note: AWS_REGION must be set before importing mem0 modules. This is due to
    a bug in Mem0's factory which uses BaseLlmConfig instead of AWSBedrockConfig,
    preventing aws_region from being passed in the config dict.
    
    Args:
        session_id: Session identifier used as user_id in Mem0 context.
        search_msg_limit: Number of recent messages to use for memory search context.
        
    Returns:
        Mem0Memory instance configured with session context.
        
    Example:
        # Set AWS_REGION before importing
        import os
        os.environ["AWS_REGION"] = "ap-southeast-2"
        
        from veris_chat.chat.retriever import get_session_memory
        memory = get_session_memory("a157")
        
        # Add a message to memory
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        memory.put(ChatMessage(role=MessageRole.USER, content="Hello"))
        
        # Get chat history with memory context
        messages = memory.get(input="What did we discuss?")
    """
    from veris_chat.utils.memory import Mem0Memory
    
    logger.info(f"[MEMORY] Creating Mem0Memory for session_id={session_id}")
    
    config = load_config()
    
    # Build Mem0 config using Qdrant as vector store
    qdrant_url = config["qdrant"].get("url") or os.getenv("QDRANT_URL", "")
    qdrant_api_key = config["qdrant"].get("api_key") or os.getenv("QDRANT_API_KEY", "")
    
    # Get Bedrock kwargs for LLM and embedder
    bedrock_kwargs = get_bedrock_kwargs(config)
    aws_region = bedrock_kwargs.get("region_name", "ap-southeast-2")
    
    # IMPORTANT: Set AWS_REGION env var for Mem0's Bedrock integration
    # Mem0's factory uses BaseLlmConfig which doesn't accept aws_region param,
    # so we must set it via environment variable. This should be set BEFORE
    # importing mem0 modules for best results.
    os.environ["AWS_REGION"] = aws_region
    
    # Mem0 config structure for local or cloud Qdrant
    if qdrant_url:
        logger.info(f"[MEMORY] Using Qdrant cloud for memory: {qdrant_url}")
        mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "url": qdrant_url,
                    "api_key": qdrant_api_key,
                    "collection_name": f"mem0_memory_{session_id}",
                    "embedding_model_dims": 1024,
                },
            },
            "embedder": {
                "provider": "aws_bedrock",
                "config": {
                    "model": config["models"].get("embedding_model", "cohere.embed-english-v3"),
                },
            },
            "llm": {
                "provider": "aws_bedrock",
                "config": {
                    "model": config["models"].get("memory_llm", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            },
        }
    else:
        logger.info("[MEMORY] Using local Qdrant for memory")
        mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": "./qdrant_local",
                    "collection_name": f"mem0_memory_{session_id}",
                    "embedding_model_dims": 1024,
                },
            },
            "embedder": {
                "provider": "aws_bedrock",
                "config": {
                    "model": config["models"].get("embedding_model", "cohere.embed-english-v3"),
                },
            },
            "llm": {
                "provider": "aws_bedrock",
                "config": {
                    "model": config["models"].get("memory_llm", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            },
        }
    
    # Context uses session_id as user_id for session isolation
    context = {"user_id": session_id}
    
    memory = Mem0Memory.from_config(
        context=context,
        config=mem0_config,
        search_msg_limit=search_msg_limit,
    )
    
    logger.info(f"[MEMORY] Mem0Memory created for session_id={session_id}")
    return memory


def format_citations_for_response(
    source_nodes: List[dict],
    style: str = "markdown_link",
) -> dict:
    """
    Format citations for API response, including both formatted strings and raw metadata.
    
    Args:
        source_nodes: List of dicts with citation metadata.
        style: Citation style - "markdown_link" (default), "inline", "bracket", or "footnote".
        
    Returns:
        Dict with:
        - citations: List of formatted citation strings
        - sources: List of raw source metadata dicts (file, page, chunk_id, url, markdown_link)
        
    Example:
        results = retrieve_for_session(query="...", session_id="a157")
        citation_data = format_citations_for_response(results, style="markdown_link")
        # {
        #     "citations": ["[doc.pdf (p.1)](https://...)", ...],
        #     "sources": [
        #         {
        #             "file": "doc.pdf", 
        #             "page": 1, 
        #             "chunk_id": "c_0", 
        #             "url": "https://...",
        #             "markdown_link": "[doc.pdf (p.1)](https://...)"
        #         }, 
        #         ...
        #     ]
        # }
    """
    formatted = format_citations(source_nodes, style=style)
    
    sources = []
    for node in source_nodes:
        filename = node.get("filename", "unknown")
        page = node.get("page_number", "?")
        url = node.get("url", "")
        
        # Always include markdown_link for frontend convenience
        if url:
            markdown_link = f"[{filename} (p.{page})]({url})"
        else:
            markdown_link = f"[{filename} (p.{page})]"
        
        sources.append({
            "file": filename,
            "page": page,
            "chunk_id": node.get("chunk_id"),
            "url": url,
            "markdown_link": markdown_link,
        })
    
    return {
        "citations": formatted,
        "sources": sources,
    }
