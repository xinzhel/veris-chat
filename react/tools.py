"""
Tool definitions for the ReAct agent.

Two tools wrapping existing rag_core components:
- ``SearchDocumentsTool``: semantic search over session documents (imports from rag_core)
- ``GetAllChunksTool``: get all chunks for a URL via Qdrant scroll (self-contained)

Usage:
    from react.tools import SearchDocumentsTool, GetAllChunksTool, STATUS_MAP

    tools = [
        SearchDocumentsTool(session_urls=urls, collection_name="veris_pdfs"),
        GetAllChunksTool(collection_name="veris_pdfs"),
    ]
"""

import logging
from typing import Optional, Set

from pydantic import BaseModel, Field
from lits.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Tool name → user-friendly status message (used by AsyncNativeReAct.stream)
STATUS_MAP = {
    "search_documents": "Searching documents...",
    "get_all_chunks": "Reading the full document...",
}


# =============================================================================
# SearchDocumentsTool — semantic search (imports from rag_core)
# =============================================================================

class SearchDocumentsInput(BaseModel):
    query: str = Field(..., description="Semantic search query over session documents")
    top_k: int = Field(5, description="Number of top chunks to return")


class SearchDocumentsTool(BaseTool):
    """Semantic search over session documents. Returns top-K relevant chunks.

    Wraps ``rag_core.chat.retriever.retrieve_with_url_filter()``.
    Requires a pre-built vector index and session URL set.
    """

    name = "search_documents"
    description = (
        "Search session documents by semantic similarity. "
        "Returns the most relevant text chunks for the given query."
    )
    args_schema = SearchDocumentsInput

    def __init__(self, session_urls: Set[str], collection_name: str = "veris_pdfs"):
        """
        Args:
            session_urls: Set of URLs for this session (from session_index).
            collection_name: Qdrant collection name.
        """
        super().__init__(client=None)
        self.session_urls = session_urls
        self.collection_name = collection_name
        self._index = None  # lazy init

    def _get_index(self):
        if self._index is None:
            from rag_core.chat.retriever import get_vector_index
            self._index = get_vector_index(collection_name=self.collection_name)
        return self._index

    def _run(self, query: str, top_k: int = 5) -> str:
        from rag_core.chat.retriever import retrieve_with_url_filter, retrieve_nodes_metadata

        if not self.session_urls:
            return "No documents in session. Ingest documents first."

        index = self._get_index()
        nodes = retrieve_with_url_filter(index, query, self.session_urls, top_k)
        results = retrieve_nodes_metadata(nodes)

        if not results:
            return "No relevant chunks found for the query."

        lines = []
        for i, r in enumerate(results, 1):
            filename = r.get("filename", "unknown")
            page = r.get("page_number", "?")
            score = r.get("score", 0)
            text = r.get("text", "")[:500]
            url = r.get("url", "")
            lines.append(f"[{i}] {filename} (p.{page}, score={score:.3f}, url={url}):\n{text}")
        return "\n\n".join(lines)


# =============================================================================
# GetAllChunksTool — Qdrant scroll by URL (self-contained, no rag_core changes)
# =============================================================================

class GetAllChunksInput(BaseModel):
    url: str = Field(..., description="Document URL to retrieve all chunks from")


class GetAllChunksTool(BaseTool):
    """Get ALL chunks for a specific document URL. Used for full-document summarization.

    Uses Qdrant ``scroll`` with URL payload filter — no embedding computation needed.
    Self-contained: does not modify or add code to ``rag_core/``.
    """

    name = "get_all_chunks"
    description = (
        "Get all text chunks of a specific document by its URL. "
        "Returns all chunks in reading order."
    )
    args_schema = GetAllChunksInput

    def __init__(self, collection_name: str = "veris_pdfs"):
        """
        Args:
            collection_name: Qdrant collection name.
        """
        super().__init__(client=None)
        self.collection_name = collection_name
        self._qdrant_client = None  # lazy init

    def _get_qdrant_client(self):
        if self._qdrant_client is None:
            from rag_core.chat.retriever import get_qdrant_client
            self._qdrant_client = get_qdrant_client()
        return self._qdrant_client

    def _run(self, url: str) -> str:
        try:
            from qdrant_client.http import models as qdrant_models
        except ImportError:
            return "Error: qdrant-client not installed"

        client = self._get_qdrant_client()

        url_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="url",
                    match=qdrant_models.MatchValue(value=url),
                )
            ]
        )

        all_chunks = []
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=url_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                payload = r.payload or {}
                all_chunks.append({
                    "chunk_index": payload.get("chunk_index", 0),
                    "filename": payload.get("filename"),
                    "page_number": payload.get("page_number"),
                    "text": payload.get("text", ""),
                })
            if offset is None:
                break

        if not all_chunks:
            return f"No chunks found for URL: {url}"

        # Sort by chunk_index for reading order
        all_chunks.sort(key=lambda c: c.get("chunk_index", 0) or 0)

        lines = []
        for c in all_chunks:
            filename = c.get("filename", "unknown")
            page = c.get("page_number", "?")
            text = c.get("text", "")
            lines.append(f"[{filename} p.{page}]\n{text}")

        logger.info(f"[GetAllChunks] {len(all_chunks)} chunks for {url}")
        return "\n\n".join(lines)
