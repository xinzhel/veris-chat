"""
RAG Retriever with PID filtering for VERIS documents.

This module provides a retriever that performs semantic search on Qdrant
with metadata filtering by PID (Project ID).

Usage:
    from veris_rag.retriever import VERISRetriever
    
    retriever = VERISRetriever(
        qdrant_url="...",
        qdrant_api_key="...",
        collection_name="veris_pdfs",
        embed_model=embed_model,
    )
    nodes = retriever.retrieve(query="What is the site status?", pid="P123", top_k=5)
"""

import os
from typing import List, Optional

import qdrant_client
from qdrant_client.http import models as qdrant_models

from llama_index.core import Settings
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore


class VERISRetriever:
    """
    Retriever for VERIS documents with PID-based metadata filtering.
    
    Uses LlamaIndex VectorStoreIndex with QdrantVectorStore backend.
    Supports filtering by PID metadata field and configurable top-K retrieval.
    
    Args:
        qdrant_url: URL of the Qdrant server.
        qdrant_api_key: API key for Qdrant authentication.
        collection_name: Name of the Qdrant collection.
        embed_model: LlamaIndex embedding model instance.
    """
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
        embed_model,
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embed_model = embed_model
        
        # Initialize Qdrant client
        self.client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Set embedding model in Settings
        Settings.embed_model = embed_model
        
        # Create vector store and index
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store
        )
    
    def retrieve(
        self,
        query: str,
        pid: Optional[str] = None,
        top_k: int = 5,
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant document chunks for a query with optional PID filtering.
        
        Args:
            query: The user query string.
            pid: Optional Project ID to filter documents. If None, no filtering applied.
            top_k: Number of top results to return.
            
        Returns:
            List of NodeWithScore objects containing retrieved chunks with scores.
        """
        # Build retriever kwargs
        retriever_kwargs = {"similarity_top_k": top_k}
        
        # Add PID filter if provided
        if pid:
            qdrant_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="PID",
                        match=qdrant_models.MatchValue(value=pid),
                    )
                ]
            )
            retriever_kwargs["vector_store_kwargs"] = {"qdrant_filters": qdrant_filter}
        
        # Create retriever with kwargs
        retriever = self.index.as_retriever(**retriever_kwargs)
        
        # Perform retrieval
        nodes = retriever.retrieve(query)
        
        return nodes
    
    def get_collection_info(self) -> dict:
        """
        Get information about the Qdrant collection.
        
        Returns:
            Dictionary with collection metadata.
        """
        collection_info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
        }
