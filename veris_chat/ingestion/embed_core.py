"""
embed_core.py

Provides unified embedding utilities for the veris_vectordb pipeline.

Supports:
- AWS Bedrock embedding models (recommended: "cohere.embed-v4")
- HuggingFace SentenceTransformer fallback

Responsibilities:
- Initialize embedding backend
- Compute embeddings for text chunks
- Batch processing
"""

from __future__ import annotations
from typing import List, Optional
import logging
import boto3
import os
import json
from pathlib import Path

import numpy as np
from botocore.exceptions import ClientError, BotoCoreError
from dotenv import load_dotenv

# Optional HF support
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


logger = logging.getLogger(__name__)

# ============================================================
# Embedding Backend: AWS Bedrock (Cohere embed-v4 recommended)
# ============================================================

class BedrockEmbedder:
    """
    Wrapper for AWS Bedrock embedding models.
    
    Supports the cohere.embed-v4:0 model for generating text embeddings.
    
    Attributes:
        model_id: The Bedrock model identifier
        client: Boto3 Bedrock runtime client
        
    Example:
        >>> embedder = BedrockEmbeddings()
        >>> embeddings = embedder.embed_texts(["Hello world", "Test text"])
        >>> print(len(embeddings))  # 2
    """
    
    def __init__(
        self,
        model_id: str = "cohere.embed-english-v3",
        region_name: str = None
    ):
        """
        Initialize the Bedrock embeddings client.
        
        Supports both IAM credentials and AWS SSO login via default credential chain.
        
        Args:
            model_id: Bedrock model ID (default: cohere.embed-english-v3)
            region_name: AWS region. If None, loads from environment
            
        Raises:
            ValueError: If AWS credentials are not configured
            ClientError: If connection to Bedrock fails
        """
        load_dotenv()
        
        # Get AWS credentials from environment (optional for SSO)
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = region_name or os.getenv("AWS_REGION", "us-east-1")
        
        self.model_id = model_id
        
        try:
            # If explicit credentials provided, use them
            if aws_access_key and aws_secret_key:
                self.client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
                logger.info(f"Initialized Bedrock client with IAM credentials, model: {model_id}")
            else:
                # Otherwise use default credential chain (supports SSO)
                self.client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=region
                )
                logger.info(f"Initialized Bedrock client with default credentials (SSO), model: {model_id}")
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
        
        # Probe embedding dimension by embedding a small example
        self.embedding_dim = self._probe_embedding_dim()

    def _probe_embedding_dim(self) -> int:
        text = "test sentence for embedding dimension"
        vec = self.embed([text])[0]
        return len(vec)
    
    def embed_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            input_type: Type of input for Cohere model 
                       ("search_document" or "search_query")
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ClientError: If the API call fails
            ValueError: If the response format is unexpected
        """
        try:
            # Prepare request body for Cohere embed model
            body = json.dumps({
                "texts": [text],
                "input_type": input_type
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract embedding from response
            if 'embeddings' in response_body and len(response_body['embeddings']) > 0:
                return response_body['embeddings'][0]
            else:
                raise ValueError("Unexpected response format from Bedrock API")
                
        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed(
        self,
        texts: List[str],
        input_type: str = "search_document",
        batch_size: int = 96
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of input texts to embed
            input_type: Type of input for Cohere model
            batch_size: Maximum number of texts per API call (Cohere limit: 96)
            
        Returns:
            List of embedding vectors, one per input text
            
        Raises:
            ClientError: If any API call fails
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                body = json.dumps({
                    "texts": batch,
                    "input_type": input_type
                })
                
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                
                response_body = json.loads(response['body'].read())
                
                if 'embeddings' in response_body:
                    all_embeddings.extend(response_body['embeddings'])
                    logger.info(f"Generated embeddings for batch {i//batch_size + 1} "
                              f"({len(batch)} texts)")
                else:
                    raise ValueError("Unexpected response format from Bedrock API")
                    
            except ClientError as e:
                logger.error(f"Bedrock API error for batch {i//batch_size + 1}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
        
        return all_embeddings
    
    def test_connection(self) -> bool:
        """
        Test the connection to Bedrock by generating a simple embedding.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_embedding = self.embed_text("test")
            logger.info("Bedrock connection test successful")
            print(f"  - Embedding dimension: {len(test_embedding)}")
            print(f"  - Sample embedding (first 5 values): {test_embedding[:5]}")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Bedrock connection test failed: {e}")
            return False


# ============================================================
# Embedding Backend: HuggingFace SentenceTransformer
# ============================================================
class HFEmbedder:
    """
    HuggingFace SentenceTransformer embedder.
    """

    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Try: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"[EMBED] Initialized HF embedder: {model_name}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()


# ============================================================
# Unified Embedder Interface
# ============================================================
class Embedder:
    """
    Unified interface for:
        - AWS Bedrock embeddings
        - HF SentenceTransformer embeddings

    Automatically selected based on model name prefix.
    """

    def __init__(self, model: str, region: str = "us-east-1"):
        """
        Args:
            model: embedding model identifier
                    - "cohere.embed-v4" → Bedrock
                    - "sentence-transformers/..." → HF model
            region: AWS region for Bedrock
        """
        self.model_name = model
        self.region_name = region

        if model.startswith("cohere.") or model.startswith("amazon."):
            self.backend = BedrockEmbedder(model, region)
        elif model.startswith("sentence-transformers/"):
            self.backend = HFEmbedder(model)
        else:
            raise ValueError(
                f"Unsupported embedding model: {model}. "
                f"Must start with 'cohere.' or 'amazon.' or 'sentence-transformers/'."
            )

        self.embedding_dim = self.backend.embedding_dim
        logger.info(f"[EMBED] Embedder ready. Dimension={self.embedding_dim}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Public embedding method.
        """
        if not isinstance(texts, list):
            texts = [texts]
        return self.backend.embed(texts)


# ============================================================
# Convenience utility for ingestion pipeline
# ============================================================
def embed_chunks(chunks: List[dict], embedder: Embedder, batch_size: int = 16) -> List[dict]:
    """
    Given chunks = [{"text": "...", metadata...}, ...]
    compute embeddings and append as:
        chunk["embedding"] = [float, float, ...]

    Returns modified list of chunks.
    """

    logger.info(f"[EMBED] Embedding {len(chunks)} chunks (batch_size={batch_size})")

    texts = [ch["text"] for ch in chunks]
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        emb = embedder.embed(batch)
        all_embeddings.extend(emb)

    assert len(all_embeddings) == len(chunks)

    # Attach embeddings to chunks
    for ch, emb in zip(chunks, all_embeddings):
        ch["embedding"] = emb

    logger.info("[EMBED] Completed embedding all chunks.")
    return chunks
