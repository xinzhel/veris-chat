"""
Qdrant client utilities for vector database operations.

Provides helper functions to build and test Qdrant connections,
supporting both local (embedded) and cloud deployments.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
except ImportError:
    raise ImportError(
        "qdrant-client is not installed. "
        "Install with: pip install qdrant-client"
    )

logger = logging.getLogger(__name__)


def build_qdrant_client(
    storage_path: str = "./qdrant_local",
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """
    Build a Qdrant client instance.
    
    Automatically detects whether to use cloud or local storage based on
    environment variables (QDRANT_URL, QDRANT_API_KEY) or provided parameters.
    
    Args:
        storage_path: Path for local embedded Qdrant storage (default: ./qdrant_local)
        url: Qdrant cloud URL (optional, overrides env var)
        api_key: Qdrant API key (optional, overrides env var)
        
    Returns:
        QdrantClient instance configured for cloud or local storage
        
    Example:
        >>> # Use cloud (from environment variables)
        >>> client = build_qdrant_client()
        
        >>> # Use local storage
        >>> client = build_qdrant_client(storage_path="./my_qdrant_db")
        
        >>> # Use cloud with explicit credentials
        >>> client = build_qdrant_client(url="https://xyz.qdrant.io", api_key="key")
    """
    load_dotenv()
    
    # Check for cloud configuration
    qdrant_url = url or os.getenv("QDRANT_URL")
    qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")
    
    if qdrant_url:
        logger.info(f"[QDRANT] Connecting to cloud instance at {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        logger.info(f"[QDRANT] Using local embedded storage at {storage_path}")
        return QdrantClient(path=storage_path)


def test_qdrant_connection(
    storage_path: str = "./qdrant_local",
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """
    Test connection to Qdrant by attempting to list collections.
    
    Args:
        storage_path: Path for local embedded Qdrant storage
        url: Qdrant cloud URL (optional)
        api_key: Qdrant API key (optional)
        
    Returns:
        True if connection successful, False otherwise
        
    Example:
        >>> if test_qdrant_connection():
        ...     print("Qdrant is accessible")
    """
    try:
        client = build_qdrant_client(
            storage_path=storage_path,
            url=url,
            api_key=api_key,
        )
        
        # Attempt to get collections as a connection test
        collections = client.get_collections()
        
        logger.info(f"[QDRANT] Connection successful. Found {len(collections.collections)} collections")
        
        # Print collection info
        if collections.collections:
            print(f"  - Found {len(collections.collections)} collection(s):")
            for col in collections.collections:
                print(f"    • {col.name}")
        else:
            print("  - No collections found (empty database)")
        
        return True
        
    except Exception as e:
        logger.error(f"[QDRANT] Connection test failed: {e}")
        return False
