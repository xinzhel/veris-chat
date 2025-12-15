"""
Test script for VERISRetriever with PID filtering.

This script tests the retriever functionality including:
- Basic retrieval without PID filtering
- Retrieval with PID metadata filtering
- Top-K configuration

Usage: Run in interactive notebook mode or as a script.
"""

import os
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract configuration
EMBEDDING_MODEL = config["models"]["embedding_model"]
QDRANT_COLLECTION = config["qdrant"]["collection_name"]

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

print("=" * 60)
print("VERIS Retriever Test")
print("=" * 60)

# -----------------------------------------------------------------------------
# Setup: Initialize Embedding Model
# -----------------------------------------------------------------------------
print("\n[Setup] Initializing embedding model...")

from llama_index.embeddings.bedrock import BedrockEmbedding

# Build kwargs based on available credentials
embed_kwargs = {"model_name": EMBEDDING_MODEL, "region_name": AWS_REGION}

if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    embed_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
    embed_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    if AWS_SESSION_TOKEN:
        embed_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    print("  Using explicit AWS credentials from .env")
else:
    print("  Using AWS SSO/default credentials")

embed_model = BedrockEmbedding(**embed_kwargs)
print("  ✓ Embedding model initialized")

# -----------------------------------------------------------------------------
# Test 1: Initialize VERISRetriever
# -----------------------------------------------------------------------------
print("\n[1/4] Initializing VERISRetriever...")

from veris_rag.retriever import VERISRetriever

try:
    retriever = VERISRetriever(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION,
        embed_model=embed_model,
    )
    print("  ✓ VERISRetriever initialized successfully")
    
    # Get collection info
    info = retriever.get_collection_info()
    print(f"    - Collection: {info['collection_name']}")
    print(f"    - Points count: {info['points_count']}")
    print(f"    - Vector size: {info['vector_size']}")
except Exception as e:
    print(f"  ✗ VERISRetriever initialization failed: {e}")
    raise

# -----------------------------------------------------------------------------
# Test 2: Basic Retrieval (no PID filter)
# -----------------------------------------------------------------------------
print("\n[2/4] Testing basic retrieval (no PID filter)...")

test_query = "What are the environmental requirements for the site?"

try:
    nodes = retriever.retrieve(query=test_query, top_k=3)
    print(f"  ✓ Retrieved {len(nodes)} nodes")
    
    for i, node in enumerate(nodes):
        print(f"\n  Node {i+1}:")
        print(f"    - Score: {node.score:.4f}")
        print(f"    - Text preview: {node.node.text[:100]}...")
        
        # Print available metadata
        metadata = node.node.metadata
        if metadata:
            print(f"    - Metadata keys: {list(metadata.keys())}")
            if "filename" in metadata:
                print(f"    - Filename: {metadata.get('filename')}")
            if "page_number" in metadata:
                print(f"    - Page: {metadata.get('page_number')}")
            if "PID" in metadata:
                print(f"    - PID: {metadata.get('PID')}")
except Exception as e:
    print(f"  ✗ Basic retrieval failed: {e}")

# -----------------------------------------------------------------------------
# Test 3: Retrieval with PID Filter
# -----------------------------------------------------------------------------
print("\n[3/4] Testing retrieval with PID filter...")

# First, let's find available PIDs from the collection
print("  Checking for available PIDs in collection...")

try:
    # Sample some points to find PIDs
    sample_points = retriever.client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=10,
        with_payload=True,
    )
    
    pids_found = set()
    for point in sample_points[0]:
        if point.payload and "PID" in point.payload:
            pids_found.add(point.payload["PID"])
    
    if pids_found:
        test_pid = list(pids_found)[0]
        print(f"  Found PIDs: {pids_found}")
        print(f"  Testing with PID: {test_pid}")
        
        nodes = retriever.retrieve(query=test_query, pid=test_pid, top_k=3)
        print(f"  ✓ Retrieved {len(nodes)} nodes with PID filter")
        
        for i, node in enumerate(nodes):
            print(f"\n  Node {i+1}:")
            print(f"    - Score: {node.score:.4f}")
            print(f"    - PID: {node.node.metadata.get('PID', 'N/A')}")
            print(f"    - Text preview: {node.node.text[:80]}...")
    else:
        print("  ⚠ No PID metadata found in collection")
        print("  Skipping PID filter test")
except Exception as e:
    print(f"  ✗ PID filter retrieval failed: {e}")

# -----------------------------------------------------------------------------
# Test 4: Top-K Configuration
# -----------------------------------------------------------------------------
print("\n[4/4] Testing top-K configuration...")

try:
    for k in [1, 3, 5]:
        nodes = retriever.retrieve(query=test_query, top_k=k)
        print(f"  ✓ top_k={k}: Retrieved {len(nodes)} nodes")
except Exception as e:
    print(f"  ✗ Top-K test failed: {e}")

print("\n" + "=" * 60)
print("Retriever test completed!")
print("=" * 60)
