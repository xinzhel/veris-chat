"""
Test script for session-scoped retrieval using LlamaIndex.

Tests:
1. Get session index from Qdrant
2. Retrieve with session_id filter
3. Verify retrieved nodes have correct session_id

Prerequisites: Run test_ingestion.py first to populate test data.

Usage: Run in interactive notebook mode or as a script.
"""

import sys
sys.path.insert(0, ".")

from veris_chat.chat.config import load_config, get_bedrock_kwargs
from veris_chat.utils.logger import setup_logging

# Setup logging
setup_logging(
    run_id="test_session_retrieval",
    result_dir="./logs",
    add_console_handler=True,
    verbose=True,
    allowed_namespaces=("veris_chat", "__main__"),
)

print("=" * 60)
print("Session-Scoped Retrieval Test")
print("=" * 60)

# -----------------------------------------------------------------------------
# Load configuration
# -----------------------------------------------------------------------------
print("\n[Setup] Loading configuration...")
config = load_config()

models_cfg = config["models"]
qdrant_cfg = config["qdrant"]

print(f"  Embedding model: {models_cfg.get('embedding_model')}")
print(f"  Collection name: {qdrant_cfg.get('collection_name')}")

# Test constants (must match test_ingestion.py)
TEST_COLLECTION = "veris_pdfs_test"
TEST_SESSION_ID = "test_session_001"
TEST_STORAGE_PATH = "./qdrant_local_test"

print(f"  Test collection: {TEST_COLLECTION}")
print(f"  Test session_id: {TEST_SESSION_ID}")

# -----------------------------------------------------------------------------
# Initialize embedding model
# -----------------------------------------------------------------------------
print("\n[1/4] Initializing embedding model...")

from llama_index.embeddings.bedrock import BedrockEmbedding

bedrock_kwargs = get_bedrock_kwargs(config)
embed_model = BedrockEmbedding(
    model_name=models_cfg.get("embedding_model", "cohere.embed-english-v3"),
    **bedrock_kwargs,
)
print("  ✓ BedrockEmbedding initialized")

# -----------------------------------------------------------------------------
# Test get_session_index
# -----------------------------------------------------------------------------
print("\n[2/4] Testing get_session_index...")

from veris_chat.chat.retriever import (
    get_vector_index,
    retrieve_with_session_filter,
    retrieve_nodes_metadata,
)

index = get_vector_index(
    collection_name=TEST_COLLECTION,
    embed_model=embed_model,
    storage_path=TEST_STORAGE_PATH,
)

print(f"  ✓ VectorStoreIndex created for collection: {TEST_COLLECTION}")
print(f"  Index type: {type(index).__name__}")

# -----------------------------------------------------------------------------
# Test retrieve_with_session_filter
# -----------------------------------------------------------------------------
print("\n[3/4] Testing retrieve_with_session_filter...")

query = "What is the purpose of this document?"
print(f"  Query: {query}")

nodes = retrieve_with_session_filter(
    index=index,
    query=query,
    session_id=TEST_SESSION_ID,
    top_k=5,
)

print(f"  ✓ Retrieved {len(nodes)} nodes")

if not nodes:
    print("  ⚠ No nodes retrieved. Make sure test_ingestion.py was run first.")
else:
    # Display retrieved nodes
    for i, node in enumerate(nodes[:3], 1):
        metadata = node.node.metadata or {}
        print(f"\n  Result {i}:")
        print(f"    Score: {node.score:.4f}")
        print(f"    session_id: {metadata.get('session_id')}")
        print(f"    filename: {metadata.get('filename')}")
        print(f"    page_number: {metadata.get('page_number')}")
        print(f"    chunk_index: {metadata.get('chunk_index')}")
        text_preview = node.node.get_content()[:100]
        print(f"    text: {text_preview}...")

# -----------------------------------------------------------------------------
# Verify session_id filtering
# -----------------------------------------------------------------------------
print("\n[4/4] Verifying session_id filtering...")

# All retrieved nodes should have the correct session_id
all_valid = True
for node in nodes:
    metadata = node.node.metadata or {}
    node_session_id = metadata.get("session_id")
    if node_session_id != TEST_SESSION_ID:
        print(f"  ✗ session_id mismatch: expected {TEST_SESSION_ID}, got {node_session_id}")
        all_valid = False

if all_valid and nodes:
    print(f"  ✓ All {len(nodes)} nodes have correct session_id: {TEST_SESSION_ID}")
elif not nodes:
    print("  ⚠ No nodes to verify (collection may be empty)")
else:
    raise AssertionError("session_id filtering verification failed")

# -----------------------------------------------------------------------------
# Test retrieve_nodes_metadata helper
# -----------------------------------------------------------------------------
print("\n[Bonus] Testing retrieve_nodes_metadata helper...")

if nodes:
    metadata_list = retrieve_nodes_metadata(nodes)
    print(f"  ✓ Extracted metadata from {len(metadata_list)} nodes")
    
    # Show first result
    if metadata_list:
        first = metadata_list[0]
        print(f"  Sample metadata:")
        print(f"    filename: {first.get('filename')}")
        print(f"    page_number: {first.get('page_number')}")
        print(f"    url: {first.get('url')[:50]}..." if first.get('url') else "    url: None")
        print(f"    score: {first.get('score'):.4f}")

# -----------------------------------------------------------------------------
# Test retrieval with non-existent session_id (should return empty)
# -----------------------------------------------------------------------------
print("\n[Extra] Testing retrieval with non-existent session_id...")

fake_session_id = "non_existent_session_xyz"
empty_nodes = retrieve_with_session_filter(
    index=index,
    query=query,
    session_id=fake_session_id,
    top_k=5,
)

if len(empty_nodes) == 0:
    print(f"  ✓ Correctly returned 0 nodes for non-existent session: {fake_session_id}")
else:
    print(f"  ⚠ Unexpected: returned {len(empty_nodes)} nodes for non-existent session")

print("\n" + "=" * 60)
print("Session-scoped retrieval test completed!")
print("=" * 60)
