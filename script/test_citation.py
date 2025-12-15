"""
Test script for Citation-Grounded Generation using CitationQueryEngine.

Tests:
1. Create CitationQueryEngine with session index
2. Query and extract source nodes with citations
3. Test citation formatting functions (inline/bracket/footnote)

Prerequisites: Run test_ingestion.py first to populate test data.

Usage: Run in interactive notebook mode or as a script.
"""

import sys
sys.path.insert(0, ".")

from veris_chat.chat.config import load_config, get_bedrock_kwargs
from veris_chat.utils.logger import setup_logging

# Setup logging
setup_logging(
    run_id="test_citation",
    result_dir="./logs",
    add_console_handler=True,
    verbose=True,
    allowed_namespaces=("veris_chat", "__main__"),
)

print("=" * 60)
print("Citation-Grounded Generation Test")
print("=" * 60)

# -----------------------------------------------------------------------------
# Load configuration
# -----------------------------------------------------------------------------
print("\n[Setup] Loading configuration...")
config = load_config()

models_cfg = config["models"]
qdrant_cfg = config["qdrant"]

print(f"  Embedding model: {models_cfg.get('embedding_model')}")
print(f"  Generation model: {models_cfg.get('generation_model')}")
print(f"  Collection name: {qdrant_cfg.get('collection_name')}")

# Test constants (must match test_ingestion.py)
TEST_COLLECTION = "veris_pdfs_test"
TEST_SESSION_ID = "test_session_001"
TEST_STORAGE_PATH = "./qdrant_local_test"

print(f"  Test collection: {TEST_COLLECTION}")
print(f"  Test session_id: {TEST_SESSION_ID}")

# -----------------------------------------------------------------------------
# Initialize models
# -----------------------------------------------------------------------------
print("\n[1/6] Initializing Bedrock models...")

from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock
from llama_index.core import Settings

bedrock_kwargs = get_bedrock_kwargs(config)

embed_model = BedrockEmbedding(
    model_name=models_cfg.get("embedding_model", "cohere.embed-english-v3"),
    **bedrock_kwargs,
)
print("  ✓ BedrockEmbedding initialized")

llm = Bedrock(
    model=models_cfg.get("generation_model", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    **bedrock_kwargs,
)
print("  ✓ Bedrock LLM initialized")

# Set global settings
Settings.embed_model = embed_model
Settings.llm = llm

# -----------------------------------------------------------------------------
# Get vector index
# -----------------------------------------------------------------------------
print("\n[2/6] Creating VectorStoreIndex...")

from veris_chat.chat.retriever import get_vector_index

index = get_vector_index(
    collection_name=TEST_COLLECTION,
    embed_model=embed_model,
    storage_path=TEST_STORAGE_PATH,
)
print(f"  ✓ VectorStoreIndex created for collection: {TEST_COLLECTION}")

# -----------------------------------------------------------------------------
# Create CitationQueryEngine
# -----------------------------------------------------------------------------
print("\n[3/6] Creating CitationQueryEngine...")

from veris_chat.utils.citation_query_engine import (
    CitationQueryEngine,
    create_session_citation_engine,
)
from veris_chat.chat.retriever import retrieve_nodes_metadata

# Method 1: Using helper function
engine = create_session_citation_engine(
    index=index,
    llm=llm,
    citation_chunk_size=512,
    similarity_top_k=5,
)
print("  ✓ CitationQueryEngine created via create_session_citation_engine()")

# Method 2: Direct from_args (alternative)
engine_direct = CitationQueryEngine.from_args(
    index=index,
    llm=llm,
    citation_chunk_size=512,
    similarity_top_k=5,
)
print("  ✓ CitationQueryEngine created via from_args() directly")

# -----------------------------------------------------------------------------
# Query with citations
# -----------------------------------------------------------------------------
print("\n[4/6] Querying with CitationQueryEngine...")

query = "What is the purpose of this document?"
print(f"  Query: {query}")

try:
    response = engine.query(query)
    print("  ✓ Query completed successfully")
    
    print(f"\n  Response:")
    print(f"  {'-' * 50}")
    response_text = str(response)
    # Truncate long responses for display
    if len(response_text) > 500:
        print(f"  {response_text}...")
    else:
        print(f"  {response_text}")
    print(f"  {'-' * 50}")
    
    # Extract source nodes
    source_nodes = response.source_nodes
    print(f"\n  Source nodes: {len(source_nodes)}")
    
except Exception as e:
    print(f"  ✗ Query failed: {e}")
    source_nodes = []

# -----------------------------------------------------------------------------
# Extract source metadata
# -----------------------------------------------------------------------------
print("\n[5/6] Extracting source metadata...")

if source_nodes:
    citations = retrieve_nodes_metadata(source_nodes)
    print(f"  ✓ Extracted {len(citations)} citation metadata entries")
    
    for i, c in enumerate(citations[:3], 1):
        print(f"\n  Citation {i}:")
        print(f"    filename: {c.get('filename')}")
        print(f"    page_number: {c.get('page_number')}")
        print(f"    chunk_id: {c.get('chunk_id')}")
        print(f"    score: {c.get('score')}")
        text_preview = c.get('text', '')[:80]
        print(f"    text: {text_preview}...")
else:
    print("  ⚠ No source nodes to extract (query may have failed or returned empty)")
    citations = []

# -----------------------------------------------------------------------------
# Test citation formatting functions
# -----------------------------------------------------------------------------
print("\n[6/6] Testing citation formatting functions...")

from veris_chat.chat.retriever import (
    format_citations,
    format_citations_for_response,
)

if citations:
    # Test inline style
    inline_citations = format_citations(citations, style="inline")
    print(f"\n  Inline style ({len(inline_citations)} citations):")
    for c in inline_citations[:2]:
        print(f"    {c}")
    
    # Test bracket style
    bracket_citations = format_citations(citations, style="bracket")
    print(f"\n  Bracket style ({len(bracket_citations)} citations):")
    for c in bracket_citations[:2]:
        print(f"    {c}")
    
    # Test footnote style
    footnote_citations = format_citations(citations, style="footnote")
    print(f"\n  Footnote style ({len(footnote_citations)} citations):")
    for c in footnote_citations[:2]:
        print(f"    {c}")
    
    # Test format_citations_for_response
    response_data = format_citations_for_response(citations, style="inline")
    print(f"\n  format_citations_for_response output:")
    print(f"    citations: {len(response_data['citations'])} items")
    print(f"    sources: {len(response_data['sources'])} items")
    if response_data['sources']:
        first_source = response_data['sources'][0]
        print(f"    First source: file={first_source.get('file')}, page={first_source.get('page')}")
    
    print("\n  ✓ All citation formatting tests passed")
else:
    print("  ⚠ Skipping formatting tests (no citations available)")
