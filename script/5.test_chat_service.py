"""
Test script for end-to-end chat service.

Tests:
1. Chat with document ingestion
2. Follow-up chat (session continuity)
3. Citation extraction and formatting
4. Timing breakdown


Relationship with Task 7.4 redesign:

This test uses the redesigned architecture but is already compatible because:

Test 1 (ingestion): Calls chat(..., document_urls=TEST_URLS) → internally uses store(url, session_id) which now adds URL to session_index and ingests if not in url_cache ✓

Test 2 (follow-up): Calls chat(..., document_urls=None) with same session_id → internally uses _create_session_retriever() which looks up URLs from session_index ✓

Test 3 (memory): Uses same TEST_SESSION_ID → documents already associated in session_index ✓

Test 4 (citation styles): Same pattern as Test 2 ✓
"""

import os
import sys

# Set AWS_REGION before imports
os.environ["AWS_REGION"] = "ap-southeast-2"

sys.path.insert(0, ".")
from veris_chat.utils.logger import setup_logging, print_timing_summary

# Setup logging
logger = setup_logging(
    run_id="test_chat_service",
    result_dir="./logs",
    add_console_handler=True,
    verbose=True,
    allowed_namespaces=("veris_chat", "__main__"),
)

# Enable DEBUG level for service module to see memory content
import logging
logging.getLogger("veris_chat.chat.service").setLevel(logging.DEBUG)


def log_print(msg: str):
    """Print to console and log to file."""
    # print(msg)
    logger.info(msg)


log_print("=" * 60)
log_print("End-to-End Chat Service Test")
log_print("=" * 60)

# -----------------------------------------------------------------------------

# Test constants
TEST_SESSION_ID = "test_chat_service_001"
TEST_URLS = [
    "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf",
]

log_print(f"  Test session_id: {TEST_SESSION_ID}")
log_print(f"  Test document URLs: {len(TEST_URLS)}")

# -----------------------------------------------------------------------------
# Clear any cached resources from previous runs
# -----------------------------------------------------------------------------
log_print("\n[Cleanup] Clearing cached resources...")
from veris_chat.chat.service import clear_cache
clear_cache()
log_print("  ✓ Cache cleared")

# -----------------------------------------------------------------------------
# Reset collection for clean test state
# -----------------------------------------------------------------------------
log_print("\n[Reset] Resetting Qdrant collection and session index...")
from veris_chat.ingestion.main_client import IngestionClient
from veris_chat.chat.config import load_config
config = load_config()
models_cfg = config["models"]
qdrant_cfg = config["qdrant"]
chunking_cfg = config["chunking"]
client = IngestionClient(
    embedding_model=models_cfg.get("embedding_model"),
    embedding_dim=qdrant_cfg.get("vector_size"),
    chunk_size=chunking_cfg.get("chunk_size", 512),
    chunk_overlap=chunking_cfg.get("overlap", 50),
)

reset_result = client.reset_collection(delete_pdfs=False)
log_print(f"  ✓ Collection reset in {reset_result['elapsed_time']:.2f}s")
log_print(f"  Deleted PDFs: {reset_result['deleted_pdfs_count']}")

# -----------------------------------------------------------------------------
# Test 1: Chat with document ingestion
# -----------------------------------------------------------------------------
log_print("\n[1/4] Testing chat with document ingestion...")

from veris_chat.chat.service import chat

try:
    response = chat(
        session_id=TEST_SESSION_ID,
        message="What is the purpose of this document?",
        document_urls=TEST_URLS,
        top_k=5,
        use_memory=False,  # Disable memory for simpler first test
        citation_style="markdown_link",
    )
    
    log_print("  ✓ Chat completed successfully")
    log_print(f"\n  Answer preview:")
    log_print(f"  {'-' * 50}")
    answer_preview = response["answer"][:500] if len(response["answer"]) > 500 else response["answer"]
    log_print(f"  {answer_preview}...")
    log_print(f"  {'-' * 50}")
    
    log_print(f"\n  Citations: {len(response['citations'])}")
    for i, citation in enumerate(response["citations"][:3], 1):
        log_print(f"    {i}. {citation}")
    
    log_print(f"\n  Sources: {len(response['sources'])}")
    for i, source in enumerate(response["sources"][:3], 1):
        log_print(f"    {i}. {source.get('file')} (p.{source.get('page')})")
    
    log_print(f"\n  Timing:")
    print_timing_summary(response["timing"], compact=True, logger=logger)
    
except Exception as e:
    log_print(f"  ✗ Chat failed: {e}")
    import traceback
    traceback.print_exc()

# -----------------------------------------------------------------------------
# Test 2: Follow-up chat (no new documents)
# -----------------------------------------------------------------------------
log_print("\n[2/4] Testing follow-up chat (no new documents)...")

try:
    response2 = chat(
        session_id=TEST_SESSION_ID,
        message="Can you tell me more about the site conditions mentioned?",
        document_urls=None,  # No new documents
        top_k=5,
        use_memory=False,
        citation_style="markdown_link",
    )
    
    log_print("  ✓ Follow-up chat completed successfully")
    log_print(f"\n  Answer preview:")
    log_print(f"  {'-' * 50}")
    answer_preview = response2["answer"][:500] if len(response2["answer"]) > 500 else response2["answer"]
    log_print(f"  {answer_preview}...")
    log_print(f"  {'-' * 50}")
    
    log_print(f"\n  Timing (no ingestion expected):")
    print_timing_summary(response2["timing"], compact=True, logger=logger)
    
except Exception as e:
    log_print(f"  ✗ Follow-up chat failed: {e}")
    import traceback
    traceback.print_exc()

# -----------------------------------------------------------------------------
# Test 3: Chat with memory enabled
# -----------------------------------------------------------------------------
log_print("\n[3/4] Testing chat with memory enabled...")

# Use the SAME session_id as test 1/2 since documents are already ingested there
# The URL cache doesn't track session_id, so we reuse the session with documents
TEST_SESSION_ID_MEMORY = TEST_SESSION_ID  # Reuse session with ingested documents

try:
    # First message with memory
    response3 = chat(
        session_id=TEST_SESSION_ID_MEMORY,
        message="My name is Alice. What is the purpose of this document?",
        document_urls=None,  # Documents already ingested for this session
        top_k=5,
        use_memory=True,
        citation_style="markdown_link",
    )
    
    log_print("  ✓ Chat with memory completed")
    log_print(f"  Answer preview: {response3['answer']}...")
    
    # Wait a bit for Mem0 to extract facts
    import time
    log_print("  ⏳ Waiting for Mem0 to extract facts (3s)...")
    time.sleep(3)
    
    # Follow-up to test memory recall
    response4 = chat(
        session_id=TEST_SESSION_ID_MEMORY,
        message="What is my name?",
        document_urls=None,
        top_k=5,
        use_memory=True,
        citation_style="markdown_link",
    )
    
    log_print(f"\n  Memory test - asking 'What is my name?':")
    log_print(f"  Answer: {response4['answer']}...")
    
    if "Alice" in response4["answer"]:
        log_print("  ✓ Memory recall working - found 'Alice' in response")
    else:
        log_print("  ⚠ Memory recall may not be working - 'Alice' not found in response")
    
except Exception as e:
    log_print(f"  ✗ Chat with memory failed: {e}")
    import traceback
    traceback.print_exc()

# -----------------------------------------------------------------------------
# Test 4: Different citation styles
# -----------------------------------------------------------------------------
log_print("\n[4/4] Testing different citation styles...")

citation_styles = ["markdown_link", "inline", "bracket", "footnote"]

for style in citation_styles:
    try:
        response_style = chat(
            session_id=TEST_SESSION_ID,
            message="What is the document about?",
            document_urls=None,
            top_k=3,
            use_memory=False,
            citation_style=style,
        )
        
        if response_style["citations"]:
            log_print(f"  {style}: {response_style['citations'][0]}")
        else:
            log_print(f"  {style}: (no citations)")
            
    except Exception as e:
        log_print(f"  {style}: Error - {e}")

# Print final timing summary from last response
if 'response' in dir() and response:
    print_timing_summary(response["timing"], title="Final Timing Summary (Test 1)", logger=logger)

log_print("\n" + "=" * 60)
log_print("End-to-end chat service test completed!")
log_print("=" * 60)
