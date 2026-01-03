"""
Test script for end-to-end chat service.

Tests:
1. Chat with document ingestion
2. Follow-up chat (session continuity)
3. Citation extraction and formatting
4. Timing breakdown

Prerequisites: Qdrant must be accessible (local or cloud).

Usage: Run in interactive notebook mode or as a script.
"""

import os
import sys

# Set AWS_REGION before imports
os.environ["AWS_REGION"] = "ap-southeast-2"

sys.path.insert(0, ".")

from veris_chat.chat.config import load_config
from veris_chat.utils.logger import setup_logging

# Setup logging
logger = setup_logging(
    run_id="test_chat_service",
    result_dir="./logs",
    add_console_handler=True,
    verbose=True,
    allowed_namespaces=("veris_chat", "__main__"),
)

print("=" * 60)
print("End-to-End Chat Service Test")
print("=" * 60)

# -----------------------------------------------------------------------------
# Load configuration
# -----------------------------------------------------------------------------
print("\n[Setup] Loading configuration...")
config = load_config()
print("  ✓ Configuration loaded")

# Test constants
TEST_SESSION_ID = "test_chat_service_001"
TEST_URLS = [
    "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf",
]

print(f"  Test session_id: {TEST_SESSION_ID}")
print(f"  Test document URLs: {len(TEST_URLS)}")

# -----------------------------------------------------------------------------
# Clear any cached resources from previous runs
# -----------------------------------------------------------------------------
print("\n[Cleanup] Clearing cached resources...")
from veris_chat.chat.service import clear_cache
clear_cache()
print("  ✓ Cache cleared")

# -----------------------------------------------------------------------------
# Test 1: Chat with document ingestion
# -----------------------------------------------------------------------------
print("\n[1/4] Testing chat with document ingestion...")

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
    
    print("  ✓ Chat completed successfully")
    print(f"\n  Answer preview:")
    print(f"  {'-' * 50}")
    answer_preview = response["answer"][:500] if len(response["answer"]) > 500 else response["answer"]
    print(f"  {answer_preview}...")
    print(f"  {'-' * 50}")
    
    print(f"\n  Citations: {len(response['citations'])}")
    for i, citation in enumerate(response["citations"][:3], 1):
        print(f"    {i}. {citation}")
    
    print(f"\n  Sources: {len(response['sources'])}")
    for i, source in enumerate(response["sources"][:3], 1):
        print(f"    {i}. {source.get('file')} (p.{source.get('page')})")
    
    print(f"\n  Timing:")
    timing = response["timing"]
    print(f"    Ingestion: {timing.get('ingestion', 0):.2f}s")
    print(f"    Retrieval: {timing.get('retrieval', 0):.2f}s")
    print(f"    Generation: {timing.get('generation', 0):.2f}s")
    print(f"    Total: {timing.get('total', 0):.2f}s")
    
except Exception as e:
    print(f"  ✗ Chat failed: {e}")
    import traceback
    traceback.print_exc()

# -----------------------------------------------------------------------------
# Test 2: Follow-up chat (no new documents)
# -----------------------------------------------------------------------------
print("\n[2/4] Testing follow-up chat (no new documents)...")

try:
    response2 = chat(
        session_id=TEST_SESSION_ID,
        message="Can you tell me more about the site conditions mentioned?",
        document_urls=None,  # No new documents
        top_k=5,
        use_memory=False,
        citation_style="markdown_link",
    )
    
    print("  ✓ Follow-up chat completed successfully")
    print(f"\n  Answer preview:")
    print(f"  {'-' * 50}")
    answer_preview = response2["answer"][:500] if len(response2["answer"]) > 500 else response2["answer"]
    print(f"  {answer_preview}...")
    print(f"  {'-' * 50}")
    
    print(f"\n  Timing (no ingestion expected):")
    timing2 = response2["timing"]
    print(f"    Ingestion: {timing2.get('ingestion', 0):.2f}s")
    print(f"    Retrieval: {timing2.get('retrieval', 0):.2f}s")
    print(f"    Generation: {timing2.get('generation', 0):.2f}s")
    print(f"    Total: {timing2.get('total', 0):.2f}s")
    
    if timing2.get("ingestion", 0) == 0:
        print("  ✓ Correctly skipped ingestion for follow-up")
    
except Exception as e:
    print(f"  ✗ Follow-up chat failed: {e}")
    import traceback
    traceback.print_exc()

# -----------------------------------------------------------------------------
# Test 3: Chat with memory enabled
# -----------------------------------------------------------------------------
print("\n[3/4] Testing chat with memory enabled...")

TEST_SESSION_ID_MEMORY = "test_chat_service_memory_001"

try:
    # First message with memory
    response3 = chat(
        session_id=TEST_SESSION_ID_MEMORY,
        message="My name is Alice. What documents do you have access to?",
        document_urls=TEST_URLS,
        top_k=5,
        use_memory=True,
        citation_style="markdown_link",
    )
    
    print("  ✓ Chat with memory completed")
    print(f"  Answer preview: {response3['answer'][:200]}...")
    
    # Follow-up to test memory recall
    response4 = chat(
        session_id=TEST_SESSION_ID_MEMORY,
        message="What is my name?",
        document_urls=None,
        top_k=5,
        use_memory=True,
        citation_style="markdown_link",
    )
    
    print(f"\n  Memory test - asking 'What is my name?':")
    print(f"  Answer: {response4['answer'][:300]}...")
    
    if "Alice" in response4["answer"]:
        print("  ✓ Memory recall working - found 'Alice' in response")
    else:
        print("  ⚠ Memory recall may not be working - 'Alice' not found in response")
    
except Exception as e:
    print(f"  ✗ Chat with memory failed: {e}")
    import traceback
    traceback.print_exc()

# -----------------------------------------------------------------------------
# Test 4: Different citation styles
# -----------------------------------------------------------------------------
print("\n[4/4] Testing different citation styles...")

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
            print(f"  {style}: {response_style['citations'][0]}")
        else:
            print(f"  {style}: (no citations)")
            
    except Exception as e:
        print(f"  {style}: Error - {e}")

print("\n" + "=" * 60)
print("End-to-end chat service test completed!")
print("=" * 60)
