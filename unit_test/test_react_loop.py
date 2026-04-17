"""
Test react/loop.py: react_chat() end-to-end with real Bedrock + Qdrant.

Requires:
- SSH tunnel to Qdrant Cloud (QDRANT_TUNNEL=true)
- AWS SSO login
- Documents already ingested in veris_pdfs collection
- KG server NOT required (we pass document_urls directly)

Run:
    python -m unit_test.test_react_loop

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.test_react_loop
"""

import asyncio
import os

os.environ.setdefault("AWS_REGION", "us-east-1")

from dotenv import load_dotenv
load_dotenv()

from react.loop import react_chat

# Known URL from previous ingestion tests
TEST_URL = "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf"
SESSION_ID = "433375739::react_test1"

SYSTEM_MESSAGE = """You are an environmental assessment assistant for Victorian land parcels.
Answer questions grounded in assessment reports. Cite sources when possible."""


async def test_react_chat_with_tool():
    """Test: ask a question that requires searching documents."""
    print("\n=== react_chat: question requiring tool use ===")
    print(f"Session: {SESSION_ID}")
    print(f"URL: {TEST_URL[:60]}...")

    events = []
    async for chunk in react_chat(
        session_id=SESSION_ID,
        message="What is the licence number in this document?",
        system_message=SYSTEM_MESSAGE,
        document_urls=[TEST_URL],
    ):
        events.append(chunk)
        if chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "status":
            print(f"\n  [{chunk['content']}]")
        elif chunk["type"] == "done":
            print(f"\n  [DONE] {chunk['token_count']} tokens, {chunk['timing']['total']}s")
        elif chunk["type"] == "error":
            print(f"\n  [ERROR] {chunk['content']}")

    types = [e["type"] for e in events]
    print(f"\nEvent types: {types}")
    breakpoint()  # inspect: events, types


async def main():
    await test_react_chat_with_tool()
    print("\n✓ All done")

if __name__ == "__main__":
    asyncio.run(main())



# === react_chat: question requiring tool use ===
# Session: 433375739::react_test1
# URL: https://drapubcdnprd.azureedge.net/publicregister/attachment...
# /Users/xinzheli/git_repo/veris-chat/.venv/lib/python3.11/site-packages/qdrant_client/qdrant_remote.py:288: UserWarning: Failed to obtain server version. Unable to check client-server compatibility. Set check_compatibility=False to skip version check.
#   show_warning(

# ======================================================================
# Initializing embedding model: cohere.embed-english-v3
# ======================================================================
# ======================================================================
# Embedder initialization complete!
# ======================================================================



# Let me search for the licence number in the document.
#   [Searching documents...]
# /Users/xinzheli/git_repo/veris-chat/.venv/lib/python3.11/site-packages/qdrant_client/qdrant_remote.py:288: UserWarning: Failed to obtain server version. Unable to check client-server compatibility. Set check_compatibility=False to skip version check.
#   show_warning(
# The licence number in this document is **OL000112921**.

# This is an **Operating Licence** issued under section 74(1)(a) of the **Environment Protection Act 2017** by the **Environment Protection Authority Victoria (EPA)**. Key details include:

# - **Licence Holder:** J.J. Richards & Sons Pty Ltd
# - **ACN:** 000805425
# - **Activity Site:** 5–11 Piper Lane, East Bendigo, VIC, 3550, AU
# - **Issue Date:** 16 April 2015
# - **Last Amended:** 3 June 2022
# - **Expiry Date:** 31 December 9999

# *(Source: OL000112921 - Statutory Document.pdf, p.1)*
#   [DONE] 41 tokens, 11.98s

# Event types: ['token', 'token', 'token', 'status', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'done']
# --Return--
# > /Users/xinzheli/git_repo/veris-chat/unit_test/test_react_loop.py(60)test_react_chat_with_tool()->None
# -> breakpoint()  # inspect: events, types
# (Pdb) c

# ✓ All done