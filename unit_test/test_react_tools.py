"""
Test react/tools.py: SearchDocumentsTool and GetAllChunksTool.

Requires:
- SSH tunnel to Qdrant Cloud (QDRANT_TUNNEL=true)
- AWS SSO login (for SearchDocumentsTool embedding via Bedrock)
- Documents already ingested in veris_pdfs collection

Run:
    python -m unit_test.test_react_tools

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.test_react_tools
"""

from dotenv import load_dotenv
load_dotenv()

from react.tools import GetAllChunksTool, SearchDocumentsTool

# Known URL from previous ingestion tests
TEST_URL = "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf"


def test_get_all_chunks():
    """GetAllChunksTool: Qdrant scroll by URL, no embedding needed."""
    print("\n=== GetAllChunksTool ===")
    tool = GetAllChunksTool(collection_name="veris_pdfs")

    # Print the schema that LLM would see
    from lits.components.policy.native_tool_use import _tools_to_schemas
    print(f"Tool schema sent to LLM:\n{_tools_to_schemas([tool])}")
    breakpoint()  # inspect: schema via p _tools_to_schemas([tool])

    result = tool._run(url=TEST_URL)
    print(f"\nResult length: {len(result)} chars")
    print(f"First 300 chars:\n{result[:300]}")
    breakpoint()  # inspect: result, len(result)


def test_search_documents():
    """SearchDocumentsTool: semantic search, needs Bedrock embedding."""
    print("\n=== SearchDocumentsTool ===")
    tool = SearchDocumentsTool(session_urls={TEST_URL}, collection_name="veris_pdfs")

    # Print embedding model and tool schema
    from rag_core.chat.config import load_config
    from lits.components.policy.native_tool_use import _tools_to_schemas
    config = load_config()
    print(f"Embedding model: {config['models'].get('embedding_model', 'unknown')}")
    print(f"Tool schema sent to LLM:\n{_tools_to_schemas([tool])}")
    breakpoint()  # inspect: schema via p _tools_to_schemas([tool])

    result = tool._run(query="What is the licence number?", top_k=3)
    print(f"\nResult:\n{result[:800]}")
    breakpoint()  # inspect: result


def test_search_empty_session():
    """SearchDocumentsTool with no URLs — should return helpful message."""
    print("\n=== SearchDocumentsTool (empty session) ===")
    tool = SearchDocumentsTool(session_urls=set(), collection_name="veris_pdfs")
    result = tool._run(query="anything")
    print(f"Result: {result}")
    breakpoint()  # inspect: result


if __name__ == "__main__":
    test_get_all_chunks()
    test_search_documents()
    test_search_empty_session()
    print("\n✓ All done")
