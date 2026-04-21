#!/bin/bash
# RAG pipeline limitation demo.
# Shows two issues that ReAct solves:
#   1) Losing sequential conversation context (Mem0 extracts facts, loses ordering)
#   2) No ability to perform actions (can only do top-K retrieval, not full-doc read)
#
# Run against deployed server:
#   API_HOST=localhost:8002 bash unit_test/test_rag_e2e.sh

API_HOST="${API_HOST:-localhost:8002}"
SESSION="433375739::rag_limitation_demo"

echo "================================================================"
echo "RAG Limitation Demo: Issues that ReAct solves"
echo "================================================================"

echo ""
echo "--- Turn 1: Ask about waste storage limits ---"
curl -s -X POST "http://${API_HOST}/rag/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"What are the waste storage limits for this site?\"}" \
  --no-buffer
echo ""

# Expected: RAG answers with top-K chunks about waste limits (300,000L etc.)
# This works fine — straightforward retrieval question.

# data: {"id": "chatcmpl-6bafce2364a7", "object": "chat.status", "created": 1776744145, "status": "Resolving parcel data..."}

# data: {"id": "chatcmpl-6bafce2364a7", "object": "chat.status", "created": 1776744145, "status": "Ingesting documents..."}

# data: {"id": "chatcmpl-6bafce2364a7", "object": "chat.status", "created": 1776744145, "status": "Retrieving memory..."}

# data: {"id": "chatcmpl-6bafce2364a7", "object": "chat.status", "created": 1776744145, "status": "Generating response..."}

# data: {"id": "chatcmpl-6bafce2364a7", "object": "chat.completion.chunk", "created": 1776744145, "choices": [{"index": 0, "delta": {"content": "Based"}, "finish_reason": null}]}

# data: {"id": "chatcmpl-6bafce2364a7", "object": "chat.completion.chunk", "created": 1776744145, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "message": 
# {"role": "assistant", "content": "Based on the EPA operating licence associated with this parcel, the site has the following waste storage limits:

# **Liquid Waste:**
# - Up to **300,000 litres** of liquid waste may be stored on the premises at any time, including waste codes J100 (waste oils, hydrocarbons, emulsions and transformer fluids), J120 (waste oil/water, hydrocarbons/water mixtures or emulsions), J130, and J160 [OL000112921 - Statutory Document.pdf (p.7)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf).

# **Additional Specific Waste Limits (storage and treatment):**
# - **F120** (solvent-based wastes): no more than **8,000 litres**
# - **J170**: no more than **24,000 litres**\n- **N250**: no more than **10,000 litres**
# - **N100** (rigid steel or plastic containers with original volume less than 200 litres): no more than **2,500 kg** (including mass of receptacles)
# - **N105**: no more than **100 containers**\n\nThese limits are outlined in conditions OL_WM1 and OL_WM1.3 of the licence [OL000112921 - Statutory Document.pdf (p.9)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf).

# **Additional Storage Requirements:**
# - Wastes that are not authorised for acceptance must be placed in a designated, sign-posted temporary storage area and sent for disposal to an appropriately licensed site **within 21 days** of receipt [OL000112921 - Statutory Document.pdf (p.7)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/"}}], "citations": ["[OL000112921 - Statutory Document.pdf (p.7)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.9)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.16)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.8)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.17)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"], "sources": [{"file": "OL000112921 - Statutory Document.pdf", "page": 7, "chunk_id": "OL000112921 - Statutory Document_p7_c4_5b3c9520", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.7)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 9, "chunk_id": "OL000112921 - Statutory Document_p9_c1_f3b9a3dd", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.9)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 16, "chunk_id": "OL000112921 - Statutory Document_p16_c0_6582a698", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.16)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 8, "chunk_id": "OL000112921 - Statutory Document_p8_c2_01c2198c", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.8)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 17, "chunk_id": "OL000112921 - Statutory Document_p17_c0_b3dcdea3", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.17)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}], "timing": {"ingestion": 0.00044125039130449295, "retrieval": 0.33227555081248283, "generation": 7.582942256703973, "memory": 3.1176887741312385, "total": 11.097004645504057}, "session_id": "433375739::rag_limitation_demo", "usage": {"completion_tokens": 112}}


echo ""
echo "Reflect Issue 1 specified in x-0421-agentic-url-handling/design.md"
echo "--- Turn 2: Ask about the structure of the previous response ---"
# ISSUE 1: Mem0 loses conversation structure.
# This question can ONLY be answered by seeing the full previous response.
# Mem0 extracts facts ("F120 limit is 8000L") but doesn't store the response structure
# (how many sections, what order, how many bullet points).
# With full conversation history (ReAct), the LLM sees its own previous response verbatim.
curl -s -X POST "http://${API_HOST}/rag/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"How many waste codes did you list in your previous answer? List them in the same order you gave them.\"}" \
  --no-buffer
echo ""

# Summary: RAG cannot reliably answer because Mem0 doesn't preserve the exact
# list ordering or count from the previous response. It may re-retrieve from
# documents and give a different list/order than what it actually said before.

# data: {"id": "chatcmpl-5fdc5338c900", "object": "chat.status", "created": 1776744676, "status": "Resolving parcel data..."}

# data: {"id": "chatcmpl-5fdc5338c900", "object": "chat.status", "created": 1776744676, "status": "Ingesting documents..."}

# data: {"id": "chatcmpl-5fdc5338c900", "object": "chat.status", "created": 1776744676, "status": "Retrieving memory..."}

# data: {"id": "chatcmpl-5fdc5338c900", "object": "chat.status", "created": 1776744676, "status": "Generating response..."}

# data: {"id": "chatcmpl-5fdc5338c900", "object": "chat.completion.chunk", "created": 1776744676, "choices": [{"index": 0, "delta": {"content": "Based"}, "finish_reason": null}]}


# ...

# data: {"id": "chatcmpl-5fdc5338c900", "object": "chat.completion.chunk", "created": 1776744676, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "message": 
# {"role": "assistant", "content": "Based on our conversation context, I did not provide a previous answer that listed waste codes. 
# This is the first question you've asked me in this conversation.\n\nHowever, if you'd like, I can provide a summary of the waste codes associated with the EPA licence (OL000112921) for this parcel. The waste acceptance table in the licence documents includes the following waste codes:\n\n1. **N250** \u2013 Absorbents contaminated with residues of hazardous substances [OL000112921 - Statutory Document.pdf (p.15)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)\n2. **J120** \u2013 Waste oil/water, hydrocarbons/water mixtures or emulsions [OL000112921 - Statutory Document.pdf (p.16)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)\n3. **N100** \u2013 Rigid steel or plastic containers with an original volume less than 200 litres [OL000112921 - Statutory Document.pdf (p.16)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)\n4. **J100** \u2013 Waste oils, hydrocarbons, emulsions and transformer fluids excluding polychlorinated biphenyls [OL000112921 - Statutory Document.pdf (p.17)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)\n5. **F120** \u2013 Solvent"}}], "citations": ["[OL000112921 - Statutory Document.pdf (p.7)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.9)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.16)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.17)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.15)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"], "sources": [{"file": "OL000112921 - Statutory Document.pdf", "page": 7, "chunk_id": "OL000112921 - Statutory Document_p7_c4_5b3c9520", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.7)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 9, "chunk_id": "OL000112921 - Statutory Document_p9_c1_f3b9a3dd", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.9)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 16, "chunk_id": "OL000112921 - Statutory Document_p16_c0_6582a698", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.16)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 17, "chunk_id": "OL000112921 - Statutory Document_p17_c0_b3dcdea3", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.17)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 15, "chunk_id": "OL000112921 - Statutory Document_p15_c0_8930ad17", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.15)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}], "timing": {"ingestion": 0.0004144106060266495, "retrieval": 0.32233693916350603, "generation": 6.712625361979008, "memory": 2.699435180053115, "total": 9.802296142093837}, "session_id": "433375739::rag_limitation_demo", "usage": {"completion_tokens": 90}}

echo ""
echo "Reflect Issue 1 & 2 specified in x-0421-agentic-url-handling/design.md"
echo "--- Turn 3: Request full document summary ---"
#  RAG can only do top-K retrieval, not full-document read.
# "every section including all appendices" needs ALL chunks, but RAG only retrieves top-5.
# Result: partial summary based on 5 most relevant chunks, missing most content.
# Compare: ReAct uses get_all_chunks → reads all 21 pages → complete 8-section summary.
curl -s -X POST "http://${API_HOST}/rag/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"Give me a complete summary of every section in this document, including all appendices\"}" \
  --no-buffer
echo ""

# Summary: 
# * Only cover p.5, p.6, p.8, p.10（from top-5 chunks）
# * No Appendix 4（waste acceptance table）、Appendix 5-7、waste storage limits、decommissioning plan ...

# data: {"id": "chatcmpl-33b07b3112fe", "object": "chat.status", "created": 1776744788, "status": "Resolving parcel data..."}

# data: {"id": "chatcmpl-33b07b3112fe", "object": "chat.status", "created": 1776744788, "status": "Ingesting documents..."}

# data: {"id": "chatcmpl-33b07b3112fe", "object": "chat.status", "created": 1776744788, "status": "Retrieving memory..."}

# data: {"id": "chatcmpl-33b07b3112fe", "object": "chat.status", "created": 1776744788, "status": "Generating response..."}

# data: {"id": "chatcmpl-33b07b3112fe", "object": "chat.completion.chunk", "created": 1776744788, "choices": [{"index": 0, "delta": {"content": "#"}, "finish_reason": null}]}

# data: {"id": "chatcmpl-33b07b3112fe", "object": "chat.completion.chunk", "created": 1776744788, "choices": [{"index": 0, "delta": {"content": " Complete"}, "finish_reason": null}]}

# ...

# data: {"id": "chatcmpl-33b07b3112fe", "object": "chat.completion.chunk", "created": 1776744788, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "message": 
# {"role": "assistant", "content": "# Complete Summary of Operating Licence OL000112921

# Based on the available source material, here is a summary of the document's structure and content:

# ## Document Overview

# This is an **Operating Licence** issued under the **Environment Protection Act 2017**, licence number **OL000112921**. The licence remains in force until its expiry date unless it is surrendered by the licence holder (with EPA consent), or suspended or revoked by EPA 
# [OL000112921 - Statutory Document.pdf (p.5)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf).
## Document Structure\n\nThe operating licence consists of multiple parts 
# [OL000112921 - Statutory Document.pdf (p.5)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf):

### 1. Conditions\nThe conditions section includes general and specific requirements, such as:

# - **OL_G4a** \u2013 The licence holder must provide the Authority with a **Permission Information and Performance Statement (PIPS)** in the form determined by the Authority within **2 months** of receiving written notification. 
# The PIPS may be released to the public in whole or in part 
# [OL000112921 - Statutory Document.pdf (p.6)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)。
# - **OL_G4b** \u2013 Information and monitoring records used for the preparation of, or inclusion in, any reporting or notification must be maintained 
# [OL000112921 - Statutory Document.pdf (p.6)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50"}}], "citations": ["[OL000112921 - Statutory Document.pdf (p.5)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.6)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.5)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.10)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)", "[OL000112921 - Statutory Document.pdf (p.8)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"], "sources": [{"file": "OL000112921 - Statutory Document.pdf", "page": 5, "chunk_id": "OL000112921 - Statutory Document_p5_c1_420dfb17", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.5)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 6, "chunk_id": "OL000112921 - Statutory Document_p6_c3_090e59c3", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.6)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 5, "chunk_id": "OL000112921 - Statutory Document_p5_c2_ababc3ec", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.5)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 10, "chunk_id": "OL000112921 - Statutory Document_p10_c0_e0ba5ff7", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.10)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}, {"file": "OL000112921 - Statutory Document.pdf", "page": 8, "chunk_id": "OL000112921 - Statutory Document_p8_c1_6ab78137", "url": "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf", "markdown_link": "[OL000112921 - Statutory Document.pdf (p.8)](https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/153856be-50b2-eb11-8236-00224814b9c3/OL000112921 - Statutory Document.pdf)"}], "timing": {"ingestion": 0.00047179125249385834, "retrieval": 0.3236389458179474, "generation": 7.270296533592045, "memory": 2.7522582104429603, "total": 10.408998974598944}, "session_id": "433375739::rag_limitation_demo", "usage": {"completion_tokens": 102}}


echo ""
echo "--- Cleanup ---"
curl -s -X DELETE "http://${API_HOST}/rag/chat/sessions/${SESSION}?clear_parcel_cache=true"
echo ""
echo ""
echo "================================================================"
echo "Compare with ReAct: bash unit_test/test_react_e2e.sh"
echo "================================================================"
