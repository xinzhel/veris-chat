#!/bin/bash
# End-to-end tests for ReAct endpoints.
#
# Requires:
#   - Server running: uvicorn main:app --reload
#   - SSH tunnels: Qdrant Cloud (6333) + Neo4j (7687)
#   - AWS SSO login
#   - KG EC2 running
#
# Usage:
#   bash unit_test/test_react_e2e.sh
# ssh -fN -L 8002:localhost:8000 -i ~/.ssh/race_lits_server.pem ec2-user@54.66.111.21

# Setup:
# curl -s http://localhost:8002/health  

API_HOST="localhost:8002"
SESSION="433375739::e2e_bash_test"


echo ""
echo "=== Test 1: ReAct stream — question with tool use ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"What is the licence number for this parcel?\"}" \
  --no-buffer
echo ""

# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "token", "content": "Based on the assessment"}

# ...

# data: {"type": "token", "content": " Document.pdf]()*"}

# data: {"type": "done", "answer": "Based on the assessment documents, the EPA licence number for this parcel (PFI: 433375739) is **OL000112921**.\n\nThis operating licence was issued under the **Environment Protection Act 2017** with the following details:\n\n- **Licence Number:** OL000112921\n- **Activity Type:** A01 \u2013 Reportable priority waste management\n- **Issue Date:** 16 April 2015\n- **Last Amended:** 3 June 2022\n- **Expiry Date:** 31 December 9999 (no expiry)\n- **Licence Holder:** J.J. Richards & Sons Pty Ltd\n\nThe licence relates to the storage and treatment of reportable priority waste at the activity site.\n\n*Source: [OL000112921 - Statutory Document.pdf]()*", "token_count": 43, "timing": {"total": 9.05}, "session_id": "433375739__e2e_bash_test"}

# data: [DONE]


# ******** Check: data/chat_state/433375739__e2e_bash_test.json ******** 

echo ""
echo "=== Test 2: Follow-up — uses conversation history ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"Who is the licence holder you just mentioned?\"}" \
  --no-buffer
echo ""


# data: {"type": "token", "content": "\n\nBased"}

# data: {"type": "token", "content": " on the assessment documents, the licence holder"}

# ...

# data: {"type": "token", "content": ")*"}

# data: {"type": "done", "answer": "\n\nBased on the assessment documents, the licence holder for EPA licence **OL000112921** is **J.J. Richards & Sons Pty Ltd**.\n\nThis information is recorded on the licence issued under the Environment Protection Act 2017.\n\n*Source: [OL000112921 - Statutory Document.pdf]()*", "token_count": 18, "timing": {"total": 3.41}, "session_id": "433375739__e2e_bash_test"}

# data: [DONE]


echo ""
echo "=== Test 3: Summarize document ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"Summarize the document\"}" \
  --no-buffer
echo ""


# data: {"type": "status", "content": "Reading the full document..."}

# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "token", "content": "Here"}

# data: {"type": "token", "content": " is"}

# ....


# data: {"type": "token", "content": "OL000112921 -"}

# data: {"type": "token", "content": " Statutory Document.pdf]()*"}

# data: {"type": "done", "answer": "Here is a summary of the document **OL000112921 - Statutory Document.pdf**:\n\n---\n\n## Summary: EPA Operating Licence OL000112921\n\n### Overview\nThis is an **EPA Operating Licence** issued under **Section 74(1)(a) of the Environment Protection Act 2017** for **reportable priority waste management (Activity Type A01)**.\n\n| Detail | Information |\n|---|---|\n| **Licence Number** | OL000112921 |\n| **Licence Holder** | J.J. Richards & Sons Pty Ltd (ACN 000805425) |\n| **Activity Site** | 5\u201311 Piper Lane, East Bendigo, VIC 3550 |\n| **Issue Date** | 16 April 2015 |\n| **Last Amended** | 3 June 2022 |\n| **Expiry Date** | 31 December 9999 (no expiry) |\n\n### Permitted Activities\nThe licence authorises the **storage and treatment** of various reportable priority wastes, including:\n- **J100** \u2013 Waste oils, hydrocarbons, emulsions and transformer fluids (excl. PCBs)\n- **J120** \u2013 Waste oil/water and hydrocarbon/water mixtures\n- **J130** \u2013 Triple interceptor waste and stormwater contaminated with oil/hydrocarbons\n- **J170** \u2013 Used oil filters\n- **F120** \u2013 Solvent-based wastes\n- **N100 / N105** \u2013 Rigid steel or plastic containers contaminated with reportable priority waste\n- **N250** \u2013 Absorbents contaminated with hazardous substances\n\nTreatment methods include storage pending permissioned activity (D15), used-oil refining (R9), and accumulation for permissioned activity (R13).\n\n### Key Storage & Treatment Limits\n- Up to **300,000 L of liquid waste** (J100, J120, J130, J160) on-site at any time\n- Additional limits on F120 (8,000 L), J170 (24,000 L), N250 (10,000 L), N100 (2,500 kg), and N105 (100 containers)\n\n### General Conditions\n- A copy of the licence must be kept on-site and accessible to all personnel\n- **Immediate notification** to EPA required for any discharge, emission, or equipment failure that may cause harm\n- All waste containers must be **labelled** with contents and waste producer details\n- Waste handling must comply with **EPA Publication 1698** (Liquid Storage and Handling Guidelines)\n- **Monthly reporting** to EPA on quantities of liquid reportable priority waste accepted\n- Waste movement records must be retained for **five years**\n- PCB content of waste oil must be monitored\n- Waste must **not burn** at the activity site\n\n### Legal Duties\nThe licence holder must comply with the **General Environmental Duty (GED)** to eliminate or minimise risks of harm to human health and the environment. Additional duties include:\n- **Duty to notify** EPA of notifiable incidents (pollution or equipment failure)\n- **Duty to notify** of contaminated land\n- **Duty to restore** affected areas after pollution incidents\n\n### Appendices\n- **No air discharge points** associated with this licence\n- **No water discharge points** associated with this licence\n- **No landfill cells** associated with this licence\n\n---\n\n*Source: [OL000112921 - Statutory Document.pdf]()*", "token_count": 258, "timing": {"total": 38.31}, "session_id": "433375739__e2e_bash_test"}

# data: [DONE]


# ******** Check: data/chat_state/433375739__e2e_bash_test.json ******** 

echo ""
echo "=== Test 4: DELETE — archive state ==="
curl -s -X DELETE "http://${API_HOST}/react/sessions/${SESSION}"
echo ""

# {"status":"cleaned","session_id":"433375739::e2e_bash_test","cleaned":{"session_index":true,"state_archived":true}}%          

# [ec2-user@ip-172-31-30-236 chat_state]$ ls
# 433375739__e2e_bash_test__20260420_061139.json

# Note 1: DELETE clears session_index but does not clear _parcel_cache
#  session_index (session_id → Set[url]) 
#  _parcel_cache (parcel_id → {document_urls, system_message, parcel_context})

# Note 2:
# 所以下次同 parcel 的新 session 不需要重新查 KG，
# 但需要重新 ingest（因为 session_index 里没有 url 了）。
# 不过 ingest 会跳过已有的 url（url_cache 检查），
# 所以实际上只是重新关联 session → url，不会重新下载/embed

echo ""
echo "=== Test 5: Fresh session after DELETE ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"Who is the licence holder you just mentioned?\"}" \
  --no-buffer
echo ""


# data: {"type": "token", "content": "\n\nLet"}

# data: {"type": "token", "content": " me search the"}

# data: {"type": "token", "content": " assessment"}

# data: {"type": "token", "content": " documents for more"}

# data: {"type": "token", "content": " details"}

# data: {"type": "token", "content": " about the EPA"}

# data: {"type": "token", "content": " licence associated"}

# data: {"type": "token", "content": " with this parcel."}

# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "token", "content": "Based on the EPA"}

# ...

# data: {"type": "token", "content": "- Statutory Document.pdf))"}

# data: {"type": "done", "answer": "Based on the EPA licence document on file, the licence holder is **J.J. Richards & Sons Pty Ltd**. This is associated with EPA Operating Licence number **OL000112921**, issued on 16 April 2015 under the *Environment Protection Act 2017*, for **reportable priority waste management (A01)** activities at this parcel.\n\nThe licence was last amended on 3 June 2022 and has no set expiry date (listed as 31 December 9999), indicating it remains active indefinitely unless revoked or amended.\n\n([Source: OL000112921 - Statutory Document.pdf, p.1](OL000112921 - Statutory Document.pdf))", "token_count": 61, "timing": {"total": 9.56}, "session_id": "433375739__e2e_bash_test"}

