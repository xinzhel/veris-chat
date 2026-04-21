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
# API_HOST="${API_HOST:-localhost:8002}"   

API_HOST="${API_HOST:-localhost:8000}"
SESSION="433375739::e2e_bash_test"

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

# ******** Check: data/chat_state/433375739__e2e_bash_test.json ******** 

echo ""
echo "=== Test 3: DELETE — archive state ==="
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
echo "=== Test 4: Fresh session after DELETE ==="
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

echo ""
echo "=== Test 5: Summarize document ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"433375739::e2e_summarize\", \"message\": \"Summarize the document\"}" \
  --no-buffer
echo ""

# data: {"type": "token", "content": "I'd"}

# ..

# data: {"type": "token", "content": " I"}

# data: {"type": "token", "content": " can assist you."}

# data: {"type": "done", "answer": "I'd be happy to help summarize a document, but I need to know which document you're referring to. Could you please provide the URL of the document you'd like me to summarize, or let me know more specifically what topic or aspect of the parcel (PFI: 433375739) you'd like me to search for?\n\nIn the meantime, I can search the available assessment documents for information related to this parcel. Would you like me to:\n\n1. Search for general information about this parcel?\n2. Search for details about the **EPA Licence (A01 - Reportable priority waste management)**?\n3. Search for information about the **planning overlays** (Airport Environs, Bushfire Management, Design and Development)?\n\nPlease let me know how I can assist you.", "token_count": 48, "timing": {"total": 5.02}, "session_id": "433375739__e2e_summarize"}



# lits_llm $ echo ""
# echo "=== Test 5: Summarize document ==="
# curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
#   -H "Content-Type: application/json" \
#   -d "{\"session_id\": \"433375739::e2e_summarize\", \"message\": \"the document you mentioned before for getting the licence holder\"}" \
#   --no-buffer
# echo ""

# === Test 5: Summarize document ===
# data: {"type": "status", "content": "Searching documents..."}

# data: {"type": "status", "content": "Reading the full document..."}

# data: {"type": "token", "content": "Here"}

# data: {"type": "token", "content": " is"}

# data: {"type": "token", "content": " a summary of the EPA Operating"}

# data: {"type": "token", "content": " Licence document"}

# data: {"type": "token", "content": " "}

# data: {"type": "token", "content": "(**OL000112921 "}

# data: {"type": "token", "content": "- Statutory Document.pdf**)"}

# data: {"type": "token", "content": " associated"}

# data: {"type": "token", "content": " with this"}

# data: {"type": "token", "content": " parcel:\n\n---"}

# data: {"type": "token", "content": "\n\n##"}

# data: {"type": "token", "content": " Summary:"}

# data: {"type": "token", "content": " EPA Operating Licence O"}

# data: {"type": "token", "content": "L000112921\n\n###"}

# data: {"type": "token", "content": " Licence"}

# data: {"type": "token", "content": " Holder"}

# data: {"type": "token", "content": "\n**J.J. Richards"}

# data: {"type": "token", "content": " & Sons Pty Ltd**"}

# data: {"type": "token", "content": "\n\n### Key"}

# data: {"type": "token", "content": " Details"}

# data: {"type": "token", "content": "\n|"}

# data: {"type": "token", "content": " Field"}

# data: {"type": "token", "content": " | Detail"}

# data: {"type": "token", "content": " |\n|---|"}

# data: {"type": "token", "content": "---|\n| **Licence Number**"}

# data: {"type": "token", "content": " | OL000112921 |"}

# data: {"type": "token", "content": "\n| **Issue Date** | "}

# data: {"type": "token", "content": "16 April 2015 |"}

# data: {"type": "token", "content": "\n| **Last"}

# data: {"type": "token", "content": " Amended** | 3 June"}

# data: {"type": "token", "content": " 2022 |\n| **"}

# data: {"type": "token", "content": "Expiry Date** | 31"}

# data: {"type": "token", "content": " December 9999 ("}

# data: {"type": "token", "content": "no"}

# data: {"type": "token", "content": " expiry) |\n| **"}

# data: {"type": "token", "content": "Issued"}

# data: {"type": "token", "content": " By** | Margaret Green, Senior"}

# data: {"type": "token", "content": " Permissioning Officer, EPA"}

# data: {"type": "token", "content": " Victoria"}

# data: {"type": "token", "content": " |"}

# data: {"type": "token", "content": "\n| **Activity"}

# data: {"type": "token", "content": " Type** | A"}

# data: {"type": "token", "content": "01 \u2013"}

# data: {"type": "token", "content": " Reportable Priority"}

# data: {"type": "token", "content": " Waste Management |\n\n### Purpose"}

# data: {"type": "token", "content": "\nThe"}

# data: {"type": "token", "content": " licence is issued under **"}

# data: {"type": "token", "content": "section 74(1)(a)"}

# data: {"type": "token", "content": " of the Environment Protection Act 2017"}

# data: {"type": "token", "content": "** and"}

# data: {"type": "token", "content": " author"}

# data: {"type": "token", "content": "ises"}

# data: {"type": "token", "content": " the management"}

# data: {"type": "token", "content": " of **"}

# data: {"type": "token", "content": "report"}

# data: {"type": "token", "content": "able priority waste** at the activity site"}

# data: {"type": "token", "content": "."}

# data: {"type": "token", "content": " EPA"}

# data: {"type": "token", "content": " issues"}

# data: {"type": "token", "content": " operating licences to"}

# data: {"type": "token", "content": " regulate"}

# data: {"type": "token", "content": " activities"}

# data: {"type": "token", "content": " that pose"}

# data: {"type": "token", "content": " potential"}

# data: {"type": "token", "content": " risks to the environment and human"}

# data: {"type": "token", "content": " health.\n\n### Permitted"}

# data: {"type": "token", "content": " Waste Types\nThe"}

# data: {"type": "token", "content": " site"}

# data: {"type": "token", "content": " is"}

# data: {"type": "token", "content": " author"}

# data: {"type": "token", "content": "ised to accept and"}

# data: {"type": "token", "content": " manage"}

# data: {"type": "token", "content": " a"}

# data: {"type": "token", "content": " range"}

# data: {"type": "token", "content": " of haz"}

# data: {"type": "token", "content": "ardous and"}

# data: {"type": "token", "content": " priority"}

# data: {"type": "token", "content": " was"}

# data: {"type": "token", "content": "tes, including:\n- **"}

# data: {"type": "token", "content": "Waste"}

# data: {"type": "token", "content": " oils,"}

# data: {"type": "token", "content": " hydrocarbons, emulsions"}

# data: {"type": "token", "content": " and transformer fluids** (J"}

# data: {"type": "token", "content": "100,"}

# data: {"type": "token", "content": " J"}

# data: {"type": "token", "content": "120)\n- **Triple"}

# data: {"type": "token", "content": " interceptor waste and"}

# data: {"type": "token", "content": " stormwater contaminated with oil/"}

# data: {"type": "token", "content": "hyd"}

# data: {"type": "token", "content": "rocarbons** (J"}

# data: {"type": "token", "content": "130)"}

# data: {"type": "token", "content": "\n- **"}

# data: {"type": "token", "content": "Sol"}

# data: {"type": "token", "content": "vent-based wastes** (F"}

# data: {"type": "token", "content": "120)"}

# data: {"type": "token", "content": "\n- **"}

# data: {"type": "token", "content": "Abs"}

# data: {"type": "token", "content": "orbents contaminated with hazardous"}

# data: {"type": "token", "content": " substances** (N250)"}

# data: {"type": "token", "content": "\n- **Rigid steel/"}

# data: {"type": "token", "content": "plastic containers**"}

# data: {"type": "token", "content": " (N100)"}

# data: {"type": "token", "content": "\n- Treatment"}

# data: {"type": "token", "content": " activities"}

# data: {"type": "token", "content": " include"}

# data: {"type": "token", "content": " **"}

# data: {"type": "token", "content": "storage"}

# data: {"type": "token", "content": " ("}

# data: {"type": "token", "content": "D15)**,"}

# data: {"type": "token", "content": " **used"}

# data: {"type": "token", "content": "-oil refining (R9)"}

# data: {"type": "token", "content": "**, and **accumulation for"}

# data: {"type": "token", "content": " permissioned activity"}

# data: {"type": "token", "content": " (R13)**\n\n### Key"}

# data: {"type": "token", "content": " Conditions"}

# data: {"type": "token", "content": "\n-"}

# data: {"type": "token", "content": " A"}

# data: {"type": "token", "content": " copy of the licence must be kept on"}

# data: {"type": "token", "content": "-"}

# data: {"type": "token", "content": "site and"}

# data: {"type": "token", "content": " accessible"}

# data: {"type": "token", "content": " to all"}

# data: {"type": "token", "content": " personnel"}

# data: {"type": "token", "content": "."}

# data: {"type": "token", "content": "\n- **"}

# data: {"type": "token", "content": "Monthly"}

# data: {"type": "token", "content": " reporting** to"}

# data: {"type": "token", "content": " EPA ("}

# data: {"type": "token", "content": "by"}

# data: {"type": "token", "content": " the 15th of"}

# data: {"type": "token", "content": " each month) on"}

# data: {"type": "token", "content": " quantities"}

# data: {"type": "token", "content": " of liquid reportable priority waste accepted."}

# data: {"type": "token", "content": "\n- **"}

# data: {"type": "token", "content": "Immediate"}

# data: {"type": "token", "content": " notification** to EPA ("}

# data: {"type": "token", "content": "1"}

# data: {"type": "token", "content": "300 EPA VIC) in"}

# data: {"type": "token", "content": " the"}

# data: {"type": "token", "content": " event of incidents"}

# data: {"type": "token", "content": "."}

# data: {"type": "token", "content": "\n- Compliance"}

# data: {"type": "token", "content": " with the"}

# data: {"type": "token", "content": " **General Environmental Duty (GED"}

# data: {"type": "token", "content": ")** and"}

# data: {"type": "token", "content": " all"}

# data: {"type": "token", "content": " duties"}

# data: {"type": "token", "content": " under"}

# data: {"type": "token", "content": " the Act."}

# data: {"type": "token", "content": "\n- **"}

# data: {"type": "token", "content": "Duty"}

# data: {"type": "token", "content": " to notify"}

# data: {"type": "token", "content": "**"}

# data: {"type": "token", "content": " EPA"}

# data: {"type": "token", "content": " of any"}

# data: {"type": "token", "content": " contaminated land,"}

# data: {"type": "token", "content": " and"}

# data: {"type": "token", "content": " to restore affected areas after"}

# data: {"type": "token", "content": " pollution"}

# data: {"type": "token", "content": " incidents as"}

# data: {"type": "token", "content": " far"}

# data: {"type": "token", "content": " as reasonably practicable."}

# data: {"type": "token", "content": "\n\n### Transfer"}

# data: {"type": "token", "content": " &"}

# data: {"type": "token", "content": " Amendment"}

# data: {"type": "token", "content": "\n- The"}

# data: {"type": "token", "content": " licence can"}

# data: {"type": "token", "content": " be **"}

# data: {"type": "token", "content": "transferred** to a new holder"}

# data: {"type": "token", "content": " under section 56 of the Act"}

# data: {"type": "token", "content": ".\n- **"}

# data: {"type": "token", "content": "Amendments** can be applied"}

# data: {"type": "token", "content": " for under"}

# data: {"type": "token", "content": " section 57,"}

# data: {"type": "token", "content": " or"}

# data: {"type": "token", "content": " initiated"}

# data: {"type": "token", "content": " by"}

# data: {"type": "token", "content": " EPA under"}

# data: {"type": "token", "content": " section 58.\n\n---\n\nThis"}

# data: {"type": "token", "content": " document confirms"}

# data: {"type": "token", "content": " that the site"}

# data: {"type": "token", "content": " is actively"}

# data: {"type": "token", "content": " licensed"}

# data: {"type": "token", "content": " for hazardous waste management operations"}

# data: {"type": "token", "content": ","}

# data: {"type": "token", "content": " which"}

# data: {"type": "token", "content": " is an"}

# data: {"type": "token", "content": " important consideration"}

# data: {"type": "token", "content": " for any"}

# data: {"type": "token", "content": " environmental assessment of"}

# data: {"type": "token", "content": " this parcel."}

# data: {"type": "done", "answer": "Here is a summary of the EPA Operating Licence document (**OL000112921 - Statutory Document.pdf**) associated with this parcel:\n\n---\n\n## Summary: EPA Operating Licence OL000112921\n\n### Licence Holder\n**J.J. Richards & Sons Pty Ltd**\n\n### Key Details\n| Field | Detail |\n|---|---|\n| **Licence Number** | OL000112921 |\n| **Issue Date** | 16 April 2015 |\n| **Last Amended** | 3 June 2022 |\n| **Expiry Date** | 31 December 9999 (no expiry) |\n| **Issued By** | Margaret Green, Senior Permissioning Officer, EPA Victoria |\n| **Activity Type** | A01 \u2013 Reportable Priority Waste Management |\n\n### Purpose\nThe licence is issued under **section 74(1)(a) of the Environment Protection Act 2017** and authorises the management of **reportable priority waste** at the activity site. EPA issues operating licences to regulate activities that pose potential risks to the environment and human health.\n\n### Permitted Waste Types\nThe site is authorised to accept and manage a range of hazardous and priority wastes, including:\n- **Waste oils, hydrocarbons, emulsions and transformer fluids** (J100, J120)\n- **Triple interceptor waste and stormwater contaminated with oil/hydrocarbons** (J130)\n- **Solvent-based wastes** (F120)\n- **Absorbents contaminated with hazardous substances** (N250)\n- **Rigid steel/plastic containers** (N100)\n- Treatment activities include **storage (D15)**, **used-oil refining (R9)**, and **accumulation for permissioned activity (R13)**\n\n### Key Conditions\n- A copy of the licence must be kept on-site and accessible to all personnel.\n- **Monthly reporting** to EPA (by the 15th of each month) on quantities of liquid reportable priority waste accepted.\n- **Immediate notification** to EPA (1300 EPA VIC) in the event of incidents.\n- Compliance with the **General Environmental Duty (GED)** and all duties under the Act.\n- **Duty to notify** EPA of any contaminated land, and to restore affected areas after pollution incidents as far as reasonably practicable.\n\n### Transfer & Amendment\n- The licence can be **transferred** to a new holder under section 56 of the Act.\n- **Amendments** can be applied for under section 57, or initiated by EPA under section 58.\n\n---\n\nThis document confirms that the site is actively licensed for hazardous waste management operations, which is an important consideration for any environmental assessment of this parcel.", "token_count": 191, "timing": {"total": 18.59}, "session_id": "433375739__e2e_summarize"}

# data: [DONE]


