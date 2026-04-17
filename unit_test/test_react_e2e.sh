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

API_HOST="${API_HOST:-localhost:8000}"
SESSION="433375739::e2e_bash_test"

echo "=== Test 1: ReAct stream — question with tool use ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"What is the licence number for this parcel?\"}" \
  --no-buffer
echo ""

echo ""
echo "=== Test 2: Follow-up — uses conversation history ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"Who is the licence holder you just mentioned?\"}" \
  --no-buffer
echo ""

echo ""
echo "=== Test 3: DELETE — archive state ==="
curl -s -X DELETE "http://${API_HOST}/react/sessions/${SESSION}"
echo ""

echo ""
echo "=== Test 4: Fresh session after DELETE ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"${SESSION}\", \"message\": \"Who is the licence holder you just mentioned?\"}" \
  --no-buffer
echo ""

echo ""
echo "=== Test 5: Summarize document ==="
curl -s -X POST "http://${API_HOST}/react/chat/stream/" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"433375739::e2e_summarize\", \"message\": \"Summarize the document\"}" \
  --no-buffer
echo ""
