# Deployment Plan: React + RAG2 Backend

## Context

- `i-04083f11879703da7` (EIP `54.66.111.21`) — original RAG-only instance, safe to override
- `i-018c87e156b4cbd8a` (EIP `54.253.127.203`) — KG Neo4j instance, keep as-is
- `16.176.180.235` — Lukesh's current frontend backend (RAG2, separate instance)
- New GitHub repo: `https://github.com/AEA-MapTalk/veris-llm-agent.git` (remote: `agent`)
- Old GitHub repo: `https://github.com/AEA-MapTalk/veris-chat.git` (remote: `deploy`)

## Tasks

- [x] Task 1: Backup source from EC2 instances
  - [x] v1 RAG from `54.66.111.21`: `backups/veris-chat-v1-rag_2026-01-19.tar.gz` (280KB)
  - [x] v2 RAG2 from `16.176.180.235`: `backups/veris-chat-v2-rag2_2026-04-14.tar.gz` (446KB, excl data/)
  - [x] v1 was commit `f80c1e1 Deployment 2026-01-19` (orphan, not in local git)

- [x] Task 2: Hard copy lits into project
  - [x] Copy `prev_projects_repo/lits_llm/lits/` → `./lits/` (project root)
  - [x] Uninstall `pip install -e` version to avoid conflicts
  - [x] Verify: `from lits.agents.chain.native_react import AsyncNativeReAct` resolves to `./lits/`
  - NOTE: `prev_projects_repo/` symlink kept for local dev convenience, but `./lits/` is the source of truth

- [x] Task 3: Setup new GitHub repo + update deploy scripts
  - [x] Add remote: `git remote add agent https://github.com/AEA-MapTalk/veris-llm-agent.git`
  - [x] Update `deploy/push_clean.sh`: push to `agent` remote, exclude `.kiro/`, `backups/`, `llm_chat/`, `unit_test/`, `prev_projects_repo/`
  - [x] Update `deploy/user_data.sh`: `uvicorn main:app` (was `app.chat_api:app`)
  - [x] lits/ is in git repo, cloned to EC2 automatically — no pip install needed
  - [x] Symlinks (chore/, prev_projects_repo/) committed as-is — broken on EC2, not imported

- [ ] Task 4: Test push to new repo
  - [ ] Run `bash deploy/push_clean.sh`
  - [ ] Verify on GitHub: `https://github.com/AEA-MapTalk/veris-llm-agent` has `main.py`, `lits/`, `react/`, `rag_app/`, `react_app/`

- [ ] Task 5: Deploy to `i-04083f11879703da7` (EIP `54.66.111.21`)
  - [ ] Option A: Terminate + relaunch with updated `user_data.sh`
  - [ ] Option B: SSH in, git pull from new repo, restart service
  - [ ] Verify: `curl http://54.66.111.21:8000/health`
  - [ ] Verify: `curl http://54.66.111.21:8000/react/chat/stream/` works
  - [ ] Verify: `curl http://54.66.111.21:8000/rag/chat/stream/` works

- [ ] Task 6: Coordinate with Lukesh
  - [ ] Notify: new endpoints are `/rag/*` and `/react/*` (not `/chat/*`)
  - [ ] Decide: does frontend switch to `54.66.111.21` or stay on `16.176.180.235`?
  - [ ] If switching: update frontend API base URL
