# Deployment Plan: React + RAG2 Backend

## Context

- `i-04083f11879703da7` (EIP `54.66.111.21`) — original RAG-only instance, safe to override
- `i-018c87e156b4cbd8a` (EIP `54.253.127.203`) — KG Neo4j instance, keep as-is
- `16.176.180.235` — Lukesh's current frontend backend (RAG2, separate instance)
- New GitHub repo: `https://github.com/AEA-MapTalk/veris-llm-agent.git`
- Old GitHub repo: `https://github.com/AEA-MapTalk/veris-chat.git` (deploy-clean was overridden by RAG2)

## Tasks

- [ ] Task 1: Backup original RAG v1 source from `i-04083f11879703da7`
  - [ ] SSH into `54.66.111.21`: `ssh -i ~/.ssh/race_lits_server.pem ec2-user@54.66.111.21`
  - [ ] Copy source: `tar czf /tmp/veris-chat-v1-backup.tar.gz -C /home/ec2-user veris-chat/`
  - [ ] SCP to local: `scp -i ~/.ssh/race_lits_server.pem ec2-user@54.66.111.21:/tmp/veris-chat-v1-backup.tar.gz ./backups/`
  - [ ] Verify backup contents locally

- [ ] Task 2: Identify original RAG v1 commit
  - [ ] Check `deploy-clean` branch: currently at `d390058 Deployment 2026-04-14` (this is RAG2)
  - [ ] Check git reflog on `i-04083f11879703da7` for the original deploy commit
  - [ ] If identifiable, tag it locally as `v1-rag-original`
  - [ ] If not identifiable, the tar backup from Task 1 serves as the v1 archive

- [ ] Task 3: Setup new GitHub repo for deploy
  - [ ] Add `veris-llm-agent` as a new remote: `git remote add agent https://github.com/AEA-MapTalk/veris-llm-agent.git`
  - [ ] Update `deploy/push_clean.sh` to push to `agent` remote instead of `deploy`
  - [ ] Test push: `bash deploy/push_clean.sh`

- [ ] Task 4: Update deployment scripts for new architecture
  - [ ] `deploy/user_data.sh`: change `uvicorn app.chat_api:app` → `uvicorn main:app`
  - [ ] `deploy/user_data.sh`: add `lits-llm` installation (copy or pip install from bundled source)
  - [ ] `deploy/push_clean.sh`: include `prev_projects_repo/lits_llm/lits/` in deploy (as actual copy, not symlink)
  - [ ] `deploy/push_clean.sh`: exclude `prev_projects_repo/lits_llm/unit_test/`, `.git/`, etc.
  - [ ] Verify `pyproject.toml` requires-python >= 3.11 matches EC2 Python version

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
