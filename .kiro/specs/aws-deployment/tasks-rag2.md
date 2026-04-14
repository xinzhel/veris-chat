# Deployment Plan: RAG Pipeline v2 (Parcel-KG Integration)

## Overview

Deploy the updated RAG pipeline with parcel-KG integration to EC2. Changes since last deployment: KG integration in `chat_api.py`, memory fixes (`::` → `_` in Qdrant collection names), `memory_llm` updated to Opus 4.6, Python 3.11, `aioboto3` + `neo4j` added to dependencies.

## Dependency Graph

```
T1 ──→ T2 ──→ T3 ──→ T4 ──→ T5
                        ↗
T6 (GitHub Actions — independent)
T7 (README — independent)
```

T1 = Start KG EC2 instance
T2 = Update deploy scripts
T3 = Push code & deploy to RAG EC2
T4 = Verify deployment
T5 = Final checkpoint
T6 = GitHub Actions CI/CD
T7 = Update README

## Tasks

- [x] Task 1: Start KG EC2 instance
  - [x] 1.1 Start the Neo4j KG instance (currently stopped)
    ```bash
    aws ec2 start-instances --instance-ids i-018c87e156b4cbd8a --region ap-southeast-2
    ```
  - [x] 1.2 Wait for instance to be running and verify Neo4j is accessible
    ```bash
    aws ec2 wait instance-running --instance-ids i-018c87e156b4cbd8a --region ap-southeast-2
    # Verify Neo4j responds (from local via SSH tunnel, or from RAG EC2 directly)
    ```

- [x] Task 2: Update deploy scripts for v2
  - [x] 2.1 Update `deploy/user_data.sh` — add `neo4j` and `aioboto3` to pip install list
  - [x] 2.2 Update `deploy/user_data.sh` — set `QDRANT_TUNNEL=false` in `.env` creation block
  - [x] 2.3 Verify `deploy/push_clean.sh` excludes `.kiro` and `deploy/push_clean.sh` (already does via `git reset HEAD`)
  - [x] 2.4 Add SSE `chat.status` events for KG resolution, ingestion, memory, generation phases
  - [x] 2.5 Local test passed: status events + streaming + session delete all working

- [ ] Task 3: Push code and deploy to RAG EC2
  - Note: Do NOT terminate old EC2 (frontend is using it). Launch a new instance without Elastic IP.
  - After verifying v2 works, check with frontend dev (see `documents/frontend_checklist.md`), then reassociate EIP.
  - [ ] 3.1 Push latest code to deploy-clean branch
    ```bash
    bash deploy/push_clean.sh
    ```
  - [ ] 3.2 Launch new EC2 instance (keep old one running)
    - Modify `ec2_launch.sh` to skip EIP association, or launch manually without `--terminate`
    - Note the new instance's public IP from launch output
  - [ ] 3.3 Wait ~5-10 min for user-data setup, then SSH in to check
    ```bash
    ssh -i ~/.ssh/race_lits_server.pem ec2-user@<NEW_IP>
    sudo tail -f /var/log/user-data.log
    systemctl status veris-chat
    ```
  - [ ] 3.4 Confirm with frontend dev (`documents/frontend_checklist.md`) that `chat.status` SSE events won't break the frontend
  - [ ] 3.5 Reassociate Elastic IP `54.66.111.21` from old EC2 to new EC2
    ```bash
    aws ec2 associate-address --region ap-southeast-2 --instance-id <NEW_INSTANCE_ID> --allocation-id eipalloc-0d54fe66102d6007c
    ```
  - [ ] 3.6 Terminate old EC2 instance

- [ ] Task 4: Verify deployment
  - [ ] 4.1 Health check
    ```bash
    curl http://54.66.111.21:8000/health
    ```
  - [ ] 4.2 Test parcel session (KG integration end-to-end)
    ```bash
    curl -X POST http://54.66.111.21:8000/chat/stream/ \
      -H "Content-Type: application/json" \
      -d '{"session_id": "433375739::deploy_test", "message": "Is this a priority site?"}' \
      --no-buffer
    ```
  - [ ] 4.3 Verify KG resolution works (check logs for "Querying KG for PFI")
    ```bash
    ssh -i ~/.ssh/race_lits_server.pem ec2-user@54.66.111.21
    journalctl -u veris-chat --no-pager | grep "KG"
    ```
  - [ ] 4.4 Clean up test session
    ```bash
    curl -X DELETE "http://54.66.111.21:8000/chat/sessions/433375739::deploy_test?clear_parcel_cache=true"
    ```

- [ ] Task 5: Final checkpoint — deployment complete when:
  - [ ] RAG EC2 running with veris-chat service active at `http://54.66.111.21:8000`
  - [ ] KG EC2 running at `54.253.127.203` (Neo4j accessible from RAG EC2)
  - [ ] Parcel session resolves KG data and streams response with citations
  - [ ] Memory works with `::` session IDs (collection name sanitized)

- [x] Task 6: Create GitHub Actions workflow for auto-deploy on push
  - [x] 6.1 Create `.github/workflows/deploy.yml`
    - Trigger: push to `deploy-clean` branch (triggered by `push_clean.sh`)
    - Steps: SSH into existing RAG EC2, git fetch + reset --hard, sed Neo4j URI, restart service
    - Secrets needed: `EC2_SSH_KEY`, `EC2_HOST`
  - [x] 6.2 Add GitHub repo secrets at https://github.com/AEA-MapTalk/veris-chat/settings/secrets/actions
    - `EC2_HOST`: `54.66.111.21`
    - `EC2_SSH_KEY`: contents of `~/.ssh/race_lits_server.pem`
    ```bash
    gh secret set EC2_HOST --repo AEA-MapTalk/veris-chat --body "54.66.111.21"
    gh secret set EC2_SSH_KEY --repo AEA-MapTalk/veris-chat < ~/.ssh/race_lits_server.pem
    ```
  - [x] 6.3 Document limitations (see Notes below)

- [x] Task 7: Update README.md
  - [x] 7.1 Add deployment section with KG EC2 info
  - [x] 7.2 Add GitHub Actions auto-deploy info
  - [x] 7.3 Update test commands with parcel session examples

## Notes

### GitHub Actions Limitations — When Auto-Deploy Won't Work

The GitHub Actions workflow does a `git pull` + `systemctl restart` on the existing EC2 instance. This will NOT work or be unreliable when:

1. **Python dependency changes** (`pyproject.toml` modified) — new packages won't be installed. Need to SSH in and run `pip3.11 install --user <new-package>` manually, or terminate and relaunch with updated `user_data.sh`.
2. **EC2 instance terminated/replaced** — the workflow SSHs to a fixed IP. If the instance is replaced, the SSH host key changes and the workflow fails. Need to update the GitHub secret or re-associate the Elastic IP.
3. **System-level changes** — anything requiring `dnf install`, systemd service file changes, or OS-level config won't be handled by git pull + restart.
4. **Neo4j config changes** — if `config.yaml` neo4j URI needs changing (e.g., KG moves to a different IP), the sed in `user_data.sh` won't re-run. Need to manually edit `config.yaml` on EC2.
5. **Environment variable changes** — new `.env` variables won't be added. Need to SSH in and edit `.env` manually.
6. **KG EC2 is stopped** — the RAG pipeline will fail to connect to Neo4j. KG instance must be started separately before deploying.
7. **Branch divergence** — if `deploy-clean` branch gets force-pushed (as `push_clean.sh` does), `git pull` may fail. The workflow uses `git fetch + git reset --hard` to handle this.
