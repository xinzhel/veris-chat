# Local Dev vs EC2 Deploy: Service Connection Setup

## Problem

Local dev (company wifi) blocks direct access to:
- Qdrant Cloud (port 6333) — blocked by firewall
- Neo4j on EC2 (port 7687) — no public access without tunnel

EC2 deploy can access both directly (same network / public internet).

## Design

Use environment variables to switch between local (tunneled) and EC2 (direct) connections.

| Service | Local Dev (via SSH tunnel) | EC2 Deploy (direct) |
|---------|--------------------------|---------------------|
| Qdrant Cloud | `localhost:6333` via tunnel, `QDRANT_TUNNEL=true` | Direct to `QDRANT_URL`, no tunnel |
| Neo4j KG | `bolt://localhost:7687` via tunnel | `bolt://54.253.127.203:7687` (set by `deploy/user_data.sh` sed) |

### SSH Tunnel (local dev)

One command starts both tunnels:
```bash
ssh -fN \
  -L 7687:localhost:7687 \
  -L 6333:629f296f-654f-4576-ab93-e335d1ab3641.ap-southeast-2-0.aws.cloud.qdrant.io:6333 \
  -i ~/.ssh/race_lits_server.pem ec2-user@54.253.127.203
```

### Why QDRANT_TUNNEL is needed

SSH tunnel forwards `localhost:6333` → Qdrant Cloud. But Qdrant Cloud uses HTTPS with a certificate signed for `629f296f-...qdrant.io`, not `localhost`. Python's SSL verification rejects the mismatch.

When `QDRANT_TUNNEL=true`:
- Connect to `host=localhost, port=6333` (through tunnel)
- Set `verify=False` to skip SSL certificate hostname check
- API key still authenticates the request

When `QDRANT_TUNNEL` is not set (EC2 deploy):
- Connect directly to `QDRANT_URL` with normal SSL verification

### Environment Variables

`.env` for local dev:
```
QDRANT_URL=https://629f296f-...qdrant.io:6333
QDRANT_API_KEY=...
QDRANT_TUNNEL=true
```

`.env` on EC2 (set by `deploy/user_data.sh`):
```
QDRANT_URL=https://629f296f-...qdrant.io:6333
QDRANT_API_KEY=...
# QDRANT_TUNNEL not set → direct connection
```

## Tasks

- [x] Task 1: Add tunnel-aware Qdrant client
  - [x] `rag_core/chat/retriever.py` `get_qdrant_client()`: if `QDRANT_TUNNEL=true`, connect via `localhost:6333` with `verify=False`
  - [x] `rag_core/ingestion/main_client.py`: same tunnel logic
  - [x] Add `QDRANT_TUNNEL=true` to `.env`
  - [x] Verified: `curl -k https://localhost:6333/collections` works through tunnel

- [ ] Task 2: Test end-to-end through tunnel
  - [ ] Restart uvicorn server
  - [ ] Test `/chat/stream/` with parcel session (KG + Qdrant + Bedrock)
  - [ ] Verify session log captures full flow

- [ ] Task 3: Create helper script `script/start_tunnels.sh`
  - [ ] Combined tunnel: Neo4j + Qdrant Cloud through EC2
  - [ ] Check if tunnels already running before starting
  - [ ] Print status after starting

- [ ] Task 4: Document in README
  - [ ] Local dev setup: run tunnels, set QDRANT_TUNNEL=true, start server
  - [ ] EC2 deploy: direct connections, no tunnels needed
