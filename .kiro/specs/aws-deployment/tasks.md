# Implementation Plan: AWS Deployment for Veris-Chat

## Overview

Deploy veris-chat on AWS EC2 with persistent Bedrock access using IAM Instance Profile. Configure Claude Opus 4.5 access via `us.` inference profile to work around RMIT SCP restrictions.

## Background & Context

### Issue 1: SSO Credentials Don't Work for Long-Running Services

**Problem**: Local development uses AWS SSO (`aws sso login`) which provides temporary credentials requiring interactive browser login. This doesn't work for deployed services.

**Root Cause**: 
- User is in IAM Identity Center (AWS SSO) at RMIT organization level
- SSO provides temporary credentials that expire and require browser re-authentication
- Long-running services need non-interactive credential refresh

**Solution**: Use EC2 Instance Profile with IAM Role
- Instance Profile automatically provides credentials via EC2 metadata service
- Credentials auto-refresh without human interaction
- IAM Role with `AmazonBedrockFullAccess` policy already created by Dale (RACE)

### Issue 2: Claude Opus 4.5 Access Blocked

**Problem**: Direct invocation of Opus 4.5 was blocked:
```
AccessDeniedException: User arn:aws:sts::554674964376:assumed-role/AWSReservedSSO_RMIT-ResearchAdmin_xxx 
is not authorized to perform: bedrock:InvokeModel on resource: 
arn:aws:bedrock:::foundation-model/anthropic.claude-opus-4-5-20251101-v1:0 
with an explicit deny in a service control policy
```

**Root Cause**: RMIT's Service Control Policy (SCP) blocks:
- Direct model invocation (no prefix)
- `global.*` inference profiles

**Solution**: Use `us.` inference profile prefix
```bash
# Blocked
aws bedrock-runtime converse --model-id anthropic.claude-opus-4-5-20251101-v1:0 ...
aws bedrock-runtime converse --model-id global.anthropic.claude-opus-4-5-20251101-v1:0 ...

# Works
aws bedrock-runtime converse --region us-east-1 --model-id us.anthropic.claude-opus-4-5-20251101-v1:0 ...
```

### AWS Account Structure Reference

```
RMIT AWS Organization
├── Management Account (RMIT IT/RACE)
│   └── IAM Identity Center (AWS SSO)
│       ├── Users (including xinzhe.li)
│       ├── Permission Sets (RMIT-ResearchAdmin)
│       └── Service Control Policies (SCPs)
│
└── Member Account: 554674964376 (ri-research-rmitmetaverse)
    └── IAM
        ├── Auto-generated Role (from SSO login) - temporary credentials
        └── Standalone Role (for EC2) - with AmazonBedrockFullAccess ← USE THIS
```

### EC2 User-Data

User-data is a script that runs automatically when an EC2 instance first boots. It's used to:
- Install software and dependencies
- Clone application code
- Create `.env` file with secrets (persists on EBS disk)
- Configure systemd service

The script runs as root and only executes once on initial launch (not on reboot).
Files created by user-data (like `.env`) persist on the EBS disk across stop/start cycles.

### Systemd Service

Systemd is Linux's service manager. A systemd service file defines how to run the application:
- Auto-start on boot (reads `.env` from disk each time)
- Auto-restart if the app crashes
- Manage logs via journalctl
- Start/stop/restart via `systemctl` commands

**Stop/Start Flow:**
1. First boot: user-data creates `.env` on disk, sets up systemd
2. Stop EC2: instance stops, EBS disk retains `.env`
3. Start EC2: systemd starts app, app reads `.env` from disk - no manual intervention needed

## Tasks

- [x] 1. Configure Claude Opus 4.5 Access
  - [x] 1.1 Update config.yaml and .env with correct model ID and region
    - Set `AWS_REGION=us-east-1` in `.env`
    - Set model ID to `us.anthropic.claude-opus-4-5-20251101-v1:0` in `config.yaml`
    - Add comments documenting the `us.` prefix requirement
  - [x] 1.2 Verify config.py handles region configuration correctly
    - Ensure `get_bedrock_kwargs()` passes region to clients from `.env`
    - Confirm SSO fallback logic works for local development
  - [x] 1.3 Test Opus 4.5 invocation locally
    - Run test script with SSO credentials (`script/test_bedrock_converse_async.py`)
    - ✓ All 4 tests passed: streaming (3.43s), async (3.18s), sync (2.77s), concurrent (3.45s)

- [x] 2. Checkpoint - Verify local Opus 4.5 access works before EC2 setup
  - ✓ Verified via Task 1.3 test results

- [x] 3. Prepare EC2 Deployment Configuration
  - [x] 3.1 Create deployment configuration file (`deploy/user_data.sh` header comments)
    - Document Instance Profile: rmit-workload-veris
    - Document security group: sg-0fed3f02e16c4f50e
    - Document instance type: t3.medium (default) or t3.large
    - Document AMI: ami-00a51cc7a8cd53e3f (Amazon Linux 2023 for ap-southeast-2)
    - Document EBS volume: 30 GB gp3
  - [x] 3.2 Create EC2 user-data startup script (`deploy/user_data.sh`)
    - Update system packages (dnf update)
    - Install Python 3.11, pip, git (AL2023 doesn't have 3.10)
    - Clone veris-chat repository from deploy-clean branch (using GIT_TOKEN via envsubst)
    - Install Python dependencies via pip
    - Create `.env` file with secrets
    - Create and enable systemd service
    - Start the application on port 8000
  - [x] 3.3 Create GitHub deployment script (`deploy/push_clean.sh`)
    - Push code to deploy-clean branch with single commit (no history)
    - Preserves local commit history on main branch
  - [x] 3.4 Create EC2 launch script (`deploy/ec2_launch.sh`)
    - Uses envsubst to substitute secrets from environment variables
    - Options: launch, --add-ip (add local IP to security group), --terminate

- [-] 4. Create EC2 Setup Scripts (merged into Task 3.2)
  - [x] 4.1 System setup included in `deploy/user_data.sh`
  - [x] 4.2 Application deployment included in `deploy/user_data.sh`
  - [x] 4.3 Systemd service file created by `deploy/user_data.sh`

- [x] 5. Checkpoint - Review deployment scripts before EC2 launch
  - ✓ `deploy/user_data.sh` - EC2 setup script using envsubst for secrets
  - ✓ `deploy/ec2_launch.sh` - Launch script with --add-ip and --terminate options
  - ✓ `deploy/push_clean.sh` - GitHub deployment script
  - ✓ `.env.example` - template for environment variables
  - ✓ Secrets stored in ~/.zshrc as environment variables (GIT_TOKEN, QDRANT_URL, QDRANT_API_KEY)

- [ ] 6. EC2 Instance Verification
  - [x] 6.1 Push code to deploy-clean branch
    - Run `bash deploy/push_clean.sh` to push latest code to `deploy` remote
    - Verify branch exists: `git ls-remote --heads deploy deploy-clean`
  - [x] 6.2 Launch EC2 instance
    - Run `bash deploy/ec2_launch.sh` to launch instance and attach Elastic IP
    - Instance: t3.medium, AMI: ami-00a51cc7a8cd53e3f (AL2023), Region: ap-southeast-2
    - Elastic IP: 54.66.111.21
  - [ ] 6.3 SSH into instance and verify setup
    - SSH: `ssh -i ~/.ssh/race_lits_server.pem ec2-user@54.66.111.21`
    - Check user-data log: `sudo cat /var/log/user-data.log`
    - Check service status: `systemctl status veris-chat`
    - Check app logs: `journalctl -u veris-chat -f`
  - [ ] 6.4 Test API endpoints
    - Health check: `curl http://54.66.111.21:8000/health`
    - Streaming chat test:
      ```bash
      curl -X POST http://54.66.111.21:8000/chat/stream/ \
        -H "Content-Type: application/json" \
        -d '{"session_id": "ec2test", "message": "Hello, are you working?"}' \
        --no-buffer
      ```
  - [ ] 6.5 Verify Bedrock access via Instance Profile
    - No AWS credentials in `.env` (empty strings)
    - Instance Profile provides credentials automatically
    - Check for "ValidationException: invalid model identifier" error
      - If occurs: verify `AWS_REGION=us-east-1` in `.env` on EC2
      - Opus 4.5 with `us.*` prefix requires us-east-1 region
  - [x] 6.6 Health check endpoint exists in `app/chat_api.py`
    - GET `/health` returns `{"status": "healthy"}`
- [ ] 7. Final Checkpoint - Deployment complete when:
  - [ ] EC2 instance running with veris-chat service active
  - [ ] API accessible at `http://54.66.111.21:8000`
  - [ ] Bedrock calls working via Instance Profile (no explicit credentials)

## Notes

- Instance Profile with `AmazonBedrockFullAccess` is already created
- No AWS credentials needed in code/config when running on EC2 with Instance Profile
- Use `us-east-1` region for Opus 4.5 access (RMIT SCP allows `us.` prefix)