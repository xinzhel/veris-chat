#!/bin/bash
# EC2 User-Data Script for Veris-Chat
# This script runs once on first boot to set up the application.
#
# This file contains secrets - keep local only (excluded in .gitignore)
#
# ============================================================================
# EC2 DEPLOYMENT CONFIGURATION
# ============================================================================
#
# Instance Configuration
# ----------------------
# - Instance type: t3.medium (2 vCPU, 4 GB RAM - suitable for light usage)
#   Alternative: t3.large (2 vCPU, 8 GB RAM - recommended for production)
# - AMI: ami-00a51cc7a8cd53e3f (Amazon Linux 2023 for ap-southeast-2)
# - Key pair: race_lits_server (for SSH access)
# - Region: ap-southeast-2 (Sydney)
#
# | Instance | vCPU | Memory | Price/hour | Monthly (24/7) |
# |----------|------|--------|------------|----------------|
# | t3.medium | 2 | 4 GB | ~$0.052 | ~$38 |
# | t3.large | 2 | 8 GB | ~$0.104 | ~$76 |
# resize later if needed:
# ```bash
# aws ec2 stop-instances --instance-id <ID>
# aws ec2 modify-instance-attribute --instance-id <ID> --instance-type t3.large
# aws ec2 start-instances --instance-id <ID>
# ```

# EBS Volume (Root Disk)
# ----------------------
# - Size: 30 GB (8 GB default is too small for Python deps + PDFs)
# - Type: gp3 (General purpose SSD, better price/performance than gp2)
#
# IAM Instance Profile
# --------------------
# - Name: rmit-workload-veris
# - Provides automatic credential rotation for Bedrock access
# - Created by Dale (RACE) with AmazonBedrockFullAccess policy
#
# Security Group
# --------------
# - ID: sg-0fed3f02e16c4f50e (already created)
#
# Application
# -----------
# - Port: 8000
# - Working directory: /home/ec2-user/veris-chat
# - Python version: 3.11
# - Process manager: systemd
# - Service name: veris-chat
#
# GitHub Repository
# -----------------
# - URL: https://github.com/AEA-MapTalk/veris-chat.git
# - Branch: deploy-clean
#
# ============================================================================
# DEPLOYMENT CHECKLIST
# ============================================================================
# [x] 1. Instance Profile: rmit-workload-veris
# [x] 2. Key pair: race_lits_server (in ap-southeast-2)
# [x] 3. Security group: sg-0fed3f02e16c4f50e
# [x] 4. AMI: ami-00a51cc7a8cd53e3f (Amazon Linux 2023 for ap-southeast-2)
#
# ============================================================================


# Exit immediately if any command fails (non-zero exit code). Prevents script from continuing after an error.
set -e

# Redirects all output (stdout and stderr) to a log file.
exec > /var/log/user-data.log 2>&1

echo "=== Veris-Chat EC2 Setup Started ==="

# ============================================================================
# CONFIGURATION - Uses environment variables from local machine
# Launch with: envsubst < deploy/user_data.sh > /tmp/user_data.sh && aws ec2 run-instances ... --user-data file:///tmp/user_data.sh
# ============================================================================
GIT_TOKEN="${GIT_TOKEN}"
QDRANT_URL="${QDRANT_URL}"
QDRANT_API_KEY="${QDRANT_API_KEY}"

APP_DIR="/home/ec2-user/veris-chat"
REPO_URL="https://${GIT_TOKEN}@github.com/AEA-MapTalk/veris-chat.git"
BRANCH="deploy-clean"

# ============================================================================
# SYSTEM SETUP
# ============================================================================
echo "=== Updating system packages ==="
dnf update -y
# want to use 3.10. But Amazon Linux 2023 的官方仓库只提供 3.9、3.11、3.12、3.13，跳过了 3.10。
echo "=== Installing Python 3.11 and dependencies ==="
dnf install -y python3.11 python3.11-pip git

# ============================================================================
# APPLICATION SETUP
# ============================================================================
# Why pip instead of conda (from environment.yaml)?
# - Install size: pip ~0 (comes with Python) vs conda ~500MB (Miniconda)
# - Setup time: pip ~2-3 min vs conda ~5-8 min
# - Complexity: pip is simple vs conda needs download, init shell, create env
# - Activation: pip not needed vs conda must activate before running app
# - Systemd: pip works direct vs conda needs wrapper script to activate env
echo "=== Cloning repository ==="
cd /home/ec2-user
git clone --depth 1 --branch ${BRANCH} ${REPO_URL} veris-chat
chown -R ec2-user:ec2-user ${APP_DIR}

echo "=== Creating .env file ==="
cat > ${APP_DIR}/.env << EOF
QDRANT_URL="${QDRANT_URL}"
QDRANT_API_KEY="${QDRANT_API_KEY}"
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_SESSION_TOKEN=""
AWS_REGION=us-east-1
EOF
chown ec2-user:ec2-user ${APP_DIR}/.env

echo "=== Installing Python dependencies ==="
cd ${APP_DIR}
sudo -u ec2-user pip3.11 install --user --upgrade pip
sudo -u ec2-user pip3.11 cache purge
sudo -u ec2-user pip3.11 install --user  -v --no-cache-dir \
    pyyaml>=6.0 \
    requests>=2.31.0 \
    llama-index>=0.9.0 \
    llama-index-llms-bedrock \
    llama-index-embeddings-bedrock \
    llama-index-vector-stores-qdrant \
    qdrant-client>=1.7.0 \
    boto3>=1.34.0 \
    python-dotenv>=1.0.0 \
    pymupdf \
    llama-index-llms-bedrock-converse \
    mem0ai \
    fastapi \
    uvicorn

# ============================================================================
# SYSTEMD SERVICE SETUP
# sudo cat > file - > 重定向是 shell 做的，不受 sudo 影响。
# 所有shell 先用你的权限打开文件，然后 sudo 运行 cat。文件打开发生在 sudo 之前，所以没权限。
# sudo tee file - tee 命令本身以 root 权限运行并打开文件，所以有权限写入。
# ============================================================================
echo "=== Creating systemd service ==="
tee /etc/systemd/system/veris-chat.service << EOF
[Unit]
Description=Veris Chat API Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=${APP_DIR}
Environment="PATH=/home/ec2-user/.local/bin:/usr/bin"
ExecStart=/home/ec2-user/.local/bin/uvicorn app.chat_api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "=== Enabling and starting service ==="
systemctl daemon-reload
systemctl enable veris-chat
systemctl start veris-chat

echo "=== Veris-Chat EC2 Setup Complete ==="
echo "Service status:"
systemctl status veris-chat --no-pager
