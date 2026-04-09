#!/bin/bash
# EC2 User-Data Script for Neo4j Knowledge Graph
# Deploys Neo4j with neosemantics (n10s) plugin and loads Victoria Unearthed KG data.
#
# ============================================================================
# EC2 DEPLOYMENT CONFIGURATION
# ============================================================================
# - Instance type: t3.small (2 vCPU, 2 GB RAM — Neo4j is read-only after load)
# - AMI: ami-00a51cc7a8cd53e3f (Amazon Linux 2023, ap-southeast-2)
# - Key pair: race_lits_server
# - Region: ap-southeast-2 (Sydney)
# - EBS: 30 GB gp3 (RDF data + Neo4j storage)
# - Security group: sg-0fed3f02e16c4f50e (reuse veris-chat SG)
# - Elastic IP: 54.253.127.203 (eipalloc-02018394cbf88895b)
#
# Access:
# - SSH: port 22 (via SSH tunnel for Neo4j access)
# - Neo4j Bolt: port 7687 (via SSH tunnel: ssh -L 7687:localhost:7687)
# - Neo4j Browser: port 7474 (via SSH tunnel: ssh -L 7474:localhost:7474)
# ============================================================================

set -e
exec > /var/log/user-data.log 2>&1

echo "=== Neo4j KG EC2 Setup Started ==="

# ============================================================================
# CONFIGURATION
# ============================================================================
GIT_TOKEN="${GIT_TOKEN}"
APP_DIR="/home/ec2-user/vic_unearthed_kg"
REPO_URL="https://${GIT_TOKEN}@github.com/AEA-MapTalk/vic_unearthed_kg.git"
BRANCH="main"

# OneDrive download URL for bulk RDF data (output/ folder)
# This URL should be set as env var before running envsubst
ONEDRIVE_URL="${ONEDRIVE_URL}"

# ============================================================================
# SYSTEM SETUP
# ============================================================================
echo "=== Updating system packages ==="
dnf update -y

echo "=== Installing Docker ==="
dnf install -y docker git
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# Install docker-compose plugin
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

echo "=== Cloning repository ==="
cd /home/ec2-user
git clone --depth 1 --branch ${BRANCH} ${REPO_URL} vic_unearthed_kg
chown -R ec2-user:ec2-user ${APP_DIR}

# ============================================================================
# DOWNLOAD RDF DATA FROM S3
# ============================================================================
echo "=== Downloading RDF data from S3 ==="
mkdir -p ${APP_DIR}/output
aws s3 cp s3://veris-kg-data/output/ ${APP_DIR}/output/ --recursive
echo "=== RDF data downloaded ==="

# ============================================================================
# BUILD AND START NEO4J
# ============================================================================
echo "=== Building Neo4j Docker image ==="
cd ${APP_DIR}
docker compose -f neo4j_docker/docker-compose.yml build

echo "=== Starting Neo4j container ==="
docker compose -f neo4j_docker/docker-compose.yml up -d

echo "=== Waiting for Neo4j to be ready ==="
for i in $(seq 1 30); do
  if docker compose -f neo4j_docker/docker-compose.yml exec -T neo4j cypher-shell -u neo4j -p neo4jpassword "RETURN 1" 2>/dev/null; then
    echo "Neo4j is ready!"
    break
  fi
  echo "Waiting for Neo4j... ($i/30)"
  sleep 5
done

# ============================================================================
# SYSTEMD SERVICE (auto-start Neo4j on reboot)
# ============================================================================
echo "=== Creating systemd service for Neo4j ==="
tee /etc/systemd/system/neo4j-kg.service << EOF
[Unit]
Description=Neo4j Knowledge Graph (Docker Compose)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=ec2-user
WorkingDirectory=${APP_DIR}
ExecStart=/usr/local/lib/docker/cli-plugins/docker-compose -f neo4j_docker/docker-compose.yml up -d
ExecStop=/usr/local/lib/docker/cli-plugins/docker-compose -f neo4j_docker/docker-compose.yml down

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable neo4j-kg

echo "=== Neo4j KG EC2 Setup Complete ==="
echo "Neo4j is running. Upload RDF data and run load_data.sh to populate."
echo "Access via SSH tunnel: ssh -L 7687:localhost:7687 -L 7474:localhost:7474 ec2-user@<IP>"
