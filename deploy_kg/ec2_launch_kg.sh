#!/bin/bash
# EC2 Launch Script for Neo4j Knowledge Graph
# Usage:
#   bash deploy_kg/ec2_launch_kg.sh           # Launch EC2 instance
#   bash deploy_kg/ec2_launch_kg.sh --add-ip  # Add local IP to security group
#   bash deploy_kg/ec2_launch_kg.sh --terminate  # Terminate neo4j-kg instance
#   bash deploy_kg/ec2_launch_kg.sh --status  # Check instance status
#
# Prerequisites:
# - AWS CLI configured with ap-southeast-2 access
# - GIT_TOKEN set in environment (for cloning private repo)
# - Key pair 'race_lits_server' exists in ap-southeast-2

set -e

REGION="ap-southeast-2"
AMI_ID="ami-00a51cc7a8cd53e3f"
INSTANCE_TYPE="t3.medium"
KEY_NAME="race_lits_server"
SECURITY_GROUP="sg-0fed3f02e16c4f50e"
INSTANCE_PROFILE="rmit-workload-veris"
ELASTIC_IP_ALLOC="eipalloc-02018394cbf88895b"
ELASTIC_IP="54.253.127.203"
INSTANCE_TAG="neo4j-kg"

echo "=== Neo4j KG EC2 Launch Script ==="

# Option: Add local IP to security group
if [ "$1" == "--add-ip" ]; then
    LOCAL_IP=$(curl -s https://checkip.amazonaws.com)
    echo "Adding local IP $LOCAL_IP to security group..."
    
    aws ec2 authorize-security-group-ingress --region $REGION \
        --group-id $SECURITY_GROUP \
        --protocol tcp --port 22 --cidr "$LOCAL_IP/32" 2>/dev/null || echo "SSH rule already exists"
    
    echo "Security group updated for IP: $LOCAL_IP"
    exit 0
fi

# Option: Check status
if [ "$1" == "--status" ]; then
    echo "Finding $INSTANCE_TAG instance..."
    aws ec2 describe-instances --region $REGION \
        --filters "Name=tag:Name,Values=$INSTANCE_TAG" "Name=instance-state-name,Values=running,pending,stopped" \
        --query "Reservations[].Instances[].[InstanceId,PublicIpAddress,State.Name,LaunchTime]" \
        --output table
    exit 0
fi

# Option: Terminate instance
if [ "$1" == "--terminate" ]; then
    echo "Finding $INSTANCE_TAG instance..."
    INSTANCE_ID=$(aws ec2 describe-instances --region $REGION \
        --filters "Name=tag:Name,Values=$INSTANCE_TAG" "Name=instance-state-name,Values=running,pending,stopped" \
        --query 'Reservations[0].Instances[0].InstanceId' --output text)
    
    if [ "$INSTANCE_ID" == "None" ] || [ -z "$INSTANCE_ID" ]; then
        echo "No $INSTANCE_TAG instance found"
        exit 1
    fi
    
    echo "Terminating instance: $INSTANCE_ID"
    aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID
    echo "Instance $INSTANCE_ID terminated"
    exit 0
fi

# Check environment variables
if [ -z "$GIT_TOKEN" ]; then
    echo "Error: GIT_TOKEN not set. Run: export GIT_TOKEN=<your-token>"
    exit 1
fi

# Generate user-data with secrets substituted
echo "Generating user-data with secrets..."
envsubst '${GIT_TOKEN}' < deploy_kg/user_data.sh > /tmp/user_data_kg.sh

# Launch EC2 instance
echo "Launching EC2 instance ($INSTANCE_TYPE)..."
INSTANCE_ID=$(aws ec2 run-instances \
  --region $REGION \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --iam-instance-profile Name=$INSTANCE_PROFILE \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --user-data file:///tmp/user_data_kg.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_TAG}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Instance ID: $INSTANCE_ID"

# Clean up temp file
rm /tmp/user_data_kg.sh

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Attach Elastic IP
echo "Attaching Elastic IP $ELASTIC_IP..."
aws ec2 associate-address --region $REGION --instance-id $INSTANCE_ID --allocation-id $ELASTIC_IP_ALLOC

echo ""
echo "=== Launch Complete ==="
echo "Instance ID: $INSTANCE_ID"
echo "Elastic IP: $ELASTIC_IP"
echo ""
echo "User-data setup takes ~3-5 minutes. Monitor with:"
echo "  ssh -i ~/.ssh/race_lits_server.pem ec2-user@$ELASTIC_IP"
echo "  sudo tail -f /var/log/user-data.log"
echo ""
echo "After setup, upload RDF data:"
echo "  scp -i ~/.ssh/race_lits_server.pem neptune_deployment/vic_unearthed_kg/output/* ec2-user@$ELASTIC_IP:~/vic_unearthed_kg/output/"
echo "  ssh -i ~/.ssh/race_lits_server.pem ec2-user@$ELASTIC_IP 'cd ~/vic_unearthed_kg && ./neo4j_docker/load_data.sh'"
echo ""
echo "Access Neo4j via SSH tunnel:"
echo "  ssh -fN -L 7687:localhost:7687 -L 7474:localhost:7474 -i ~/.ssh/race_lits_server.pem ec2-user@$ELASTIC_IP"
echo "  # Then: bolt://localhost:7687 or http://localhost:7474"
