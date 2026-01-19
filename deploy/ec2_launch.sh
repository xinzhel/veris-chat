#!/bin/bash
# EC2 Launch Script for Veris-Chat
# Usage: 
#   bash deploy/ec2_launch.sh           # Launch EC2 instance
#   bash deploy/ec2_launch.sh --add-ip  # Add local IP to security group
#   bash deploy/ec2_launch.sh --terminate  # Terminate veris-chat instance
#
# Prerequisites:
# - AWS CLI configured with ap-southeast-2 access
# - Environment variables set in ~/.zshrc: GIT_TOKEN, QDRANT_URL, QDRANT_API_KEY
# - Key pair 'race_lits_server' exists in ap-southeast-2

set -e

REGION="ap-southeast-2"
AMI_ID="ami-00a51cc7a8cd53e3f"
INSTANCE_TYPE="t3.medium"
KEY_NAME="race_lits_server"
SECURITY_GROUP="sg-0fed3f02e16c4f50e"
INSTANCE_PROFILE="rmit-workload-veris"
ELASTIC_IP_ALLOC="eipalloc-0d54fe66102d6007c"
ELASTIC_IP="54.66.111.21"

echo "=== Veris-Chat EC2 Launch Script ==="

# Option: Add local IP to security group only
if [ "$1" == "--add-ip" ]; then
    LOCAL_IP=$(curl -s https://checkip.amazonaws.com)
    echo "Adding local IP $LOCAL_IP to security group..."
    
    # Add SSH (22) and API (8000) access for local IP
    aws ec2 authorize-security-group-ingress --region $REGION \
        --group-id $SECURITY_GROUP \
        --protocol tcp --port 22 --cidr "$LOCAL_IP/32" 2>/dev/null || echo "SSH rule already exists"
    
    aws ec2 authorize-security-group-ingress --region $REGION \
        --group-id $SECURITY_GROUP \
        --protocol tcp --port 8000 --cidr "$LOCAL_IP/32" 2>/dev/null || echo "API rule already exists"
    
    echo "Security group updated for IP: $LOCAL_IP"
    exit 0
fi

# Option: Terminate veris-chat instance
if [ "$1" == "--terminate" ]; then
    echo "Finding veris-chat instance..."
    INSTANCE_ID=$(aws ec2 describe-instances --region $REGION \
        --filters "Name=tag:Name,Values=veris-chat" "Name=instance-state-name,Values=running,pending,stopped" \
        --query 'Reservations[0].Instances[0].InstanceId' --output text)
    
    if [ "$INSTANCE_ID" == "None" ] || [ -z "$INSTANCE_ID" ]; then
        echo "No veris-chat instance found"
        exit 1
    fi
    
    echo "Terminating instance: $INSTANCE_ID"
    aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID
    echo "Instance $INSTANCE_ID terminated"
    exit 0
fi

# Check environment variables
if [ -z "$GIT_TOKEN" ] || [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ]; then
    echo "Error: Missing environment variables. Run: source ~/.zshrc"
    exit 1
fi

# Generate user-data with secrets substituted
echo "Generating user-data with secrets..."
envsubst '${GIT_TOKEN} ${QDRANT_URL} ${QDRANT_API_KEY}' < deploy/user_data.sh > /tmp/user_data.sh

# Launch EC2 instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --region $REGION \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --iam-instance-profile Name=$INSTANCE_PROFILE \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --user-data file:///tmp/user_data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=veris-chat}]' \
  --query 'Instances[0].InstanceId' --output text)

echo "Instance ID: $INSTANCE_ID"

# Clean up temp file
rm /tmp/user_data.sh

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
echo "User-data setup takes ~5-10 minutes. Monitor with:"
echo "  ssh-keygen -R $ELASTIC_IP"
echo "  ssh -i ~/.ssh/race_lits_server.pem ec2-user@$ELASTIC_IP"
echo "  sudo tail -f /var/log/user-data.log"
echo ""
echo "Test API:"
echo "  curl http://$ELASTIC_IP:8000/health"
