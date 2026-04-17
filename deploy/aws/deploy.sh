#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== ML Inference AWS Deployment ==="
echo "1. Initializing Terraform..."
terraform init

echo "2. Building and pushing Docker images..."
# Get ECR login
REGION=$(terraform output -raw region 2>/dev/null || echo "us-east-1")
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "3. Planning infrastructure..."
terraform plan -out=tfplan

echo "4. Applying infrastructure..."
terraform apply tfplan

echo ""
echo "=== Deployment Complete ==="
echo "ALB URL: $(terraform output -raw alb_dns_name)"
echo "Test: curl -F 'file=@test.jpg' http://$(terraform output -raw alb_dns_name)/api/v1/classify"
echo ""
echo "To destroy: ./destroy.sh"
