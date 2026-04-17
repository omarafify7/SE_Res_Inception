output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "inference_ecr_repo" {
  description = "ECR repository URI for the inference service"
  value       = aws_ecr_repository.inference.repository_url
}

output "gateway_ecr_repo" {
  description = "ECR repository URI for the gateway service"
  value       = aws_ecr_repository.gateway.repository_url
}

output "region" {
  description = "AWS region"
  value       = var.region
}
