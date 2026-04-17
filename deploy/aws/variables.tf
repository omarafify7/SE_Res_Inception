variable "region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "ml-inference"
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU inference workers"
  type        = string
  default     = "g6.xlarge"
}

variable "inference_image" {
  description = "ECR image URI for inference service"
  type        = string
}

variable "gateway_image" {
  description = "ECR image URI for gateway service"
  type        = string
}
