# Kubernetes Manifests - ML Inference System

## Architecture

This system deploys two services: an **Inference** service (Python/FastAPI with GPU) that runs the ML model, and a **Gateway** service (Go/Fiber, CPU-only) that provides the public API. The gateway forwards classification requests to the inference service over the cluster network. An HPA auto-scales the gateway from 2 to 10 replicas based on CPU load.

## Prerequisites

- `kubectl` configured with access to a Kubernetes cluster
- A node pool with NVIDIA GPUs and the [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) installed
- An nginx ingress controller deployed in the cluster (for the Ingress resource)

## Building Images

```bash
# From the project root
docker build -t ml-inference:latest -f inference/Dockerfile .
docker build -t ml-gateway:latest -f gateway/Dockerfile .
```

Push to your container registry and update the image references in the deployment manifests (or use Kustomize / CI to override).

## Local Validation (Dry Run)

Kind and other local clusters do not support GPUs. You can still validate manifest syntax:

```bash
kubectl apply --dry-run=client -f k8s/
```

## Deploying

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
```

Verify rollout:

```bash
kubectl -n ml-inference get pods
kubectl -n ml-inference get svc
kubectl -n ml-inference get ingress
```

## Cloud Deployment Notes

- **AWS**: Use `g6.xlarge` instances (L4 GPU) in a managed node group with the NVIDIA GPU Operator.
- **Azure**: Use `NC4as_T4_v3` VMs in an AKS GPU node pool.
- GPU scheduling requires the `nvidia.com/gpu` resource to be advertised by nodes. Without the GPU Operator, inference pods will remain Pending.
- For production deployment scripts and Terraform/Helm configurations, see the `deploy/` directory.
