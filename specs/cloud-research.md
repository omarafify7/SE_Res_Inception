# Cloud GPU Deployment Research — GO/NO-GO Report

## Local Environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| GPU | NVIDIA RTX 5070 Ti (Blackwell, SM_120) |
| Docker base image | `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime` |
| CUDA requirement | 12.6+ (driver >= 525, recommended R570+) |

**Note**: SM_120 requires CUDA 12.8+ for local dev. Cloud targets (A10G=SM_86, L4=SM_89, T4=SM_75) are all compatible with CUDA 12.6.

---

## Compatibility Matrix

### Cloud GPU Instance Comparison

| Platform | Instance | GPU | vCPU | RAM | On-Demand $/hr | Spot $/hr (est.) | Monthly OD (730h) | Monthly Spot (~30%) |
|---|---|---|---|---|---|---|---|---|
| AWS | g5.xlarge | A10G 24GB (SM_86) | 4 | 16 GiB | $1.006 | ~$0.30–0.40 | ~$735 | ~$66–88 |
| AWS | g6.xlarge | L4 24GB (SM_89) | 4 | 16 GiB | $0.805 | ~$0.24–0.32 | ~$588 | ~$53–70 |
| AWS | inf2.xlarge | Inferentia2 | 4 | 16 GiB | $0.758 | N/A | ~$554 | — |
| Azure | NC4as T4 v3 | T4 16GB (SM_75) | 4 | 28 GiB | $0.526 | ~$0.19 | ~$384 | ~$42 |
| Azure | NC6s v3 | V100 16GB | 6 | 112 GiB | $3.060 | ~$0.57 | ~$2,234 | — |
| RunPod | Serverless T4 | T4 16GB | shared | shared | $0.40/hr per-sec | N/A | pay-per-use | ~$0–12 |
| Modal | Serverless T4 | T4 16GB | shared | shared | $0.59/hr per-sec | N/A | pay-per-use | ~$0 (free $30/mo) |
| Modal | Serverless A10G | A10G 24GB | shared | shared | $1.10/hr per-sec | N/A | pay-per-use | ~$0 (free $30/mo) |

### Gateway Hosting Comparison

| Platform | Option | Monthly Idle Cost | Notes |
|---|---|---|---|
| AWS | ECS Fargate (0.25 vCPU, 0.5 GB) | ~$4–6 | Always-on; no GPU |
| Azure | Container Apps (scale-to-zero) | ~$0–3 | Per-request billing |
| Fly.io | Free tier shared-CPU | $0 | 3 shared VMs included |

---

## GO/NO-GO Per Path

| Path | Verdict | Rationale |
|---|---|---|
| AWS g6.xlarge (L4) + ECS Fargate gateway + EC2 inference | **GO** | Best AWS price/performance; proven VPC pattern |
| AWS g5.xlarge (A10G) + ECS | CONDITIONAL GO | Valid alternative; $0.20/hr more expensive than g6 |
| AWS Inferentia inf2.xlarge | **NO-GO** | Neuron SDK compilation overhead unjustified for 5.6M param model |
| Azure AKS + NC4as T4 v3 GPU node pool | **GO** | Cheapest always-on cloud GPU; AKS managed GPU simplifies setup |
| Azure NC6s v3 (V100) | **NO-GO** | Massively overpriced |
| Azure Container Apps Serverless GPU | CONDITIONAL GO | Good scale-to-zero; pricing opacity is risk |
| RunPod Serverless (T4) | **GO** (demo/budget) | Best for <=30/mo; plain Docker image; per-second billing |
| Modal Serverless (T4 or A10G) | **GO** (demo/budget) | $30/mo free credit; excellent DX; thin SDK wrapper required |
| kind + GPU local Kubernetes | **NO-GO** | GPU passthrough not natively supported |

---

## Recommended Production Architecture (AWS)

```
Internet -> [ALB] -> [ECS Fargate: Go/Fiber Gateway, 0.25 vCPU, 512 MB]
                        |  (internal VPC, AWS Cloud Map, port 8000)
                        v
                   [ECS EC2 on g6.xlarge: FastAPI Inference, 1x L4 GPU]
```

**Cost estimate** (demo, 8h/day spot): g6.xlarge spot ~$0.28/hr x 8h x 30 = ~$67/mo + Fargate ~$5/mo + ALB ~$18/mo = **~$90/month**.

---

## Budget Path: <= $30/Month Idle

```
Internet -> [Fly.io free tier: Go/Fiber Gateway, ~$0/month]
                |  (HTTPS to Modal endpoint)
                v
           [Modal Serverless: FastAPI Inference, T4 GPU, scale-to-zero]
```

- Modal provides $30/month free compute credit (no credit card required)
- At $0.59/hr, covers ~50 hours of active GPU inference
- Scales to zero; cold starts 2-4s (warm) to 15-30s (cold with model load)
- **Total idle cost: ~$0**
- Requires adding `modal_app.py` (~20 lines) using `@app.asgi_app()`

**Alternative**: RunPod Serverless (T4, $0.40/hr) — push Docker image directly, no SDK wrapper needed. $30 = ~75 hours.

---

## NVIDIA Container Toolkit on Windows + Docker Desktop

### Prerequisites
- Windows NVIDIA driver >= 572.47 WHQL
- Docker Desktop 4.x+ with WSL2 backend

### WSL2 Setup (run once inside Ubuntu WSL2)
1. Add NVIDIA Container Toolkit apt repository
2. `sudo apt-get install -y nvidia-container-toolkit`
3. `sudo nvidia-ctk runtime configure --runtime=docker`
4. `sudo systemctl restart docker`
5. Validate: `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`

**Important**: Do NOT install a separate Linux NVIDIA driver inside WSL2.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| PyTorch cu126 has no SM_120 kernels (RTX 5070 Ti) | Cloud targets don't need SM_120. Use cu128 wheel only for local dev |
| Spot instance interruption -> 502 errors | Exponential backoff retries in gateway (max 3). ECS auto-replaces |
| Modal/RunPod cold starts (15-30s) | Set gateway timeout to 60s. Use `keep_warm=1` during demos |
| kind GPU on Windows WSL2 unsupported | Use `docker compose` for local GPU dev. K8s validation is dry-run only |

---

## Recommendation for builder-devops

1. Use `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime` as inference Dockerfile base
2. Implement two deploy paths: `deploy/aws/` (Terraform: VPC + ECS + g6.xlarge + ALB) and document Modal as the budget alternative
3. In k8s manifests, set `nvidia.com/gpu: 1` resource limit; note in k8s/README.md that GPU scheduling requires cloud node pool
4. Docker Compose GPU reservation stanza: `driver: nvidia, count: 1, capabilities: [gpu]`
