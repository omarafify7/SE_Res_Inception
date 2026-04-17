# SE-Res-Inception: Portfolio Build Documentation

**A complete record of transforming a single-machine research project into a portfolio-grade ML platform deliverable.**

Date: April 2026
Author: Omar Afify

---

## 1. What This Project Is

A custom **8.86M-parameter convolutional neural network** (SE-Res-Inception V2) fused from three architectural ideas — **Inception modules, residual connections, and Squeeze-and-Excitation attention** — trained from scratch on CIFAR-100 and scaled to COCO.

The project was built as a three-pillar portfolio piece targeting **ML platform / cloud engineering** roles, demonstrating:

1. **Scientific rigor** — a fair, reproducible comparison against stock ResNet-18 and GoogLeNet under identical training recipes.
2. **Cross-dataset generalization** — the same architecture trained on CIFAR-100 (32×32, 100 classes), Tiny ImageNet (64×64, 200 classes), and COCO-Crops (128×128, 80 classes).
3. **Production systems** — refactored from a monolithic script into a two-service microservice (Python/FastAPI + Go/Fiber), containerized, deployable to Kubernetes and AWS.

---

## 2. Final Results

### CIFAR-100 Fair Comparison (100 epochs, identical hyperparameters)

| Model              | Params | Top-1   | Top-5   | GPU Latency |
|--------------------|--------|---------|---------|-------------|
| **SE-Res-Inception** | 8.86M | **78.85%** | **95.15%** | 7.92 ms     |
| ResNet-18          | 11.23M | 60.25%  | 83.32%  | 2.05 ms     |
| GoogLeNet          | 5.70M  | 58.17%  | 83.14%  | 5.81 ms     |

**Headline**: SE-Res-Inception beats ResNet-18 by **+18.6 pp top-1** with 21% fewer parameters.

### Scaling Story

| Dataset       | Input   | Classes | Top-1 (best)  | Status        |
|---------------|---------|---------|---------------|---------------|
| CIFAR-100     | 32×32   | 100     | 78.85%        | Converged     |
| Tiny ImageNet | 64×64   | 200     | ~55% (prior)  | Converged     |
| COCO-Crops    | 128×128 | 80      | in progress   | 50-epoch run  |

### Deployment Artifacts

- `inference/` — FastAPI GPU inference microservice
- `gateway/` — Go/Fiber public gateway with MIME validation + error handling
- `docker-compose.yml` — dev multi-container orchestration
- `k8s/` — 8 Kubernetes manifests (Deployment, Service, HPA, Ingress)
- `deploy/aws/` — Terraform for VPC + ECS Fargate + g6.xlarge GPU
- `.github/workflows/ci.yml` — lint + test + Docker build pipeline

---

## 3. Build Phases (Chronological)

### Phase 1 — Scientific rigor

Goal: produce a fair side-by-side comparison of our model vs. two well-known baselines.

- Created `baselines/train_baseline.py` that instantiates `torchvision.models.resnet18(weights=None)` and `googlenet(weights=None)` with matching CIFAR-100 FC heads and reuses the exact training utilities from `train.py` (Mixup α=1.0, CutMix α=1.0, AdamW lr=1e-3, cosine anneal, 100 epochs, AMP, label smoothing 0.1).
- Created `baselines/plot_comparison.py` that evaluates each checkpoint, measures CPU + GPU inference latency (100 passes, middle 80 averaged), and emits `outputs/baseline_comparison.{csv,png}`.
- Created `visualize_narrative.py` — Grad-CAM narrative gallery hooking the final Inception block's post-SE feature maps, contrasting confidently correct vs. confidently wrong predictions across 10 superclasses.

### Phase 2 — Scale to larger datasets

- Created `datasets/coco_crops.py` — a `torch.utils.data.Dataset` that parses COCO 2014 `instances_train2014.json` via `pycocotools`, extracts bbox crops with 10% padding, filters tiny boxes (`area < 1024`, `w or h < 16`), and maps COCO's non-contiguous category IDs to contiguous 0–79 labels. **Training set: 347,133 crops across 80 classes.**
- Added `configs/coco.yaml` and extended `train.py` with `--dataset {cifar100, tiny_imagenet, coco_crops} --config <path>` so the same training loop handles all three datasets via YAML override.

### Phase 3 — Microservice refactor

- Moved monolithic `server.py` → `inference/main.py`. Added Pydantic response schemas, renamed `/predict` → `/internal/predict`, bound `0.0.0.0:8000`, killed the public docs endpoint (`docs_url=None` unless `DEV_MODE=1`), and made checkpoint path / dataset profile / num_classes configurable via environment.
- Created `gateway/main.go` in Go with the Fiber framework: public `POST /api/v1/classify` endpoint, multipart parsing, `net/http.DetectContentType` MIME sniffing, configurable max upload size, and typed error responses (`415` for bad MIME, `413` for oversized, `502` for inference-service unreachable, 30-second downstream timeout).
- Added `gateway/handlers_test.go` with 8+ table-driven test cases against an `httptest.NewServer` inference stub.

### Phase 4 — Containerization

- `inference/Dockerfile`: `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`, slim Python deps, `uvicorn --workers 1`, healthcheck on `/health`.
- `gateway/Dockerfile`: multi-stage `golang:1.22-alpine` → `alpine:3.20`, `CGO_ENABLED=0`, ~15MB final image.
- `docker-compose.yml`: `ml-net` bridge, GPU device reservation stanza, `./checkpoints:/app/checkpoints:ro` volume, gateway depends on inference healthcheck, no public ports on inference (internal-only).

### Phase 5 — Cloud research and deploy

- `specs/cloud-research.md`: GO/NO-GO matrix across AWS (g5/g6/Fargate), Azure (AKS + NC-series), and serverless GPU (Modal, RunPod). Recommended: AWS g6.xlarge (L4) + ECS Fargate gateway for production, Modal + Fly.io for the ≤$30/month demo path.
- `deploy/aws/`: Terraform modules for VPC, subnets, NAT, ECR, ECS cluster with mixed Fargate + EC2 capacity providers, ALB, security groups, CloudWatch logs, and Cloud Map service discovery. Includes `deploy.sh` and `destroy.sh` (teardown as first-class deliverable).

### Phase 6 — Kubernetes

- `k8s/namespace.yaml`, `k8s/inference-deployment.yaml` (with `nvidia.com/gpu: 1`), `inference-service.yaml` (ClusterIP, internal-only), `gateway-deployment.yaml` (2 replicas), `gateway-service.yaml`, `gateway-ingress.yaml` (nginx), `gateway-hpa.yaml` (2–10 replicas @ 70% CPU).
- Manifests pass `kubectl apply --dry-run=client` validation.

### Phase 7 — Polish and portfolio assets

- `README.md`: full rewrite with Mermaid architecture diagram, results table, Grad-CAM narrative embed, docker-compose quickstart.
- `.github/workflows/ci.yml`: parallel Python (ruff + pytest + py_compile), Go (fmt + vet + test -race), Docker (both images), Terraform (fmt + validate) jobs with an aggregate `ci-success` gate.

---

## 4. Engineering Challenges Solved

### Challenge 1: RTX 5070 Ti (Blackwell SM_120) + PyTorch

**Symptom**: Every CUDA kernel launch failed with `cudaErrorNoKernelImageForDevice`. Baselines wouldn't start. Existing trained checkpoints couldn't be used for inference.

**Root cause**: PyTorch stable builds through 2.6.x ship with SM_50–SM_90 kernels only. SM_120 requires CUDA 12.8, which is only available in nightly wheels as of April 2026.

**Fix**: Replaced `torch 2.10.dev+cu126` with `torch 2.12.dev+cu128` nightly. One command:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
```

**Lesson**: For any new hardware, verify CUDA compatibility via a trivial op (`torch.randn(2,3).cuda().sum()`) before running a multi-hour training.

### Challenge 2: COCO training was 300× slower than expected

**Symptom**: 12.87 s/iter on the 2-epoch smoke test, implying ~19 hours per epoch. GPU at 100% util but only 57W power draw (vs. 285W TDP). System RAM usage climbing to 15 GB.

**Investigation process**:
1. Initial hypothesis: data pipeline bottleneck (only 4 DataLoader workers on a 16-core CPU). Bumped to 12 with `persistent_workers=True, prefetch_factor=4`. No improvement.
2. Isolated the model by benchmarking it on random tensors (no DataLoader in the loop):

   | Input | Time/iter | Peak VRAM |
   |-------|-----------|-----------|
   | 128×128 | 12,549 ms | 33.23 GB |
   | 96×96   | 8,112 ms | 18.79 GB |
   | 64×64   | 2,843 ms | 8.47 GB |

   Peak VRAM at 128×128 was **33 GB** — exceeding the 16 GB physical limit. PyTorch was silently spilling to system RAM via Windows unified memory, which is orders of magnitude slower than VRAM.

3. **Root cause**: the model's stem was `Conv 3×3, stride=1`, designed for CIFAR's 32×32 input. On 128×128, the first two Inception blocks operated at full 128×128 resolution with 256→480 channels, exploding activation memory (each layer's activation is `batch × channels × H × W × 2 bytes` at FP16, doubled for gradients, times ~10 live tensors per block).

**Fix**: Added an input-size-aware stem to `SEResInception`:

```python
if input_size <= 64:
    # Legacy stride=1 (preserves CIFAR / Tiny ImageNet checkpoints)
    self.stem = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
    )
else:
    # ImageNet-style: /4 spatial in stem
    self.stem = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
    )
```

**Post-fix benchmark at 128×128**: 41.8 ms/iter, 2.38 GB peak. **300× speedup**, 14× memory reduction. All existing CIFAR-100 and Tiny ImageNet checkpoints remain valid (default branch is unchanged).

**Lesson**: CNNs silently spill to system RAM on Windows when activations exceed VRAM — no OOM, just a 300× slowdown. Always benchmark `(model, input_size, batch_size)` in isolation before committing to multi-hour training. The failure mode is fundamentally invisible without explicit memory instrumentation.

### Challenge 3: Baseline comparison fairness

**Problem**: A pretrained ImageNet ResNet-18 would have been transfer learning, not a baseline. Using it would invalidate any "my model beats ResNet-18" claim.

**Fix**: All baselines train **from scratch** with the exact same pipeline as SE-Res-Inception: same optimizer (AdamW lr=1e-3, wd=5e-4), same augmentation (Mixup α=1.0 + CutMix α=1.0, 50% probability each), same cosine LR schedule, same label smoothing (0.1), same 100 epochs, same batch size (80), same AMP. The only variable is the model.

### Challenge 4: GoogLeNet auxiliary classifiers

**Problem**: Stock GoogLeNet returns a `GoogLeNetOutputs` named tuple during training (with `logits`, `aux_logits1`, `aux_logits2`) but a plain tensor during eval. The existing training loop didn't handle this.

**Fix**: `baselines/train_baseline.py` branches on `hasattr(output, "logits")`; in training, the aux losses are added with weight 0.3 (stock paper config). For inference in `plot_comparison.py`, builds GoogLeNet with `aux_logits=False` and filters aux keys from the state dict (`strict=False`).

---

## 5. File Structure

```
Goog-ResSkips-SE_Net/
├── model.py                      # SEResInception(num_classes, input_size)
├── train.py                      # unified trainer (--dataset + --config)
├── visualize_narrative.py        # Grad-CAM narrative gallery
├── requirements.txt
│
├── baselines/
│   ├── train_baseline.py         # ResNet-18 / GoogLeNet trainers
│   ├── plot_comparison.py        # CSV + grouped bar chart
│   └── __init__.py
│
├── datasets/
│   ├── coco_crops.py             # CocoCropsDataset + get_coco_dataloaders
│   └── test_coco_crops.py        # pytest smoke test
│
├── inference/
│   ├── main.py                   # FastAPI internal service
│   ├── Dockerfile                # CUDA 12.6 runtime base
│   └── requirements.txt
│
├── gateway/
│   ├── main.go                   # Go/Fiber public gateway
│   ├── handlers_test.go          # 8+ table-driven tests
│   ├── Dockerfile                # multi-stage alpine ~15MB
│   ├── go.mod, go.sum
│
├── configs/
│   └── coco.yaml                 # COCO training hyperparameters
│
├── k8s/
│   ├── namespace.yaml
│   ├── inference-deployment.yaml # GPU resource
│   ├── inference-service.yaml
│   ├── gateway-deployment.yaml   # 2 replicas
│   ├── gateway-service.yaml
│   ├── gateway-ingress.yaml
│   ├── gateway-hpa.yaml
│   └── README.md
│
├── deploy/aws/
│   ├── main.tf                   # VPC + ECS + ALB + EC2 GPU
│   ├── variables.tf, outputs.tf, versions.tf
│   ├── deploy.sh, destroy.sh
│   └── README.md
│
├── docker-compose.yml
├── .github/workflows/ci.yml
│
├── checkpoints/                  # trained weights (gitignored)
│   ├── best_model.pth            # SE-Res-Inception CIFAR-100 (78.85%)
│   ├── baselines/
│   │   ├── resnet18_best.pth     # 60.25%
│   │   └── googlenet_best.pth    # 58.17%
│   ├── coco/                     # populated by full COCO run
│   └── tiny_imagenet/
│
├── outputs/
│   ├── baseline_comparison.csv
│   ├── baseline_comparison.png
│   ├── gradcam_narrative.png
│   └── gradcam_narrative_caption.md
│
├── docs/
│   ├── PORTFOLIO_BUILD.md        # this document
│   └── architecture.html         # architecture explainer (for portfolio site)
│
└── specs/
    ├── portfolio-polish-microservices-cloud.md  # original 7-phase plan
    ├── cloud-research.md                         # GO/NO-GO matrix
    └── validation-report.md                      # end-of-build validation
```

---

## 6. How to Reproduce

### Prerequisites
- NVIDIA GPU with CUDA 12.8+ support (required for Blackwell / RTX 50-series)
- Docker Desktop with WSL2 + NVIDIA Container Toolkit (for local microservice testing)
- Python 3.11+, Go 1.22+

### Install
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

### Train SE-Res-Inception on CIFAR-100
```bash
python train.py  # default config: CIFAR-100, 100 epochs
```

### Train baselines
```bash
python -m baselines.train_baseline --model resnet18 --epochs 100
python -m baselines.train_baseline --model googlenet --epochs 100
python -m baselines.plot_comparison  # produces outputs/baseline_comparison.{csv,png}
```

### Generate Grad-CAM narrative
```bash
python visualize_narrative.py  # produces outputs/gradcam_narrative.png
```

### Train on COCO-Crops (scaling story)
```bash
python train.py --dataset coco_crops --config configs/coco.yaml
```

### Run the microservices locally
```bash
docker compose up --build
# then:
curl -F "file=@outputs/gradcam_narrative.png" http://localhost:8080/api/v1/classify
```

### Deploy to AWS
```bash
cd deploy/aws
./deploy.sh   # provisions VPC + ECS + ALB + GPU EC2
./destroy.sh  # tears down (to cap costs)
```

---

## 7. Status

- **Phase 1–6**: complete
- **COCO 50-epoch training**: in progress (~5 hours expected)
- **Cloud deployment**: scripts written + tested with `terraform validate`, not yet deployed to AWS
- **Demo recording (≤90s GIF)**: not yet recorded

---

## 8. Key Learnings

1. **Verify hardware + software compat before committing**. A 30-minute CUDA install saved a week of confused debugging later.
2. **Model architecture is tied to input size**. Spatial dimensions propagate through every layer; a model designed for 32×32 doesn't gracefully scale to 128×128 without reworking the stem.
3. **Silent performance degradation is the worst failure mode**. A 300× slowdown with no error is harder to debug than a crash. Instrument peak memory explicitly.
4. **Fair comparisons require discipline**. The temptation to use pretrained weights for "speed" would have destroyed the baseline's credibility. Running baselines from scratch cost 1.5 hours of GPU time and earned a defensible claim.
5. **Microservices aren't free**. Splitting the monolith added complexity (service discovery, healthchecks, `depends_on`, cross-process error handling) but earned the cloud-native story that's the portfolio's main sell for platform roles.
