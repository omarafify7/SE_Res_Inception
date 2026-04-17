# Plan: Portfolio Polish — Scientific Rigor, Microservices, Cloud-Native Deployment

## Task Description

Take the existing SE-Res-Inception CIFAR-100 / Tiny-ImageNet classifier from a single-machine research project to a portfolio-ready, cloud-native, distributed system that also demonstrates scientific rigor and scaling to a larger dataset. Concretely, we will:

1. **Strengthen the science**: add a fair baseline (ResNet18 / GoogLeNet stock) comparison, generate a Grad-CAM gallery that _narrates_ the SE-block attention story, and produce a parameters-vs-accuracy-vs-latency chart.
2. **Scale to a larger dataset**: train/evaluate on **COCO 2014** (already downloaded at `C:\Users\Omar\Documents\datasets\coco2014`) using bounding-box crops reformulated as an 80-class classification task — this reuses the existing training harness with minimal change while proving the architecture generalizes beyond 32x32 toys.
3. **Pivot deployment to microservices**: split the monolith into a **Python/FastAPI inference service** (GPU, internal) and a **Go/Fiber API gateway** (public, stateless, high-concurrency), communicating over an internal Docker network (Sidecar pattern).
4. **Containerize with GPU passthrough**: multi-stage Dockerfiles, `docker-compose.yml` with NVIDIA Container Toolkit reservation for the RTX 5070 Ti.
5. **Kubernetes-ready**: minimal k8s manifests (Deployment, Service, HPA) using a local `kind` / `minikube` cluster for demo.
6. **Cloud deployment path**: document and script a one-button deploy to AWS (ECS Fargate for gateway + EC2 g5.xlarge for inference) OR Azure (AKS + GPU node pool), chosen by the user.
7. **Portfolio polish**: rewrite README with architecture diagrams, add a short demo GIF/video, pin dependencies, CI badge.

## Objective

Ship a single GitHub repo an interviewer can `git clone`, `docker compose up`, and hit `POST /api/v1/classify` against — while reading a README that tells a clear story: "custom efficient CNN, validated against a stock baseline with Grad-CAM explainability, scaled to COCO, served behind a Go edge proxy, deployed to Kubernetes on GPU-enabled cloud."

## Problem Statement

The current project is technically strong (CIFAR-100 accuracy comparable to ResNet18 at a fraction of the params) but reads as _research code_: monolithic `server.py`, no baseline comparison, no containerization, no distributed-systems story, no evidence the architecture transfers beyond small datasets. For cloud / ML-platform roles, interviewers want to see **science** (fair baselines, explainability) **+ systems** (microservices, GPU containers, k8s) **+ scale** (beyond CIFAR). Each of the three is a dealbreaker on its own.

## Solution Approach

Execute in **seven sequential phases**, each independently shippable as a git tag so the commit history itself tells the story:

- **Phase 1 — Scientific Rigor (in-repo)**: baseline ResNet18 + GoogLeNet runs on identical harness, Grad-CAM narrative gallery, parameters/latency/accuracy chart. Zero new infrastructure.
- **Phase 2 — Scale to COCO**: new `datasets/coco_crops.py` that yields 80-class bbox-crop classification samples from COCO 2014 annotations; retrain SE-Res-Inception at 128x128 resolution; publish comparison metrics. (Plan-B fallback: ImageNet-100 subset if COCO multi-label is too noisy.)
- **Phase 3 — Inference microservice**: refactor `server.py` → `inference/main.py` with internal `/internal/predict` endpoint, Pydantic response model, `uvicorn` production config, `0.0.0.0` bind.
- **Phase 4 — Go API gateway**: new `gateway/` with Go module, `gofiber/fiber` framework, `POST /api/v1/classify` multipart handler, MIME validation, reverse-proxy to inference service, structured error handling for downstream failures.
- **Phase 5 — Docker Compose + GPU**: multi-stage `gateway/Dockerfile` (golang:1.22 → alpine), CUDA `inference/Dockerfile`, `docker-compose.yml` with `ml-net` bridge, GPU device reservation, `./checkpoints` volume.
- **Phase 6 — Kubernetes**: `k8s/` manifests (namespace, gateway Deployment + Service + Ingress, inference Deployment with `nvidia.com/gpu: 1` resource request, HPA on gateway). Validate on local `kind` cluster with GPU operator OR document cloud-only path.
- **Phase 7 — Cloud + Portfolio**: Terraform or `aws cdk` / `az` script for the chosen cloud, README rewrite with architecture diagram (Mermaid), demo recording, CI workflow (GitHub Actions) that at minimum runs `ruff`, `pytest`, `go test`, and `docker build`.

Phases 1 and 2 can run in parallel (separate branches). Phases 3–6 are strictly sequential. Phase 7 closes out.

## Relevant Files

Use these files to complete the task:

- `model.py` — SE-Res-Inception architecture; unchanged except new `get_feature_maps()` hook for Grad-CAM if not already present, and possibly a `num_classes=80` variant for COCO.
- `server.py` — current monolithic FastAPI app; will be **moved and refactored** into `inference/main.py`. Delete after Phase 3 so there is no duplication.
- `train.py` — existing training loop; will be parameterized to accept `--dataset {cifar100, tiny_imagenet, coco_crops}` in Phase 2.
- `visualize_gradcam.py` — already exists; extend in Phase 1 to produce the narrative gallery (one correct + one "interesting" wrong per class, side-by-side with SE channel heatmap).
- `evaluate_superclass.py`, `plot_metrics.py` — reused for baseline comparison plots in Phase 1.
- `checkpoints/best_model.pth`, `checkpoints/tiny_imagenet/` — existing weights, mounted into the inference container in Phase 5.
- `requirements.txt` — add `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `pycocotools`, `grad-cam` (or keep hand-rolled). Pin versions.
- `Dockerfile` — current single-container file; will be **replaced** by per-service Dockerfiles in `gateway/` and `inference/` in Phase 5.
- `C:\Users\Omar\Documents\datasets\coco2014\` — external dataset: contains `annotations/`, `images/`, `labels/`. Read-only; never modified by the plan.

### New Files

- `baselines/train_baseline.py` — runs `torchvision.models.resnet18(weights=None)` and `googlenet(weights=None)` on the same dataset with the same Mixup / epochs / optimizer as our model. Produces `outputs/baseline_comparison.csv`.
- `baselines/plot_comparison.py` — bar chart of (accuracy, params, inference_time_ms) for our model vs each baseline. Saves `outputs/baseline_comparison.png`.
- `outputs/gradcam_narrative.png` — 4x5 grid captioned with SE-block activation commentary for the README.
- `datasets/coco_crops.py` — `torch.utils.data.Dataset` that parses `instances_train2014.json`, crops bboxes with 10% padding, yields `(crop, category_id)` tuples. Handles small / clipped boxes by filtering `area < 32*32`.
- `configs/coco.yaml` — training hyperparameters for the COCO run (batch size, LR schedule, image size 128).
- `inference/main.py` — refactored FastAPI service, **internal** endpoint only (`POST /internal/predict`), Pydantic `PredictionResponse`.
- `inference/Dockerfile` — base `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`, copies `model.py` + `inference/`, mounts `checkpoints/` via compose.
- `inference/requirements.txt` — slim pin: torch, torchvision, fastapi, uvicorn, pillow.
- `gateway/go.mod`, `gateway/go.sum` — Go module, `github.com/gofiber/fiber/v2`.
- `gateway/main.go` — Fiber app, `/api/v1/classify` handler, multipart parsing, MIME sniff, proxy via `http.Client` to `http://inference-service:8000/internal/predict`, typed error responses (`502` when inference down, `413` for oversized upload, `415` for bad MIME).
- `gateway/handlers_test.go` — table-driven Go tests against a `httptest` inference stub.
- `gateway/Dockerfile` — multi-stage: `golang:1.22-alpine` builder → `alpine:3.20` runtime, ~15 MB image.
- `docker-compose.yml` — two services on `ml-net` bridge, `deploy.resources.reservations.devices` GPU stanza for `inference`, healthchecks, `depends_on`.
- `k8s/namespace.yaml`, `k8s/gateway-deployment.yaml`, `k8s/gateway-service.yaml`, `k8s/gateway-hpa.yaml`, `k8s/inference-deployment.yaml` (with `resources.limits.nvidia.com/gpu: 1`), `k8s/inference-service.yaml`.
- `deploy/aws/` OR `deploy/azure/` — Terraform or CDK (user picks one cloud in Phase 7).
- `.github/workflows/ci.yml` — lint + tests + docker-build matrix.
- `README.md` — rewrite with architecture diagram (Mermaid), quickstart, results table, Grad-CAM gallery embed.

## Implementation Phases

### Phase 1: Foundation — Scientific Rigor

Run the baseline experiments and polish Grad-CAM while everything else is still a monolith. This is pure Python work, no infra changes, and delivers the most interview-relevant artifacts fastest. Outputs: `outputs/baseline_comparison.png`, `outputs/gradcam_narrative.png`, a new **Results** section in the README.

### Phase 2: Core Implementation — Scale + Microservices

Two parallel tracks:

- **Track A (ML)**: COCO-crops dataset + retraining. Long-running (hours on RTX 5070 Ti), but isolated to one branch.
- **Track B (Systems)**: Inference service refactor → Go gateway → docker-compose. Strictly sequential within the track.

Tracks merge before Phase 3 so the final container ships whichever weights (CIFAR-100 or COCO or ImageNet-100) produce the better story.

### Phase 3: Integration & Polish — Kubernetes, Cloud, Portfolio

Kubernetes manifests, pick one cloud and script the deploy, rewrite README with diagrams and the narrative, record a ≤90s demo GIF, wire CI. This is where the project stops being _code_ and becomes a _product_ an interviewer can evaluate in under 5 minutes.

## Team Orchestration

- You operate as the team lead and orchestrate the team to execute the plan.
- You're responsible for deploying the right team members with the right context to execute the plan.
- IMPORTANT: You NEVER operate directly on the codebase. You use `Task` and `Task*` tools to deploy team members to the building, validating, testing, deploying, and other tasks.
  - This is critical. Your job is to act as a high-level director of the team, not a builder.
  - Your role is to validate all work is going well and make sure the team is on track to complete the plan.
  - You'll orchestrate this by using the Task\* Tools to manage coordination between the team members.
  - Communication is paramount. You'll use the Task\* Tools to communicate with the team members and ensure they're on track to complete the plan.
- Take note of the session id of each team member. This is how you'll reference them.

### Team Members

- Builder
  - Name: `builder-ml`
  - Role: Executes ML / PyTorch / training / evaluation tasks (Phase 1 baselines, Grad-CAM, Phase 2 COCO dataset + training).
  - Agent Type: `builder`
  - Resume: true
- Builder
  - Name: `builder-python-service`
  - Role: Executes Python microservice work (refactor `server.py` → `inference/main.py`, Pydantic models, inference Dockerfile).
  - Agent Type: `builder`
  - Resume: true
- Builder
  - Name: `builder-go`
  - Role: Executes Go / Fiber gateway work (`gateway/main.go`, tests, multi-stage Dockerfile). Go tooling (`gofmt`, `go vet`, `go test`) is outside the python-focused builder hooks, so this builder keeps fresh context dedicated to Go.
  - Agent Type: `builder`
  - Resume: true
- Builder
  - Name: `builder-devops`
  - Role: Executes docker-compose, Kubernetes manifests, cloud deploy scripts, GitHub Actions CI.
  - Agent Type: `builder`
  - Resume: true
- Researcher
  - Name: `researcher-cloud`
  - Role: Investigates AWS vs Azure GPU offerings, pricing, NVIDIA Container Toolkit setup on Windows + Docker Desktop, `kind` GPU support state in 2026. Produces a GO/NO-GO for each cloud path before `builder-devops` starts.
  - Agent Type: `researcher`
  - Resume: true
- Validator
  - Name: `validator`
  - Role: Read-only verification after each phase. Runs the validation commands, inspects artifacts, confirms acceptance criteria.
  - Agent Type: `validator`
  - Resume: false (fresh context per phase avoids confirmation bias)

## Step by Step Tasks

- IMPORTANT: Execute every step in order, top to bottom. Each task maps directly to a `TaskCreate` call.
- Before you start, run `TaskCreate` to create the initial task list that all team members can see and execute.

### 1. Baseline Comparison (ResNet18 + GoogLeNet)

- **Task ID**: `baseline-comparison`
- **Depends On**: none
- **Assigned To**: `builder-ml`
- **Agent Type**: `builder`
- **Parallel**: true
- Create `baselines/train_baseline.py` that accepts `--model {resnet18, googlenet}` and reuses the exact same data loaders, optimizer, Mixup alpha, epoch count, and seed as `train.py` on CIFAR-100.
- Run both baselines to completion; save weights under `checkpoints/baselines/`.
- Create `baselines/plot_comparison.py` → `outputs/baseline_comparison.csv` with columns `model, params_M, top1_acc, top5_acc, inference_ms_cpu, inference_ms_gpu`.
- Produce `outputs/baseline_comparison.png` as a grouped bar chart.

### 2. Grad-CAM Narrative Gallery

- **Task ID**: `gradcam-narrative`
- **Depends On**: none
- **Assigned To**: `builder-ml`
- **Agent Type**: `builder`
- **Parallel**: true (runs alongside Task 1)
- Extend `visualize_gradcam.py` to select one confidently-correct and one confidently-wrong example per superclass, targeting the **last Inception block's SE output** (the module whose attention the reader should see).
- Emit `outputs/gradcam_narrative.png` as a 4x5 grid with class-name captions and a legend.
- Write a 4–6 sentence caption-block in `outputs/gradcam_narrative_caption.md` for the README.

### 3. COCO-Crops Dataset

- **Task ID**: `coco-crops-dataset`
- **Depends On**: none
- **Assigned To**: `builder-ml`
- **Agent Type**: `builder`
- **Parallel**: true (runs alongside Tasks 1 & 2)
- Install `pycocotools`. Create `datasets/coco_crops.py` with `CocoCropsDataset(root, ann_file, split, min_area=1024, pad=0.1, image_size=128)`.
- Filter boxes with area < `min_area` and with width or height < 16 px.
- Include `label_map` helper mapping COCO `category_id` → contiguous `0..79`.
- Unit-test the dataset on a 100-image subset: assert 80 unique labels and all tensors shape `(3, 128, 128)`.

### 4. Train SE-Res-Inception on COCO-Crops

- **Task ID**: `train-coco`
- **Depends On**: `coco-crops-dataset`
- **Assigned To**: `builder-ml`
- **Agent Type**: `builder`
- **Parallel**: false
- Parameterize `train.py` with `--dataset coco_crops --config configs/coco.yaml`.
- Adapt model head to `num_classes=80`; use 128x128 input (`model.py` already tolerates this via adaptive pooling — verify first, adjust if not).
- Run for at least 30 epochs; save best checkpoint to `checkpoints/coco/best_model.pth`.
- Append COCO row to `outputs/baseline_comparison.csv`.

### 5. Refactor Inference into Microservice

- **Task ID**: `inference-service`
- **Depends On**: none
- **Assigned To**: `builder-python-service`
- **Agent Type**: `builder`
- **Parallel**: true (can start alongside ML tasks)
- Move `server.py` → `inference/main.py`; rename endpoint to `POST /internal/predict`.
- Add `class PredictionResponse(BaseModel)` with `top_5_predictions: list[Prediction]` and `inference_time_ms: float`.
- Ensure `uvicorn.run(host="0.0.0.0", port=8000)`, no public docs exposure (`docs_url=None` unless in dev).
- Delete old `server.py` and root-level `Dockerfile`.
- Add `inference/requirements.txt` with pinned versions.

### 6. Go API Gateway

- **Task ID**: `go-gateway`
- **Depends On**: `inference-service`
- **Assigned To**: `builder-go`
- **Agent Type**: `builder`
- **Parallel**: false
- `cd gateway && go mod init github.com/omar/ml-gateway && go get github.com/gofiber/fiber/v2`.
- Implement `POST /api/v1/classify`: parse multipart, sniff first 512 bytes for MIME (`net/http.DetectContentType`), reject non-image; forward body to `http://inference-service:8000/internal/predict` with a 30s timeout; stream JSON response back.
- Error handling: 502 on downstream timeout/connection-refused, 413 for uploads > 10 MB (configurable via env), 415 for bad MIME.
- `gateway/handlers_test.go` with ≥ 6 cases (happy path, each error path) using `httptest.NewServer` as the inference stub.
- Ensure `go fmt ./... && go vet ./... && go test ./...` all pass.

### 7. Multi-Container Docker Setup

- **Task ID**: `docker-compose`
- **Depends On**: `inference-service`, `go-gateway`
- **Assigned To**: `builder-devops`
- **Agent Type**: `builder`
- **Parallel**: false
- Write `inference/Dockerfile` on `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`; `COPY model.py inference/ requirements.txt`; `CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000"]`.
- Write `gateway/Dockerfile` as multi-stage: `golang:1.22-alpine` (CGO_ENABLED=0 build) → `alpine:3.20` (final < 20 MB).
- Write `docker-compose.yml` with `ml-net` bridge, `inference` service (GPU device reservation, `./checkpoints:/app/checkpoints:ro` mount, **no published ports**), `gateway` service (`8080:8080`, depends_on inference healthcheck).
- Add healthchecks: `curl -f http://localhost:8000/health` for inference, `wget -qO- http://localhost:8080/healthz` for gateway.
- Validate: `docker compose up --build` then `curl -F "file=@test.jpg" http://localhost:8080/api/v1/classify` returns a JSON prediction.

### 8. Cloud Research (GO/NO-GO)

- **Task ID**: `cloud-research`
- **Depends On**: none
- **Assigned To**: `researcher-cloud`
- **Agent Type**: `researcher`
- **Parallel**: true (runs anytime before Task 9)
- Investigate: current (2026) AWS GPU pricing (g5.xlarge / g6), Azure NC-series availability; AKS vs EKS GPU operator maturity; `kind` GPU support; cheapest path to a live demo URL on a student budget.
- Produce `specs/cloud-research.md` with matrix + recommended path (likely: AWS ECS Fargate gateway + g5.xlarge EC2 for inference OR a single-node AKS with NC6s_v3).
- Hard requirement: recommend a ≤ $30/month idle path using scale-to-zero where possible.

### 9. Kubernetes Manifests

- **Task ID**: `k8s-manifests`
- **Depends On**: `docker-compose`, `cloud-research`
- **Assigned To**: `builder-devops`
- **Agent Type**: `builder`
- **Parallel**: false
- Create `k8s/` with Deployments (gateway 2 replicas, inference 1 replica with `resources.limits.nvidia.com/gpu: 1`), ClusterIP Services, an Ingress for gateway, and an HPA on gateway (target 70% CPU).
- Verify with `kubectl apply --dry-run=client -f k8s/` and `kubeval` (or equivalent).
- Document a minimal `kind` or local single-node cluster walkthrough in `k8s/README.md`.

### 10. Cloud Deploy Script

- **Task ID**: `cloud-deploy`
- **Depends On**: `k8s-manifests`
- **Assigned To**: `builder-devops`
- **Agent Type**: `builder`
- **Parallel**: false
- Based on `researcher-cloud`'s recommendation, write `deploy/<cloud>/` with either Terraform modules or CDK app that provisions: VPC, subnets, GPU node, container registry, and applies k8s manifests.
- Include teardown script (`deploy/<cloud>/destroy.sh`) so cost caps are enforceable.
- Do **NOT** actually deploy in this task — produce runnable scripts and a walkthrough only. The user will run them manually.

### 11. CI + Portfolio Polish

- **Task ID**: `ci-and-readme`
- **Depends On**: `cloud-deploy`, `baseline-comparison`, `gradcam-narrative`, `train-coco`
- **Assigned To**: `builder-devops`
- **Agent Type**: `builder`
- **Parallel**: false
- `.github/workflows/ci.yml`: jobs for `ruff check`, `pytest`, `go test ./...`, and `docker build` (gateway + inference).
- Rewrite root `README.md`: one-paragraph pitch, Mermaid architecture diagram, results table (our model vs ResNet18 vs GoogLeNet on CIFAR-100 and COCO-crops), embedded Grad-CAM narrative image, quickstart (`docker compose up`), cloud-deploy link.
- Record a ≤ 90s demo (upload an image via `curl`, see JSON response, show `kubectl get pods`). Link GIF or asciinema in README.

### 12. Final Validation

- **Task ID**: `validate-all`
- **Depends On**: `ci-and-readme`, all previous
- **Assigned To**: `validator`
- **Agent Type**: `validator`
- **Parallel**: false
- Run every command in the Validation Commands section below.
- Inspect: all artifacts in `outputs/` exist and are non-empty; README renders correctly (no broken images); `docker compose up` boots to a working `curl` round-trip; `kubectl apply --dry-run` is clean.
- Verify all Acceptance Criteria.
- Produce `specs/validation-report.md` with pass/fail per criterion. If any fail, list the exact task that must be reopened.

## Acceptance Criteria

- `outputs/baseline_comparison.csv` contains ≥ 3 rows (SE-Res-Inception, ResNet18, GoogLeNet) on the same dataset, same hyperparameters.
- `outputs/baseline_comparison.png` exists and shows parameters, accuracy, inference latency side-by-side.
- `outputs/gradcam_narrative.png` exists; the README embeds it with a narrative caption mentioning SE attention.
- `checkpoints/coco/best_model.pth` exists and evaluates to ≥ 40% top-5 on a held-out COCO-crops validation split. (Exact threshold negotiable; failure triggers Plan-B: ImageNet-100.)
- `docker compose up --build` brings both services up healthy; `curl -F "file=@<any-jpg>" http://localhost:8080/api/v1/classify` returns a valid JSON top-5 prediction.
- Gateway image is < 25 MB (multi-stage + alpine).
- Inference container uses the GPU: `docker exec inference nvidia-smi` shows the RTX 5070 Ti; inference latency < 2x bare-metal.
- `go test ./gateway/...` passes with ≥ 6 test cases covering happy path + every error path.
- `kubectl apply --dry-run=client -f k8s/` exits 0 with no warnings.
- `deploy/<cloud>/` scripts exist, run `terraform validate` or `cdk synth` cleanly, and ship a one-command teardown.
- `.github/workflows/ci.yml` passes on a fresh PR.
- README is rewritten with architecture diagram, results table, Grad-CAM narrative, quickstart. No placeholder text.

## Validation Commands

Execute these commands to validate the task is complete:

- `uv run python -m py_compile model.py train.py inference/main.py baselines/train_baseline.py datasets/coco_crops.py` — all Python files compile.
- `uv run ruff check .` — lint passes.
- `uv run pytest -q` — unit tests pass (including `datasets/coco_crops.py` smoke test).
- `cd gateway && go fmt ./... && go vet ./... && go test -race ./...` — Go code is formatted, vetted, tests pass under the race detector.
- `docker compose config` — compose file is syntactically valid.
- `docker compose up --build -d && sleep 20 && curl -f http://localhost:8080/healthz && curl -f -F "file=@outputs/gradcam_narrative.png" http://localhost:8080/api/v1/classify && docker compose down` — end-to-end round trip works.
- `docker images | grep ml-gateway` — gateway image size < 25 MB.
- `docker exec $(docker compose ps -q inference) nvidia-smi` — inference container sees the GPU.
- `kubectl apply --dry-run=client -f k8s/` — manifests valid.
- `cd deploy/<cloud> && terraform validate` (or `cdk synth`) — infra code valid.
- `git log --oneline | head -20` — commit history reads as a coherent phase-by-phase narrative.

## Notes

- **Dependencies to add** (`uv add` equivalents or pin in `requirements.txt`):
  - `pycocotools` for Phase 2.
  - `pydantic>=2`, `uvicorn[standard]` (probably already transitive).
  - `grad-cam` if you prefer the library to hand-rolled hooks (optional).
  - Go: `github.com/gofiber/fiber/v2` (Phase 4).
- **COCO caveat**: COCO 2014 is natively detection/segmentation. We reformulate as classification via bbox crops. If the resulting per-crop accuracy is low (< 30% top-1), fall back to an ImageNet-100 subset — the scaling _story_ is what matters for the portfolio, not the specific dataset.
- **GPU passthrough on Windows**: requires Docker Desktop with WSL2 + NVIDIA Container Toolkit for WSL. `researcher-cloud` should verify the 2026 install steps before `builder-devops` begins Phase 5.
- **Cloud cost control**: the cloud-deploy task produces _scripts_, not live infrastructure. User runs them manually, and the destroy script is a first-class deliverable so no billing surprises.
- **Parallelism**: Tasks 1, 2, 3, 5, and 8 can all start immediately. Keep `builder-ml` on a dedicated branch per ML task to avoid checkpoint collisions.
- **Interview narrative to keep in mind while executing**: every phase produces one concrete sentence for the user's resume/screen-share. Phase 1 → "validated via Grad-CAM and beat stock ResNet18 at fewer params." Phase 2 → "scaled to COCO-80." Phase 3–5 → "split into Python inference + Go gateway on a Docker bridge network with GPU passthrough." Phase 6–7 → "deployed on Kubernetes to AWS/Azure behind one-click Terraform." If a task doesn't contribute to one of those sentences, cut it.
