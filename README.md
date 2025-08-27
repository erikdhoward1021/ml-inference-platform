# ML Inference Platform
A production-ready machine learning inference service built with FastAPI and sentence-transformers, designed to run on Kubernetes with observability, autoscaling, and GitOps best practices.

## Project Structure
```
ml-inference-platform
├─ .dockerignore
├─ .python-version
├─ Dockerfile
├─ Makefile
├─ README.md
├─ docker-compose.yml
├─ k8s
│  ├─ base
│  │  ├─ configmap.yaml
│  │  ├─ deployment.yaml
│  │  ├─ hpa.yaml
│  │  ├─ kustomization.yaml
│  │  ├─ namespace.yaml
│  │  ├─ poddisruptionbudget.yaml
│  │  └─ service.yaml
│  └─ overlays
│     └─ dev
│        └─ kustomization.yaml
├─ notebooks
│  └─ endpoint_test.ipynb
├─ pyproject.toml
├─ requirements-dev.txt
├─ requirements.txt
├─ src
│  ├─ __init__.py
│  ├─ api
│  │  ├─ __init__.py
│  │  ├─ main.py
│  │  └─ schemas.py
│  ├─ models
│  │  ├─ __init__.py
│  │  └─ sentence_transformers.py
│  └─ utils
│     ├─ __init__.py
│     └─ config.py
├─ tests
│  ├─ __init__.py
│  ├─ test_api.py
│  └─ test_model.py
└─ uv.lock
```

## Setup
1. Kubernetes Cluster
   - Tested locally on Docker Desktop (M4 Max).
   - Requires kubectl and optionally helm for monitoring stack.
2. Deployment

        `make k8s-deploy`
    - Deploys model-server with:
        - ConfigMap-based configuration
        - Resource requests/limits
        - Liveness, readiness, and startup probes
        - PodDisruptionBudget
        - Horizontal Pod Autoscaler (HPA) based on CPU/memory
        - Service for HTTP + metrics ports
3. Observability
    - model-server exposes Prometheus metrics at /metrics (port 9090).
    - Prometheus scrapes pod and HPA metrics.
    - Grafana dashboards visualize:
        - Current vs desired replicas
        - Pod CPU/memory usage
        - HPA scaling events in real time
4. GitOps & CI/CD
    - Manifests are fully declarative with Kustomize.
    - Can be integrated with ArgoCD for automated deployment:
        - Application points to Git repo, syncs changes automatically.
        - Supports rollback and self-healing.
        - Optional GitHub Actions workflow:
    - Build & push Docker image
    - Update Kustomize image tag
    - Trigger ArgoCD to deploy new version
5. Testing
    - Unit tests: pytest for API and model modules.
    - Load testing can demonstrate HPA behavior:

        `hey -z 2m -q 50 -c 10 http://<service-ip>/predict`
6. Production-readiness Highlights
    - HPA scales based on CPU/memory usage.
    - Configurable via ConfigMap without rebuilding containers.
    - PodDisruptionBudget ensures minimum availability during node maintenance.
    - Resource requests/limits prevent overcommit.
    - Observability via Prometheus + Grafana.
    - Declarative manifests + GitOps enable repeatable, auditable deployments.