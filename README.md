# ML Inference Platform

A production-ready machine learning inference service built with FastAPI and sentence-transformers.

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

(Coming soon)

## Development

(Coming soon)

## Testing

(Coming soon)

