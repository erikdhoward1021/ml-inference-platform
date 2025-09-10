# LLM-Augmented Book Recommendation Platform

A production-ready hybrid recommendation system combining collaborative filtering with fine-tuned language models for context-aware book recommendations. Built with FastAPI, Kubernetes deployment, and observability best practices.

## Architecture

**Hybrid Approach:**
- Traditional collaborative filtering for user-item patterns
- Fine-tuned small language model (mlx-lm) for contextual recommendations
- Fusion layer combining both approaches for enhanced personalization

**Dataset:** [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) (Books, Ratings, Users)

## Project Structure
```
llm-recommendation-platform
├─ data/
│  ├─ processed/              # Cleaned, processed datasets
│  └─ raw/                    # Raw Kaggle datasets
├─ scripts/
│  ├─ download_data.py        # Kaggle dataset download
│  ├─ prepare_data.py         # Data preprocessing pipeline
│  ├─ train_cf_model.py       # Collaborative filtering training
│  ├─ finetune_llm.py         # LLM fine-tuning (mlx-lm)
│  └─ evaluate_models.py      # Model evaluation
├─ src/
│  ├─ data/
│  │  ├─ loaders.py          # Data loading utilities (Polars)
│  │  └─ processors.py       # Data cleaning and feature engineering
│  ├─ models/
│  │  ├─ collaborative_filtering.py  # Matrix factorization
│  │  ├─ llm_recommender.py         # Fine-tuned LLM inference
│  │  └─ hybrid_recommender.py      # Fusion model
│  ├─ training/              # Training pipelines
│  ├─ evaluation/            # Recommendation metrics
│  └─ api/                   # FastAPI endpoints
├─ k8s/                      # Kubernetes manifests
├─ notebooks/                # Data exploration and analysis
└─ tests/                    # Test suites
```

## Quick Start

1. **Data Setup**
   ```bash
   # Install kaggle CLI and set up credentials (~/.kaggle/kaggle.json)
   python scripts/download_data.py
   ```

2. **Development**
   ```bash
   pip install -r requirements.txt
   jupyter notebook  # Explore data/notebooks/
   ```

3. **Production Deployment**
   ```bash
   make k8s-deploy
   ```

## Production Features

- **Kubernetes-native**: HPA, resource limits, health checks
- **Observability**: Prometheus metrics, Grafana dashboards
- **GitOps**: ArgoCD integration, declarative manifests
- **Testing**: Unit tests, load testing with `hey`
- **Data Processing**: Fast Polars-based pipeline
- **Model Serving**: Multi-model inference (CF + LLM)

## Technology Stack

- **ML**: PyTorch, mlx-lm, scikit-learn, Polars
- **API**: FastAPI, Pydantic
- **Infrastructure**: Kubernetes, Docker, ArgoCD
- **Monitoring**: Prometheus, Grafana
- **Data**: Kaggle Book Recommendation Dataset