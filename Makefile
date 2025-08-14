# Makefile for ML Inference Platform
# Provides convenient commands for common operations

# Variables
NAMESPACE := ml-platform
IMAGE_NAME := ml-inference
IMAGE_TAG := latest
FULL_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)
PORT := 8000

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

.PHONY: help
help: ## Show this help message
	@echo '${YELLOW}Usage:${NC}'
	@echo '  ${GREEN}make${NC} ${YELLOW}<target>${NC}'
	@echo ''
	@echo '${YELLOW}Targets:${NC}'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-20s${NC} %s\n", $$1, $$2}'

# Docker commands
.PHONY: docker-build
docker-build: ## Build the Docker image
	@echo "${GREEN}Building Docker image ${FULL_IMAGE}...${NC}"
	docker build -t $(FULL_IMAGE) .
	@echo "${GREEN}✓ Docker image built successfully${NC}"

.PHONY: docker-run
docker-run: ## Run the Docker container locally
	@echo "${GREEN}Running Docker container...${NC}"
	docker run -p $(PORT):$(PORT) --rm $(FULL_IMAGE)

.PHONY: docker-test
docker-test: ## Test the Docker container
	@echo "${GREEN}Testing Docker container...${NC}"
	docker run --rm $(FULL_IMAGE) python -m pytest tests/

# Kubernetes commands
.PHONY: k8s-create-namespace
k8s-create-namespace: ## Create the Kubernetes namespace
	@echo "${GREEN}Creating namespace...${NC}"
	kubectl apply -f k8s/base/namespace.yaml
	@echo "${GREEN}✓ Namespace created${NC}"

.PHONY: k8s-deploy
k8s-deploy: ## Deploy to Kubernetes using Kustomize
	@echo "${GREEN}Deploying to Kubernetes...${NC}"
	kubectl apply -k k8s/base/
	@echo "${GREEN}✓ Deployment complete${NC}"
	@echo "${YELLOW}Waiting for pods to be ready...${NC}"
	kubectl wait --for=condition=ready pod -l app=model-server -n $(NAMESPACE) --timeout=120s
	@echo "${GREEN}✓ All pods are ready${NC}"

.PHONY: k8s-deploy-dev
k8s-deploy-dev: ## Deploy to Kubernetes with development overlay
	@echo "${GREEN}Deploying to Kubernetes (dev environment)...${NC}"
	kubectl apply -k k8s/overlays/dev/
	@echo "${GREEN}✓ Development deployment complete${NC}"

.PHONY: k8s-delete
k8s-delete: ## Delete all Kubernetes resources
	@echo "${RED}Deleting Kubernetes resources...${NC}"
	kubectl delete -k k8s/base/ --ignore-not-found=true
	@echo "${GREEN}✓ Resources deleted${NC}"

.PHONY: k8s-status
k8s-status: ## Show status of Kubernetes resources
	@echo "${GREEN}Namespace: $(NAMESPACE)${NC}"
	@echo "\n${YELLOW}Pods:${NC}"
	@kubectl get pods -n $(NAMESPACE) -o wide
	@echo "\n${YELLOW}Services:${NC}"
	@kubectl get svc -n $(NAMESPACE)
	@echo "\n${YELLOW}Deployments:${NC}"
	@kubectl get deployments -n $(NAMESPACE)
	@echo "\n${YELLOW}ConfigMaps:${NC}"
	@kubectl get configmaps -n $(NAMESPACE)

.PHONY: k8s-logs
k8s-logs: ## Show logs from all pods
	@echo "${GREEN}Showing logs from all pods...${NC}"
	kubectl logs -n $(NAMESPACE) -l app=model-server --tail=50 --prefix=true

.PHONY: k8s-logs-follow
k8s-logs-follow: ## Follow logs from all pods
	@echo "${GREEN}Following logs from all pods (Ctrl+C to stop)...${NC}"
	kubectl logs -n $(NAMESPACE) -l app=model-server --tail=10 --prefix=true -f

.PHONY: k8s-port-forward
k8s-port-forward: ## Port forward to access the service locally
	@echo "${GREEN}Port forwarding to service (http://localhost:$(PORT))...${NC}"
	@echo "${YELLOW}Press Ctrl+C to stop${NC}"
	kubectl port-forward -n $(NAMESPACE) service/model-service $(PORT):80

.PHONY: k8s-shell
k8s-shell: ## Open a shell in a running pod
	@echo "${GREEN}Opening shell in pod...${NC}"
	@POD=$$(kubectl get pods -n $(NAMESPACE) -l app=model-server -o jsonpath='{.items[0].metadata.name}'); \
	kubectl exec -it -n $(NAMESPACE) $$POD -- /bin/bash

.PHONY: k8s-describe
k8s-describe: ## Describe all resources
	@echo "${GREEN}Describing pods...${NC}"
	kubectl describe pods -n $(NAMESPACE) -l app=model-server
	@echo "\n${GREEN}Describing service...${NC}"
	kubectl describe service model-service -n $(NAMESPACE)

.PHONY: k8s-events
k8s-events: ## Show recent events in namespace
	@echo "${GREEN}Recent events in namespace $(NAMESPACE):${NC}"
	kubectl get events -n $(NAMESPACE) --sort-by='.lastTimestamp' | tail -20

# Testing commands
.PHONY: test-health
test-health: ## Test health endpoints
	@echo "${GREEN}Testing health endpoints...${NC}"
	@echo "Testing /health/live:"
	curl -s http://localhost:$(PORT)/health/live | jq .
	@echo "\nTesting /health/ready:"
	curl -s http://localhost:$(PORT)/health/ready | jq .

.PHONY: test-predict
test-predict: ## Test prediction endpoint
	@echo "${GREEN}Testing prediction endpoint...${NC}"
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "Testing ML inference platform on Kubernetes"}' | jq .

.PHONY: test-batch
test-batch: ## Test batch prediction endpoint
	@echo "${GREEN}Testing batch prediction endpoint...${NC}"
	curl -s -X POST http://localhost:$(PORT)/predict/batch \
		-H "Content-Type: application/json" \
		-d '{"texts": ["First text", "Second text", "Third text"]}' | jq .

# Monitoring commands
.PHONY: k8s-top
k8s-top: ## Show resource usage
	@echo "${GREEN}Resource usage by pods:${NC}"
	kubectl top pods -n $(NAMESPACE)
	@echo "\n${GREEN}Resource usage by nodes:${NC}"
	kubectl top nodes

# Scaling commands
.PHONY: k8s-scale
k8s-scale: ## Scale deployment (use REPLICAS=n)
	@echo "${GREEN}Scaling deployment to ${REPLICAS} replicas...${NC}"
	kubectl scale deployment/model-server -n $(NAMESPACE) --replicas=${REPLICAS}
	@echo "${GREEN}✓ Scaled to ${REPLICAS} replicas${NC}"

# Development commands
.PHONY: dev
dev: ## Run FastAPI in development mode with hot reload
	@echo "${GREEN}Starting development server...${NC}"
	cd src && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT)

.PHONY: test
test: ## Run tests locally
	@echo "${GREEN}Running tests...${NC}"
	python -m pytest tests/ -v

.PHONY: lint
lint: ## Run linting
	@echo "${GREEN}Running linting...${NC}"
	python -m flake8 src/ tests/
	python -m black --check src/ tests/
	python -m mypy src/

.PHONY: format
format: ## Format code
	@echo "${GREEN}Formatting code...${NC}"
	python -m black src/ tests/

# Cleanup commands
.PHONY: clean
clean: ## Clean up generated files and caches
	@echo "${GREEN}Cleaning up...${NC}"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "${GREEN}✓ Cleanup complete${NC}"

# Combined commands
.PHONY: all
all: docker-build k8s-deploy k8s-port-forward ## Build, deploy, and connect

.PHONY: restart
restart: k8s-delete k8s-deploy ## Restart the deployment

.PHONY: redeploy
redeploy: docker-build k8s-delete k8s-deploy ## Rebuild and redeploy everything