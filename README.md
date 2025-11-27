# LLMSecOps Workshop – HuggingFace QA + GitHub + Kubernetes

This repository implements a simple **Question Answering (QA) API** using
**FastAPI** and a **HuggingFace** model, then secures and deploys it with a
**Docker → Trivy → Kubernetes → GitHub Actions** pipeline.

## Architecture

- **Model:** `distilbert-base-uncased-distilled-squad` from HuggingFace
- **API:** FastAPI app exposing:
  - `GET /` – health/info endpoint
  - `POST /chat` – accepts `{ "question": "...", "context": "..." }`
- **Container:** Docker image built from `app/Dockerfile`
- **Security:** Trivy image scan (HIGH/CRITICAL severities)
- **Orchestration:** Kubernetes Deployment + Service under `k8s/`
- **CI/CD:** GitHub Actions workflow `.github/workflows/llmsecops.yml`

## Local Development

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
