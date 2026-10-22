# Churn Risk Detection Agent

An AI-powered Customer Success dashboard that detects at-risk SaaS customers, explains churn signals via two independent methods, retrieves matching retention playbooks, and drafts personalized outreach — all routed through a human-in-the-loop approval flow.

> Built by Sarthak as a personal portfolio project.

---

## Architecture

```
frontend/index.html         (Vanilla JS SPA, same-origin)
        │  REST /api/*
        ▼
FastAPI  app/main.py        (lifespan: init_db → rag.warm → ml health; CORS; StaticFiles)
        │
app/routers/
  customers.py              list / get / trigger analysis
  analysis.py               fetch latest analysis results
  health.py                 /api/health liveness probe
  auth.py                   session-based auth (SessionMiddleware)
        │
app/services/
  signals.py                get_signals(session, id) → aggregated usage/feedback dict
  ml_model.py               predict_from_customer(row) → {churn_probability, band, top_features (SHAP)}
  scoring.py                risk_breakdown(signals) → {health_score 0-100, contributors[], risk_band}
  action_catalog.py         pick_base_action(signals, churn_prob) → {action, priority P0-P3, playbook_query}
  rag.py                    retrieve_best(query) → playbook snippet (cosine similarity + lexical fallback)
  embeddings.py             embed_one / embed_many — backend: ollama (local) | gemini (cloud)
  llm.py                    analyze(...) → {summary, top_reasons, recommended_action, draft_message}
  prompts.py                SYSTEM_INSTRUCTION + build_user_prompt
  orchestrator.py           analyze_customer(id, force) — full pipeline entrypoint
        │
app/db.py                   SQLAlchemy engine, SessionLocal, Base
app/models.py               Customer, UsageDaily, Feedback, Analysis, Meta
app/schemas.py              Pydantic v2 request/response models
app/deps.py                 get_db, require_auth
```

**Offline artifacts (built once):**

```
load_dataset.py             Kaggle CSV → SQLite churn_agent.db (ETL pipeline)
ml/features.py              Feature engineering shared between training and serving
ml/train_model.py           XGBoost training → churn_model.pkl + model_card.json
```

---

## ML Model

| Property | Value |
|---|---|
| Algorithm | XGBoost (`binary:logistic`) |
| Training rows | 10,000 |
| Features | 25 engineered features |
| Dataset churn rate | 10.21% |
| Test AUC (ROC) | **0.881** |
| Test PR-AUC | 0.634 |
| Precision / Recall (churn class) | 56.8% / 57.4% |
| Decision threshold | 0.698 (chosen by max-F1 on PR curve) |
| Trained at | 2026-08-30 |

**Top 5 features by gain:**

| Feature | Gain |
|---|---|
| `recent_inactivity_flag` | 0.359 |
| `last_login_days` | 0.195 |
| `payment_failures` | 0.052 |
| `tenure_days` | 0.041 |
| `usage_per_tenure` | 0.034 |

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested with conda env `sarthak_env` |
| [Ollama](https://ollama.ai) | Local embedding server — pull `nomic-embed-text` |
| Google Gemini API key | Free tier at [ai.google.dev](https://ai.google.dev) — used for LLM analysis |

### 1. Install dependencies

```bash
cd Churn-risk-detection-agent-clean
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in:

```env
GEMINI_API_KEY=your_key_here
SECRET_KEY=any-random-string
EMBED_BACKEND=ollama           # or "gemini" for cloud deploy
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
GEMINI_MODEL=models/gemini-2.5-flash
```

### 3. Start Ollama

```bash
ollama serve        # If not already running
ollama pull nomic-embed-text
```

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

### 5. Run tests

```bash
pytest tests/ -v
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/customers` | List customers with risk bands (sorted by churn probability) |
| `GET` | `/api/customers/{id}` | Full customer profile + latest analysis |
| `POST` | `/api/customers/{id}/analyze` | Run full pipeline for one customer |
| `POST` | `/api/customers/{id}/approve` | Approve/edit/dismiss analysis |
| `GET` | `/api/analysis/{id}` | Fetch stored analysis result |

---

## Deployment

Target: **Hugging Face Spaces** (Docker SDK, free tier)

Key settings for cloud:
- `EMBED_BACKEND=gemini` — Ollama not available in HF Spaces
- Set all secrets via HF Spaces environment variables
- SQLite `churn_agent.db` is bundled; swap to Postgres via `DATABASE_URL` for production

---

## Credits & Attribution

This project uses the following open-source libraries and services:
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosted tree classifier
- [SHAP](https://shap.readthedocs.io/) — model explainability
- [Google Gemini API](https://ai.google.dev/) — LLM analysis and cloud embeddings
- [Ollama](https://ollama.ai/) + `nomic-embed-text` — local embeddings
- [FastAPI](https://fastapi.tiangolo.com/) — backend REST API
- Dataset: Synthetic telco-churn dataset (Kaggle-style, 10,000 rows)
