# Lab 1 — Agro Scoring API

## Description

A FastAPI prototype for agricultural enterprise risk scoring.

The project uses a rule-based scoring function instead of a trained machine learning model.

## Endpoints

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `GET /predictions/{request_id}`
- `GET /predictions`

## Run

```bash
python -m uvicorn main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Project structure

- `main.py` — FastAPI endpoints
- `schemas.py` — Pydantic models
- `model.py` — risk calculation
- `services.py` — business logic
- `storage.py` — temporary in-memory storage