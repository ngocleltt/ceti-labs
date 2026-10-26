# Agro Scoring API — Labs 1 & 2

## Description

A FastAPI prototype for agricultural enterprise risk scoring.

The service uses a rule-based scoring function, not a trained machine
learning model. Inference latency is simulated for batching and timeout tests.

Lab 2 extends the service with:
- Request ID middleware, logging and processing-time headers.
- Persistent SQLite storage through aiosqlite.
- Prediction history and weekday pagination.
- Explicit and dynamic batching.
- Queue backpressure and inference timeouts.
- External LLM integration through httpx.AsyncClient.

The LLM generates explanations only. It does not calculate or overwrite
risk_score or risk_level.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Check API availability |
| GET | /model-info | Get scoring component metadata |
| POST | /predict | Score one farm and save the result |
| POST | /batch-predict | Score 1–8 farms in one request |
| POST | /predict-queued | Score one farm through dynamic batching |
| GET | /queue-status | Get queue size and batching settings |
| GET | /predictions | List recent predictions; supports limit and risk_level |
| GET | /predictions/by-weekday | Filter by weekday with page and per_page |
| GET | /predictions/{request_id} | Get a single prediction |
| GET | /requests/{request_id}/predictions | Get all predictions for a request |
| POST | /llm/chat | Send a prompt to the external LLM |
| POST | /predict-with-llm | Score a farm and generate an explanation |

Successful prediction endpoints return HTTP 200 after saving their results.

An explicit batch shares one request_id across its predictions.
GET /predictions/{request_id} returns HTTP 409 if multiple results exist;
use GET /requests/{request_id}/predictions instead.

## Setup and run — Windows PowerShell

Tested with Python 3.11.9.

Run the following commands from the directory containing main.py
and requirements.txt:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

If .venv already exists, skip the first command.

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

OpenAPI specification:

```text
http://127.0.0.1:8000/openapi.json
```

## Request tracking

The middleware accepts X-Request-ID from the client or generates a UUID.

Responses include:
- X-Request-ID — the current HTTP request identifier.
- X-Process-Time — processing time in seconds.

Error responses include detail and request_id.

Request IDs provide correlation, not idempotency:
repeating a POST with the same ID does not prevent duplicate records.

## Storage

Predictions are stored in agro_scoring.db next to storage.py.
The database and table are initialized automatically at startup.

Results remain available after restarting the application.

The predictions table contains:
id, request_id, farm_id, risk_score, risk_level,
recommendation, model_version, source and created_at.

## Configuration

Settings are defined in config.py.

| Setting | Standard value |
|---|---|
| MAX_BATCH_SIZE | 8 |
| MAX_BATCH_WAIT_MS | 50 |
| QUEUE_MAX_SIZE | 100 |
| INFERENCE_TIMEOUT_SECONDS | 2.0 |
| MODEL_FIXED_OVERHEAD_SECONDS | 0.020 |
| MODEL_PER_ITEM_SECONDS | 0.003 |

The queue belongs to one application process. Use one Uvicorn worker
for the supplied concurrency tests.

## LLM integration

The service calls the OpenAI-compatible Vireonix endpoint:

```text
[https://vireonix.ai/v1/chat/completions](https://vireonix.ai/v1/chat/completions)
```

The configured model is auto. Both /llm/chat and /predict-with-llm
were tested successfully with HTTP 200.

External API availability is not guaranteed.
Use synthetic educational data only.

For /predict-with-llm, scoring is saved before calling the LLM.
If the LLM call fails, the prediction remains stored and can be retrieved
using the request_id returned with the error.

## Error handling

| Code | Meaning |
|---|---|
| 400 | Business validation error or invalid weekday |
| 404 | Prediction or page not found |
| 409 | Multiple predictions found for a single-result lookup |
| 422 | Request validation error |
| 429 | Inference queue is full |
| 500 | Unhandled internal error |
| 502 | LLM provider error or invalid response |
| 503 | Model, worker, storage or LLM connection unavailable |
| 504 | Inference or LLM timeout |

## Concurrency test

Keep the server running and execute in a second terminal:

```powershell
.\.venv\Scripts\python.exe test_dynamic_batch.py
```

The script sends 16 concurrent requests to /predict-queued,
checks the returned IDs and scores, and prints a summary.

Each successful request creates a prediction in SQLite.

For backpressure testing, temporarily reduce QUEUE_MAX_SIZE
and increase simulated inference latency.

For timeout testing, temporarily set INFERENCE_TIMEOUT_SECONDS
to 0.001.

Restart the server after changing test settings and restore
the standard configuration afterwards.

The script expects HTTP 200 during normal operation.
Responses intentionally rejected with HTTP 429 are marked FAIL
by its normal-operation checks.

## Project structure

- main.py — application, lifespan, middleware, error handlers and endpoints
- config.py — batching, timeout and LLM settings
- schemas.py — Pydantic request and response models
- model.py — rule-based risk calculation and simulated batch inference
- services.py — business validation, postprocessing and single inference
- storage.py — asynchronous SQLite access and pagination
- batching.py — queue, batch worker and Future-based result delivery
- llm.py — external LLM calls and explanation prompts
- test_dynamic_batch.py — concurrent request test
- requirements.txt — dependencies

## Limitations

This is an educational prototype, not a production banking system.

It uses rule-based scoring, simulated inference latency,
an in-process queue and an external LLM provider.