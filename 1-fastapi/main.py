from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, status

from schemas import FarmRequest, PredictionResponse
from services import MODEL_VERSION, create_prediction
from storage import predictions


app = FastAPI(
    title="Agro Scoring API",
    description=(
        "Prototype REST API for agricultural risk scoring. "
        "The score is rule-based, not a trained ML model."
    ),
    version="1.0.0",
)


@app.get(
    "/health",
    summary="Check API health",
    description="Checks whether the API is responding.",
)
def health():
    return {"status": "ok"}


@app.get(
    "/model-info",
    summary="Get model information",
    description="Returns metadata about the scoring component.",
)
def model_info():
    return {
        "model_name": "agro-risk-model",
        "model_version": MODEL_VERSION,
        "model_type": "risk-scoring",
        "status": "ready",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "Unknown region"},
        422: {"description": "Validation error"},
    },
    summary="Score a farm",
    description=(
        "Validates farm data, calculates a risk score, "
        "and stores the prediction."
    ),
)
def predict(request: FarmRequest):
    request_id = str(uuid4())
    result = create_prediction(request, request_id)
    predictions[request_id] = result
    return result


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    responses={404: {"description": "Prediction not found"}},
    summary="Get prediction by ID",
)
def get_prediction(request_id: str):
    if request_id not in predictions:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return predictions[request_id]


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    summary="List recent predictions",
)
def get_predictions(
    limit: int = Query(default=10, ge=1, le=100),
    risk_level: Literal["low", "medium", "high"] | None = None,
):
    values = list(reversed(list(predictions.values())))

    if risk_level is not None:
        values = [
            item for item in values
            if item.risk_level == risk_level
        ]

    return values[:limit]