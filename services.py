from fastapi import HTTPException

from model import calculate_risk
from schemas import FarmRequest, PredictionResponse


MODEL_VERSION = "1.0"

ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol",
}


def risk_level(score: float) -> str:
    if score < 0.3:
        return "low"

    if score < 0.7:
        return "medium"

    return "high"


def recommendation(level: str) -> str:
    if level == "low":
        return "Continue standard review."

    if level == "medium":
        return "Request additional documents and review the case."

    return "Refer the case for manual risk review."


def create_prediction(
    data: FarmRequest,
    request_id: str,
) -> PredictionResponse:
    if data.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code=400,
            detail="Unknown region",
        )

    score = calculate_risk(data)
    level = risk_level(score)

    return PredictionResponse(
        request_id=request_id,
        farm_id=data.farm_id,
        risk_score=score,
        risk_level=level,
        recommendation=recommendation(level),
        model_version=MODEL_VERSION,
    )