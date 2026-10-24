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
    
def validate_batch_regions(
    items: list[FarmRequest],
) -> None:
    for index, item in enumerate(items):
        if item.region not in ALLOWED_REGIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown region at items[{index}]: "
                    f"{item.region}"
                ),
            )


def build_batch_predictions(
    items: list[FarmRequest],
    scores: list[float],
    request_id: str,
) -> list[PredictionResponse]:
    if len(items) != len(scores):
        raise RuntimeError(
            "Batch model returned an unexpected number of scores"
        )

    results = []

    for item, score in zip(items, scores):
        level = risk_level(score)

        results.append(
            PredictionResponse(
                request_id=request_id,
                farm_id=item.farm_id,
                risk_score=score,
                risk_level=level,
                recommendation=recommendation(level),
                model_version=MODEL_VERSION,
            )
        )

    return results

def build_prediction_from_score(
    data: FarmRequest,
    score: float,
    request_id: str,
) -> PredictionResponse:
    level = risk_level(score)

    return PredictionResponse(
        request_id=request_id,
        farm_id=data.farm_id,
        risk_score=score,
        risk_level=level,
        recommendation=recommendation(level),
        model_version=MODEL_VERSION,
    )