from fastapi import HTTPException
from model import calculate_risk, model_predict_batch
from schemas import FarmRequest, PredictionResponse
import asyncio
import logging
import config

logger = logging.getLogger(__name__)


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
    
async def run_single_prediction(
    data: FarmRequest,
    request_id: str,
) -> PredictionResponse:
    validate_batch_regions([data])

    logger.info(
        "Single inference started | request_id=%s",
        request_id,
    )

    try:
        scores = await asyncio.wait_for(
            model_predict_batch([data]),
            timeout=config.INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError as exc:
        logger.warning(
            "Single inference timeout | request_id=%s | timeout=%s",
            request_id,
            config.INFERENCE_TIMEOUT_SECONDS,
        )

        raise HTTPException(
            status_code=504,
            detail="Inference timeout",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Single inference failed | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=503,
            detail="Scoring model is unavailable",
        ) from exc

    if len(scores) != 1:
        logger.error(
            "Invalid single inference result | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=503,
            detail="Scoring model returned an invalid result",
        )

    result = build_prediction_from_score(
        data=data,
        score=scores[0],
        request_id=request_id,
    )

    logger.info(
        "Single inference completed | request_id=%s | "
        "risk_score=%s | risk_level=%s",
        request_id,
        result.risk_score,
        result.risk_level,
    )

    return result