import logging
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from schemas import FarmRequest, PredictionResponse
from services import MODEL_VERSION, create_prediction
from contextlib import asynccontextmanager
from storage import (
    find_predictions,
    init_db,
    list_predictions,
    save_prediction,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="Agro Scoring API",
    description=(
        "Prototype REST API for agricultural risk scoring. "
        "The score is rule-based, not a trained ML model."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()

    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id

    logger.info(
        "Request started | request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Unhandled error | request_id=%s",
            request_id,
        )

        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )

    process_time = time.perf_counter() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.6f}"

    logger.info(
        "Request completed | request_id=%s | status=%s | time=%.6fs",
        request_id,
        response.status_code,
        process_time,
    )

    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(
            {
                "detail": exc.detail,
                "request_id": getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            }
        ),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(
            {
                "detail": exc.errors(),
                "request_id": getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            }
        ),
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
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Unknown region"},
        422: {"description": "Validation error"},
        503: {"description": "Prediction storage is unavailable"},
    },
    summary="Score a farm",
    description=(
        "Validates farm data, calculates a risk score, "
        "and returns success only after saving it to SQLite."
    ),
)
async def predict(
    data: FarmRequest,
    request: Request,
):
    request_id = request.state.request_id

    result = create_prediction(data, request_id)

    await save_prediction(
        result,
        source="single",
    )

    return result


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    responses={
        404: {"description": "Prediction not found"},
        409: {"description": "Multiple predictions for this request"},
        503: {"description": "Prediction storage is unavailable"},
    },
    summary="Get one prediction by request ID",
)
async def get_prediction(request_id: str):
    results = await find_predictions(request_id)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    if len(results) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                "This request_id contains multiple predictions. "
                f"Use /requests/{request_id}/predictions"
            ),
        )

    return results[0]


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    responses={
        503: {"description": "Prediction storage is unavailable"},
    },
    summary="List recent predictions from SQLite",
)
async def get_predictions(
    limit: int = Query(default=10, ge=1, le=100),
    risk_level: Literal["low", "medium", "high"] | None = None,
):
    return await list_predictions(
        limit=limit,
        risk_level=risk_level,
    )
    
@app.get(
    "/requests/{request_id}/predictions",
    response_model=list[PredictionResponse],
    responses={
        404: {"description": "Prediction not found"},
        503: {"description": "Prediction storage is unavailable"},
    },
    summary="Get all predictions belonging to a request",
)
async def get_request_predictions(request_id: str):
    results = await find_predictions(request_id)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return results