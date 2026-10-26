import asyncio
import logging
import time

from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import INFERENCE_TIMEOUT_SECONDS
from model import model_predict_batch

import config
from batching import DynamicBatcher
import httpx
from llm import build_explanation_prompt, call_llm

from schemas import (
    BatchPredictionResponse,
    BatchRequest,
    FarmRequest,
    LLMChatRequest,
    LLMChatResponse,
    PaginatedPredictionsResponse,
    PredictionResponse,
    PredictWithLLMResponse,
)

from services import (
    MODEL_VERSION,
    build_batch_predictions,
    build_prediction_from_score,
    run_single_prediction,
    validate_batch_regions,
)

from storage import (
    find_predictions,
    get_weekday_predictions,
    init_db,
    list_predictions,
    save_prediction,
    save_predictions_batch,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


WEEKDAY_MAP = {
    "sunday": "0",
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    batcher = DynamicBatcher()
    app.state.batcher = batcher

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(config.LLM_TIMEOUT_SECONDS),
    ) as llm_client:
        app.state.llm_client = llm_client

        await batcher.start()

        try:
            yield

        finally:
            await batcher.stop()


app = FastAPI(
    title="Agro Scoring API",
    description=(
        "Prototype REST API for agricultural risk scoring. "
        "Includes request tracking, SQLite storage, "
        "weekday pagination and explicit batching. "
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
        503: {"description": "Model or prediction storage unavailable"},
        504: {"description": "Inference timeout"},
    },
    summary="Score a farm",
    description=(
        "Validates farm data, runs scoring with an inference timeout "
        "and returns success only after saving the result to SQLite."
    ),
)
async def predict(
    data: FarmRequest,
    request: Request,
):
    result = await run_single_prediction(
        data=data,
        request_id=request.state.request_id,
    )

    await save_prediction(
        result,
        source="single",
    )

    return result


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Unknown region in batch"},
        422: {"description": "Invalid input or batch size"},
        503: {"description": "Prediction storage is unavailable"},
        504: {"description": "Batch inference timeout"},
    },
    summary="Score multiple farms in one request",
    description=(
        "Processes 1 to 8 farms with one batch model call "
        "and saves all predictions in one SQLite transaction."
    ),
)
async def batch_predict(
    data: BatchRequest,
    request: Request,
):
    started = time.perf_counter()
    request_id = request.state.request_id

    validate_batch_regions(data.items)

    try:
        scores = await asyncio.wait_for(
            model_predict_batch(data.items),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Batch inference timeout",
        ) from exc

    results = build_batch_predictions(
        items=data.items,
        scores=scores,
        request_id=request_id,
    )

    await save_predictions_batch(results)

    return BatchPredictionResponse(
        request_id=request_id,
        batch_size=len(results),
        processing_time_ms=round(
            (time.perf_counter() - started) * 1000,
            2,
        ),
        items=results,
    )

@app.get(
    "/predictions/by-weekday",
    response_model=PaginatedPredictionsResponse,
    responses={
        400: {"description": "Unknown weekday"},
        404: {"description": "Page not found"},
        422: {"description": "Validation error"},
        503: {"description": "Prediction storage is unavailable"},
    },
    summary="List predictions by weekday with pagination",
    description=(
        "Filters predictions by the weekday of created_at "
        "and returns a page with navigation metadata."
    ),
)
async def predictions_by_weekday(
    request: Request,
    weekday: str = Query(
        ...,
        description=(
            "monday, tuesday, wednesday, thursday, "
            "friday, saturday or sunday"
        ),
    ),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=5, ge=1, le=100),
):
    weekday = weekday.strip().lower()

    if weekday not in WEEKDAY_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "weekday must be one of: monday, tuesday, "
                "wednesday, thursday, friday, saturday, sunday"
            ),
        )

    data, total_items = await get_weekday_predictions(
        weekday_number=WEEKDAY_MAP[weekday],
        page=page,
        per_page=per_page,
    )

    total_pages = (total_items + per_page - 1) // per_page

    if page > max(total_pages, 1):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Page not found",
        )

    next_page = None

    if page < total_pages:
        next_page = str(
            request.url.include_query_params(
                weekday=weekday,
                page=page + 1,
                per_page=per_page,
            )
        )

    prev_page = None

    if page > 1:
        prev_page = str(
            request.url.include_query_params(
                weekday=weekday,
                page=page - 1,
                per_page=per_page,
            )
        )

    return {
        "data": data,
        "meta": {
            "weekday": weekday,
            "current_page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "next_page": next_page,
            "prev_page": prev_page,
        },
    }


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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    if len(results) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This request_id contains multiple predictions. "
                f"Use /requests/{request_id}/predictions"
            ),
        )

    return results[0]


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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    return results

@app.post(
    "/predict-queued",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Unknown region"},
        422: {"description": "Validation error"},
        429: {"description": "Inference queue is full"},
        503: {"description": "Model, worker or storage unavailable"},
        504: {"description": "Inference timeout"},
    },
    summary="Score one farm through dynamic batching",
    description=(
        "Queues one farm for dynamic batching, waits for its "
        "own score and saves the prediction before returning."
    ),
)
async def predict_queued(
    data: FarmRequest,
    request: Request,
):
    request_id = request.state.request_id

    validate_batch_regions([data])

    score = await request.app.state.batcher.submit(
        data=data,
        request_id=request_id,
    )

    result = build_prediction_from_score(
        data=data,
        score=score,
        request_id=request_id,
    )

    await save_prediction(
        result,
        source="queued",
    )

    return result

@app.get(
    "/queue-status",
    summary="Get queue size and batching settings",
)
async def queue_status(request: Request):
    batcher = request.app.state.batcher

    return {
        "queue_size": batcher.queue.qsize(),
        "queue_max_size": batcher.queue.maxsize,
        "max_batch_size": config.MAX_BATCH_SIZE,
        "max_batch_wait_ms": config.MAX_BATCH_WAIT_MS,
        "inference_timeout_seconds": (
            config.INFERENCE_TIMEOUT_SECONDS
        ),
    }
    
@app.post(
    "/llm/chat",
    response_model=LLMChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Validation error"},
        502: {"description": "Invalid response or error from LLM provider"},
        503: {"description": "LLM service is unavailable"},
        504: {"description": "LLM timeout"},
    },
    summary="Send a prompt to the external LLM",
    description=(
        "Calls the configured OpenAI-compatible LLM API. "
        "Use synthetic educational data only."
    ),
)
async def llm_chat(
    data: LLMChatRequest,
    request: Request,
):
    return await call_llm(
        client=request.app.state.llm_client,
        prompt=data.prompt,
        request_id=request.state.request_id,
        model=data.model,
        temperature=data.temperature,
        max_tokens=data.max_tokens,
    )
    
@app.post(
    "/predict-with-llm",
    response_model=PredictWithLLMResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Unknown region"},
        422: {"description": "Validation error"},
        502: {"description": "LLM provider error"},
        503: {"description": "Model, storage or LLM unavailable"},
        504: {"description": "Inference or LLM timeout"},
    },
    summary="Score a farm and explain the result with an LLM",
    description=(
        "Runs scoring with a timeout and saves the prediction first. "
        "The external LLM then generates an explanation without "
        "overwriting risk_score or risk_level. "
        "If the LLM fails, the prediction remains stored."
    ),
)
async def predict_with_llm(
    data: FarmRequest,
    request: Request,
):
    request_id = request.state.request_id

    prediction = await run_single_prediction(
        data=data,
        request_id=request_id,
    )

    await save_prediction(
        prediction,
        source="single+llm",
    )

    llm_result = await call_llm(
        client=request.app.state.llm_client,
        prompt=build_explanation_prompt(prediction),
        request_id=request_id,
    )

    return PredictWithLLMResponse(
        prediction=prediction,
        llm=llm_result,
    )