import asyncio
import logging
import time

import httpx

from fastapi import HTTPException

import config

from schemas import LLMChatResponse, PredictionResponse


logger = logging.getLogger(__name__)


async def call_llm(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: str,
    model: str = config.LLM_MODEL,
    temperature: float = config.LLM_DEFAULT_TEMPERATURE,
    max_tokens: int = config.LLM_DEFAULT_MAX_TOKENS,
) -> LLMChatResponse:
    url = (
        f"{config.LLM_BASE_URL.rstrip('/')}"
        "/chat/completions"
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник учебного сервиса агроскоринга. "
                    "Отвечай кратко и понятно на русском языке. "
                    "Если передан результат скоринга, не изменяй "
                    "risk_score и risk_level. "
                    "Не придумывай факторы риска, которых нет "
                    "в предоставленных данных. "
                    "Не представляй результат как окончательное "
                    "решение банка."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    started = time.perf_counter()

    logger.info(
        "LLM request started | request_id=%s | model=%s",
        request_id,
        model,
    )

    try:
        response = await asyncio.wait_for(
            client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id,
                },
            ),
            timeout=config.LLM_TIMEOUT_SECONDS,
        )

    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        logger.warning(
            "LLM timeout | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=504,
            detail="LLM request timeout",
        ) from exc

    except httpx.RequestError as exc:
        logger.warning(
            "LLM connection failed | request_id=%s | error=%s",
            request_id,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="LLM service is unavailable",
        ) from exc

    if response.status_code != 200:
        logger.warning(
            "LLM provider error | request_id=%s | "
            "upstream_status=%s | content_type=%s | body=%r",
            request_id,
            response.status_code,
            response.headers.get("content-type"),
            response.text[:1000],
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "LLM provider returned an error "
                f"(HTTP {response.status_code})"
            ),
        )

    try:
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Response must be an object")

        choices = data.get("choices")

        if not isinstance(choices, list) or not choices:
            raise ValueError("Missing choices")

        choice = choices[0]

        if not isinstance(choice, dict):
            raise ValueError("Invalid choice")

        message = choice.get("message")

        if not isinstance(message, dict):
            raise ValueError("Invalid message")

        answer = message.get("content")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Missing text content")

        returned_model = data.get("model")
        finish_reason = choice.get("finish_reason")

        if returned_model is not None and not isinstance(
            returned_model, str
        ):
            raise ValueError("Invalid model field")

        if finish_reason is not None and not isinstance(
            finish_reason, str
        ):
            raise ValueError("Invalid finish_reason field")

    except (ValueError, TypeError) as exc:
        logger.warning(
            "Invalid LLM response | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=502,
            detail="LLM provider returned an invalid response",
        ) from exc

    latency_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )

    logger.info(
        "LLM request completed | request_id=%s | "
        "returned_model=%s | latency_ms=%s",
        request_id,
        returned_model,
        latency_ms,
    )

    return LLMChatResponse(
        request_id=request_id,
        requested_model=model,
        returned_model=returned_model,
        answer=answer.strip(),
        finish_reason=finish_reason,
        latency_ms=latency_ms,
    )


def build_explanation_prompt(
    prediction: PredictionResponse,
) -> str:
    return (
        "Объясни оператору банка результат учебного агроскоринга.\n"
        f"risk_score: {prediction.risk_score}\n"
        f"risk_level: {prediction.risk_level}\n"
        f"recommendation: {prediction.recommendation}\n"
        f"model_version: {prediction.model_version}\n\n"
        "Напиши 2–4 предложения на русском языке. "
        "Сохрани указанные числовую оценку и категорию риска. "
        "Не делай выводов о причинах риска, поскольку исходные "
        "признаки хозяйства тебе не предоставлены. "
        "Укажи, что окончательное решение принимается "
        "по правилам банка и при необходимости требует "
        "проверки оператором."
    )