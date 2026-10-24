from schemas import FarmRequest
import asyncio

from config import (
    MODEL_FIXED_OVERHEAD_SECONDS,
    MODEL_PER_ITEM_SECONDS,
)


def calculate_risk(data: FarmRequest) -> float:
    score = 0.1

    if data.payment_delay_days > 30:
        score += 0.3

    if data.previous_defaults > 0:
        score += 0.3

    if data.debt > 5_000_000:
        score += 0.2

    if data.precipitation_mm < 100:
        score += 0.1

    return round(min(score, 1.0), 2)

async def model_predict_batch(
    items: list[FarmRequest],
) -> list[float]:
    await asyncio.sleep(
        MODEL_FIXED_OVERHEAD_SECONDS
        + MODEL_PER_ITEM_SECONDS * len(items)
    )

    return [calculate_risk(item) for item in items]