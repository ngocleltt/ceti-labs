from pydantic import BaseModel, Field
from config import (
    MAX_BATCH_SIZE,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_DEFAULT_TEMPERATURE,
    LLM_MODEL,
)


class FarmRequest(BaseModel):
    farm_id: str
    region: str
    crop_type: str
    area_ha: float = Field(gt=0)
    temperature_avg: float
    precipitation_mm: float = Field(ge=0)
    payment_delay_days: int = Field(ge=0)
    previous_defaults: int = Field(ge=0)
    debt: float = Field(ge=0)


class PredictionResponse(BaseModel):
    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str
    
class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: str | None
    prev_page: str | None


class PaginatedPredictionsResponse(BaseModel):
    data: list[PredictionResponse]
    meta: PaginationMeta

class BatchRequest(BaseModel):
    items: list[FarmRequest] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )


class BatchPredictionResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: list[PredictionResponse]
    
class LLMChatRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=5000,
        description="Use demo data only; do not send confidential data.",
    )

    model: str = Field(
        default=LLM_MODEL,
        min_length=1,
        max_length=200,
    )

    temperature: float = Field(
        default=LLM_DEFAULT_TEMPERATURE,
        ge=0,
        le=2,
    )

    max_tokens: int = Field(
        default=LLM_DEFAULT_MAX_TOKENS,
        ge=1,
        le=2000,
    )


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str | None
    answer: str
    finish_reason: str | None
    latency_ms: float


class PredictWithLLMResponse(BaseModel):
    prediction: PredictionResponse
    llm: LLMChatResponse