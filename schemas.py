from pydantic import BaseModel, Field


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