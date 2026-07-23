from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = Field(default=0.7, ge=0)


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    queue_wait_ms: float
    scheduler_policy: str
    batch_size: int
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_name: str
    model_loaded: bool
    context_window_tokens: int | None = None
