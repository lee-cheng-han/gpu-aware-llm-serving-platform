from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stop_token_ids: list[int] = Field(default_factory=list, max_length=32)
    priority: int = Field(default=0, ge=0, le=100)
    deadline_seconds: float | None = Field(default=None, gt=0)

    @field_validator("stop_token_ids")
    @classmethod
    def validate_stop_token_ids(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("stop token identifiers must be non-negative")
        if len(set(values)) != len(values):
            raise ValueError("stop token identifiers must be unique")
        return values


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


class RequestStatusResponse(BaseModel):
    request_id: str
    model_id: str
    status: str
    assigned_worker_id: str | None
    attempt_count: int
    priority: int
    retry_reasons: list[str]
    transition_timestamps: dict[str, float]
