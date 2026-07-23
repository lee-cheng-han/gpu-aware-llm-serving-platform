from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class APIError(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ):
        super().__init__(
            status_code=status_code,
            detail={"code": code, "message": message, "details": details},
        )
