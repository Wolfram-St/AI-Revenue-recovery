"""Domain exception to HTTP error translation.

Maps domain-specific exceptions to structured HTTP error responses without
leaking stack traces, secrets, or internal details.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ml.portfolio_optimizer import PortfolioOptimizationError, PortfolioProblemTooLargeError


class CaseNotFoundError(Exception):
    """Raised when a recovery case is not found."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        super().__init__(f"Recovery case {case_id!r} was not found")


class AnalysisError(Exception):
    """Raised when case analysis fails due to a dependency or domain issue."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ModelNotAvailableError(Exception):
    """Raised when a required ML model or artifact is unavailable."""

    def __init__(self, message: str = "Required model artifact is not available") -> None:
        super().__init__(message)


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


async def case_not_found_handler(request: Request, exc: CaseNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=_error_body("CASE_NOT_FOUND", str(exc)),
    )


async def analysis_error_handler(request: Request, exc: AnalysisError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=_error_body("ANALYSIS_ERROR", str(exc)),
    )


async def model_not_available_handler(request: Request, exc: ModelNotAvailableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=_error_body("MODEL_NOT_AVAILABLE", str(exc)),
    )


async def portfolio_optimization_error_handler(request: Request, exc: PortfolioOptimizationError) -> JSONResponse:
    if isinstance(exc, PortfolioProblemTooLargeError):
        return JSONResponse(
            status_code=422,
            content=_error_body("PORTFOLIO_PROBLEM_TOO_LARGE", str(exc)),
        )
    return JSONResponse(
        status_code=400,
        content=_error_body("PORTFOLIO_OPTIMIZATION_ERROR", str(exc)),
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_error_body("VALIDATION_ERROR", str(exc)),
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=_error_body("INTERNAL_ERROR", "An unexpected internal error occurred"),
    )
