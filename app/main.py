"""FastAPI application factory.

Creates the application with CORS, error handlers, and route registration.
The core domain modules are NOT imported here -- they are injected through
the dependency layer to maintain the unidirectional dependency rule.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # Load .env before anything reads os.environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.errors import (
    CaseNotFoundError,
    ModelNotAvailableError,
    AnalysisError,
    PortfolioOptimizationError,
    PortfolioProblemTooLargeError,
    case_not_found_handler,
    analysis_error_handler,
    model_not_available_handler,
    portfolio_optimization_error_handler,
    value_error_handler,
    generic_error_handler,
)
from app.api.routes.health import router as health_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.cases import router as cases_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.portfolio import router as portfolio_router
from app.api.routes.audit import router as audit_router
from app.api.routes.recovery import router as recovery_router


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()

    app = FastAPI(
        title="RecoverAI API",
        description="AI-powered revenue recovery decision support system",
        version="0.1.0",
        docs_url="/docs" if config.demo_mode else None,
        redoc_url="/redoc" if config.demo_mode else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(CaseNotFoundError, case_not_found_handler)
    app.add_exception_handler(AnalysisError, analysis_error_handler)
    app.add_exception_handler(ModelNotAvailableError, model_not_available_handler)
    app.add_exception_handler(PortfolioOptimizationError, portfolio_optimization_error_handler)
    app.add_exception_handler(PortfolioProblemTooLargeError, portfolio_optimization_error_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    app.include_router(health_router)
    app.include_router(dashboard_router)
    app.include_router(cases_router)
    app.include_router(analysis_router)
    app.include_router(portfolio_router)
    app.include_router(audit_router)
    app.include_router(recovery_router)

    return app


# Module-level app for uvicorn: `uvicorn app.main:app`
app = create_app()
