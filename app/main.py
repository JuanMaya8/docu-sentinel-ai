"""FastAPI wiring. All logic lives in `service.py` / `engine.py`.

Run:  uvicorn app.main:app --reload --port 8000
Env:  MODELS_DIR (persist models), AI_API_KEY (require X-API-Key), CORS_ORIGINS (comma list),
      MAX_BODY_MB (request body limit, default 64), LOG_LEVEL (default INFO)
"""

from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .errors import ServiceError
from .middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from .schemas import (
    DetectRequest,
    DetectResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ScoreRequest,
    ScoreResponse,
    TrainRequest,
)
from .service import AnomalyService
from .store import ModelStore


DEFAULT_MAX_BODY_MB = 64


def _max_body_bytes() -> int:
    raw = os.environ.get("MAX_BODY_MB", "").strip()
    try:
        mb = float(raw) if raw else DEFAULT_MAX_BODY_MB
    except ValueError:
        mb = DEFAULT_MAX_BODY_MB
    return int(mb * 1024 * 1024)


def create_app(service: AnomalyService | None = None, max_body_bytes: int | None = None) -> FastAPI:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(message)s")
    svc =service if service is not None else AnomalyService(ModelStore(os.environ.get("MODELS_DIR") or None))
    api_key = os.environ.get("AI_API_KEY") or None

    async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and x_api_key != api_key:
            raise ServiceError("UNAUTHORIZED", "API key inválida o ausente.", 401)

    app = FastAPI(
        title="docu-sentinel-ai",
        version=__version__,
        description="Servicio de detección de anomalías (Isolation Forest) para docu-sentinel.",
    )
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
        )

    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=max_body_bytes if max_body_bytes is not None else _max_body_bytes(),
    )
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=exc.to_dict())

    errors = {
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    }

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return svc.health()

    @app.post(
        "/v1/models",
        response_model=ModelInfo,
        status_code=201,
        tags=["models"],
        dependencies=[Depends(require_api_key)],
        responses=errors,
    )
    def train(req: TrainRequest) -> ModelInfo:
        return svc.train(req)

    @app.get("/v1/models", response_model=list[ModelInfo], tags=["models"],
             dependencies=[Depends(require_api_key)])
    def list_models() -> list[ModelInfo]:
        return svc.list_models()

    @app.get("/v1/models/{model_id}", response_model=ModelInfo, tags=["models"],
             dependencies=[Depends(require_api_key)], responses=errors)
    def get_model(model_id: str) -> ModelInfo:
        return svc.get_model(model_id)

    @app.delete("/v1/models/{model_id}", status_code=204, tags=["models"],
                dependencies=[Depends(require_api_key)], responses=errors)
    def delete_model(model_id: str) -> None:
        svc.delete_model(model_id)

    @app.post("/v1/models/{model_id}/score", response_model=ScoreResponse, tags=["models"],
              dependencies=[Depends(require_api_key)], responses=errors)
    def score(model_id: str, req: ScoreRequest) -> ScoreResponse:
        return svc.score(model_id, req)

    @app.post("/v1/detect", response_model=DetectResponse, tags=["detect"],
              dependencies=[Depends(require_api_key)], responses=errors)
    def detect(req: DetectRequest) -> DetectResponse:
        return svc.detect(req)

    return app


app = create_app()
