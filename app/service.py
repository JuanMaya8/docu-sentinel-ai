"""Application service. Framework-free: takes/returns pydantic models.

`main.py` only wires HTTP to these methods, so this file is what the unit tests cover.
"""

from __future__ import annotations

from . import __version__, engine
from .errors import ServiceError
from .schemas import (
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ModelInfo,
    ScoreRequest,
    ScoreResponse,
    TrainRequest,
)
from .store import ModelStore


class AnomalyService:
    def __init__(self, store: ModelStore | None = None) -> None:
        self.store = store if store is not None else ModelStore()

    # ---- helpers -----------------------------------------------------------------
    @staticmethod
    def _info(model_id: str, m: engine.FittedModel) -> ModelInfo:
        return ModelInfo(
            model_id=model_id,
            name=m.name,
            columns=m.columns,
            kinds=m.kinds,
            rows=m.rows,
            threshold=round(m.threshold, 6),
            contamination=m.contamination,
            seed=m.seed,
            trained_at=m.trained_at,
            schema_signature=m.schema_signature,
            engine=engine.ENGINE_NAME,
        )

    @staticmethod
    def _explain_indexes(mode: str, flags) -> list[int]:
        if mode == "none":
            return []
        if mode == "all":
            return list(range(len(flags)))
        return [i for i, f in enumerate(flags) if f]

    def _train(self, req: TrainRequest) -> tuple[engine.FittedModel, object]:
        try:
            return engine.fit(
                columns=req.columns,
                matrix=req.matrix,
                kinds=req.kinds,
                contamination=req.contamination,
                n_estimators=req.n_estimators,
                seed=req.seed,
                name=req.name,
                schema_signature=req.schema_signature,
            )
        except ValueError as exc:
            raise ServiceError("INVALID_TRAINING_DATA", str(exc), 422) from exc

    def _score_with(
        self, model: engine.FittedModel, matrix, explain: str, top_k: int, threshold: float | None
    ) -> tuple[ScoreResponse, object]:
        scores, flags, x = engine.score(model, matrix, threshold)
        contributions = engine.explain_rows(model, x, self._explain_indexes(explain, flags), top_k)
        resp = ScoreResponse(
            model_id=None,
            threshold=round(threshold if threshold is not None else model.threshold, 6),
            scores=[round(float(s), 6) for s in scores],
            flags=[bool(f) for f in flags],
            contributions=contributions,  # type: ignore[arg-type]
        )
        return resp, x

    # ---- use cases ---------------------------------------------------------------
    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok", version=__version__, engine=engine.ENGINE_NAME, models=len(self.store)
        )

    def train(self, req: TrainRequest) -> ModelInfo:
        model, _ = self._train(req)
        model_id = self.store.add(model)
        return self._info(model_id, model)

    def score(self, model_id: str, req: ScoreRequest) -> ScoreResponse:
        model = self.store.get(model_id)
        if req.columns != model.columns:
            missing = [c for c in model.columns if c not in req.columns]
            extra = [c for c in req.columns if c not in model.columns]
            raise ServiceError(
                "SCHEMA_MISMATCH",
                "Las columnas no coinciden con el modelo entrenado"
                + (f"; faltan: {', '.join(missing[:8])}" if missing else "")
                + (f"; sobran: {', '.join(extra[:8])}" if extra else "")
                + (". El orden de las columnas también debe coincidir." if not missing and not extra else "."),
                422,
            )
        resp, _ = self._score_with(model, req.matrix, req.explain, req.top_k, req.threshold)
        resp.model_id = model_id
        return resp

    def detect(self, req: DetectRequest) -> DetectResponse:
        """Stateless unsupervised mode: train on the batch and score the same batch."""
        model, _ = self._train(req)
        resp, _ = self._score_with(model, req.matrix, req.explain, req.top_k, None)
        return DetectResponse(**resp.model_dump(), rows=model.rows, engine=engine.ENGINE_NAME)

    def list_models(self) -> list[ModelInfo]:
        return [self._info(mid, m) for mid, m in self.store.items()]

    def get_model(self, model_id: str) -> ModelInfo:
        return self._info(model_id, self.store.get(model_id))

    def delete_model(self, model_id: str) -> None:
        self.store.delete(model_id)
