"""Model store: in-memory registry with optional on-disk persistence (joblib).

Security note: joblib uses pickle. Only load files written by this service from a
directory that untrusted parties cannot write to.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import joblib

from .engine import FittedModel
from .errors import ServiceError

MAX_MODELS = 100


class ModelStore:
    def __init__(self, directory: str | Path | None = None) -> None:
        self._models: dict[str, FittedModel] = {}
        self._lock = threading.Lock()
        self._dir = Path(directory) if directory else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    def _load_all(self) -> None:
        assert self._dir is not None
        for path in sorted(self._dir.glob("mdl_*.joblib")):
            try:
                self._models[path.stem] = joblib.load(path)
            except Exception:  # corrupted file: skip, never crash the service on boot
                continue

    def add(self, model: FittedModel) -> str:
        with self._lock:
            if len(self._models) >= MAX_MODELS:
                raise ServiceError(
                    "MODEL_LIMIT",
                    f"Se alcanzó el máximo de {MAX_MODELS} modelos; elimina alguno antes de entrenar otro.",
                    409,
                )
            model_id = f"mdl_{uuid.uuid4().hex[:12]}"
            self._models[model_id] = model
            if self._dir:
                joblib.dump(model, self._dir / f"{model_id}.joblib")
            return model_id

    def get(self, model_id: str) -> FittedModel:
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            raise ServiceError("MODEL_NOT_FOUND", "El modelo solicitado no existe.", 404)
        return model

    def delete(self, model_id: str) -> None:
        with self._lock:
            if model_id not in self._models:
                raise ServiceError("MODEL_NOT_FOUND", "El modelo solicitado no existe.", 404)
            del self._models[model_id]
            if self._dir:
                (self._dir / f"{model_id}.joblib").unlink(missing_ok=True)

    def items(self) -> list[tuple[str, FittedModel]]:
        with self._lock:
            return sorted(self._models.items(), key=lambda kv: kv[1].trained_at, reverse=True)

    def __len__(self) -> int:
        return len(self._models)
