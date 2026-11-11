"""Model store: in-memory registry with optional persistence.

Backends (picked by `ModelStore.from_env()`):
  * DATABASE_URL set  -> PostgreSQL table `ds_ai_models` (needed on hosts with an ephemeral disk, e.g. Render free)
  * MODELS_DIR set    -> one joblib file per model on disk
  * neither           -> memory only
MODEL_TTL_DAYS > 0 deletes models older than that many days (on boot and before each new model).

Security note: models are serialised with joblib/pickle. Only load them from a directory or database that untrusted
parties cannot write to.
"""

from __future__ import annotations

import io
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import joblib

from .engine import FittedModel
from .errors import ServiceError

MAX_MODELS = 100


class _Backend(Protocol):
    def load_all(self) -> dict[str, FittedModel]: ...
    def save(self, model_id: str, model: FittedModel) -> None: ...
    def delete(self, model_id: str) -> None: ...


def _dumps(model: FittedModel) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.getvalue()


def _loads(data: bytes) -> FittedModel:
    return joblib.load(io.BytesIO(data))


class DiskBackend:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> dict[str, FittedModel]:
        out: dict[str, FittedModel] = {}
        for path in sorted(self.dir.glob("mdl_*.joblib")):
            try:
                out[path.stem] = joblib.load(path)
            except Exception:  # corrupted file: skip, never crash the service on boot
                continue
        return out

    def save(self, model_id: str, model: FittedModel) -> None:
        tmp = self.dir / f"{model_id}.joblib.tmp"
        joblib.dump(model, tmp)
        tmp.replace(self.dir / f"{model_id}.joblib")

    def delete(self, model_id: str) -> None:
        (self.dir / f"{model_id}.joblib").unlink(missing_ok=True)


class PostgresBackend:
    """Stores each model as a compressed joblib blob in `ds_ai_models` (created on first use)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS ds_ai_models (
      id          text PRIMARY KEY,
      trained_at  timestamptz NOT NULL,
      payload     bytea NOT NULL
    )"""

    def __init__(self, url: str) -> None:
        import psycopg  # imported lazily: only needed when DATABASE_URL is used

        self._psycopg = psycopg
        self.url = url
        with self._connect() as conn:
            conn.execute(self.SCHEMA)

    def _connect(self):
        return self._psycopg.connect(self.url, autocommit=True, connect_timeout=10)

    def load_all(self) -> dict[str, FittedModel]:
        out: dict[str, FittedModel] = {}
        with self._connect() as conn:
            for model_id, payload in conn.execute("SELECT id, payload FROM ds_ai_models ORDER BY trained_at"):
                try:
                    out[model_id] = _loads(bytes(payload))
                except Exception:
                    continue
        return out

    def save(self, model_id: str, model: FittedModel) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ds_ai_models (id, trained_at, payload) VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET trained_at = EXCLUDED.trained_at, payload = EXCLUDED.payload",
                (model_id, model.trained_at, _dumps(model)),
            )

    def delete(self, model_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ds_ai_models WHERE id = %s", (model_id,))


class ModelStore:
    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        database_url: str | None = None,
        ttl_days: float = 0,
        backend: _Backend | None = None,
    ) -> None:
        self._models: dict[str, FittedModel] = {}
        self._lock = threading.Lock()
        self.ttl_days = ttl_days
        if backend is not None:
            self._backend: _Backend | None = backend
        elif database_url:
            self._backend = PostgresBackend(database_url)
        elif directory:
            self._backend = DiskBackend(directory)
        else:
            self._backend = None
        self.kind = (
            "postgres" if isinstance(self._backend, PostgresBackend) else "disk" if isinstance(self._backend, DiskBackend) else "memory"
        )
        if self._backend is not None:
            self._models.update(self._backend.load_all())
        self.purge_expired()

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ModelStore":
        env = dict(os.environ) if env is None else env
        try:
            ttl = float(env.get("MODEL_TTL_DAYS", "") or 0)
        except ValueError:
            ttl = 0
        return cls(env.get("MODELS_DIR") or None, database_url=env.get("DATABASE_URL") or None, ttl_days=ttl)

    def purge_expired(self, now: datetime | None = None) -> list[str]:
        """Deletes models older than `ttl_days` (no-op when ttl_days <= 0). Returns the removed ids."""
        if not self.ttl_days or self.ttl_days <= 0:
            return []
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=self.ttl_days)
        removed: list[str] = []
        with self._lock:
            for model_id, model in list(self._models.items()):
                try:
                    trained = datetime.fromisoformat(model.trained_at)
                except ValueError:
                    continue
                if trained.tzinfo is None:
                    trained = trained.replace(tzinfo=timezone.utc)
                if trained < cutoff:
                    del self._models[model_id]
                    if self._backend is not None:
                        self._backend.delete(model_id)
                    removed.append(model_id)
        return removed

    def add(self, model: FittedModel) -> str:
        self.purge_expired()
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
