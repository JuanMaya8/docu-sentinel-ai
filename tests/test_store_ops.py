"""Model store backends (disk, PostgreSQL), TTL, /metrics and rate limiting."""

import importlib.util
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from app import engine
from app.store import DiskBackend, ModelStore
from tests.helpers import make_normal

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None
PG_URL = os.environ.get("TEST_DATABASE_URL")


def _model(trained_at: str | None = None):
    columns, rows = make_normal(120)
    model, _ = engine.fit(columns, rows, seed=1)
    if trained_at:
        model.trained_at = trained_at
    return model


class StoreTests(unittest.TestCase):
    def test_disk_backend_survives_restart_and_writes_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            store = ModelStore(d)
            self.assertEqual(store.kind, "disk")
            mid = store.add(_model())
            self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(d)))
            again = ModelStore(d)
            self.assertEqual(again.get(mid).rows, 120)
            again.delete(mid)
            self.assertEqual(len(ModelStore(d)), 0)

    def test_ttl_purges_old_models_on_boot_and_on_add(self):
        with tempfile.TemporaryDirectory() as d:
            plain = ModelStore(d)
            old_id = plain.add(_model((datetime.now(timezone.utc) - timedelta(days=40)).isoformat(timespec="seconds")))
            fresh_id = plain.add(_model())
            store = ModelStore(d, ttl_days=30)
            self.assertEqual([mid for mid, _ in store.items()], [fresh_id])
            self.assertFalse(os.path.exists(os.path.join(d, f"{old_id}.joblib")))
            self.assertEqual(store.purge_expired(now=datetime.now(timezone.utc) + timedelta(days=31)), [fresh_id])

    def test_from_env_picks_the_backend(self):
        self.assertEqual(ModelStore.from_env({}).kind, "memory")
        with tempfile.TemporaryDirectory() as d:
            s = ModelStore.from_env({"MODELS_DIR": d, "MODEL_TTL_DAYS": "7"})
            self.assertEqual((s.kind, s.ttl_days), ("disk", 7.0))
            self.assertEqual(ModelStore.from_env({"MODEL_TTL_DAYS": "nope"}).ttl_days, 0)

    def test_corrupted_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "mdl_000000000bad.joblib"), "wb") as f:
                f.write(b"not a model")
            self.assertEqual(len(DiskBackend(d).load_all()), 0)


@unittest.skipUnless(PG_URL, "TEST_DATABASE_URL not set")
class PostgresStoreTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        with psycopg.connect(PG_URL, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS ds_ai_models")

    def test_models_persist_in_postgres(self):
        store = ModelStore(database_url=PG_URL)
        self.assertEqual(store.kind, "postgres")
        mid = store.add(_model())
        again = ModelStore(database_url=PG_URL)
        restored = again.get(mid)
        self.assertEqual(restored.rows, 120)
        scores, _, _ = engine.score(restored, make_normal(10)[1])
        self.assertEqual(len(scores), 10)
        again.delete(mid)
        self.assertEqual(len(ModelStore(database_url=PG_URL)), 0)

    def test_ttl_in_postgres(self):
        store = ModelStore(database_url=PG_URL)
        store.add(_model((datetime.now(timezone.utc) - timedelta(days=10)).isoformat(timespec="seconds")))
        self.assertEqual(len(ModelStore(database_url=PG_URL, ttl_days=5)), 0)
        self.assertEqual(len(ModelStore(database_url=PG_URL)), 0, "expired model was deleted from the table")


@unittest.skipUnless(HAS_FASTAPI, "fastapi/httpx not installed")
class OpsApiTests(unittest.TestCase):
    def test_metrics_endpoint(self):
        from fastapi.testclient import TestClient

        from app.main import create_app
        from app.service import AnomalyService

        client = TestClient(create_app(AnomalyService()))
        columns, rows = make_normal(100)
        model_id = client.post("/v1/models", json={"columns": columns, "matrix": rows}).json()["model_id"]
        client.post(f"/v1/models/{model_id}/score", json={"columns": columns, "matrix": rows[:10]})
        client.get("/health")
        text = client.get("/metrics").text
        self.assertIn('ds_ai_http_requests_total{method="POST",route="/v1/models/{model_id}/score",status="200"} 1', text)
        self.assertIn("ds_ai_models_trained_total 1", text)
        self.assertIn("ds_ai_rows_scored_total 10", text)
        self.assertIn("ds_ai_models 1", text)
        self.assertEqual(client.get("/health").json()["store"], "memory")

    def test_metrics_token(self):
        from fastapi.testclient import TestClient

        from app.main import create_app
        from app.service import AnomalyService

        os.environ["METRICS_TOKEN"] = "tok"
        try:
            client = TestClient(create_app(AnomalyService()))
        finally:
            del os.environ["METRICS_TOKEN"]
        self.assertEqual(client.get("/metrics").status_code, 401)
        self.assertEqual(client.get("/metrics", headers={"Authorization": "Bearer tok"}).status_code, 200)

    def test_rate_limit(self):
        from fastapi.testclient import TestClient

        from app.main import create_app
        from app.service import AnomalyService

        client = TestClient(create_app(AnomalyService(), rate_limit_per_min=3))
        codes = [client.get("/v1/models").status_code for _ in range(5)]
        self.assertEqual(codes, [200, 200, 200, 429, 429])
        r = client.get("/v1/models")
        self.assertEqual(r.json()["code"], "RATE_LIMITED")
        self.assertIn("retry-after", r.headers)
        self.assertEqual(client.get("/health").status_code, 200, "health is never limited")
        other = client.get("/v1/models", headers={"X-API-Key": "otra"})
        self.assertEqual(other.status_code, 200, "limits are per API key / client")


if __name__ == "__main__":
    unittest.main()
