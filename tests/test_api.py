"""HTTP-level tests. Require fastapi + httpx (see requirements-dev.txt); skipped otherwise."""

import importlib.util
import unittest

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None


@unittest.skipUnless(HAS_FASTAPI, "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.main import create_app
        from app.service import AnomalyService
        from tests.helpers import make_normal, with_outliers

        self.client = TestClient(create_app(AnomalyService()))
        self.columns, normal = make_normal(300)
        self.normal = normal
        self.rows, self.bad = with_outliers(normal)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_full_flow(self):
        r = self.client.post("/v1/models", json={"columns": self.columns, "matrix": self.normal})
        self.assertEqual(r.status_code, 201, r.text)
        model_id = r.json()["model_id"]
        r = self.client.post(
            f"/v1/models/{model_id}/score", json={"columns": self.columns, "matrix": self.rows}
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for i in self.bad:
            self.assertTrue(body["flags"][i])
        self.assertEqual(self.client.delete(f"/v1/models/{model_id}").status_code, 204)
        self.assertEqual(self.client.get(f"/v1/models/{model_id}").status_code, 404)

    def test_null_values_accepted(self):
        rows = [list(r) for r in self.normal]
        rows[0][0] = None
        r = self.client.post("/v1/detect", json={"columns": self.columns, "matrix": rows})
        self.assertEqual(r.status_code, 200, r.text)

    def test_error_shape(self):
        r = self.client.post("/v1/models/mdl_x/score", json={"columns": ["a"], "matrix": [[1.0]]})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["code"], "MODEL_NOT_FOUND")

    def test_threshold_floor_round_trip(self):
        r = self.client.post(
            "/v1/models",
            json={"columns": self.columns, "matrix": self.normal, "threshold_floor": 0.65},
        )
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["threshold_floor"], 0.65)
        self.assertGreaterEqual(body["threshold"], 0.65)
        r = self.client.post(
            "/v1/models", json={"columns": self.columns, "matrix": self.normal, "threshold_floor": 0.2}
        )
        self.assertEqual(r.status_code, 422)

    def test_coherence_contribution_in_response(self):
        r = self.client.post("/v1/models", json={"columns": self.columns, "matrix": self.normal})
        model_id = r.json()["model_id"]
        broken = [2.0, 100.0, 19 * 100 * 1.19, 0.4, 0.0]
        r = self.client.post(
            f"/v1/models/{model_id}/score", json={"columns": self.columns, "matrix": [broken]}
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["flags"][0])
        first = body["contributions"]["0"][0]
        self.assertEqual(first["kind"], "coherence")
        self.assertIsNotNone(first["expected"])

    def test_schema_mismatch(self):
        r = self.client.post("/v1/models", json={"columns": self.columns, "matrix": self.normal})
        model_id = r.json()["model_id"]
        cols = list(reversed(self.columns))
        r = self.client.post(
            f"/v1/models/{model_id}/score", json={"columns": cols, "matrix": [[1.0] * len(cols)]}
        )
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["code"], "SCHEMA_MISMATCH")

    def test_request_id_is_echoed_or_created(self):
        r = self.client.get("/health", headers={"X-Request-ID": "abc-123"})
        self.assertEqual(r.headers["x-request-id"], "abc-123")
        r = self.client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
        self.assertNotEqual(r.headers["x-request-id"], "bad id with spaces")
        self.assertTrue(r.headers["x-request-id"])

    def test_body_size_limit(self):
        from fastapi.testclient import TestClient

        from app.main import create_app
        from app.service import AnomalyService

        small = TestClient(create_app(AnomalyService(), max_body_bytes=2_000))
        r = small.post("/v1/detect", json={"columns": self.columns, "matrix": self.normal})
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.json()["code"], "PAYLOAD_TOO_LARGE")
        self.assertIn("tamaño máximo", r.json()["message"])

    def test_api_key_required_when_configured(self):
        import os

        from fastapi.testclient import TestClient

        from app.main import create_app
        from app.service import AnomalyService

        os.environ["AI_API_KEY"] = "secret"
        try:
            client = TestClient(create_app(AnomalyService()))
        finally:
            del os.environ["AI_API_KEY"]
        self.assertEqual(client.get("/v1/models").status_code, 401)
        self.assertEqual(client.get("/v1/models", headers={"X-API-Key": "secret"}).status_code, 200)
        self.assertEqual(client.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
