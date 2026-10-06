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


if __name__ == "__main__":
    unittest.main()
