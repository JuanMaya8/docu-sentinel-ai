import tempfile
import unittest

from app.errors import ServiceError
from app.schemas import DetectRequest, ScoreRequest, TrainRequest
from app.service import AnomalyService
from app.store import ModelStore
from tests.helpers import make_normal, with_outliers


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.columns, self.normal = make_normal()
        self.rows, self.bad = with_outliers(self.normal)

    def test_train_score_roundtrip(self):
        svc = AnomalyService()
        info = svc.train(TrainRequest(name="base", columns=self.columns, matrix=self.normal, seed=3))
        self.assertTrue(info.model_id.startswith("mdl_"))
        self.assertEqual(info.rows, len(self.normal))
        self.assertEqual(info.kinds, ["numeric", "numeric", "numeric", "rarity", "missing"])
        resp = svc.score(info.model_id, ScoreRequest(columns=self.columns, matrix=self.rows))
        self.assertEqual(resp.model_id, info.model_id)
        self.assertEqual(len(resp.scores), len(self.rows))
        for i in self.bad:
            self.assertTrue(resp.flags[i])
            self.assertIn(str(i), resp.contributions)
            self.assertGreaterEqual(len(resp.contributions[str(i)]), 1)
        # only flagged rows are explained by default
        self.assertEqual(set(resp.contributions), {str(i) for i, f in enumerate(resp.flags) if f})

    def test_explain_modes(self):
        svc = AnomalyService()
        info = svc.train(TrainRequest(columns=self.columns, matrix=self.normal))
        r_none = svc.score(info.model_id, ScoreRequest(columns=self.columns, matrix=self.rows, explain="none"))
        r_all = svc.score(info.model_id, ScoreRequest(columns=self.columns, matrix=self.rows[:50], explain="all"))
        self.assertEqual(r_none.contributions, {})
        self.assertEqual(len(r_all.contributions), 50)

    def test_schema_mismatch(self):
        svc = AnomalyService()
        info = svc.train(TrainRequest(columns=self.columns, matrix=self.normal))
        bad_cols = list(reversed(self.columns))
        with self.assertRaises(ServiceError) as ctx:
            svc.score(info.model_id, ScoreRequest(columns=bad_cols, matrix=[r[::-1] for r in self.rows]))
        self.assertEqual(ctx.exception.code, "SCHEMA_MISMATCH")
        self.assertEqual(ctx.exception.status, 422)

    def test_unknown_model(self):
        svc = AnomalyService()
        with self.assertRaises(ServiceError) as ctx:
            svc.get_model("mdl_nope")
        self.assertEqual(ctx.exception.status, 404)

    def test_detect_stateless(self):
        svc = AnomalyService()
        resp = svc.detect(DetectRequest(columns=self.columns, matrix=self.rows, contamination=0.02))
        self.assertEqual(resp.rows, len(self.rows))
        self.assertIsNone(resp.model_id)
        self.assertEqual(len(svc.list_models()), 0)
        for i in self.bad:
            self.assertTrue(resp.flags[i])

    def test_invalid_training_data(self):
        svc = AnomalyService()
        with self.assertRaises(ServiceError) as ctx:
            svc.train(TrainRequest(columns=self.columns, matrix=self.normal[:5]))
        self.assertEqual(ctx.exception.code, "INVALID_TRAINING_DATA")

    def test_schema_validation(self):
        with self.assertRaises(ValueError):
            TrainRequest(columns=["a", "b"], matrix=[[1.0]])
        with self.assertRaises(ValueError):
            TrainRequest(columns=["a", "a"], matrix=[[1.0, 2.0]])
        with self.assertRaises(ValueError):
            TrainRequest(columns=["a"], kinds=["numeric", "date"], matrix=[[1.0]])

    def test_persistence_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = AnomalyService(ModelStore(tmp))
            info = svc.train(TrainRequest(columns=self.columns, matrix=self.normal, seed=9))
            before = svc.score(info.model_id, ScoreRequest(columns=self.columns, matrix=self.rows))
            svc2 = AnomalyService(ModelStore(tmp))  # simulated restart
            self.assertEqual([m.model_id for m in svc2.list_models()], [info.model_id])
            after = svc2.score(info.model_id, ScoreRequest(columns=self.columns, matrix=self.rows))
            self.assertEqual(before.scores, after.scores)
            svc2.delete_model(info.model_id)
            self.assertEqual(len(AnomalyService(ModelStore(tmp)).list_models()), 0)

    def test_health(self):
        self.assertEqual(AnomalyService().health().status, "ok")


if __name__ == "__main__":
    unittest.main()
