import math
import unittest

import numpy as np

from app import engine
from app.formatting import fmt_epoch_days, fmt_number, fmt_percent
from tests.helpers import make_normal, with_outliers


class FormattingTests(unittest.TestCase):
    def test_fmt_number_es_co(self):
        self.assertEqual(fmt_number(1234.5), "1.234,5")
        self.assertEqual(fmt_number(1200), "1.200")
        self.assertEqual(fmt_number(0.25), "0,25")
        self.assertEqual(fmt_number(-7.2, 1), "-7,2")

    def test_fmt_epoch_and_percent(self):
        self.assertEqual(fmt_epoch_days(0), "1970-01-01")
        self.assertEqual(fmt_epoch_days(20_000), "2024-10-04")
        self.assertEqual(fmt_percent(0.5), "50%")
        self.assertEqual(fmt_percent(0.035), "3,5%")


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.columns, self.normal = make_normal()
        self.rows, self.bad = with_outliers(self.normal)

    def test_detects_injected_outliers(self):
        model, _ = engine.fit(self.columns, self.rows, contamination=0.02, seed=1)
        scores, flags, _ = engine.score(model, self.rows)
        for i in self.bad:
            self.assertTrue(flags[i], f"row {i} should be flagged (score={scores[i]:.3f})")
        flagged = int(flags.sum())
        self.assertLessEqual(flagged, 30, "must not flag a large share of normal rows")

    def test_clean_data_flags_little(self):
        model, _ = engine.fit(self.columns, self.normal, contamination=0.01, seed=1)
        _, flags, _ = engine.score(model, self.normal)
        self.assertLessEqual(flags.mean(), 0.03)

    def test_deterministic_with_seed(self):
        m1, _ = engine.fit(self.columns, self.rows, seed=5)
        m2, _ = engine.fit(self.columns, self.rows, seed=5)
        s1, _, _ = engine.score(m1, self.rows)
        s2, _, _ = engine.score(m2, self.rows)
        np.testing.assert_allclose(s1, s2)

    def test_scores_in_unit_interval(self):
        model, _ = engine.fit(self.columns, self.rows)
        scores, _, _ = engine.score(model, self.rows)
        self.assertTrue(((scores > 0) & (scores < 1)).all())

    def test_handles_none_values(self):
        rows = [list(r) for r in self.normal]
        rows[10][0] = None
        rows[11] = [None] * len(self.columns)
        model, _ = engine.fit(self.columns, rows)
        scores, _, _ = engine.score(model, rows)
        self.assertTrue(np.isfinite(scores).all())

    def test_too_few_rows(self):
        with self.assertRaises(ValueError):
            engine.fit(self.columns, self.normal[:5])

    def test_constant_column_does_not_break(self):
        rows = [[r[0], 5.0, r[2], r[3], r[4]] for r in self.normal]
        model, _ = engine.fit(self.columns, rows)
        self.assertTrue(math.isfinite(float(model.scale[1])))

    def test_explanations_name_the_right_fields_in_spanish(self):
        model, _ = engine.fit(self.columns, self.rows, contamination=0.02, seed=1)
        _, _, x = engine.score(model, self.rows)
        out = engine.explain_rows(model, x, [0, 1, 2], top_k=2)
        self.assertEqual(out["0"][0]["field"], "total")
        msg0 = out["0"][0]["message"]
        self.assertTrue(
            "desviaciones robustas por encima de la mediana" in msg0 or "no es coherente" in msg0, msg0
        )
        self.assertEqual(out["1"][0]["field"], "unit_price")
        fields_row2 = {c["field"] for c in out["2"]}
        self.assertTrue({"city", "email"} & fields_row2)
        msgs = " ".join(c["message"] for c in out["2"])
        self.assertTrue("poco frecuente" in msgs or "está vacío" in msgs)

    def test_baseline_mode_extrapolation_guard(self):
        """Train on clean history, score a NEW batch: far-out values must be flagged and
        fresh normal data must not be."""
        model, _ = engine.fit(self.columns, self.normal, contamination=0.01, seed=2)
        _, fresh = make_normal(500, seed=99)
        _, flags_fresh, _ = engine.score(model, fresh)
        self.assertLessEqual(flags_fresh.mean(), 0.04)
        _, flags_bad, _ = engine.score(model, self.rows)
        for i in self.bad:
            self.assertTrue(flags_bad[i], f"row {i} not flagged in baseline mode")

    def test_threshold_override(self):
        model, _ = engine.fit(self.columns, self.rows, seed=1)
        scores, flags_default, _ = engine.score(model, self.rows)
        limit = float(np.sort(scores)[-1]) + 1e-9
        _, flags_hi, _ = engine.score(model, self.rows, threshold=min(limit, 0.9999999))
        self.assertEqual(int(flags_hi.sum()), 0)
        _, flags_lo, _ = engine.score(model, self.rows, threshold=0.5)
        self.assertGreaterEqual(int(flags_lo.sum()), int(flags_default.sum()))

    def test_threshold_floor_is_respected(self):
        model, _ = engine.fit(self.columns, self.normal, seed=1, threshold_floor=0.7)
        self.assertGreaterEqual(model.threshold, 0.7)
        self.assertEqual(model.threshold_floor, 0.7)
        default, _ = engine.fit(self.columns, self.normal, seed=1)
        self.assertEqual(default.threshold_floor, engine.ABS_THRESHOLD_FLOOR)


class ParityTests(unittest.TestCase):
    """Features ported from docu-sentinel-frontend/src/engine/ml (log scale + coherence)."""

    def setUp(self):
        self.columns, self.normal = make_normal()

    def test_should_log_detects_skewed_positive_money(self):
        rng = np.random.default_rng(3)
        money = rng.lognormal(mean=7.0, sigma=1.2, size=500)
        self.assertTrue(engine.should_log(money, "numeric"))
        symmetric = rng.normal(100.0, 5.0, 500)
        self.assertFalse(engine.should_log(symmetric, "numeric"))
        with_negatives = np.concatenate([money, -np.ones(20)])
        self.assertFalse(engine.should_log(with_negatives, "numeric"))
        self.assertFalse(engine.should_log(money, "rarity"))

    def test_log_scale_keeps_non_positive_values_finite(self):
        rng = np.random.default_rng(4)
        money = rng.lognormal(7.0, 1.2, 400)
        rows = [[float(v), float(v) * 0.19, 0.0] for v in money]
        cols = ["subtotal", "tax", "notes__missing"]
        model, _ = engine.fit(cols, rows, seed=1)
        self.assertTrue(bool(model.log[0]))
        scores, _, x = engine.score(model, [[-5.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertTrue(np.isfinite(scores).all())
        engine.explain_rows(model, x, [0, 1])  # must not raise

    def test_coherence_catches_broken_relation_between_plausible_values(self):
        model, _ = engine.fit(self.columns, self.normal, contamination=0.01, seed=2)
        self.assertIsNotNone(model.coherence)
        # every value is inside its usual range, but total != quantity x unit_price x 1.19
        broken = [2.0, 100.0, 19 * 100 * 1.19, 0.4, 0.0]
        scores, flags, x = engine.score(model, [broken])
        self.assertTrue(flags[0], f"score={scores[0]:.3f}")
        items = engine.explain_rows(model, x, [0], top_k=3)["0"]
        self.assertEqual(items[0]["kind"], "coherence")
        self.assertIn(items[0]["field"], {"total", "quantity"})
        self.assertIn("no es coherente", items[0]["message"])
        self.assertIsNotNone(items[0]["expected"])

    def test_coherence_does_not_flag_fresh_normal_data(self):
        model, _ = engine.fit(self.columns, self.normal, contamination=0.01, seed=2)
        _, fresh = make_normal(800, seed=123)
        _, flags, _ = engine.score(model, fresh)
        self.assertLessEqual(flags.mean(), 0.03)

    def test_extreme_single_value_is_explained_on_its_own(self):
        model, _ = engine.fit(self.columns, self.normal, contamination=0.01, seed=2)
        row = list(self.normal[0])
        row[1] = 9_000.0  # far outside the training range
        _, flags, x = engine.score(model, [row])
        self.assertTrue(flags[0])
        items = engine.explain_rows(model, x, [0], top_k=2)["0"]
        self.assertEqual(items[0]["field"], "unit_price")
        self.assertNotEqual(items[0]["kind"], "coherence")

    def test_scores_never_reach_one(self):
        model, _ = engine.fit(self.columns, self.normal, seed=2)
        rows = [[1e9, 1e9, -1e9, 50.0, 1.0]]
        scores, _, _ = engine.score(model, rows)
        self.assertLess(float(scores[0]), 1.0)

    def test_old_pickles_without_new_fields_still_load(self):
        import pickle

        model, _ = engine.fit(self.columns, self.normal, seed=2)
        state = dict(model.__dict__)
        for key in ("log", "threshold_floor", "coherence"):
            state.pop(key)
        old = engine.FittedModel.__new__(engine.FittedModel)
        old.__setstate__(state)
        restored = pickle.loads(pickle.dumps(old))
        self.assertIsNone(restored.coherence)
        self.assertEqual(restored.threshold_floor, engine.ABS_THRESHOLD_FLOOR)
        scores, _, _ = engine.score(restored, self.normal[:10])
        self.assertEqual(scores.shape, (10,))


if __name__ == "__main__":
    unittest.main()
