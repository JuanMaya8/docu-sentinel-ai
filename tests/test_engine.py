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
        self.assertIn("desviaciones robustas por encima de la mediana", out["0"][0]["message"])
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
        _, flags_hi, _ = engine.score(model, self.rows, threshold=0.99)
        self.assertEqual(int(flags_hi.sum()), 0)


if __name__ == "__main__":
    unittest.main()
