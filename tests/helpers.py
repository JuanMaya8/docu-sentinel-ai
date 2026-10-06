"""Synthetic data helpers for tests."""

from __future__ import annotations

import numpy as np


def make_normal(n: int = 600, seed: int = 7):
    """3 numeric features with correlation + a rarity feature + a missing indicator."""
    rng = np.random.default_rng(seed)
    qty = rng.integers(1, 20, n).astype(float)
    price = rng.normal(100.0, 10.0, n)
    total = qty * price * 1.19 + rng.normal(0, 5.0, n)
    rarity = rng.choice([0.4, 0.7, 1.2], size=n, p=[0.6, 0.3, 0.1])
    missing = np.zeros(n)
    columns = ["quantity", "unit_price", "total", "city__rarity", "email__missing"]
    matrix = [
        [float(qty[i]), float(price[i]), float(total[i]), float(rarity[i]), float(missing[i])]
        for i in range(n)
    ]
    return columns, matrix


def with_outliers(matrix: list[list[float]]):
    rows = [list(r) for r in matrix]
    # row 0: absurd total, row 1: absurd price, row 2: rare city + missing email
    rows[0][2] = 250_000.0
    rows[1][1] = 9_000.0
    rows[2][3] = 6.9
    rows[2][4] = 1.0
    return rows, [0, 1, 2]
