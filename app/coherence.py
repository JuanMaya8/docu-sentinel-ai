"""Coherence model: a robust Gaussian model of the joint distribution of the robust-scaled features.

Port of docu-sentinel-frontend/src/engine/ml/coherence.ts (same maths, same constants).

An Isolation Forest splits ONE feature at a time, so it is weak at spotting rows whose values are all
plausible on their own but whose COMBINATION is impossible (e.g. a total that does not match quantity x price).
The Mahalanobis distance uses the covariance between fields and amplifies deviations along low-variance
directions, which is exactly where "broken relations" live.

* Robustness: trimmed fit (the most distant rows are dropped and the model is refit).
* Attribution: with the precision matrix P the standardised conditional residual of feature j given the rest is
  t_j = (P d)_j / sqrt(P_jj) and the value the other features predict is z_j - (P d)_j / P_jj.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_FEATURES = 3
MIN_ROWS = 50


@dataclass
class CoherenceModel:
    mean: np.ndarray
    precision: np.ndarray
    dev0: float


def _mean_precision(z: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mean = z.mean(axis=0)
    d = z - mean[None, :]
    cov = (d.T @ d) / max(1, z.shape[0] - 1)
    trace = float(np.trace(cov))
    if not trace > 1e-12:
        return None
    values, vectors = np.linalg.eigh(cov)
    floor = max(1e-10 * trace, 1e-12)
    inv = 1.0 / np.maximum(values, floor)
    precision = (vectors * inv[None, :]) @ vectors.T
    return mean, precision


def distances(model: CoherenceModel, z: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of every row of `z`."""
    d = z - model.mean[None, :]
    sq = np.einsum("ij,jk,ik->i", d, model.precision, d)
    return np.sqrt(np.maximum(sq, 0.0))


def fit_coherence(z: np.ndarray, trim: float = 0.03) -> CoherenceModel | None:
    n, width = z.shape
    if width < MIN_FEATURES or n < MIN_ROWS:
        return None
    first = _mean_precision(z)
    if first is None:
        return None
    model = CoherenceModel(mean=first[0], precision=first[1], dev0=0.0)
    dist = distances(model, z)
    cut = np.sort(dist)[min(n - 1, int(np.floor((1.0 - trim) * (n - 1))))]
    keep = dist <= cut
    zk = z
    if MIN_ROWS <= int(keep.sum()) < n:
        zk = z[keep]
        second = _mean_precision(zk)
        if second is not None:
            model = CoherenceModel(mean=second[0], precision=second[1], dev0=0.0)
    kept = np.sort(distances(model, zk))
    q999 = kept[min(len(kept) - 1, int(np.floor(0.999 * (len(kept) - 1))))]
    model.dev0 = float(max(np.sqrt(width) + 4.0, 1.5 * q999))
    return model


def coherence_scores(model: CoherenceModel, z: np.ndarray) -> np.ndarray:
    """0 below dev0, then 0.5 -> 1 (aligned with Isolation Forest scores)."""
    dist = distances(model, z)
    x = (dist - model.dev0) / model.dev0
    return np.where(dist > model.dev0, 0.5 + 0.5 * (1.0 - np.exp(-(x**2))), 0.0)


def attribution(model: CoherenceModel, zrow: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(t, expected_z, distance) for one row."""
    d = zrow - model.mean
    q = model.precision @ d
    dist = float(np.sqrt(max(float(d @ q), 0.0)))
    pjj = np.diag(model.precision)
    safe = np.where(pjj > 0, pjj, 1.0)
    t = np.where(pjj > 0, q / np.sqrt(safe), 0.0)
    expected = np.where(pjj > 0, zrow - q / safe, zrow)
    return t, expected, dist
