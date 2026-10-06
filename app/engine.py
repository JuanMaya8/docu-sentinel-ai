"""Anomaly engine: robust preprocessing + Isolation Forest + per-feature explanations.

Pure numpy/scikit-learn. No web framework imports, so it is unit-testable anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from .explain_es import describe_contribution
from .schemas import FeatureKind, infer_kind

ENGINE_NAME = "sklearn-isolation-forest"
MAX_ABS_Z = 50.0
ABS_THRESHOLD_FLOOR = 0.55  # an Isolation Forest score near 0.5 means "normal"
UNSEEN_INDICATOR_EXT = 3.0  # a field that was never empty in training is now empty
EXT_SCALE = 4.0  # robust-z units beyond the training range for the guard to reach ~0.63


@dataclass
class FittedModel:
    columns: list[str]
    kinds: list[FeatureKind]
    median: np.ndarray
    scale: np.ndarray
    forest: IsolationForest
    threshold: float
    contamination: float
    seed: int
    rows: int
    trained_at: str
    name: str | None = None
    schema_signature: str | None = None
    # training prevalence of 1.0 for missing-indicator columns (used for explanations)
    indicator_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # training range in robust-z space, used by the extrapolation guard
    z_min: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z_max: np.ndarray = field(default_factory=lambda: np.zeros(0))


def to_array(matrix: Sequence[Sequence[float | None]], width: int) -> np.ndarray:
    """None -> NaN, ragged input rejected earlier by the schema."""
    if len(matrix) == 0:
        return np.zeros((0, width), dtype=np.float64)
    return np.array(
        [[np.nan if v is None else float(v) for v in row] for row in matrix],
        dtype=np.float64,
    )


def _robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = x.shape[1]
    median = np.zeros(width)
    scale = np.ones(width)
    for j in range(width):
        col = x[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        median[j] = float(np.median(col))
        q1, q3 = np.percentile(col, [25, 75])
        s = (q3 - q1) / 1.349
        if s <= 1e-12:
            s = float(np.std(col))
        scale[j] = s if s > 1e-12 else 1.0
    return median, scale


def _transform(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(x), median[None, :], x)
    z = (filled - median[None, :]) / scale[None, :]
    return np.clip(z, -MAX_ABS_Z, MAX_ABS_Z)


def fit(
    columns: list[str],
    matrix: Sequence[Sequence[float | None]],
    kinds: list[FeatureKind] | None = None,
    contamination: float = 0.01,
    n_estimators: int = 200,
    seed: int = 42,
    name: str | None = None,
    schema_signature: str | None = None,
) -> tuple[FittedModel, np.ndarray]:
    """Train on `matrix`. Returns (model, training_scores)."""
    x = to_array(matrix, len(columns))
    if x.shape[0] < 20:
        raise ValueError("Se necesitan al menos 20 filas para entrenar un modelo")
    kinds = kinds or [infer_kind(c) for c in columns]
    median, scale = _robust_stats(x)
    z = _transform(x, median, scale)
    forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(256, x.shape[0]),
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(z)
    train_scores = -forest.score_samples(z)
    quantile = float(np.quantile(train_scores, 1.0 - contamination))
    threshold = max(ABS_THRESHOLD_FLOOR, quantile)
    indicator_rate = np.zeros(len(columns))
    for j, kind in enumerate(kinds):
        if kind == "missing":
            col = x[:, j]
            col = col[~np.isnan(col)]
            indicator_rate[j] = float(np.mean(col > 0.5)) if col.size else 0.0
    model = FittedModel(
        columns=list(columns),
        kinds=list(kinds),
        median=median,
        scale=scale,
        forest=forest,
        threshold=threshold,
        contamination=contamination,
        seed=seed,
        rows=int(x.shape[0]),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        name=name,
        schema_signature=schema_signature,
        indicator_rate=indicator_rate,
        z_min=z.min(axis=0),
        z_max=z.max(axis=0),
    )
    return model, train_scores


def score(
    model: FittedModel,
    matrix: Sequence[Sequence[float | None]],
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (scores in ~[0,1], flags, raw matrix) for the given rows."""
    x = to_array(matrix, len(model.columns))
    if x.shape[0] == 0:
        return np.zeros(0), np.zeros(0, dtype=bool), x
    z = _transform(x, model.median, model.scale)
    scores = -model.forest.score_samples(z)
    scores = np.maximum(scores, _extrapolation_score(model, z))
    limit = model.threshold if threshold is None else threshold
    return scores, scores >= limit, x


def _extrapolation_score(model: FittedModel, z: np.ndarray) -> np.ndarray:
    """Isolation Forests under-score points far OUTSIDE the training range (they follow the
    same branch as the training extreme). This guard scores how far a row lies beyond the
    training min/max, in robust-z units, and maps it smoothly to [0.5, 1)."""
    below = model.z_min[None, :] - z
    above = z - model.z_max[None, :]
    ext = np.maximum(np.maximum(below, above), 0.0)
    for j, kind in enumerate(model.kinds):
        if kind == "missing":
            # indicator columns have no meaningful z-distance: handle "never empty before"
            ext[:, j] = np.where(
                (model.indicator_rate[j] == 0.0) & (z[:, j] > model.z_max[j] + 1e-9),
                UNSEEN_INDICATOR_EXT,
                0.0,
            )
    worst = ext.max(axis=1) if ext.shape[1] else np.zeros(z.shape[0])
    guard = 0.5 + 0.5 * (1.0 - np.exp(-((worst / EXT_SCALE) ** 2)))
    return np.where(worst > 0.0, guard, 0.0)


def explain_rows(
    model: FittedModel,
    x: np.ndarray,
    row_indexes: Sequence[int],
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Top-k feature contributions per row, as dictionaries matching `Contribution`."""
    out: dict[str, list[dict]] = {}
    for i in row_indexes:
        row = x[i]
        candidates: list[tuple[float, int, float]] = []
        for j, kind in enumerate(model.kinds):
            value = row[j]
            if kind == "missing":
                if np.isnan(value) or value < 0.5:
                    continue
                rate = float(model.indicator_rate[j])
                if rate > 0.2:
                    continue
                strength = min(MAX_ABS_Z, -math.log(max(rate, 1e-4)) * 1.5)
                candidates.append((strength, j, strength))
                continue
            if np.isnan(value):
                continue
            z = float((value - model.median[j]) / model.scale[j])
            z = max(-MAX_ABS_Z, min(MAX_ABS_Z, z))
            if kind == "rarity" and z <= 0:
                continue  # common categories are not suspicious
            candidates.append((abs(z), j, z))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        items = []
        for _, j, dev in candidates[:top_k]:
            kind = model.kinds[j]
            col = model.columns[j]
            value = None if np.isnan(row[j]) else float(row[j])
            items.append(
                {
                    "column": col,
                    "field": _field_name(col),
                    "kind": kind,
                    "value": value,
                    "median": float(model.median[j]),
                    "deviation": float(dev),
                    "message": describe_contribution(
                        field=_field_name(col),
                        kind=kind,
                        value=value,
                        median=float(model.median[j]),
                        deviation=float(dev),
                        indicator_rate=float(model.indicator_rate[j]),
                    ),
                }
            )
        out[str(i)] = items
    return out


def _field_name(column: str) -> str:
    for suffix in ("__missing", "__rarity"):
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column
