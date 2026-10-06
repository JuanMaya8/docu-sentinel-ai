"""Public contract (pydantic v2). Mirrors docs/CONTRACT.md.

Feature naming convention shared with docu-sentinel-frontend:
  <field>            numeric value (or a date as epoch days when kind == 'date')
  <field>__missing   1 when the field is empty, else 0   (kind == 'missing')
  <field>__rarity    -ln(relative frequency of the category) (kind == 'rarity')
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

FeatureKind = Literal["numeric", "date", "missing", "rarity"]
ExplainMode = Literal["flagged", "all", "none"]

MAX_ROWS = 200_000
MAX_CELLS = 8_000_000


def infer_kind(column: str) -> FeatureKind:
    if column.endswith("__missing"):
        return "missing"
    if column.endswith("__rarity"):
        return "rarity"
    return "numeric"


class _MatrixMixin(BaseModel):
    columns: list[str] = Field(min_length=1, max_length=512)
    kinds: list[FeatureKind] | None = None
    matrix: list[list[float | None]]

    @model_validator(mode="after")
    def _check_shape(self) -> "_MatrixMixin":
        width = len(self.columns)
        if self.kinds is not None and len(self.kinds) != width:
            raise ValueError("kinds debe tener la misma longitud que columns")
        if len(set(self.columns)) != width:
            raise ValueError("columns contiene nombres repetidos")
        if len(self.matrix) > MAX_ROWS:
            raise ValueError(f"máximo {MAX_ROWS} filas por solicitud")
        if len(self.matrix) * width > MAX_CELLS:
            raise ValueError(f"máximo {MAX_CELLS} celdas por solicitud")
        for i, row in enumerate(self.matrix):
            if len(row) != width:
                raise ValueError(f"la fila {i} tiene {len(row)} valores y se esperaban {width}")
        return self


class TrainRequest(_MatrixMixin):
    name: str | None = Field(default=None, max_length=120)
    schema_signature: str | None = Field(default=None, max_length=200)
    contamination: float = Field(default=0.01, gt=0.0, le=0.25)
    n_estimators: int = Field(default=200, ge=10, le=1000)
    seed: int = 42


class ScoreRequest(_MatrixMixin):
    explain: ExplainMode = "flagged"
    top_k: int = Field(default=3, ge=1, le=10)
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)


class DetectRequest(TrainRequest):
    explain: ExplainMode = "flagged"
    top_k: int = Field(default=3, ge=1, le=10)


class Contribution(BaseModel):
    column: str
    field: str
    kind: FeatureKind
    value: float | None
    median: float
    deviation: float = Field(description="Signed robust z-score; strength for missing/rarity")
    message: str = Field(description="Spanish explanation")


class ModelInfo(BaseModel):
    model_id: str
    name: str | None
    columns: list[str]
    kinds: list[FeatureKind]
    rows: int
    threshold: float
    contamination: float
    seed: int
    trained_at: str
    schema_signature: str | None = None
    engine: str


class ScoreResponse(BaseModel):
    model_id: str | None
    threshold: float
    scores: list[float]
    flags: list[bool]
    contributions: dict[str, list[Contribution]] = Field(
        default_factory=dict,
        description="Keyed by row index (as string); only rows selected by `explain`",
    )


class DetectResponse(ScoreResponse):
    rows: int
    engine: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    engine: str
    models: int


class ErrorResponse(BaseModel):
    code: str
    message: str
