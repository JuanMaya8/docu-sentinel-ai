"""Spanish explanation templates for feature contributions."""

from __future__ import annotations

import math

from .formatting import fmt_epoch_days, fmt_number, fmt_percent


def describe_contribution(
    field: str,
    kind: str,
    value: float | None,
    median: float,
    deviation: float,
    indicator_rate: float = 0.0,
) -> str:
    if kind == "missing":
        return (
            f"El campo «{field}» está vacío y casi nunca falta en los datos de referencia "
            f"({fmt_percent(indicator_rate)} de los registros)."
        )
    if kind == "rarity":
        # rarity = -ln(freq) -> freq = exp(-rarity)
        freq = math.exp(-value) if value is not None else 0.0
        return (
            f"El valor en «{field}» es poco frecuente en los datos de referencia "
            f"(aparece en ~{fmt_percent(freq)} de los registros)."
        )
    shown = fmt_epoch_days(value) if kind == "date" and value is not None else fmt_number(value or 0.0)
    ref = fmt_epoch_days(median) if kind == "date" else fmt_number(median)
    direction = "por encima" if deviation > 0 else "por debajo"
    return (
        f"El valor {shown} en «{field}» está {fmt_number(abs(deviation), 1)} desviaciones "
        f"robustas {direction} de la mediana ({ref})."
    )


def describe_coherence(field: str, feature_kind: str, value: float, expected: float) -> str:
    """A field whose value does not fit what the other fields of the same record predict."""
    fmt = fmt_epoch_days if feature_kind == "date" else fmt_number
    return (
        f"El valor {fmt(value)} en «{field}» no es coherente con el resto del registro: "
        f"según los demás campos se esperaría alrededor de {fmt(expected)}."
    )
