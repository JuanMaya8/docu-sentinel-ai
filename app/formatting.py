"""Spanish (es-CO) number/date formatting used in explanations."""

from __future__ import annotations

import math
from datetime import date, timedelta

_EPOCH = date(1970, 1, 1)


def fmt_number(value: float, max_decimals: int = 2) -> str:
    """Format as es-CO: thousands '.', decimal ','. Example: 1234.5 -> '1.234,5'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    text = f"{value:,.{max_decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    # swap separators: ',' -> '.' and '.' -> ','
    return text.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")


def fmt_epoch_days(days: float) -> str:
    """Epoch days (since 1970-01-01) -> ISO date string."""
    try:
        return (_EPOCH + timedelta(days=int(round(days)))).isoformat()
    except (OverflowError, ValueError):
        return fmt_number(days)


def fmt_percent(fraction: float) -> str:
    pct = fraction * 100.0
    if pct >= 10:
        return f"{pct:.0f}%"
    if pct >= 1:
        return f"{pct:.1f}%".replace(".", ",")
    return f"{pct:.2f}%".replace(".", ",")
