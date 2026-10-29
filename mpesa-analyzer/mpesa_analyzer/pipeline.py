"""High-level entry points: raw input -> parsed -> categorized -> enriched DataFrame."""
from __future__ import annotations

from datetime import date

import pandas as pd

from . import analytics, generator, parser
from .categorizer import Categorizer


def _finish(df: pd.DataFrame, categorizer: Categorizer | None) -> pd.DataFrame:
    if df.empty:
        return df
    categorizer = categorizer or Categorizer()
    return analytics.enrich(categorizer.categorize(df))


def from_text(text: str, categorizer: Categorizer | None = None) -> tuple[pd.DataFrame, list[str]]:
    df, unparsed = parser.parse_text(text)
    return _finish(df, categorizer), unparsed


def from_statement(source, categorizer: Categorizer | None = None) -> tuple[pd.DataFrame, list[str]]:
    return _finish(parser.parse_statement(source), categorizer), []


def from_synthetic(months: int = 3, seed: int = 42, categorizer: Categorizer | None = None,
                   start: date = date(2026, 1, 1)) -> tuple[pd.DataFrame, list[str]]:
    """Generate fake SMS and push them through the *real* parsing pipeline."""
    sms = generator.sms_text(generator.generate(months=months, seed=seed, start=start))
    return from_text(sms, categorizer)
