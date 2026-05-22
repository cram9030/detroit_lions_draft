from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

import polars as pl


class PredictionResult(TypedDict):
    position: str
    observed_years: list[int]
    observed_av: list[float]
    predicted_years: list[int]
    y_pred: list[float]
    y_upper: list[float]
    y_lower: list[float]


@runtime_checkable
class CareerAVModel(Protocol):
    """Strategy interface for career AV trajectory models.

    trajectory_df schema for fit(): output of _aggregate_career_av_by_position
    with columns [Player, Pos, Draft Year, years_from_draft, AV.1]. The Pick
    column is included when available and used by models that support it (KNN).

    predict() accepts an optional pick parameter. Models that don't use it
    accept and ignore it for protocol compliance.
    """

    def fit(self, trajectory_df: pl.DataFrame) -> None: ...

    def predict(
        self, position: str, observed_av: list[float], pick: int | None = None
    ) -> PredictionResult: ...

    def save(self, model_dir: str | Path) -> None: ...

    def load(self, model_dir: str | Path) -> None: ...
