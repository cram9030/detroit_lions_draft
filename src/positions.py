"""Position normalization constants and helpers for NFL draft analysis."""

from __future__ import annotations

import polars as pl


_POSITION_GROUPS: dict[str, str] = {
    "FL":"WR",
    "FB": "RB", "RH":"RB", "LH":"RB",
    "LDE": "DE", "RDE": "DE",
    "NT": "DT", "LDT":"DT", "RDT":"DT",
    "LG": "OG", "RG": "OG", "LT": "OT", "RT": "OT", "T":"OT", "G":"OG", "C":"OC",
    "LCB": "CB", "RCB": "CB",
    "LILB": "LB", "RILB": "LB", "LOLB": "LB", "ROLB": "LB", "LLB": "LB", "ILB":"LB", "OLB":"LB", "RLB":"LB","MLB":"LB",
    "FS": "S", "SS": "S", "DB": "S",
}

_SPECALIST: list[str] = ['K', 'KR', 'P', 'PR', 'LS']

_GENERALIST: list[str] = ['DL', 'OL']

_POSITION_ORDER: list[str] = ["QB", "WR", "TE", "RB", "OC", "OG", "OT", "DE", "DT", "LB", "CB", "S"]


def normalize_pos(
    player: str,
    pos: str | None,
    overrides: dict[str, str] | None = None,
) -> str:
    """Return the normalized position group, applying per-player overrides first."""
    if overrides and player in overrides:
        return overrides[player]
    if not pos:
        return "UNK"
    first = pos.replace("-", "/").split("/")[0].strip()
    return _POSITION_GROUPS.get(first, first)


def canonicalize_positions(df: pl.DataFrame) -> pl.DataFrame:
    """Resolve each player to a single canonical position across all seasons."""
    _generalist_set: frozenset[str] = frozenset(_GENERALIST)

    yr0_pos = (
        df.filter(pl.col("years_from_draft") == 0)
        .select(["Player", "Pick", "Draft Year", "Pos"])
        .rename({"Pos": "yr0_pos"})
    )

    pos_counts = (
        df.filter(~pl.col("Pos").is_in(list(_generalist_set)))
        .group_by(["Player", "Pick", "Draft Year", "Pos"])
        .agg(pl.len().alias("count"))
    )

    if pos_counts.is_empty():
        canonical = yr0_pos.rename({"yr0_pos": "canonical_pos"})
    else:
        best = (
            pos_counts
            .join(yr0_pos, on=["Player", "Pick", "Draft Year"], how="left")
            .with_columns(
                (pl.col("Pos") == pl.col("yr0_pos")).cast(pl.Int8).alias("is_yr0_int")
            )
            .sort(
                ["Player", "Pick", "Draft Year", "count", "is_yr0_int", "Pos"],
                descending=[False, False, False, True, True, False],
            )
            .unique(
                subset=["Player", "Pick", "Draft Year"],
                keep="first",
                maintain_order=False,
            )
            .select(["Player", "Pick", "Draft Year", pl.col("Pos").alias("canonical_pos")])
        )
        canonical = (
            yr0_pos
            .join(best, on=["Player", "Pick", "Draft Year"], how="left")
            .with_columns(
                pl.coalesce(["canonical_pos", "yr0_pos"]).alias("canonical_pos")
            )
            .select(["Player", "Pick", "Draft Year", "canonical_pos"])
        )

    return (
        df
        .join(canonical, on=["Player", "Pick", "Draft Year"])
        .with_columns(pl.col("canonical_pos").alias("Pos"))
        .drop("canonical_pos")
    )
