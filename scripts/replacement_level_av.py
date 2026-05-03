"""Compute descriptive statistics for replacement-level players' season AV.

Replacement level is defined as a 1-year contract within 120% of the NFL
minimum salary for the player's experience tier.

Usage
-----
    python scripts/replacement_level_av.py
    python scripts/replacement_level_av.py --replacement-percent 1.10
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.annual_av_analysis import prepare_av_data
from src.contracts import get_replacement_level_av, get_replacement_level_contracts
from src.data_ingest import load_parquets_from_dir

RAW_DIR = PROJECT_ROOT / "data/raw/stathead/annual_av"


def main(replacement_percent: float) -> None:
    av_lf = prepare_av_data(load_parquets_from_dir(RAW_DIR))
    contracts_df = get_replacement_level_contracts(replacement_percent)
    av_df = get_replacement_level_av(av_lf, contracts_df)

    print(f"Replacement-level AV population ({replacement_percent:.0%} of NFL minimum salary)")
    print(f"  Player-seasons: {len(av_df)}")
    print(f"  Contracts matched: {len(contracts_df)}")
    print(f"  Year range: {av_df['year_signed'].min()} – {av_df['year_signed'].max()}")
    print()
    print(av_df.select("AV.1").describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replacement-level AV statistics")
    parser.add_argument(
        "--replacement-percent",
        type=float,
        default=1.20,
        help="Ceiling multiplier on the minimum salary (default: 1.20)",
    )
    args = parser.parse_args()
    main(args.replacement_percent)
