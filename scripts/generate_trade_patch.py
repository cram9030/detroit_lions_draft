"""CLI script: generate an extended trade patch JSON from a draft order JSON.

Usage:
    python scripts/generate_trade_patch.py data/processed/nfl_draft_2026.json 2026
    python scripts/generate_trade_patch.py data/processed/nfl_draft_2026.json 2026 \
        --output data/processed/trade_patch_2026_v2.json

Writes the patch to data/processed/trade_patch_{year}.json by default.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.draft_integration import generate_trade_patch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trade patch JSON from draft order JSON.")
    parser.add_argument("draft_json", type=Path, help="Path to draft order JSON (e.g. nfl_draft_2026.json)")
    parser.add_argument("year", type=int, help="Draft year (e.g. 2026)")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path (default: data/processed/trade_patch_{year}.json)",
    )
    args = parser.parse_args()

    output_path: Path = args.output or (
        Path(__file__).resolve().parents[1] / "data" / "processed" / f"trade_patch_{args.year}.json"
    )

    print(f"Loading draft JSON from {args.draft_json} ...")
    patch = generate_trade_patch(args.draft_json, args.year)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(patch, f, indent=2)

    print(
        f"Wrote {len(patch['patches'])} patches, "
        f"{len(patch['new_trades'])} new trades, "
        f"{len(patch['warnings'])} warnings → {output_path}"
    )


if __name__ == "__main__":
    main()
