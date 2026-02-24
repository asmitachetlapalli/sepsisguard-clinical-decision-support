#!/usr/bin/env python3
"""
Explore patient data from .psv (pipe-separated) files.
Searches data/ and data/archive/ (e.g. training_setA, training_setB).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)

# Optional: data root where .psv files live (archive/training_setA, training_setB).
# Override via first argument or env DATA_ROOT. Example:
#   python data/explore_data.py /Users/sujandm/Desktop/sepsisguard-clinical-decision-support/data
DATA_ROOT_FALLBACK = Path("/Users/sujandm/Desktop/sepsisguard-clinical-decision-support/data")


def get_data_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    if os.environ.get("DATA_ROOT"):
        return Path(os.environ["DATA_ROOT"]).resolve()
    return Path(__file__).resolve().parent


def find_all_psv(data_dir: Path) -> list[Path]:
    out = []
    for root in (data_dir, data_dir / "archive"):
        if root.exists():
            out.extend(sorted(root.rglob("*.psv")))
    return out


def main() -> None:
    data_dir = get_data_dir()
    print("=" * 60)
    print("Patient data exploration")
    print("=" * 60)
    print(f"Data directory: {data_dir}\n")

    psv_files = find_all_psv(data_dir)
    if not psv_files and data_dir == Path(__file__).resolve().parent:
        data_dir = DATA_ROOT_FALLBACK
        if data_dir.exists():
            psv_files = find_all_psv(data_dir)
    if not psv_files:
        print("No .psv files found under data/ or data/archive/.")
        print("Tip: pass the data root as first argument or set DATA_ROOT env var.")
        return

    print(f"Using data root: {data_dir}\n")
    n_set_a = sum(1 for p in psv_files if "training_setA" in p.parts)
    n_set_b = sum(1 for p in psv_files if "training_setB" in p.parts)
    print(f"Found {len(psv_files)} patient file(s) total:")
    print(f"  training_setA: {n_set_a}")
    print(f"  training_setB: {n_set_b}")
    print()
    for i, p in enumerate(psv_files[:20], 1):
        try:
            rel = p.relative_to(data_dir)
        except ValueError:
            rel = p
        print(f"  {i}. {rel}")
    if len(psv_files) > 20:
        print(f"  ... and {len(psv_files) - 20} more")
    print()

    first_file = psv_files[0]
    try:
        df = pd.read_csv(first_file, sep="|", low_memory=False)
    except Exception as e:
        print(f"Error loading {first_file}: {e}")
        return

    print(f"Sample file: {first_file.relative_to(data_dir)}")
    print()

    print("--- Shape ---")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}\n")

    print("--- Column names ---")
    print(list(df.columns))
    print()

    print("--- Data types ---")
    print(df.dtypes.to_string())
    print()

    print("--- First 10 rows ---")
    print(df.head(10).to_string())
    print()

    print("--- Basic statistics (numeric columns) ---")
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        print("No numeric columns.")
    else:
        print(numeric.describe().loc[["mean", "std", "min", "max"]].to_string())
    print()

    print("--- SepsisLabel ---")
    if "SepsisLabel" not in df.columns:
        print("Column 'SepsisLabel' not found.")
    else:
        n_sepsis = (df["SepsisLabel"] == 1).sum()
        total = len(df)
        print(f"Hours with SepsisLabel=1: {n_sepsis} (of {total} rows)")
        if total:
            print(f"Fraction: {100 * n_sepsis / total:.2f}%")


if __name__ == "__main__":
    main()
