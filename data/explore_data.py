#!/usr/bin/env python3
"""
Explore patient data from .psv (pipe-separated) files.
Searches data/ and data/archive/ (e.g. training_setA, training_setB).

Usage:
    python data/explore_data.py                        
    python data/explore_data.py /path/to/data          
    DATA_ROOT=/path/to/data python data/explore_data.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas and numpy are required.")
    print("Install with: pip install pandas numpy")
    sys.exit(1)

# Override via CLI argument or DATA_ROOT env var.
DATA_ROOT_FALLBACK = Path(
    "/Users/sujandm/Desktop/sepsisguard-clinical-decision-support/data"
)

# PhysioNet 2019 feature groups
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp", "EtCO2"]
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]


def get_data_dir() -> Path:
    """Resolve data directory from CLI arg > env var > script location > fallback."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    if os.environ.get("DATA_ROOT"):
        return Path(os.environ["DATA_ROOT"]).resolve()
    script_dir = Path(__file__).resolve().parent
    if list(script_dir.rglob("*.psv")):
        return script_dir
    return DATA_ROOT_FALLBACK


def find_all_psv(data_dir: Path) -> list[Path]:
    """Recursively find all .psv files under data_dir and data_dir/archive."""
    out: list[Path] = []
    for root in (data_dir, data_dir / "archive"):
        if root.exists():
            out.extend(sorted(root.rglob("*.psv")))
    return out


def print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def explore_single_patient(path: Path, data_dir: Path) -> pd.DataFrame | None:
    """Load and display summary for a single patient file."""
    try:
        df = pd.read_csv(path, sep="|", low_memory=False)
    except Exception as e:
        print(f"  Error loading {path.name}: {e}")
        return None

    try:
        rel = path.relative_to(data_dir)
    except ValueError:
        rel = path

    print(f"  File: {rel}")
    print(f"  Rows (hours): {len(df)},  Columns: {len(df.columns)}")

    if "SepsisLabel" in df.columns:
        has_sepsis = (df["SepsisLabel"] == 1).any()
        onset_hour = df.loc[df["SepsisLabel"] == 1, "ICULOS"].min() if has_sepsis else None
        status = f"YES (onset at hour {onset_hour})" if has_sepsis else "NO"
        print(f"  Sepsis: {status}")

    return df


def dataset_level_summary(psv_files: list[Path], max_sample: int = 500) -> None:
    """Compute dataset-level statistics from a sample of patient files."""
    print_section("Dataset-Level Summary")

    sample_files = psv_files[:max_sample]
    print(f"  Sampling {len(sample_files)} of {len(psv_files)} patients...\n")

    total_hours = 0
    sepsis_patients = 0
    stay_lengths: list[int] = []
    sepsis_hours = 0

    for p in sample_files:
        try:
            df = pd.read_csv(p, sep="|", low_memory=False)
        except Exception:
            continue

        n = len(df)
        total_hours += n
        stay_lengths.append(n)

        if "SepsisLabel" in df.columns:
            n_pos = (df["SepsisLabel"] == 1).sum()
            sepsis_hours += n_pos
            if n_pos > 0:
                sepsis_patients += 1

    n_patients = len(stay_lengths)
    if n_patients == 0:
        print("  No patients loaded.")
        return

    arr = np.array(stay_lengths)
    print(f"  Patients loaded:     {n_patients}")
    print(f"  Total hours:         {total_hours:,}")
    print(f"  Sepsis patients:     {sepsis_patients} ({100 * sepsis_patients / n_patients:.1f}%)")
    print(f"  Sepsis hours:        {sepsis_hours:,} ({100 * sepsis_hours / total_hours:.2f}%)")
    print(f"  Stay length (hours): mean={arr.mean():.1f}, "
          f"median={np.median(arr):.0f}, min={arr.min()}, max={arr.max()}")


def missingness_report(path: Path) -> None:
    """Show per-column missing value percentages for one patient file."""
    print_section("Missing-Value Profile (sample patient)")
    try:
        df = pd.read_csv(path, sep="|", low_memory=False)
    except Exception as e:
        print(f"  Error: {e}")
        return

    total = len(df)
    if total == 0:
        print("  Empty file.")
        return

    groups = [("Vitals", VITAL_COLS), ("Labs", LAB_COLS), ("Demographics", DEMO_COLS)]
    for group_name, cols in groups:
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        print(f"\n  {group_name}:")
        for c in present:
            pct = 100 * df[c].isna().sum() / total
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {c:<20s} {bar} {pct:5.1f}% missing")


def main() -> None:
    data_dir = get_data_dir()

    print("=" * 60)
    print("  SepsisGuard — Patient Data Exploration")
    print("=" * 60)
    print(f"  Data directory: {data_dir}\n")

    psv_files = find_all_psv(data_dir)
    if not psv_files and data_dir == Path(__file__).resolve().parent:
        data_dir = DATA_ROOT_FALLBACK
        if data_dir.exists():
            psv_files = find_all_psv(data_dir)

    if not psv_files:
        print("  No .psv files found under data/ or data/archive/.")
        print("  Tip: pass the data root as first argument or set DATA_ROOT env var.")
        return

    # File counts 
    n_set_a = sum(1 for p in psv_files if "training_setA" in p.parts)
    n_set_b = sum(1 for p in psv_files if "training_setB" in p.parts)
    print(f"  Found {len(psv_files):,} patient files total")
    print(f"    training_setA: {n_set_a:,}")
    print(f"    training_setB: {n_set_b:,}")

    # Sample patient
    print_section("Sample Patient")
    sample_df = explore_single_patient(psv_files[0], data_dir)

    if sample_df is not None:
        print(f"\n  Columns ({len(sample_df.columns)}):")
        print(f"    {list(sample_df.columns)}")
        print(f"\n  First 5 rows:")
        print(sample_df.head().to_string(index=False))

    # Missingness
    missingness_report(psv_files[0])

    # Dataset-level stats
    dataset_level_summary(psv_files, max_sample=500)

    print("\n" + "=" * 60)
    print("  Exploration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
