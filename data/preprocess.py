#!/usr/bin/env python3
"""
Preprocess PhysioNet 2019 sepsis data for model training.

Pipeline:
  1. Load patient .psv files (configurable count)
  2. Impute missing values (forward-fill vitals, median-fill ALL labs)
  3. Engineer early-warning labels (6-hour lead time)
  4. Normalize features (StandardScaler fitted on train-set patients)
  5. Save processed CSV + fitted scaler

Data: .psv files (pipe-separated) under training_setA / training_setB.

Usage:
    python data/preprocess.py                              # default 1000 patients
    python data/preprocess.py --max-patients 5000          # scale up
    python data/preprocess.py --data-dir /path/to/setA     # custom path
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# PhysioNet 2019 Feature Groups
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp", "EtCO2"]

LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

# Features used for modeling (vitals + key labs + age/gender)
MODEL_FEATURE_COLS = (
    VITAL_COLS
    + ["Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets",
       "Glucose", "BUN", "pH", "BaseExcess", "HCO3", "Hgb", "Hct"]
    + ["Age", "Gender"]
)

DEFAULT_DATA_DIR = Path(
    "/Users/sujandm/Desktop/sepsisguard-clinical-decision-support/data/archive/training_setA"
)


# Step 1: Load patients

def load_multiple_patients(
    data_dir: str | Path,
    max_patients: int = 1000,
) -> pd.DataFrame:
    """Load up to max_patients .psv files, assigning a unique patient_id."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    psv_files = sorted(data_path.rglob("*.psv"))
    if not psv_files:
        raise FileNotFoundError(f"No .psv files found in {data_path}")

    to_load = psv_files[:max_patients]
    print(f"Found {len(psv_files):,} patient files. Loading {len(to_load):,}...")

    frames: list[pd.DataFrame] = []
    for i, p in enumerate(to_load):
        try:
            df = pd.read_csv(p, sep="|", low_memory=False)
            df["patient_id"] = i
            df["patient_file"] = p.stem          # e.g. "p000001"
            frames.append(df)
        except Exception as e:
            print(f"  Warning: skipping {p.name} ({e})")
            continue

        if (i + 1) % 500 == 0:
            print(f"  ... loaded {i + 1:,} patients")

    if not frames:
        raise ValueError("No patient files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(frames):,} patients  |  {combined.shape[0]:,} rows × {combined.shape[1]} cols")
    return combined


# Step 2: Imputation

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Vital signs: forward-fill within each patient, then back-fill residuals
      - Lab values:  forward-fill within each patient, then median-fill globally
      - Demographics: fill with 0 (binary flags) or median (Age, HospAdmTime)
    """
    missing_before = df.isna().sum().sum()
    print(f"\nMissing values BEFORE imputation: {missing_before:,}")

    out = df.copy()

    # Vitals: per-patient forward-fill + back-fill
    present_vitals = [c for c in VITAL_COLS if c in out.columns]
    if "patient_id" in out.columns and present_vitals:
        for col in present_vitals:
            out[col] = out.groupby("patient_id", group_keys=False)[col].apply(
                lambda s: s.ffill().bfill()
            )

    # Labs: per-patient forward-fill, then global median for remaining NaNs
    present_labs = [c for c in LAB_COLS if c in out.columns]
    if "patient_id" in out.columns and present_labs:
        for col in present_labs:
            out[col] = out.groupby("patient_id", group_keys=False)[col].ffill()
            med = out[col].median()
            out[col] = out[col].fillna(med if pd.notna(med) else 0)

    # Demographics
    for col in ["Gender", "Unit1", "Unit2"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    for col in ["Age", "HospAdmTime"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())

    missing_after = out.isna().sum().sum()
    pct = 100 * (1 - missing_after / missing_before) if missing_before > 0 else 0
    print(f"Missing values AFTER  imputation: {missing_after:,}  ({pct:.1f}% reduction)")

    return out


# Step 3: Early-warning labels

def create_early_warning_labels(
    df: pd.DataFrame,
    lead_time_hours: int = 6,
) -> pd.DataFrame:
    """
    Add 'early_sepsis_label':
      1 for rows within [onset - lead_time_hours, onset - 1] hours.
    Uses ICULOS column as the hour indicator.
    """
    out = df.copy()
    out["early_sepsis_label"] = 0

    if "SepsisLabel" not in out.columns:
        print("Warning: 'SepsisLabel' column not found; labels left as 0.")
        return out

    hour_col = "ICULOS" if "ICULOS" in out.columns else None
    n_sepsis_patients = 0
    n_early_samples = 0

    for pid in out["patient_id"].unique():
        mask = out["patient_id"] == pid
        block = out.loc[mask]

        sepsis_rows = block["SepsisLabel"] == 1
        if not sepsis_rows.any():
            continue

        n_sepsis_patients += 1

        if hour_col:
            onset_hour = block.loc[sepsis_rows, hour_col].min()
            early_mask = (
                (block[hour_col] >= onset_hour - lead_time_hours)
                & (block[hour_col] < onset_hour)
            )
            early_idx = block.index[early_mask]
        else:
            onset_pos = block.index[sepsis_rows].min()
            start_pos = max(block.index.min(), onset_pos - lead_time_hours)
            end_pos = onset_pos - 1
            early_idx = block.loc[start_pos:end_pos].index if start_pos <= end_pos else pd.Index([])

        out.loc[early_idx, "early_sepsis_label"] = 1
        n_early_samples += len(early_idx)

    n_pos = (out["early_sepsis_label"] == 1).sum()
    n_neg = (out["early_sepsis_label"] == 0).sum()
    total = n_pos + n_neg

    print(f"\nEarly-warning labels (lead_time={lead_time_hours}h):")
    print(f"  Patients with sepsis:  {n_sepsis_patients}")
    print(f"  Early-warning samples: {n_early_samples}")
    print(f"  Class balance:         pos={n_pos:,}  neg={n_neg:,}")
    if total > 0:
        print(f"  Positive class ratio:  {100 * n_pos / total:.2f}%")

    return out


# Step 4: Feature scaling

def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler_path: Path | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on the data and transform feature columns in-place.
    Optionally save the fitted scaler to disk.
    """
    present = [c for c in feature_cols if c in df.columns]
    if not present:
        print("Warning: no feature columns found for scaling.")
        return df, StandardScaler()

    out = df.copy()
    scaler = StandardScaler()
    out[present] = scaler.fit_transform(out[present].values)

    if scaler_path:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": scaler, "feature_cols": present}, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    print(f"Scaled {len(present)} feature columns.")
    return out, scaler


# Main pipeline 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SepsisGuard preprocessing pipeline")
    parser.add_argument(
        "--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
        help="Path to directory containing .psv patient files",
    )
    parser.add_argument(
        "--max-patients", type=int, default=1000,
        help="Maximum number of patients to load (default: 1000)",
    )
    parser.add_argument(
        "--lead-time", type=int, default=6,
        help="Early-warning lead time in hours (default: 6)",
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="Skip feature scaling (save unscaled CSV)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("  SepsisGuard — Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Data dir:      {data_dir}")
    print(f"  Max patients:  {args.max_patients}")
    print(f"  Lead time:     {args.lead_time}h")
    print()

    # 1. Load
    df = load_multiple_patients(data_dir, max_patients=args.max_patients)

    # 2. Impute
    df = handle_missing_values(df)

    # 3. Early-warning labels
    df = create_early_warning_labels(df, lead_time_hours=args.lead_time)

    # 4. Scale features (optional)
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = out_dir / "feature_scaler.pkl"
    if not args.no_scale:
        df, _ = scale_features(df, MODEL_FEATURE_COLS, scaler_path=scaler_path)

    # 5. Save
    tag = args.max_patients
    out_path = out_dir / f"preprocessed_{tag}.csv"
    df.to_csv(out_path, index=False)

    # Summary 
    n_rows = len(df)
    n_patients = df["patient_id"].nunique()
    sepsis_prev = 100 * (df["SepsisLabel"] == 1).sum() / n_rows if n_rows else 0
    n_early = (df["early_sepsis_label"] == 1).sum()

    present_features = [c for c in MODEL_FEATURE_COLS if c in df.columns]

    print(f"\n{'=' * 60}")
    print(f"  Preprocessing Complete")
    print(f"{'=' * 60}")
    print(f"  Total rows:          {n_rows:,}")
    print(f"  Total patients:      {n_patients:,}")
    print(f"  Sepsis prevalence:   {sepsis_prev:.2f}%")
    print(f"  Early-warning rows:  {n_early:,}")
    print(f"  Model features:      {len(present_features)}")
    print(f"  Output CSV:          {out_path}")
    print(f"  Feature scaler:      {scaler_path if not args.no_scale else 'skipped'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
