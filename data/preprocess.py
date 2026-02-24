#!/usr/bin/env python3
"""
Preprocess PhysioNet sepsis data: load patients, handle missing values,
and create early warning labels.
Data: .psv files (pipe-separated) under training_setA (or given directory).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_multiple_patients(data_dir: str | Path, max_patients: int = 1000) -> pd.DataFrame:
    """
    Load up to max_patients .psv files from data_dir, add patient_id and
    patient_file, combine into one dataframe.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    psv_files = sorted(data_path.rglob("*.psv"))
    if not psv_files:
        raise FileNotFoundError(f"No .psv files found in {data_path}")

    to_load = psv_files[:max_patients]
    print(f"Found {len(psv_files)} patient file(s). Loading up to {max_patients}...")

    frames = []
    for i, p in enumerate(to_load):
        try:
            df = pd.read_csv(p, sep="|", low_memory=False)
            df["patient_id"] = i
            df["patient_file"] = p.name
            frames.append(df)
        except Exception as e:
            print(f"Warning: failed to load {p.name}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1} patients...")

    if not frames:
        raise ValueError("No patient files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(frames)} patients. Combined shape: {combined.shape}")
    return combined


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill vital signs per patient; fill lab columns with column median.
    Print missing-value stats before and after.
    """
    vital_cols = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp"]
    lab_cols = ["Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets"]

    missing_before = df.isna().sum().sum()
    print("\n--- Missing values (before) ---")
    print(f"Total missing values: {missing_before}")

    out = df.copy()

    if "patient_id" not in out.columns:
        print("Warning: 'patient_id' not found; skipping per-patient forward fill.")
    else:
        for col in vital_cols:
            if col in out.columns:
                out[col] = out.groupby("patient_id", group_keys=False)[col].ffill()

    for col in lab_cols:
        if col in out.columns:
            med = out[col].median()
            out[col] = out[col].fillna(med)

    missing_after = out.isna().sum().sum()
    print("\n--- Missing values (after) ---")
    print(f"Total missing values: {missing_after}")
    if missing_before > 0:
        pct = 100 * (1 - missing_after / missing_before)
        print(f"Percentage reduction: {pct:.1f}%")
    else:
        print("Percentage reduction: N/A (no missing before)")

    return out


def create_early_warning_labels(df: pd.DataFrame, lead_time_hours: int = 6) -> pd.DataFrame:
    """
    Add 'early_sepsis_label': 1 for rows 1 to lead_time_hours before sepsis onset.
    Uses ICULOS as hour; if missing, uses row order within each patient.
    """
    out = df.copy()
    out["early_sepsis_label"] = 0

    if "SepsisLabel" not in out.columns:
        print("Warning: 'SepsisLabel' not found; early_sepsis_label left as 0.")
        return out

    patient_ids = out["patient_id"].unique()
    n_patients = len(patient_ids)
    n_sepsis_patients = 0
    n_early_samples = 0

    hour_col = "ICULOS" if "ICULOS" in out.columns else None

    for pid in patient_ids:
        mask = out["patient_id"] == pid
        block = out.loc[mask]

        sepsis_rows = block["SepsisLabel"] == 1
        if not sepsis_rows.any():
            continue

        n_sepsis_patients += 1

        if hour_col and hour_col in block.columns:
            onset_hour = block.loc[sepsis_rows, hour_col].min()
            # 1 to lead_time_hours hours before onset
            early_hours = (block[hour_col] >= onset_hour - lead_time_hours) & (
                block[hour_col] <= onset_hour - 1
            )
            early_idx = block.index[early_hours]
        else:
            onset_idx = block.index[sepsis_rows].min()
            start_idx = max(block.index.min(), onset_idx - lead_time_hours)
            end_idx = onset_idx - 1
            if start_idx <= end_idx:
                early_idx = block.loc[start_idx : end_idx].index
            else:
                early_idx = pd.Index([])

        out.loc[out.index.isin(early_idx), "early_sepsis_label"] = 1
        n_early_samples += len(early_idx)

    n_positive = (out["early_sepsis_label"] == 1).sum()
    n_negative = (out["early_sepsis_label"] == 0).sum()
    print("\n--- Early warning labels ---")
    print(f"Total patients processed: {n_patients}")
    print(f"Patients with sepsis: {n_sepsis_patients}")
    print(f"Total early warning samples (early_sepsis_label==1): {n_early_samples}")
    print(f"Class balance: positive={n_positive}, negative={n_negative}")
    if n_positive + n_negative > 0:
        pct_pos = 100 * n_positive / (n_positive + n_negative)
        print(f"Positive class: {pct_pos:.2f}%")

    return out


def main() -> None:
    data_dir = Path(
        "/Users/sujandm/Desktop/sepsisguard-clinical-decision-support/data/archive/training_setA"
    )

    print("Starting preprocessing pipeline...")
    print(f"Data directory: {data_dir}\n")

    try:
        df = load_multiple_patients(data_dir, max_patients=1000)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"Combined data shape: {df.shape}\n")

    df = handle_missing_values(df)
    df = create_early_warning_labels(df, lead_time_hours=6)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessed_1000.csv"

    try:
        df.to_csv(out_path, index=False)
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

    n_rows = len(df)
    n_patients = df["patient_id"].nunique()
    sepsis_prevalence = (df["SepsisLabel"] == 1).sum() / n_rows * 100 if n_rows else 0
    n_early = (df["early_sepsis_label"] == 1).sum()

    print("\n--- Final summary ---")
    print(f"Total rows in processed data: {n_rows}")
    print(f"Total patients: {n_patients}")
    print(f"Sepsis prevalence (rows with SepsisLabel=1): {sepsis_prevalence:.2f}%")
    print(f"Early warning samples: {n_early}")
    print("\nPreprocessing complete! Saved to data/processed/preprocessed_1000.csv")


if __name__ == "__main__":
    main()
