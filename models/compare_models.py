#!/usr/bin/env python3
"""
Generate side-by-side ROC comparison of XGBoost vs Logistic Regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    csvs = sorted(processed_dir.glob("preprocessed_*.csv"), key=lambda p: int(p.stem.split("_")[1]), reverse=True)
    data_path = csvs[0] if csvs else processed_dir / "preprocessed_1000.csv"

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Load models
    xgb_data = joblib.load(project_root / "models" / "xgboost_model.pkl")
    lr_data = joblib.load(project_root / "models" / "baseline_lr.pkl")

    # Same patient-level split as training
    patient_ids = df["patient_id"].unique()
    _, test_pids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    df_test = df[df["patient_id"].isin(test_pids)]

    target = "early_sepsis_label"
    y_test = df_test[target].values

    # XGBoost predictions (fill NaN with median, same as training)
    xgb_features = xgb_data["feature_cols"]
    for col in xgb_features:
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(df[col].median())
    X_xgb = df_test[xgb_features].values
    xgb_proba = xgb_data["model"].predict_proba(X_xgb)[:, 1]
    xgb_auroc = roc_auc_score(y_test, xgb_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)

    # LR predictions (only on rows with no NaN in LR features)
    lr_features = lr_data["feature_cols"]
    df_lr = df_test.dropna(subset=lr_features)
    X_lr = df_lr[lr_features].values
    y_lr = df_lr[target].values
    lr_proba = lr_data["model"].predict_proba(X_lr)[:, 1]
    lr_auroc = roc_auc_score(y_lr, lr_proba)
    lr_fpr, lr_tpr, _ = roc_curve(y_lr, lr_proba)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xgb_fpr, xgb_tpr, "b-", linewidth=2, label=f"XGBoost (AUROC = {xgb_auroc:.3f})")
    ax.plot(lr_fpr, lr_tpr, "r-", linewidth=2, label=f"Logistic Regression (AUROC = {lr_auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Model Comparison — ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = project_root / "results" / "model_comparison_roc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"XGBoost AUROC: {xgb_auroc:.4f}")
    print(f"LR AUROC:      {lr_auroc:.4f}")
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
