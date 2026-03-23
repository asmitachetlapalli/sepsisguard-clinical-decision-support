#!/usr/bin/env python3
"""
Train XGBoost model for early sepsis prediction.
Uses the same patient-level split and features as the LR baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as PlattLR
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_FEATURES = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp"]
LAB_FEATURES = ["Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets",
                "BUN", "Glucose", "Hgb", "Hct", "pH"]
DEMO_FEATURES = ["Age", "Gender", "HospAdmTime", "ICULOS"]
TARGET_COL = "early_sepsis_label"


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "preprocessed_1000.csv"

    print("=" * 60)
    print("XGBoost Training — Early Sepsis Prediction")
    print("=" * 60)

    # ── 1. Load data ────────────────────────────────────────────────────
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run data/preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {df['patient_id'].nunique()} patients")

    # Collect all available features
    all_candidates = BASE_FEATURES + LAB_FEATURES + DEMO_FEATURES
    FEATURE_COLS = [c for c in all_candidates if c in df.columns]
    print(f"Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")

    # Fill remaining NaN (XGBoost can handle NaN, but fill for consistency)
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    # ── 2. Patient-level split (no data leakage) ───────────────────────
    patient_ids = df["patient_id"].unique()
    train_pids, test_pids = train_test_split(
        patient_ids, test_size=0.2, random_state=42
    )

    train_mask = df["patient_id"].isin(train_pids)
    test_mask = df["patient_id"].isin(test_pids)

    X_train = df.loc[train_mask, FEATURE_COLS].values
    y_train = df.loc[train_mask, TARGET_COL].values
    X_test = df.loc[test_mask, FEATURE_COLS].values
    y_test = df.loc[test_mask, TARGET_COL].values

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)  # same as sklearn's class_weight="balanced"

    print(f"Train: {len(X_train):,} samples ({n_pos} pos, {n_neg} neg)")
    print(f"Test:  {len(X_test):,} samples ({int(y_test.sum())} pos)")
    print(f"scale_pos_weight: {scale_pos:.1f}")

    # ── 3. Train XGBoost ───────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.01,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric="auc",
        random_state=42,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=25,
    )

    # ── 4. Evaluate ────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_proba)
    print(f"\nProbability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresh = float(thresholds[optimal_idx])
    y_pred = (y_proba >= optimal_thresh).astype(int)

    print(f"\n{'='*60}")
    print(f"RESULTS  (AUROC: {auroc:.4f})")
    print(f"{'='*60}")
    print(f"Optimal threshold: {optimal_thresh:.4f}\n")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Sepsis", "Sepsis"],
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── 5. Feature importance ──────────────────────────────────────────
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature Importance:")
    for i in sorted_idx:
        print(f"  {FEATURE_COLS[i]:10s}: {importances[i]:.4f}")

    # ── 6. Save model ─────────────────────────────────────────────────
    save_path = project_root / "models" / "xgboost_model.pkl"
    joblib.dump({
        "model": model,
        "feature_cols": list(FEATURE_COLS),
        "target_col": TARGET_COL,
        "auroc": auroc,
        "optimal_threshold": optimal_thresh,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # ── 7. Plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, "b-", label=f"XGBoost (AUROC={auroc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].scatter(fpr[optimal_idx], tpr[optimal_idx], c="red", s=80, zorder=5,
                    label=f"Threshold={optimal_thresh:.2f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — XGBoost")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Feature importance
    axes[1].barh(
        [FEATURE_COLS[i] for i in sorted_idx[::-1]],
        importances[sorted_idx[::-1]],
        color="steelblue",
    )
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Feature Importance")
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plot_path = project_root / "results" / "xgboost_results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plots saved to {plot_path}")
    print("Done!")


if __name__ == "__main__":
    main()
