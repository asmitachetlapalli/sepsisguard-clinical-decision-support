#!/usr/bin/env python3
"""
Train a logistic regression baseline for early sepsis prediction.
Uses patient-level train/test split to prevent data leakage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    project_root = get_project_root()
    data_path = project_root / "data" / "processed" / "preprocessed_1000.csv"

    print("=" * 60)
    print("Logistic Regression Baseline - Early Sepsis Prediction")
    print("=" * 60)

    # 1. Load preprocessed data
    if not data_path.exists():
        print(f"Error: Preprocessed data not found at {data_path}")
        print("Run data/preprocess.py first to generate preprocessed_1000.csv")
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"\nLoaded data shape: {df.shape}")
    print("First few rows:")
    print(df.head())
    print()

    # 2. Prepare features
    feature_cols = ["HR", "O2Sat", "Temp", "MAP", "Resp", "Age"]
    target_col = "early_sepsis_label"

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing feature columns: {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        print("Error: No feature columns found in data.")
        sys.exit(1)

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        sys.exit(1)

    df_clean = df[feature_cols + [target_col, "patient_id"]].dropna(
        subset=feature_cols
    )
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print(f"Features shape after cleaning: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Feature columns: {feature_cols}")
    print()

    # 3. Split by patient (no leakage)
    patient_ids = df_clean["patient_id"].unique()
    train_patients, test_patients = train_test_split(
        patient_ids, test_size=0.2, random_state=42
    )

    train_mask = df_clean["patient_id"].isin(train_patients)
    test_mask = df_clean["patient_id"].isin(test_patients)

    X_train = X.loc[train_mask].values
    y_train = y.loc[train_mask].values
    X_test = X.loc[test_mask].values
    y_test = y.loc[test_mask].values

    n_train, n_test = len(X_train), len(X_test)
    pct_train = 100 * (y_train == 1).sum() / n_train if n_train else 0
    pct_test = 100 * (y_test == 1).sum() / n_test if n_test else 0

    print("--- Train/Test split (by patient) ---")
    print(f"Train: {n_train} samples ({len(train_patients)} patients)")
    print(f"Test:  {n_test} samples ({len(test_patients)} patients)")
    print(f"Train positive class: {pct_train:.2f}%")
    print(f"Test positive class:  {pct_test:.2f}%")
    print()

    # 4. Train model
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Model training complete.")
    print()

    # 5. Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc = roc_auc_score(y_test, y_pred_proba)
    print("--- Evaluation (test set) ---")
    print(f"AUROC: {auroc:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUROC = {auroc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression Baseline")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    roc_path = results_dir / "baseline_lr_roc.png"
    try:
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"ROC curve saved to {roc_path}")
    except Exception as e:
        print(f"Warning: Could not save ROC plot: {e}")
        plt.close()
    print()

    # 6. Save model
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_lr.pkl"
    try:
        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target_col": target_col,
            },
            model_path,
        )
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)
    print()

    # 7. Final summary
    print("=" * 60)
    print("Final summary")
    print("=" * 60)
    print(f"Model type: Logistic Regression (sklearn)")
    print(f"AUROC score: {auroc:.4f}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    print(f"Train size: {n_train}, Test size: {n_test}")
    print(f"Model saved to: {model_path}")
    print(f"ROC plot saved to: {roc_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
