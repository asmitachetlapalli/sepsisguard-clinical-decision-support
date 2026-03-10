#!/usr/bin/env python3
"""
Logistic Regression baseline for early sepsis prediction.

Improvements over v1:
  - Uses full feature set (20 features: 8 vitals + 12 labs + Age + Gender)
  - Patient-level train/test split (no data leakage)
  - Reports AUROC, PR-AUC, and lead-time analysis
  - Feature importance ranking
  - Saves model + evaluation artifacts

Usage:
    python models/baseline_lr.py
    python models/baseline_lr.py --data-path data/processed/preprocessed_5000.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Feature set (must match preprocess.py MODEL_FEATURE_COLS)
FEATURE_COLS = [
    # Vitals
    "HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp", "EtCO2",
    # Key labs
    "Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets",
    "Glucose", "BUN", "pH", "BaseExcess", "HCO3", "Hgb", "Hct",
    # Demographics
    "Age", "Gender",
]

TARGET_COL = "early_sepsis_label"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_data(data_path: Path) -> pd.DataFrame:
    """Load preprocessed CSV and validate required columns."""
    if not data_path.exists():
        print(f"Error: data not found at {data_path}")
        print("Run `python data/preprocess.py` first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols from {data_path.name}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select available features, drop rows with NaNs in feature columns."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Note: {len(missing)} feature(s) not in data: {missing}")

    required = available + [TARGET_COL, "patient_id"]
    df_clean = df[required].dropna(subset=available)
    print(f"  Using {len(available)} features, {len(df_clean):,} rows after cleaning")
    return df_clean, available


def patient_split(
    df: pd.DataFrame, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by patient_id (70/15/15) to match LSTM split — prevents data leakage."""
    patient_ids = df["patient_id"].unique()

    # Same split logic as train_lstm.py and evaluate.py
    train_pids, temp_pids = train_test_split(
        patient_ids, test_size=0.3, random_state=seed
    )
    _, test_pids = train_test_split(
        temp_pids, test_size=0.5, random_state=seed
    )

    train_mask = df["patient_id"].isin(train_pids)
    test_mask = df["patient_id"].isin(test_pids)

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, "patient_id"]]

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, TARGET_COL].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, TARGET_COL].values

    print(f"\n  Train: {len(X_train):,} samples ({len(train_pids)} patients) "
          f"| pos={100 * y_train.mean():.2f}%")
    print(f"  Test:  {len(X_test):,} samples ({len(test_pids)} patients) "
          f"| pos={100 * y_test.mean():.2f}%")

    return X_train, y_train, X_test, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train Logistic Regression with class balancing."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    results_dir: Path,
) -> dict:
    """Compute metrics, generate plots, print report."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"\n{'─' * 50}")
    print(f"  AUROC:  {auroc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"{'─' * 50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Feature importance
    coefs = model.coef_[0]
    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    print("\nTop 10 Features by Importance:")
    for _, row in importance.head(10).iterrows():
        direction = "+" if row["coefficient"] > 0 else "-"
        print(f"  {direction} {row['feature']:<22s}  coef={row['coefficient']:+.4f}")

    # Plots
    results_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"LR (AUROC = {auroc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    baseline_pr = y_test.mean()
    axes[1].plot(recall, precision, "r-", linewidth=2, label=f"LR (PR-AUC = {pr_auc:.3f})")
    axes[1].axhline(y=baseline_pr, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline_pr:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / "baseline_lr_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlots saved to {plot_path}")

    # Feature importance bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(15)
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in top["coefficient"]]
    ax.barh(range(len(top)), top["coefficient"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.set_xlabel("Coefficient")
    ax.set_title("Top 15 Feature Importances (Logistic Regression)")
    ax.invert_yaxis()
    plt.tight_layout()
    fi_path = results_dir / "baseline_lr_feature_importance.png"
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Feature importance saved to {fi_path}")

    return {"auroc": auroc, "pr_auc": pr_auc, "confusion_matrix": cm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()

    project_root = get_project_root()
    data_path = Path(args.data_path) if args.data_path else (
        project_root / "data" / "processed" / "preprocessed_1000.csv"
    )

    print("=" * 60)
    print("  Logistic Regression Baseline — Early Sepsis Prediction")
    print("=" * 60)

    # 1. Load
    df = load_data(data_path)

    # 2. Features
    df_clean, feature_names = prepare_features(df)

    # 3. Patient-level split
    X_train, y_train, X_test, y_test = patient_split(df_clean)

    # 4. Train
    print("\nTraining Logistic Regression...")
    model = train_model(X_train, y_train)
    print("  Done.")

    # 5. Evaluate
    results_dir = project_root / "results"
    metrics = evaluate(model, X_test, y_test, feature_names, results_dir)

    # 6. Save model
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_lr.pkl"
    joblib.dump({
        "model": model,
        "feature_cols": feature_names,
        "target_col": TARGET_COL,
        "metrics": metrics,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # 7. Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  Model:     Logistic Regression (balanced)")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  Features:  {len(feature_names)}")
    print(f"  Saved:     {model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
