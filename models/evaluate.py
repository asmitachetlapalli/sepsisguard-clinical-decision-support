#!/usr/bin/env python3
"""
Evaluate and compare all trained models (Logistic Regression vs LSTM).

Generates:
  - Side-by-side ROC and PR curves
  - Model comparison table
  - Per-threshold analysis (sensitivity at various specificity targets)

Usage:
    python models/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lstm_model import SepsisLSTM
from train_lstm import SepsisSequenceDataset, FEATURE_COLS, TARGET_COL


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_lr_predictions(
    df: pd.DataFrame,
    model_path: Path,
    test_pids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load LR model and generate predictions on test patients."""
    if not model_path.exists():
        print(f"  LR model not found at {model_path}")
        return None

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    available = [c for c in feature_cols if c in df.columns]
    test_df = df[df["patient_id"].isin(test_pids)].dropna(subset=available)

    if len(test_df) == 0:
        print("  No test data for LR model.")
        return None

    X_test = test_df[available].values
    y_test = test_df[TARGET_COL].values
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_test, y_prob


def load_lstm_predictions(
    df: pd.DataFrame,
    model_path: Path,
    test_pids: np.ndarray,
    seq_len: int = 24,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load LSTM model and generate predictions on test patients."""
    if not model_path.exists():
        print(f"  LSTM model not found at {model_path}")
        return None

    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    n_features = checkpoint.get("n_features", 22)
    feature_cols = checkpoint.get("feature_cols", FEATURE_COLS)
    model_args = checkpoint.get("args", {})

    model = SepsisLSTM(
        input_size=n_features,
        hidden_size=model_args.get("hidden_size", 64),
        num_layers=model_args.get("num_layers", 2),
        dropout=0.0,  # no dropout at inference
        use_attention=not model_args.get("no_attention", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    available = [c for c in feature_cols if c in df.columns]
    test_ds = SepsisSequenceDataset(df, test_pids, available, TARGET_COL, seq_len)

    if len(test_ds) == 0:
        print("  No test sequences for LSTM.")
        return None

    loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds = model(X_b)
            all_probs.extend(preds.squeeze(1).tolist())
            all_labels.extend(y_b.tolist())

    return np.array(all_labels), np.array(all_probs)


def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    """Print sensitivity at various specificity targets."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr

    print(f"\n  {model_name} — Threshold Analysis:")
    print(f"  {'Specificity':>12s} | {'Sensitivity':>12s} | {'Threshold':>10s}")
    print(f"  {'─' * 42}")

    for target_spec in [0.95, 0.90, 0.85, 0.80, 0.70]:
        idx = np.argmin(np.abs(specificity - target_spec))
        print(f"  {specificity[idx]:12.3f} | {tpr[idx]:12.3f} | {thresholds[idx]:10.4f}")


def main() -> None:
    project_root = get_project_root()
    data_path = project_root / "data" / "processed" / "preprocessed_1000.csv"
    lr_model_path = project_root / "models" / "baseline_lr.pkl"
    lstm_model_path = project_root / "models" / "lstm_best.pt"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SepsisGuard — Model Comparison")
    print("=" * 60)

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {df['patient_id'].nunique()} patients\n")

    # Use same split as training (seed=42, 70/15/15)
    all_pids = df["patient_id"].unique()
    train_pids, temp_pids = train_test_split(all_pids, test_size=0.3, random_state=42)
    _, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42)
    print(f"Test set: {len(test_pids)} patients\n")

    # Collect predictions
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print("Loading Logistic Regression...")
    lr_result = load_lr_predictions(df, lr_model_path, test_pids)
    if lr_result:
        results["Logistic Regression"] = lr_result

    print("Loading LSTM...")
    lstm_result = load_lstm_predictions(df, lstm_model_path, test_pids)
    if lstm_result:
        results["LSTM"] = lstm_result

    if not results:
        print("No models found to evaluate. Train models first.")
        sys.exit(1)

    # Metrics table
    print(f"\n{'─' * 60}")
    print(f"  {'Model':<25s} | {'AUROC':>8s} | {'PR-AUC':>8s} | {'Samples':>8s}")
    print(f"{'─' * 60}")

    for name, (y_true, y_prob) in results.items():
        auroc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        print(f"  {name:<25s} | {auroc:8.4f} | {pr_auc:8.4f} | {len(y_true):8,}")

    print(f"{'─' * 60}")

    # Threshold analysis
    for name, (y_true, y_prob) in results.items():
        threshold_analysis(y_true, y_prob, name)

    # Comparison plots
    colors = {"Logistic Regression": "#3498db", "LSTM": "#e74c3c"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ROC
    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, color=colors.get(name, "gray"), linewidth=2,
                     label=f"{name} (AUROC={auroc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title("ROC Curve Comparison", fontsize=14)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # PR
    for name, (y_true, y_prob) in results.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        axes[1].plot(rec, prec, color=colors.get(name, "gray"), linewidth=2,
                     label=f"{name} (PR-AUC={pr_auc:.3f})")

    # Baseline
    first_y_true = list(results.values())[0][0]
    baseline = first_y_true.mean()
    axes[1].axhline(y=baseline, color="k", linestyle="--", alpha=0.4,
                     label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Curve Comparison", fontsize=14)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    comp_path = results_dir / "model_comparison.png"
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"\nComparison plots saved to {comp_path}")

    print(f"\n{'=' * 60}")
    print("  Evaluation complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
