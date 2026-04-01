#!/usr/bin/env python3
"""
Generate model comparison visualizations: ROC curves, confusion matrices, threshold analysis.
XGBoost vs Logistic Regression on the same dataset and features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score, f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    csvs = sorted(processed_dir.glob("preprocessed_*.csv"),
                  key=lambda p: int(p.stem.split("_")[1]), reverse=True)
    data_path = csvs[0] if csvs else processed_dir / "preprocessed_1000.csv"

    print(f"Data: {data_path.name}")
    df = pd.read_csv(data_path)

    # Load models
    xgb_data = joblib.load(project_root / "models" / "xgboost_model.pkl")
    lr_data = joblib.load(project_root / "models" / "baseline_lr.pkl")

    # Same patient split
    patient_ids = df["patient_id"].unique()
    _, test_pids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    df_test = df[df["patient_id"].isin(test_pids)].copy()

    target = xgb_data["target_col"]
    y_test = df_test[target].values

    # Fill NaN
    all_features = set(xgb_data["feature_cols"] + lr_data["feature_cols"])
    for col in all_features:
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(df[col].median())

    # Add temporal features if needed
    for col in ["HR", "O2Sat", "Temp", "MAP", "Resp", "SBP", "DBP"]:
        if f"{col}_delta" in xgb_data["feature_cols"] and f"{col}_delta" not in df_test.columns:
            df_test[f"{col}_delta"] = df_test.groupby("patient_id")[col].diff().fillna(0)
        if f"{col}_roll3" in xgb_data["feature_cols"] and f"{col}_roll3" not in df_test.columns:
            df_test[f"{col}_roll3"] = df_test.groupby("patient_id")[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean())

    # XGBoost predictions
    X_xgb = df_test[xgb_data["feature_cols"]].fillna(0).values
    xgb_proba = xgb_data["model"].predict_proba(X_xgb)[:, 1]
    xgb_auroc = roc_auc_score(y_test, xgb_proba)
    xgb_fpr, xgb_tpr, xgb_thresh = roc_curve(y_test, xgb_proba)
    xgb_pr_auc = average_precision_score(y_test, xgb_proba)

    # LR predictions
    X_lr = df_test[lr_data["feature_cols"]].fillna(0).values
    lr_proba = lr_data["model"].predict_proba(X_lr)[:, 1]
    lr_auroc = roc_auc_score(y_test, lr_proba)
    lr_fpr, lr_tpr, lr_thresh = roc_curve(y_test, lr_proba)
    lr_pr_auc = average_precision_score(y_test, lr_proba)

    print(f"XGBoost AUROC: {xgb_auroc:.4f}  PR-AUC: {xgb_pr_auc:.4f}")
    print(f"LR AUROC:      {lr_auroc:.4f}  PR-AUC: {lr_pr_auc:.4f}")

    # Optimal thresholds
    xgb_opt = xgb_thresh[np.argmax(xgb_tpr - xgb_fpr)]
    lr_opt = lr_thresh[np.argmax(lr_tpr - lr_fpr)]

    xgb_pred = (xgb_proba >= xgb_opt).astype(int)
    lr_pred = (lr_proba >= lr_opt).astype(int)

    xgb_cm = confusion_matrix(y_test, xgb_pred)
    lr_cm = confusion_matrix(y_test, lr_pred)

    # ── Create figure with 4 subplots ───────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # 1. ROC Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(xgb_fpr, xgb_tpr, "b-", linewidth=2.5, label=f"XGBoost (AUROC = {xgb_auroc:.3f})")
    ax1.plot(lr_fpr, lr_tpr, "r-", linewidth=2.5, label=f"Logistic Regression (AUROC = {lr_auroc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="lower right")
    ax1.grid(alpha=0.3)

    # 2. PR Curve Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    xgb_prec, xgb_rec, _ = precision_recall_curve(y_test, xgb_proba)
    lr_prec, lr_rec, _ = precision_recall_curve(y_test, lr_proba)
    baseline = y_test.mean()
    ax2.plot(xgb_rec, xgb_prec, "b-", linewidth=2.5, label=f"XGBoost (PR-AUC = {xgb_pr_auc:.3f})")
    ax2.plot(lr_rec, lr_prec, "r-", linewidth=2.5, label=f"LR (PR-AUC = {lr_pr_auc:.3f})")
    ax2.axhline(baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curve Comparison", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    # 3. Confusion Matrices
    ax3 = fig.add_subplot(gs[1, 0])
    labels = ["No Sepsis", "Sepsis"]
    im = ax3.imshow(xgb_cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            val = xgb_cm[i, j]
            ax3.text(j, i, f"{val:,}", ha="center", va="center",
                     fontsize=14, color="white" if val > xgb_cm.max() / 2 else "black")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel("Predicted", fontsize=12)
    ax3.set_ylabel("Actual", fontsize=12)
    ax3.set_title(f"XGBoost Confusion Matrix (threshold={xgb_opt:.3f})", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax3, shrink=0.8)

    # 4. Threshold Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    # Calculate sensitivity and specificity at various thresholds for XGBoost
    thresholds_to_test = np.arange(0.05, 0.95, 0.01)
    sensitivities = []
    specificities = []
    f1_scores = []
    for t in thresholds_to_test:
        pred = (xgb_proba >= t).astype(int)
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, pred, zero_division=0)
        sensitivities.append(sens)
        specificities.append(spec)
        f1_scores.append(f1)

    ax4.plot(thresholds_to_test, sensitivities, "b-", linewidth=2, label="Sensitivity (Recall)")
    ax4.plot(thresholds_to_test, specificities, "r-", linewidth=2, label="Specificity")
    ax4.plot(thresholds_to_test, f1_scores, "g-", linewidth=2, label="F1 Score")
    ax4.axvline(xgb_opt, color="black", linestyle="--", alpha=0.7, label=f"Optimal threshold ({xgb_opt:.3f})")
    ax4.set_xlabel("Decision Threshold", fontsize=12)
    ax4.set_ylabel("Score", fontsize=12)
    ax4.set_title("XGBoost Threshold Analysis", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    plt.savefig(project_root / "results" / "model_comparison_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved to results/model_comparison_roc.png")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25s} {'XGBoost':>12s} {'LR':>12s}")
    print(f"{'-'*49}")
    print(f"{'AUROC':<25s} {xgb_auroc:>12.4f} {lr_auroc:>12.4f}")
    print(f"{'PR-AUC':<25s} {xgb_pr_auc:>12.4f} {lr_pr_auc:>12.4f}")

    xgb_tn, xgb_fp, xgb_fn, xgb_tp = xgb_cm.ravel()
    lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()

    xgb_sens = xgb_tp / (xgb_tp + xgb_fn)
    lr_sens = lr_tp / (lr_tp + lr_fn)
    xgb_spec = xgb_tn / (xgb_tn + xgb_fp)
    lr_spec = lr_tn / (lr_tn + lr_fp)
    xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
    lr_f1 = f1_score(y_test, lr_pred, zero_division=0)

    print(f"{'Sensitivity':<25s} {xgb_sens:>12.4f} {lr_sens:>12.4f}")
    print(f"{'Specificity':<25s} {xgb_spec:>12.4f} {lr_spec:>12.4f}")
    print(f"{'F1 Score':<25s} {xgb_f1:>12.4f} {lr_f1:>12.4f}")
    print(f"{'Features':<25s} {len(xgb_data['feature_cols']):>12d} {len(lr_data['feature_cols']):>12d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
