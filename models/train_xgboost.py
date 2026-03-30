#!/usr/bin/env python3
"""
Train XGBoost model for early sepsis prediction with Optuna hyperparameter tuning.
Includes full evaluation metrics and calibrated risk thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

BASE_FEATURES = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp"]
LAB_FEATURES = ["Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets",
                "BUN", "Glucose", "Hgb", "Hct", "pH"]
DEMO_FEATURES = ["Age", "Gender", "HospAdmTime", "ICULOS"]
TARGET_COL = "early_sepsis_label"


def main():
    project_root = Path(__file__).resolve().parent.parent
    # Auto-detect the largest preprocessed file
    processed_dir = project_root / "data" / "processed"
    csvs = sorted(processed_dir.glob("preprocessed_*.csv"), key=lambda p: int(p.stem.split("_")[1]), reverse=True)
    data_path = csvs[0] if csvs else processed_dir / "preprocessed_1000.csv"

    print("=" * 60)
    print("XGBoost Training — Early Sepsis Prediction")
    print("=" * 60)

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {df['patient_id'].nunique()} patients")

    # Collect available features
    all_candidates = BASE_FEATURES + LAB_FEATURES + DEMO_FEATURES
    FEATURE_COLS = [c for c in all_candidates if c in df.columns]
    print(f"Using {len(FEATURE_COLS)} features")

    # Fill NaN with median
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Patient-level split: 80% train, 20% test
    patient_ids = df["patient_id"].unique()
    train_pids, test_pids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_mask = df["patient_id"].isin(train_pids)
    test_mask = df["patient_id"].isin(test_pids)

    X_train = df.loc[train_mask, FEATURE_COLS].values
    y_train = df.loc[train_mask, TARGET_COL].values
    X_test = df.loc[test_mask, FEATURE_COLS].values
    y_test = df.loc[test_mask, TARGET_COL].values

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"Train: {len(X_train):,} ({n_pos} pos, {n_neg} neg)")
    print(f"Test:  {len(X_test):,} ({int(y_test.sum())} pos)")

    # ── Optuna Hyperparameter Tuning ────────────────────────────────────
    if HAS_OPTUNA:
        print("\nRunning Optuna hyperparameter tuning (50 trials)...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3),
            }
            model = xgb.XGBClassifier(
                **params, eval_metric="auc", random_state=42, early_stopping_rounds=20,
            )
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                model.fit(
                    X_train[train_idx], y_train[train_idx],
                    eval_set=[(X_train[val_idx], y_train[val_idx])],
                    verbose=False,
                )
                pred = model.predict_proba(X_train[val_idx])[:, 1]
                scores.append(roc_auc_score(y_train[val_idx], pred))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        best_params = study.best_params
        print(f"Best AUROC (CV): {study.best_value:.4f}")
        print(f"Best params: {best_params}")
    else:
        print("\nOptuna not available, using default params")
        best_params = {
            "n_estimators": 500, "max_depth": 3, "learning_rate": 0.01,
            "scale_pos_weight": n_neg / max(n_pos, 1),
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "gamma": 0.1,
            "reg_alpha": 0.5, "reg_lambda": 2.0,
        }

    # ── Train final model with best params ──────────────────────────────
    print("\nTraining final model...")
    model = xgb.XGBClassifier(
        **best_params, eval_metric="auc", random_state=42, early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)

    # ── Evaluate ────────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresh = float(thresholds[optimal_idx])
    y_pred = (y_proba >= optimal_thresh).astype(int)

    # Risk level thresholds from distribution
    risk_low = float(np.percentile(y_proba, 50))
    risk_high = float(np.percentile(y_proba, 90))

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"AUROC:  {auroc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Optimal threshold: {optimal_thresh:.4f}")
    print(f"Risk thresholds: LOW < {risk_low:.3f} < MODERATE < {risk_high:.3f} < HIGH")

    print(f"\nClassification Report (at optimal threshold {optimal_thresh:.3f}):")
    print(classification_report(y_test, y_pred, target_names=["No Sepsis", "Sepsis"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Confusion Matrix:\n{cm}")
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"PPV (Precision):      {ppv:.4f}")
    print(f"NPV:                  {npv:.4f}")
    print(f"F1 Score:             {f1:.4f}")

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature Importance (top 10):")
    for i in sorted_idx[:10]:
        print(f"  {FEATURE_COLS[i]:20s}: {importances[i]:.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    save_path = project_root / "models" / "xgboost_model.pkl"
    joblib.dump({
        "model": model,
        "feature_cols": list(FEATURE_COLS),
        "target_col": TARGET_COL,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "optimal_threshold": optimal_thresh,
        "risk_threshold_low": risk_low,
        "risk_threshold_high": risk_high,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "best_params": best_params,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # ── Plots ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ROC
    axes[0, 0].plot(fpr, tpr, "b-", linewidth=2, label=f"XGBoost (AUROC={auroc:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], "k--")
    axes[0, 0].scatter(fpr[optimal_idx], tpr[optimal_idx], c="red", s=80, zorder=5,
                       label=f"Threshold={optimal_thresh:.2f}")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[0, 1].plot(rec, prec, "r-", linewidth=2, label=f"XGBoost (PR-AUC={pr_auc:.3f})")
    baseline = y_test.mean()
    axes[0, 1].axhline(baseline, color="k", linestyle="--", label=f"Baseline ({baseline:.3f})")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Feature importance
    top_n = min(15, len(FEATURE_COLS))
    top_idx = sorted_idx[:top_n][::-1]
    axes[1, 0].barh([FEATURE_COLS[i] for i in top_idx], importances[top_idx], color="steelblue")
    axes[1, 0].set_xlabel("Importance (Gain)")
    axes[1, 0].set_title("Feature Importance")
    axes[1, 0].grid(alpha=0.3, axis="x")

    # Score distribution
    axes[1, 1].hist(y_proba[y_test == 0], bins=50, alpha=0.6, label="No Sepsis", density=True)
    axes[1, 1].hist(y_proba[y_test == 1], bins=50, alpha=0.6, label="Sepsis", density=True, color="red")
    axes[1, 1].axvline(optimal_thresh, color="black", linestyle="--", label=f"Threshold={optimal_thresh:.2f}")
    axes[1, 1].set_xlabel("Risk Score")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Score Distribution")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = project_root / "results" / "xgboost_results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plots saved to {plot_path}")
    print("Done!")


if __name__ == "__main__":
    main()
