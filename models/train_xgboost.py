#!/usr/bin/env python3
"""
Train XGBoost for sepsis prediction — optimized version.
- Drops features with >80% missing (noise)
- Adds temporal features (rolling means, deltas)
- Uses SepsisLabel for more positive samples
- Optuna hyperparameter tuning
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

# Features with <80% missing — these actually have signal
GOOD_FEATURES = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp",
                 "Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets",
                 "Age", "Gender", "HospAdmTime", "ICULOS"]
TARGET_COL = "SepsisLabel"  # more positives than early_sepsis_label


def add_temporal_features(df, base_vitals):
    """Add rolling means and deltas for vital signs."""
    df = df.copy()
    new_cols = []
    for col in base_vitals:
        if col not in df.columns:
            continue
        # Delta from previous hour
        delta = f"{col}_delta"
        df[delta] = df.groupby("patient_id")[col].diff().fillna(0)
        new_cols.append(delta)

        # 3-hour rolling mean
        roll = f"{col}_roll3"
        df[roll] = df.groupby("patient_id")[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        new_cols.append(roll)
    return df, new_cols


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    csvs = sorted(processed_dir.glob("preprocessed_*.csv"),
                  key=lambda p: int(p.stem.split("_")[1]), reverse=True)
    data_path = csvs[0] if csvs else processed_dir / "preprocessed_1000.csv"

    print("=" * 60)
    print("XGBoost Training — Optimized")
    print("=" * 60)
    print(f"Data: {data_path.name}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {df['patient_id'].nunique():,} patients")

    # Use only features with reasonable data quality
    FEATURE_COLS = [c for c in GOOD_FEATURES if c in df.columns]

    # Fill NaN
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Add temporal features for vitals
    base_vitals = ["HR", "O2Sat", "Temp", "MAP", "Resp", "SBP", "DBP"]
    df, temporal_cols = add_temporal_features(df, base_vitals)
    ALL_FEATURES = FEATURE_COLS + temporal_cols
    print(f"Features: {len(FEATURE_COLS)} base + {len(temporal_cols)} temporal = {len(ALL_FEATURES)} total")

    # Patient-level split
    patient_ids = df["patient_id"].unique()
    train_pids, test_pids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    X_train = df.loc[df["patient_id"].isin(train_pids), ALL_FEATURES].values
    y_train = df.loc[df["patient_id"].isin(train_pids), TARGET_COL].values
    X_test = df.loc[df["patient_id"].isin(test_pids), ALL_FEATURES].values
    y_test = df.loc[df["patient_id"].isin(test_pids), TARGET_COL].values

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"Target: {TARGET_COL}")
    print(f"Train: {len(X_train):,} ({n_pos:,} pos, {n_neg:,} neg)")
    print(f"Test:  {len(X_test):,} ({int(y_test.sum()):,} pos)")

    # Optuna
    if HAS_OPTUNA:
        print("\nOptuna tuning (50 trials)...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "gamma": trial.suggest_float("gamma", 0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5),
            }
            model = xgb.XGBClassifier(**params, eval_metric="auc", random_state=42, early_stopping_rounds=20)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for tr_idx, val_idx in cv.split(X_train, y_train):
                model.fit(X_train[tr_idx], y_train[tr_idx],
                          eval_set=[(X_train[val_idx], y_train[val_idx])], verbose=False)
                scores.append(roc_auc_score(y_train[val_idx], model.predict_proba(X_train[val_idx])[:, 1]))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        best_params = study.best_params
        print(f"Best CV AUROC: {study.best_value:.4f}")
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 5, "learning_rate": 0.02,
            "scale_pos_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "gamma": 0.1, "reg_alpha": 0.5, "reg_lambda": 2.0,
        }

    # Train final
    print("\nTraining final model...")
    model = xgb.XGBClassifier(**best_params, eval_metric="auc", random_state=42, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    opt_idx = np.argmax(tpr - fpr)
    opt_thresh = float(thresholds[opt_idx])
    y_pred = (y_proba >= opt_thresh).astype(int)

    risk_low = float(np.percentile(y_proba, 50))
    risk_high = float(np.percentile(y_proba, 90))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"RESULTS (AUROC: {auroc:.4f})")
    print(f"{'='*60}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"Threshold:   {opt_thresh:.4f}")
    print(f"Risk levels: LOW < {risk_low:.3f} < MOD < {risk_high:.3f} < HIGH")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Sepsis', 'Sepsis'], zero_division=0)}")
    print(f"Confusion Matrix:\n{cm}")

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 15 Features:")
    for i in sorted_idx[:15]:
        print(f"  {ALL_FEATURES[i]:20s}: {importances[i]:.4f}")

    # Save
    save_path = project_root / "models" / "xgboost_model.pkl"
    joblib.dump({
        "model": model,
        "feature_cols": list(ALL_FEATURES),
        "base_features": list(FEATURE_COLS),
        "temporal_features": temporal_cols,
        "target_col": TARGET_COL,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "optimal_threshold": opt_thresh,
        "risk_threshold_low": risk_low,
        "risk_threshold_high": risk_high,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "best_params": best_params,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes[0, 0].plot(fpr, tpr, "b-", lw=2, label=f"XGBoost (AUROC={auroc:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], "k--")
    axes[0, 0].scatter(fpr[opt_idx], tpr[opt_idx], c="red", s=80, zorder=5)
    axes[0, 0].set_xlabel("FPR"); axes[0, 0].set_ylabel("TPR")
    axes[0, 0].set_title("ROC Curve"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[0, 1].plot(rec, prec, "r-", lw=2, label=f"PR-AUC={pr_auc:.3f}")
    axes[0, 1].axhline(y_test.mean(), color="k", linestyle="--", label=f"Baseline ({y_test.mean():.3f})")
    axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    top_n = min(20, len(ALL_FEATURES))
    top_idx = sorted_idx[:top_n][::-1]
    axes[1, 0].barh([ALL_FEATURES[i] for i in top_idx], importances[top_idx], color="steelblue")
    axes[1, 0].set_xlabel("Importance"); axes[1, 0].set_title("Feature Importance"); axes[1, 0].grid(alpha=0.3, axis="x")

    axes[1, 1].hist(y_proba[y_test == 0], bins=50, alpha=0.6, label="No Sepsis", density=True)
    axes[1, 1].hist(y_proba[y_test == 1], bins=50, alpha=0.6, label="Sepsis", density=True, color="red")
    axes[1, 1].axvline(opt_thresh, color="black", linestyle="--", label=f"Threshold={opt_thresh:.2f}")
    axes[1, 1].set_xlabel("Risk Score"); axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Score Distribution"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(project_root / "results" / "xgboost_results.png", dpi=150)
    plt.close()
    print("Plots saved. Done!")


if __name__ == "__main__":
    main()
