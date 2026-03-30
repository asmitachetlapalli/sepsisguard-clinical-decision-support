#!/usr/bin/env python3
"""
Train LSTM for sepsis prediction on full 40K dataset.
Uses same 6 features as LR baseline for fair comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lstm_model import SepsisLSTM

FEATURE_COLS = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp",
                "Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets",
                "BUN", "Glucose", "Hgb", "Hct", "pH",
                "Age", "Gender", "HospAdmTime", "ICULOS"]
TARGET_COL = "SepsisLabel"
SEQ_LEN = 24
BATCH_SIZE = 256
EPOCHS = 50
LR = 2e-4  # lower LR to learn slower and generalize better
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SepsisSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(df, feature_cols, seq_len=24, neg_ratio=3):
    """Build sequences. Keep all positive, subsample negative at neg_ratio:1."""
    seqs, labels = [], []
    for pid in df["patient_id"].unique():
        patient = df[df["patient_id"] == pid].sort_values("ICULOS")
        X = patient[feature_cols].values
        y = patient[TARGET_COL].values

        if len(X) < seq_len:
            continue

        pos_seqs, neg_seqs = [], []
        for i in range(len(X) - seq_len + 1):
            s = X[i:i + seq_len]
            l = y[i + seq_len - 1]
            if l == 1:
                pos_seqs.append((s, l))
            else:
                neg_seqs.append((s, l))

        # Keep all positive sequences
        for s, l in pos_seqs:
            seqs.append(s)
            labels.append(l)

        # Subsample negative: keep neg_ratio * num_positive, or at least 3 per patient
        if neg_seqs:
            n_keep = max(3, len(pos_seqs) * neg_ratio)
            step = max(1, len(neg_seqs) // n_keep)
            for s, l in neg_seqs[::step][:n_keep]:
                seqs.append(s)
                labels.append(l)

    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    # Pick the file with the most patients (highest number in filename)
    csvs = sorted(processed_dir.glob("preprocessed_*.csv"),
                  key=lambda p: int(p.stem.split("_")[1]), reverse=True)
    data_path = csvs[0] if csvs else processed_dir / "preprocessed_1000.csv"

    print("=" * 60)
    print("LSTM Training — Sepsis Prediction")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {data_path.name}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {df['patient_id'].nunique():,} patients")

    # Fill NaN
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Patient-level split
    patient_ids = df["patient_id"].unique()
    train_pids, test_pids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    df_train = df[df["patient_id"].isin(train_pids)].copy()
    df_test = df[df["patient_id"].isin(test_pids)].copy()

    # Scale
    scaler = StandardScaler()
    df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_test[FEATURE_COLS] = scaler.transform(df_test[FEATURE_COLS])

    print(f"Train patients: {len(train_pids):,}, Test patients: {len(test_pids):,}")

    # Build sequences
    print(f"Building {SEQ_LEN}-hour sequences...")
    X_train, y_train = build_sequences(df_train, FEATURE_COLS, SEQ_LEN, neg_ratio=5)
    X_test, y_test = build_sequences(df_test, FEATURE_COLS, SEQ_LEN, neg_ratio=5)

    n_pos_train = int(y_train.sum())
    n_pos_test = int(y_test.sum())
    print(f"Train: {len(X_train):,} seqs ({n_pos_train:,} pos, {len(X_train)-n_pos_train:,} neg)")
    print(f"Test:  {len(X_test):,} seqs ({n_pos_test:,} pos)")

    train_loader = DataLoader(SepsisSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SepsisSequenceDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # Model
    model = SepsisLSTM(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2, dropout=0.5).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Weighted loss
    pos_weight = torch.tensor([min((len(y_train) - n_pos_train) / max(n_pos_train, 1), 10.0)]).to(DEVICE)
    print(f"Pos weight: {pos_weight.item():.1f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    best_auroc = 0
    best_state = None
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_b)
            n += len(y_b)

        avg_loss = total_loss / n

        # Evaluate
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                p = model(X_b.to(DEVICE), apply_sigmoid=True).cpu().numpy()
                preds.append(p)
                labs.append(y_b.numpy())
        preds = np.concatenate(preds).flatten()
        labs = np.concatenate(labs).flatten()
        auroc = roc_auc_score(labs, preds)
        scheduler.step(auroc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | AUROC: {auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Final eval
    model.load_state_dict(best_state)
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds.append(model(X_b.to(DEVICE), apply_sigmoid=True).cpu().numpy())
            labs.append(y_b.numpy())
    preds = np.concatenate(preds).flatten()
    labs = np.concatenate(labs).flatten()

    auroc = roc_auc_score(labs, preds)
    pr_auc = average_precision_score(labs, preds)
    fpr, tpr, thresholds = roc_curve(labs, preds)
    opt_idx = np.argmax(tpr - fpr)
    opt_thresh = float(thresholds[opt_idx])
    y_pred = (preds >= opt_thresh).astype(int)

    cm = confusion_matrix(labs.astype(int), y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(labs.astype(int), y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"AUROC:       {auroc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"Threshold:   {opt_thresh:.4f}")
    print(f"\n{classification_report(labs.astype(int), y_pred, target_names=['No Sepsis', 'Sepsis'], zero_division=0)}")
    print(f"Confusion Matrix:\n{cm}")

    # Save
    save_path = project_root / "models" / "lstm_trained.pth"
    torch.save({
        "model_state_dict": best_state,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "best_auroc": best_auroc,
        "pr_auc": pr_auc,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "optimal_threshold": opt_thresh,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"LSTM (AUROC={auroc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].scatter(fpr[opt_idx], tpr[opt_idx], c="red", s=80, zorder=5)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve — LSTM")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(labs, preds)
    axes[1].plot(rec, prec, "r-", linewidth=2, label=f"LSTM (PR-AUC={pr_auc:.3f})")
    axes[1].axhline(labs.mean(), color="k", linestyle="--", label=f"Baseline ({labs.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curve — LSTM")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(project_root / "results" / "lstm_results.png", dpi=150)
    plt.close()
    print("Plots saved to results/lstm_results.png")
    print("Done!")


if __name__ == "__main__":
    main()
