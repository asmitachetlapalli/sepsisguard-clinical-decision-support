#!/usr/bin/env python3
"""
Train the SepsisLSTM model on preprocessed PhysioNet data.

Pipeline:
  1. Load preprocessed CSV
  2. Create 24-hour sliding-window sequences per patient
  3. Patient-level train/val/test split (no leakage)
  4. Train with BCE loss, Adam optimizer, early stopping on val AUROC
  5. Evaluate on held-out test set
  6. Save trained model + results

Usage:
    python models/train_lstm.py
    python models/train_lstm.py --epochs 50 --batch-size 64 --seq-len 24
    python models/train_lstm.py --data-path data/processed/preprocessed_5000.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Import model from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lstm_model import SepsisLSTM


# Feature columns (must match preprocess.py MODEL_FEATURE_COLS)
FEATURE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp", "EtCO2",
    "Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets",
    "Glucose", "BUN", "pH", "BaseExcess", "HCO3", "Hgb", "Hct",
    "Age", "Gender",
]

TARGET_COL = "early_sepsis_label"


# Dataset class

class SepsisSequenceDataset(Dataset):
    """
    Creates fixed-length sliding-window sequences from patient time-series.

    For each patient with T hours of data, generates (T - seq_len + 1)
    sequences of shape (seq_len, n_features). The label for each window
    is the target value at the LAST timestep of the window.

    Patients shorter than seq_len are padded from the left with zeros
    (and a single sequence is created).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        patient_ids: np.ndarray,
        feature_cols: list[str],
        target_col: str = TARGET_COL,
        seq_len: int = 24,
    ):
        self.seq_len = seq_len
        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []

        available = [c for c in feature_cols if c in df.columns]

        for pid in patient_ids:
            patient_data = df[df["patient_id"] == pid]
            features = patient_data[available].values.astype(np.float32)
            targets = patient_data[target_col].values.astype(np.float32)

            T = len(features)

            if T == 0:
                continue

            if T < seq_len:
                # Pad shorter stays with zeros on the left
                pad = np.zeros((seq_len - T, features.shape[1]), dtype=np.float32)
                padded_feat = np.vstack([pad, features])
                self.sequences.append(padded_feat)
                self.labels.append(int(targets[-1]))
            else:
                # Sliding window
                for start in range(T - seq_len + 1):
                    end = start + seq_len
                    self.sequences.append(features[start:end])
                    self.labels.append(int(targets[end - 1]))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# Training utilities

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weight(dataset: SepsisSequenceDataset) -> float:
    """Compute positive class weight for imbalanced BCE loss."""
    labels = np.array(dataset.labels)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train_one_epoch(
    model: SepsisLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch; return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(
    model: SepsisLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model; return (loss, auroc, pr_auc)."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        total_loss += loss.item()
        n_batches += 1

        all_probs.extend(y_pred.cpu().squeeze(1).tolist())
        all_labels.extend(y_batch.cpu().squeeze(1).tolist())

    avg_loss = total_loss / max(n_batches, 1)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    if len(np.unique(y_true)) < 2:
        return avg_loss, 0.5, 0.0

    auroc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    return avg_loss, auroc, pr_auc


# Main training pipeline

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SepsisLSTM")
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--seq-len", type=int, default=24, help="Sequence length in hours")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-attention", action="store_true", default=False,
                   help="Disable temporal attention (use last timestep only)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_path) if args.data_path else (
        project_root / "data" / "processed" / "preprocessed_1000.csv"
    )

    print("=" * 60)
    print("  SepsisLSTM Training Pipeline")
    print("=" * 60)

    device = get_device()
    print(f"  Device:     {device}")
    print(f"  Seq length: {args.seq_len}h")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Patience:   {args.patience}")
    print()

    # 1. Load data
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    n_features = len(available_features)
    print(f"Loaded {len(df):,} rows, {df['patient_id'].nunique()} patients")
    print(f"Using {n_features} features: {available_features}\n")

    # 2. Patient-level split: 70% train / 15% val / 15% test
    all_pids = df["patient_id"].unique()
    train_pids, temp_pids = train_test_split(all_pids, test_size=0.3, random_state=args.seed)
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=args.seed)

    print(f"Patient split: train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")

    # 3. Create sequence datasets
    print("Creating sequence datasets...")
    t0 = time.time()
    train_ds = SepsisSequenceDataset(df, train_pids, available_features, TARGET_COL, args.seq_len)
    val_ds   = SepsisSequenceDataset(df, val_pids,   available_features, TARGET_COL, args.seq_len)
    test_ds  = SepsisSequenceDataset(df, test_pids,  available_features, TARGET_COL, args.seq_len)
    elapsed = time.time() - t0

    train_pos = sum(train_ds.labels)
    val_pos = sum(val_ds.labels)
    test_pos = sum(test_ds.labels)

    print(f"  Train: {len(train_ds):,} sequences ({train_pos:,} positive)")
    print(f"  Val:   {len(val_ds):,} sequences ({val_pos:,} positive)")
    print(f"  Test:  {len(test_ds):,} sequences ({test_pos:,} positive)")
    print(f"  Created in {elapsed:.1f}s\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 4. Model + optimizer
    use_attention = not args.no_attention
    model = SepsisLSTM(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_attention=use_attention,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SepsisLSTM ({total_params:,} parameters)")
    print(f"  Attention: {'yes' if use_attention else 'no'}")
    print()

    # Weighted BCE for class imbalance
    # Model outputs sigmoid probabilities, so we use binary_cross_entropy
    # with per-sample weights: positive samples get weight = n_neg / n_pos.
    pw = compute_pos_weight(train_ds)
    print(f"  Positive class weight: {pw:.2f}")
    pos_weight_tensor = torch.tensor([pw], device=device)

    def weighted_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted binary cross-entropy (handles class imbalance)."""
        weight = torch.where(target == 1, pos_weight_tensor, torch.ones(1, device=pred.device))
        return nn.functional.binary_cross_entropy(pred, target, weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    # 5. Training loop with early stopping
    print("Training...")
    print(f"{'Epoch':>5s} | {'Train Loss':>10s} | {'Val Loss':>10s} | {'Val AUROC':>10s} | {'Val PR-AUC':>10s} | {'Time':>6s}")
    print("─" * 70)

    best_val_auroc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_pr_auc": []}

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "lstm_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss_total = 0.0
        n_train_batches = 0
        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device).unsqueeze(1)

            optimizer.zero_grad()
            y_pred = model(X_b)
            loss = weighted_bce(y_pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            n_train_batches += 1

        train_loss = train_loss_total / max(n_train_batches, 1)

        # Validate
        model.eval()
        val_probs, val_labels = [], []
        val_loss_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                y_b = y_b.to(device).unsqueeze(1)
                y_pred = model(X_b)
                loss = weighted_bce(y_pred, y_b)
                val_loss_total += loss.item()
                n_val_batches += 1
                val_probs.extend(y_pred.cpu().squeeze(1).tolist())
                val_labels.extend(y_b.cpu().squeeze(1).tolist())

        val_loss = val_loss_total / max(n_val_batches, 1)
        val_labels_arr = np.array(val_labels)
        val_probs_arr = np.array(val_probs)

        if len(np.unique(val_labels_arr)) >= 2:
            val_auroc = roc_auc_score(val_labels_arr, val_probs_arr)
            val_pr_auc = average_precision_score(val_labels_arr, val_probs_arr)
        else:
            val_auroc, val_pr_auc = 0.5, 0.0

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_pr_auc"].append(val_pr_auc)

        marker = ""
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auroc": val_auroc,
                "val_pr_auc": val_pr_auc,
                "args": vars(args),
                "n_features": n_features,
                "feature_cols": available_features,
            }, best_model_path)
            marker = " ★"
        else:
            patience_counter += 1

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | "
              f"{val_auroc:10.4f} | {val_pr_auc:10.4f} | {elapsed:5.1f}s{marker}")

        scheduler.step(val_auroc)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nBest validation AUROC: {best_val_auroc:.4f}")
    print(f"Best model saved to {best_model_path}")

    # 6. Test evaluation
    print(f"\n{'=' * 60}")
    print("  Test Set Evaluation")
    print(f"{'=' * 60}")

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_probs, test_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            y_pred = model(X_b)
            test_probs.extend(y_pred.cpu().squeeze(1).tolist())
            test_labels.extend(y_b.tolist())

    y_true = np.array(test_labels)
    y_prob = np.array(test_probs)
    y_pred_binary = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) >= 2:
        test_auroc = roc_auc_score(y_true, y_prob)
        test_pr_auc = average_precision_score(y_true, y_prob)
    else:
        test_auroc, test_pr_auc = 0.5, 0.0

    print(f"\n  Test AUROC:  {test_auroc:.4f}")
    print(f"  Test PR-AUC: {test_pr_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=["negative", "positive"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_binary))

    # 7. Plots
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs_range, history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["val_auroc"], "g-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Validation AUROC")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history["val_pr_auc"], "m-", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PR-AUC")
    axes[2].set_title("Validation PR-AUC")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = results_dir / "lstm_training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"\nTraining curves saved to {curves_path}")

    # ROC + PR curves (test set)
    if len(np.unique(y_true)) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"LSTM (AUROC={test_auroc:.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].set_title("ROC Curve — LSTM")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        baseline_pr = y_true.mean()
        axes[1].plot(rec, prec, "r-", linewidth=2, label=f"LSTM (PR-AUC={test_pr_auc:.3f})")
        axes[1].axhline(y=baseline_pr, color="k", linestyle="--", alpha=0.5,
                         label=f"Baseline ({baseline_pr:.3f})")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("PR Curve — LSTM")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        test_curves_path = results_dir / "lstm_test_curves.png"
        plt.savefig(test_curves_path, dpi=150)
        plt.close()
        print(f"Test curves saved to {test_curves_path}")

    # 8. Final summary
    print(f"\n{'=' * 60}")
    print(f"  Training Complete — Summary")
    print(f"{'=' * 60}")
    print(f"  Model:          SepsisLSTM (attention={'yes' if use_attention else 'no'})")
    print(f"  Parameters:     {total_params:,}")
    print(f"  Features:       {n_features}")
    print(f"  Sequence len:   {args.seq_len}h")
    print(f"  Best val AUROC: {best_val_auroc:.4f}")
    print(f"  Test AUROC:     {test_auroc:.4f}")
    print(f"  Test PR-AUC:    {test_pr_auc:.4f}")
    print(f"  Model saved:    {best_model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
