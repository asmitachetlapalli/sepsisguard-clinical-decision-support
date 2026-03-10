#!/usr/bin/env python3
"""
LSTM model for early sepsis prediction from ICU time-series sequences.

Architecture:
  - Multi-layer LSTM encoder (captures temporal vital-sign patterns)
  - Optional temporal attention (weighted combination of all timesteps)
  - Fully connected classifier head → sigmoid risk probability

Input:  (batch_size, sequence_length, input_size)
Output: (batch_size, 1) — sepsis risk score in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Learnable attention over LSTM timesteps.
    Instead of using only the last hidden state, compute a weighted
    combination of ALL hidden states so the model can focus on the
    most informative hours in the sequence.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
        """
        scores = self.attn(lstm_output).squeeze(-1)       # (batch, seq_len)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)   # (batch, seq_len, 1)
        context = (lstm_output * weights).sum(dim=1)       # (batch, hidden_size)
        return context


class SepsisLSTM(nn.Module):
    """
    LSTM-based model for early sepsis prediction.

    Args:
        input_size:  Number of features per timestep (default 22 for full feature set)
        hidden_size: LSTM hidden dimension
        num_layers:  Number of stacked LSTM layers
        dropout:     Dropout rate for regularization
        use_attention: If True, use temporal attention; otherwise use last timestep
    """

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Temporal attention (optional)
        if use_attention:
            self.attention = TemporalAttention(hidden_size)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            risk: (batch_size, 1) — sepsis risk probability
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        if self.use_attention:
            context = self.attention(lstm_out)       # (batch, hidden_size)
        else:
            context = lstm_out[:, -1, :]             # (batch, hidden_size)

        return self.classifier(context)              # (batch, 1)


# Quick architecture test

if __name__ == "__main__":
    print("=" * 55)
    print("  SepsisLSTM Architecture Test")
    print("=" * 55)

    for use_attn in [True, False]:
        tag = "with attention" if use_attn else "last-timestep only"
        print(f"\n--- {tag} ---")
        model = SepsisLSTM(
            input_size=22,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            use_attention=use_attn,
        )
        print(model)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Total parameters:     {total:,}")
        print(f"  Trainable parameters: {trainable:,}")

        # Dummy forward pass: batch=8, seq_len=24, features=22
        dummy = torch.randn(8, 24, 22)
        out = model(dummy)
        preds = out[:3].squeeze(1).tolist()
        print(f"  Input shape:  {tuple(dummy.shape)}")
        print(f"  Output shape: {tuple(out.shape)}")
        print(f"  Sample preds: [{preds[0]:.4f}, {preds[1]:.4f}, {preds[2]:.4f}]")

    print(f"\n{'=' * 55}")
    print("  Architecture verified successfully!")
    print(f"{'=' * 55}")
