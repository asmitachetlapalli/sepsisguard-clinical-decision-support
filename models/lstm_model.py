#!/usr/bin/env python3
"""
LSTM model for sepsis prediction from vital sign sequences.
Uses PyTorch; expects input (batch_size, sequence_length, input_size).
"""

import torch
import torch.nn as nn


class SepsisLSTM(nn.Module):
    """
    LSTM-based model for early sepsis prediction from vital sign time series.

    Predicts sepsis risk from a sequence of vital signs (e.g. 24 hours of
    HR, O2Sat, Temp, MAP, Resp, Age). The LSTM captures temporal patterns
    across timesteps; the final hidden state is passed through fully
    connected layers to produce a single risk score per sample.

    Input format: (batch_size, sequence_length, input_size)
        - batch_size: number of samples in the batch
        - sequence_length: number of timesteps (e.g. 24 for 24 hours)
        - input_size: number of features per timestep (e.g. 6 vitals)

    Output format: (batch_size, 1)
        - One risk score in [0, 1] per sample (probability of early sepsis).
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM: only apply dropout between stacked layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Classifier head: hidden -> 32 -> 1 with ReLU and dropout
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, apply_sigmoid: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, sequence_length, input_size)
            apply_sigmoid: if True, apply sigmoid to output (use at inference).
                           During training, use BCEWithLogitsLoss instead.

        Returns:
            Logits (batch_size, 1) or probabilities if apply_sigmoid=True.
        """
        # LSTM output: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Use last timestep only
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if apply_sigmoid:
            out = torch.sigmoid(out)

        return out  # (batch_size, 1)


if __name__ == "__main__":
    print("============ LSTM Architecture ============")
    model = SepsisLSTM(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    )
    print(model)
    print()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    print("============ Testing Forward Pass ============")
    # batch_size=8, sequence_length=24 hours, input_size=6 vitals
    dummy_input = torch.randn(8, 24, 6)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    sample_preds = output[:3].squeeze(1).tolist()
    print(f"Sample predictions: [{sample_preds[0]:.3f}, {sample_preds[1]:.3f}, {sample_preds[2]:.3f}]")
    print("✓ Architecture verified successfully!")
