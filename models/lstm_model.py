#!/usr/bin/env python3
"""LSTM model for sepsis prediction from vital sign sequences."""

import torch
import torch.nn as nn


class SepsisLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, apply_sigmoid=False):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if apply_sigmoid:
            out = torch.sigmoid(out)
        return out
