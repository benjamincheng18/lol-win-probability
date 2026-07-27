import torch
import torch.nn as nn


class WinProbLSTM(nn.Module):
    def __init__(self, n_features=10, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, time, features)
        out, _ = self.lstm(x)      # out: (batch, time, hidden_size); we ignore the final (h,c) tuple
        out = self.dropout(out)
        logits = self.head(out)    # (batch, time, 1)
        return logits.squeeze(-1)  # (batch, time) — one logit per frame