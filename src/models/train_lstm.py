import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
import pandas as pd

from src.models.evaluate import report_metrics, plot_by_minute, plot_calibration
from src.models.lstm_model import WinProbLSTM
from src.models.sequence_data import build_sequences, scale_sequences, make_splits, FEATURES_PATH, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # will be cpu on your Mac


def masked_bce_loss(logits, targets, mask):
    per_frame = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_frame = per_frame * mask
    return per_frame.sum() / mask.sum()


def make_loader(split, batch_size=64, shuffle=False):
    """Turn a (X, y, mask) numpy tuple into a batched DataLoader of tensors."""
    X, y, mask = split
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model, loader):
    """Compute average masked loss over a loader, no gradient updates."""
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for X, y, mask in loader:
            X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            logits = model(X)
            loss = masked_bce_loss(logits, y, mask)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches


def train(model, train_loader, val_loader, epochs=20, lr=1e-3, patience=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()                      # dropout ON
        for X, y, mask in train_loader:
            X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = masked_bce_loss(logits, y, mask)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}  val_loss={val_loss:.4f}")

        # early stopping: keep best model, stop if no improvement for `patience` epochs
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()    # snapshot best weights
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)          # restore best weights
    return model


def lstm_predictions(model, split):
    """
    Run the LSTM on a split, flatten to per-frame rows, drop padding.
    Returns a DataFrame: minute, y_true, y_prob, y_pred — same format as baseline preds.
    """
    X, y, mask = split
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits = model(X_t).cpu().numpy()        # (games, frames)
    probs = 1 / (1 + np.exp(-logits))            # sigmoid -> P(win), shape (games, frames)

    rows = []
    n_games, n_frames = mask.shape
    for g in range(n_games):
        for f in range(n_frames):
            if mask[g, f] == 0:
                continue                          # skip padded frames
            rows.append({
                "minute": f,                      # frame index = minute
                "y_true": int(y[g, f]),
                "y_prob": float(probs[g, f]),
                "y_pred": int(probs[g, f] >= 0.5),
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(FEATURES_PATH)
    X, y, mask, match_ids = build_sequences(df)
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    X = scale_sequences(X, mask, scaler)
    splits = make_splits(X, y, mask, match_ids)

    train_loader = make_loader(splits["train"], shuffle=True)
    val_loader = make_loader(splits["val"], shuffle=False)

    model = WinProbLSTM().to(DEVICE)
    model = train(model, train_loader, val_loader)

    torch.save(model.state_dict(), f"{MODEL_DIR}/lstm.pt")
    print("LSTM trained and saved")

    # evaluate on the test set right after training
    preds = lstm_predictions(model, splits["test"])
    print("LSTM test rows:", len(preds))
    report_metrics(preds, "lstm")
    plot_by_minute(preds, "lstm")
    plot_calibration(preds, "lstm")


if __name__ == "__main__":
    main()