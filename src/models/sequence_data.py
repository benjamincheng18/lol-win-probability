import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

FEATURES_PATH = "data/processed/features.csv"
MODEL_DIR = "data/processed/models"

FEATURE_COLS = [
    "minute", "gold_diff", "xp_diff", "level_diff", "cs_diff",
    "tower_diff", "dragon_diff", "baron_diff", "herald_diff", "grub_diff",
]
LABEL_COL = "won"
MAX_LEN = 38


def build_sequences(df, max_len=MAX_LEN):
    """
    Convert flat feature table into padded sequences.
    Returns:
        X:    (n_games, max_len, n_features)  float array, zero-padded
        y:    (n_games, max_len)              label repeated per real frame, 0 in padding
        mask: (n_games, max_len)              1.0 for real frames, 0.0 for padding
        match_ids: (n_games,)                 aligned match id per sequence, for splitting
    """
    n_features = len(FEATURE_COLS)
    X_list, y_list, mask_list, id_list = [], [], [], []

    for match_id, game in df.groupby("match_id"):
        game = game.sort_values("minute")          # ensure temporal order within game
        feats = game[FEATURE_COLS].to_numpy(dtype=np.float32)   # (game_len, n_features)
        label = int(game[LABEL_COL].iloc[0])       # constant per game

        game_len = len(game)
        game_len = min(game_len, max_len)
        feats = feats[:max_len]

        x_pad = np.zeros((max_len, n_features), dtype=np.float32)
        x_pad[:game_len] = feats
        y_pad = np.zeros((max_len, ), dtype=np.float32)
        y_pad[:game_len] = label
        
        m = np.zeros((max_len, ), dtype=np.float32)
        m[:game_len] = 1.0
        
        X_list.append(x_pad)
        y_list.append(y_pad)
        mask_list.append(m)
        id_list.append(match_id)

    X = np.stack(X_list)
    y = np.stack(y_list)
    mask = np.stack(mask_list)
    match_ids = np.array(id_list)
    return X, y, mask, match_ids


def scale_sequences(X, mask, scaler):
    """
    Apply the SAME StandardScaler from the baseline to sequence features.
    Only real (unmasked) frames should inform nothing new — scaler is already fit;
    we just transform. Padded frames stay ~0 after scaling is handled by re-masking.
    """
    n_games, max_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_flat).reshape(n_games, max_len, n_features)
    X_scaled = X_scaled * mask[:, :, None]
    return X_scaled


def make_splits(X, y, mask, match_ids, random_state=42):
    train_ids = np.load(f"{MODEL_DIR}/train_ids.npy", allow_pickle=True)
    test_ids = np.load(f"{MODEL_DIR}/test_ids.npy", allow_pickle=True)

    # split baseline train into train/val
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=random_state)

    def select(ids):
        sel = np.isin(match_ids, ids)
        return X[sel], y[sel], mask[sel]

    return {
        "train": select(train_ids),
        "val": select(val_ids),
        "test": select(test_ids),
    }


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    X, y, mask, match_ids = build_sequences(df)
    print("X:", X.shape, "y:", y.shape, "mask:", mask.shape)

    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    X = scale_sequences(X, mask, scaler)
    print("scaled X range:", round(float(X.min()), 2), round(float(X.max()), 2))

    splits = make_splits(X, y, mask, match_ids)
    for name, (Xs, ys, ms) in splits.items():
        print(f"{name}: X {Xs.shape}, y {ys.shape}, mask {ms.shape}")