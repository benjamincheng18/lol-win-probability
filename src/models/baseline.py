import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

FEATURES_PATH = "data/processed/features.csv"
MODEL_DIR = "data/processed/models"

FEATURE_COLS = [
    "minute", "gold_diff", "xp_diff", "level_diff", "cs_diff",
    "tower_diff", "dragon_diff", "baron_diff", "herald_diff", "grub_diff",
]
LABEL_COL = "won"


def data_preparation(df, test_size=0.2, random_state=42):
    """
    Split by match (not row) into train/test, then separate features (X) and label (y).
    Returns X_train, X_test, y_train, y_test.
    """
    match_ids = df["match_id"].unique()
    train_ids, test_ids = train_test_split(match_ids, test_size=test_size, random_state=random_state)
    train_df = df[df["match_id"].isin(train_ids)]
    test_df  = df[df["match_id"].isin(test_ids)]
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[LABEL_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[LABEL_COL]
    return X_train, X_test, y_train, y_test


def train_logistic(X_train, y_train):
    """
    Standardize features, then fit logistic regression.
    Returns (fitted_model, fitted_scaler) — scaler needed to transform future data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform in one call
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(X_train_scaled, y_train)
    return logistic_regression, scaler

def train_xgboost(X_train, y_train):
    """Fit gradient boosting. No scaling needed (trees are scale-invariant)."""
    xgboost = XGBClassifier(eval_metric="logloss", random_state=42)
    xgboost.fit(X_train, y_train)
    return xgboost


def main():
    df = pd.read_csv(FEATURES_PATH)
    X_train, X_test, y_train, y_test = data_preparation(df)

    logreg, scaler = train_logistic(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(logreg, f"{MODEL_DIR}/logistic.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(xgb, f"{MODEL_DIR}/xgboost.joblib")

    # save the test split so Module 4 evaluates on the identical matches
    X_test.to_csv(f"{MODEL_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{MODEL_DIR}/y_test.csv", index=False)

    print(f"Trained on {len(X_train)} rows, test set {len(X_test)} rows")
    print("Models + scaler + test split saved")


if __name__ == "__main__":
    main()