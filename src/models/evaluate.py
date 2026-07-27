import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import calibration_curve

MODEL_DIR = "data/processed/models"
FIG_DIR = "reports/figures"
FEATURES_PATH = "data/processed/features.csv"

FEATURE_COLS = [
    "minute", "gold_diff", "xp_diff", "level_diff", "cs_diff",
    "tower_diff", "dragon_diff", "baron_diff", "herald_diff", "grub_diff",
]


def compute_predictions(model, X, y, minutes, scaler=None):
    if scaler is not None:
        X_in = scaler.transform(X)
    else:
        X_in = X
    y_prob = model.predict_proba(X_in)[:, 1]
    y_pred = model.predict(X_in)
    return pd.DataFrame({
        "minute": minutes, 
        "y_true": y, 
        "y_prob": y_prob, 
        "y_pred": y_pred
    })


def report_metrics(preds, model_name):
    ll = log_loss(preds["y_true"], preds["y_prob"])
    acc = accuracy_score(preds["y_true"], preds["y_pred"])
    print(f"{model_name}: \nLog Loss: {ll}\nAccuracy: {acc}")
    return (ll, acc)

def plot_by_minute(preds, model_name):
    """Bucket by minute (per-min to 35, 36+ grouped), compute acc + log loss per bucket."""
    preds = preds.copy()
    preds["bucket"] = preds["minute"].clip(upper=36)

    buckets, ll_list, acc_list = [], [], []
    for b, group in preds.groupby("bucket"):
        if len(group) == 0:
            continue
        buckets.append(b)
        ll_list.append(log_loss(group["y_true"], group["y_prob"], labels=[0, 1]))
        acc_list.append(accuracy_score(group["y_true"], group["y_pred"]))

    # --- accuracy plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(buckets, acc_list, marker="o")
    plt.title(f"Accuracy by minute — {model_name}")
    plt.xlabel("Game minute (36 = 36+)")
    plt.ylabel("Accuracy")
    plt.ylim(0.4, 1.0)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/{model_name}_accuracy_by_minute.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- log loss plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(buckets, ll_list, marker="o", color="darkred")
    plt.title(f"Log loss by minute — {model_name}")
    plt.xlabel("Game minute (36 = 36+)")
    plt.ylabel("Log loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/{model_name}_logloss_by_minute.png", dpi=150, bbox_inches="tight")
    plt.close()

    return pd.DataFrame({"bucket": buckets, "log_loss": ll_list, "accuracy": acc_list})


def plot_calibration(preds, model_name, n_bins=10):
    """Reliability diagram: predicted prob vs actual win rate, against the diagonal."""
    true_rate, pred_mean = calibration_curve(preds["y_true"], preds["y_prob"], n_bins=n_bins)
    plt.figure(figsize=(8, 5))
    plt.plot(pred_mean, true_rate, marker="o", label=model_name)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.gca().set_aspect("equal")
    plt.xlabel("Predicted mean")
    plt.ylabel("True rate")
    plt.grid(True, alpha=0.3)
    plt.axline((0, 0), slope=1, color="red", linestyle="--", label="perfect calibration")
    plt.legend()
    plt.savefig(f"{FIG_DIR}/{model_name}_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


def feature_importance(xgb_model, feature_names):
    importances = xgb_model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False)
    print(fi.to_string(index=False))
    return fi


def logloss_by_minute_table(preds):
    preds = preds.copy()
    preds["bucket"] = preds["minute"].clip(upper=36)
    buckets, ll_list = [], []
    for b, group in preds.groupby("bucket"):
        buckets.append(b)
        ll_list.append(log_loss(group["y_true"], group["y_prob"], labels=[0, 1]))
    return pd.DataFrame({"bucket": buckets, "log_loss": ll_list})


def compare_models(preds_dict):
    """
    preds_dict: {model_name: preds_df}. Overlay per-minute log loss for all models.
    """
    plt.figure(figsize=(9, 6))
    for name, preds in preds_dict.items():
        table = logloss_by_minute_table(preds)
        plt.plot(table["bucket"], table["log_loss"], marker="o", markersize=4, label=name)

    plt.axhline(0.693, color="grey", linestyle=":", label="coinflip (0.693)")
    plt.title("Log loss by minute — model comparison")
    plt.xlabel("Game minute (36 = 36+)")
    plt.ylabel("Log loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/model_comparison_by_minute.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_comparison():
    import joblib, torch, numpy as np
    from src.models.lstm_model import WinProbLSTM
    from src.models.sequence_data import build_sequences, scale_sequences, make_splits
    from src.models.train_lstm import lstm_predictions
    # baseline preds
    logreg = joblib.load(f"{MODEL_DIR}/logistic.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    xgb = joblib.load(f"{MODEL_DIR}/xgboost.joblib")
    X_test = pd.read_csv(f"{MODEL_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{MODEL_DIR}/y_test.csv").squeeze()

    log_preds = compute_predictions(logreg, X_test, y_test, X_test["minute"], scaler=scaler)
    xgb_preds = compute_predictions(xgb, X_test, y_test, X_test["minute"], scaler=None)

    # lstm preds (same test matches)
    df = pd.read_csv(FEATURES_PATH)
    X, y, mask, match_ids = build_sequences(df)
    X = scale_sequences(X, mask, scaler)
    splits = make_splits(X, y, mask, match_ids)
    model = WinProbLSTM()
    model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm.pt"))
    lstm_preds = lstm_predictions(model, splits["test"])

    compare_models({"logistic": log_preds, "xgboost": xgb_preds, "lstm": lstm_preds})
    print("Comparison plot saved")


def main():
    logreg = joblib.load(f"{MODEL_DIR}/logistic.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    xgb = joblib.load(f"{MODEL_DIR}/xgboost.joblib")

    X_test = pd.read_csv(f"{MODEL_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{MODEL_DIR}/y_test.csv").squeeze()
    minutes = X_test["minute"]

    os.makedirs(FIG_DIR, exist_ok=True)

    # logistic: scaled | xgboost: raw
    logreg_preds = compute_predictions(logreg, X_test, y_test, minutes, scaler=scaler)
    xgb_preds = compute_predictions(xgb, X_test, y_test, minutes, scaler=None)

    for preds, name in [(logreg_preds, "logistic"), (xgb_preds, "xgboost")]:
        report_metrics(preds, name)
        plot_by_minute(preds, name)
        plot_calibration(preds, name)

    feature_importance(xgb, FEATURE_COLS)


if __name__ == "__main__":
    main()