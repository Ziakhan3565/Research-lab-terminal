import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from src.feature_pipeline import FEATURES

MARKET_DATA_FILE = "market_data_log.csv"
TRADE_FEEDBACK_FILE = "trade_feedback.csv"
MODEL_FILE = "xgboost_obi_model.pkl"
MODEL_META_FILE = "model_metadata.json"
CANDIDATE_MODEL_FILE = "xgboost_candidate.pkl"

RETRAIN_EVERY = 20
MIN_MARKET_ROWS = 200
MIN_FEEDBACK_ROWS = 20
MIN_ACCEPTABLE_ACCURACY = 0.52


def clean_frame(df):
    df = df.copy()
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def market_training_data():
    if not os.path.exists(MARKET_DATA_FILE):
        return pd.DataFrame(), pd.Series(dtype=int)

    df = pd.read_csv(MARKET_DATA_FILE)
    if len(df) < MIN_MARKET_ROWS:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = clean_frame(df)
    if "symbol" not in df or "timestamp" not in df or "current_price" not in df:
        return pd.DataFrame(), pd.Series(dtype=int)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Five collector observations ~= 25 seconds at a 5-second interval.
    # This is only a baseline market target; trade feedback is the stronger signal.
    df["future_price"] = df.groupby("symbol")["current_price"].shift(-5)
    df["future_return"] = (
        (df["future_price"] - df["current_price"])
        / (df["current_price"] + 1e-8)
    )
    df["target"] = (df["future_return"] > 0.0005).astype(int)
    df = df.dropna(subset=FEATURES + ["future_price"])

    return df[FEATURES], df["target"].astype(int)


def feedback_training_data():
    if not os.path.exists(TRADE_FEEDBACK_FILE):
        return pd.DataFrame(), pd.Series(dtype=int)

    df = pd.read_csv(TRADE_FEEDBACK_FILE)
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = df[df.get("status", "") == "CLOSED"].copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = clean_frame(df)

    # Model predicts the direction that would have been correct at entry.
    # Long WIN -> bullish(1), Long LOSS -> bearish(0)
    # Short WIN -> bearish(0), Short LOSS -> bullish(1)
    direction = df["direction"].astype(str).str.upper()
    outcome = df["outcome"].astype(str).str.upper()
    df["target"] = np.where(
        direction.str.contains("LONG"),
        (outcome == "WIN").astype(int),
        (outcome == "LOSS").astype(int),
    )

    return df[FEATURES], df["target"].astype(int)


def combined_training_data():
    market_X, market_y = market_training_data()
    feedback_X, feedback_y = feedback_training_data()

    parts_x, parts_y, weights = [], [], []

    if not market_X.empty:
        parts_x.append(market_X)
        parts_y.append(market_y)
        weights.extend([1.0] * len(market_X))

    if not feedback_X.empty:
        # Real completed trades get stronger influence than generic market samples.
        parts_x.append(feedback_X)
        parts_y.append(feedback_y)
        weights.extend([5.0] * len(feedback_X))

    if not parts_x:
        return pd.DataFrame(columns=FEATURES), pd.Series(dtype=int), np.array([])

    X = pd.concat(parts_x, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.concat(parts_y, ignore_index=True).astype(int)
    return X, y, np.asarray(weights, dtype=float)


def chronological_split(X, y, weights):
    order = np.arange(len(X))
    # Combined data is already chronological within each source; keep a final holdout.
    n = len(X)
    train_end = max(1, int(n * 0.70))
    valid_end = max(train_end + 1, int(n * 0.85))
    valid_end = min(valid_end, n)

    return (
        X.iloc[:train_end], y.iloc[:train_end], weights[:train_end],
        X.iloc[train_end:valid_end], y.iloc[train_end:valid_end],
        X.iloc[valid_end:], y.iloc[valid_end:],
    )


def create_model():
    return XGBClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.5,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )


def evaluate_model(model, X, y):
    if len(X) == 0 or y.nunique() < 2:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def train_model():
    X, y, weights = combined_training_data()
    if len(X) < MIN_MARKET_ROWS and len(X) < MIN_FEEDBACK_ROWS:
        print("Not enough training data.")
        return False

    if y.nunique() < 2:
        print("Need both bullish and bearish targets before training.")
        return False

    (
        X_train, y_train, w_train,
        X_val, y_val,
        X_test, y_test,
    ) = chronological_split(X, y, weights)

    if y_train.nunique() < 2:
        print("Training split contains only one class.")
        return False

    model = create_model()
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)] if len(X_val) else None,
        verbose=False,
    )

    metrics = evaluate_model(model, X_test, y_test)
    joblib.dump(model, MODEL_FILE)

    metadata = {
        "version": 2,
        "features": FEATURES,
        "metrics": metrics,
        "samples": int(len(X)),
        "feedback_samples": int(len(feedback_training_data()[0])),
    }
    Path(MODEL_META_FILE).write_text(json.dumps(metadata, indent=2))

    print("Initial/normal model trained:", metrics)
    return True


def retrain_after_20_trades():
    if not os.path.exists(TRADE_FEEDBACK_FILE):
        return False

    feedback = pd.read_csv(TRADE_FEEDBACK_FILE)
    closed = feedback[feedback.get("status", "") == "CLOSED"]
    if len(closed) < RETRAIN_EVERY:
        return False

    X, y, weights = combined_training_data()
    if len(X) < MIN_MARKET_ROWS and len(closed) < MIN_FEEDBACK_ROWS:
        return False
    if y.nunique() < 2:
        print("Retrain skipped: only one target class.")
        return False

    (
        X_train, y_train, w_train,
        X_val, y_val,
        X_test, y_test,
    ) = chronological_split(X, y, weights)

    if y_train.nunique() < 2:
        return False

    candidate = create_model()
    candidate.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)] if len(X_val) else None,
        verbose=False,
    )

    candidate_metrics = evaluate_model(candidate, X_test, y_test)
    candidate_score = (
        candidate_metrics["accuracy"]
        + candidate_metrics["precision"]
        + candidate_metrics["f1"]
    )

    old_score = -1.0
    old_metrics = None

    if os.path.exists(MODEL_FILE):
        try:
            old = joblib.load(MODEL_FILE)
            old_metrics = evaluate_model(old, X_test, y_test)
            old_score = (
                old_metrics["accuracy"]
                + old_metrics["precision"]
                + old_metrics["f1"]
            )
        except Exception as exc:
            print("Old model could not be evaluated:", exc)

    accepted = (
        not os.path.exists(MODEL_FILE)
        or (
            candidate_metrics["accuracy"] >= MIN_ACCEPTABLE_ACCURACY
            and candidate_score > old_score
        )
    )

    joblib.dump(candidate, CANDIDATE_MODEL_FILE)

    if accepted:
        if os.path.exists(MODEL_FILE):
            shutil.copy2(MODEL_FILE, MODEL_FILE + ".backup")
        os.replace(CANDIDATE_MODEL_FILE, MODEL_FILE)
        print("NEW MODEL ACCEPTED:", candidate_metrics)
    else:
        print("OLD MODEL KEPT. Candidate:", candidate_metrics, "Old:", old_metrics)

    metadata = {
        "version": 2,
        "last_retrain_closed_trades": int(len(closed)),
        "candidate_metrics": candidate_metrics,
        "old_metrics": old_metrics,
        "accepted": bool(accepted),
        "features": FEATURES,
    }
    Path(MODEL_META_FILE).write_text(json.dumps(metadata, indent=2))
    return accepted


def count_completed_trades():
    if not os.path.exists(TRADE_FEEDBACK_FILE):
        return 0
    try:
        df = pd.read_csv(TRADE_FEEDBACK_FILE)
        return int((df.get("status", "") == "CLOSED").sum())
    except Exception:
        return 0


def check_and_retrain():
    count = count_completed_trades()
    if count < RETRAIN_EVERY or count % RETRAIN_EVERY != 0:
        return False
    return retrain_after_20_trades()


if __name__ == "__main__":
    if os.path.exists(MODEL_FILE):
        retrain_after_20_trades()
    else:
        train_model()
