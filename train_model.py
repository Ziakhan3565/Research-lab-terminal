import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


# ============================================================
# FILES
# ============================================================

MARKET_DATA_FILE = "market_data_log.csv"
TRADE_FEEDBACK_FILE = "trade_feedback.csv"

MODEL_FILE = "xgboost_obi_model.pkl"
MODEL_META_FILE = "model_metadata.json"

CANDIDATE_MODEL_FILE = "xgboost_candidate.pkl"

RETRAIN_EVERY = 20


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "top20_bid_sum",
    "top20_ask_sum",
    "top50_bid_sum",
    "top50_ask_sum",
    "obi_top20",
    "obi_top50",
    "spread",
    "spread_pct",
    "BOOK_IMB",
    "OFI",
    "TAKER_FLOW",
    "QUANT_IMPLY",
    "ADAPT_CONF",
    "BAYESIAN",
    "FOURIER_TREND",
    "EMA_TREND",
    "VWAP_DISTANCE",
    "VOLATILITY",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    df = df.copy()

    if "symbol" not in df.columns:
        raise ValueError(
            "symbol column missing"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # BASIC FEATURES
    # --------------------------------------------------------

    df["bid_ask_ratio"] = (
        df["top20_bid_sum"]
        /
        (
            df["top20_ask_sum"]
            + 1e-8
        )
    )

    df["total_depth20"] = (
        df["top20_bid_sum"]
        +
        df["top20_ask_sum"]
    )

    df["total_depth50"] = (
        df["top50_bid_sum"]
        +
        df["top50_ask_sum"]
    )

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    df["returns"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .pct_change()
        .fillna(0)
    )

    df["realized_vol"] = (
        df.groupby("symbol")[
            "returns"
        ]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=5,
            ).std()
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # BOOK IMBALANCE
    # --------------------------------------------------------

    df["BOOK_IMB"] = (
        df["top20_bid_sum"]
        -
        df["top20_ask_sum"]
    ) / (
        df["top20_bid_sum"]
        +
        df["top20_ask_sum"]
        +
        1e-8
    )

    # --------------------------------------------------------
    # OFI
    # --------------------------------------------------------

    prev_bid = (
        df.groupby("symbol")[
            "top20_bid_sum"
        ]
        .shift(1)
    )

    prev_ask = (
        df.groupby("symbol")[
            "top20_ask_sum"
        ]
        .shift(1)
    )

    raw_ofi = (
        (
            df["top20_bid_sum"]
            - prev_bid
        )
        -
        (
            df["top20_ask_sum"]
            - prev_ask
        )
    )

    depth_scale = (
        df["top20_bid_sum"]
        +
        df["top20_ask_sum"]
        +
        1e-8
    )

    df["OFI"] = (
        raw_ofi
        / depth_scale
    ).clip(
        -1,
        1,
    ).fillna(0)

    # --------------------------------------------------------
    # TAKER FLOW
    #
    # Collector currently does not have true trade-by-trade
    # aggressor data. Therefore this remains conservative.
    # --------------------------------------------------------

    delta = (
        df.groupby("symbol")[
            "current_price"
        ]
        .diff()
        .fillna(0)
    )

    estimated_buy = np.where(
        delta > 0,
        1.0,
        0.3,
    )

    estimated_sell = np.where(
        delta <= 0,
        1.0,
        0.3,
    )

    df["TAKER_FLOW"] = (
        estimated_buy
        -
        estimated_sell
    ) / (
        estimated_buy
        +
        estimated_sell
        +
        1e-8
    )

    # --------------------------------------------------------
    # QUANT IMPLY
    # --------------------------------------------------------

    best_bid_size = (
        df["top20_bid_sum"]
        / 20.0
    )

    best_ask_size = (
        df["top20_ask_sum"]
        / 20.0
    )

    depth_skew = (
        best_bid_size
        -
        best_ask_size
    ) / (
        best_bid_size
        +
        best_ask_size
        +
        1e-8
    )

    df["QUANT_IMPLY"] = (
        depth_skew * 1.5
    ).clip(
        -1,
        1,
    )

    # --------------------------------------------------------
    # ADAPT CONF
    # --------------------------------------------------------

    ma3 = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=1,
            ).mean()
        )
    )

    ma10 = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.rolling(
                10,
                min_periods=1,
            ).mean()
        )
    )

    df["ADAPT_CONF"] = (
        (
            ma3 - ma10
        )
        /
        (
            df["realized_vol"]
            *
            df["current_price"]
            +
            1e-8
        )
    ).clip(
        -1,
        1,
    )

    # --------------------------------------------------------
    # BAYESIAN
    # --------------------------------------------------------

    prior = 0.745

    likelihood = np.where(
        df["BOOK_IMB"] > 0,
        1.0,
        0.25,
    )

    numerator = (
        likelihood * prior
    )

    denominator = (
        numerator
        +
        (
            (1 - likelihood)
            *
            (1 - prior)
        )
        +
        1e-8
    )

    posterior = (
        numerator
        /
        denominator
    )

    df["BAYESIAN"] = (
        (posterior - 0.5)
        * 2
    ).clip(
        -1,
        1,
    )

    # --------------------------------------------------------
    # ROLLING FOURIER
    # --------------------------------------------------------

    def rolling_fourier(series):

        values = series.values.astype(float)

        output = np.zeros(
            len(values)
        )

        window = 32

        for i in range(
            len(values)
        ):

            start = max(
                0,
                i - window + 1,
            )

            segment = values[
                start:i + 1
            ]

            if len(segment) < 15:
                continue

            centered = (
                segment
                -
                np.mean(segment)
            )

            fft_values = np.fft.fft(
                centered
            )

            n = len(
                fft_values
            )

            keep = max(
                1,
                int(
                    n * 0.15
                ),
            )

            filtered = np.zeros_like(
                fft_values
            )

            filtered[:keep] = (
                fft_values[:keep]
            )

            filtered[-keep:] = (
                fft_values[-keep:]
            )

            curve = np.real(
                np.fft.ifft(
                    filtered
                )
            )

            trend = (
                curve[-1]
                -
                curve[-2]
            )

            output[i] = trend

        return pd.Series(
            output,
            index=series.index,
        )

    df["FOURIER_TREND"] = (
        df.groupby(
            "symbol"
        )[
            "current_price"
        ]
        .transform(
            rolling_fourier
        )
    )

    # Normalize Fourier

    df["FOURIER_TREND"] = (
        df["FOURIER_TREND"]
        /
        (
            df["current_price"]
            *
            df["realized_vol"]
            +
            1e-8
        )
    ).clip(
        -1,
        1,
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.ewm(
                span=20,
                adjust=False,
            ).mean()
        )
    )

    df["ema50"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.ewm(
                span=50,
                adjust=False,
            ).mean()
        )
    )

    df["EMA_TREND"] = (
        (
            df["ema20"]
            -
            df["ema50"]
        )
        /
        (
            df["current_price"]
            +
            1e-8
        )
    ) * 100

    df["EMA_TREND"] = (
        df["EMA_TREND"]
        .clip(
            -1,
            1,
        )
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    # Collector currently has no actual volume.
    # Use depth as a conservative proxy.

    volume_proxy = (
        df["top20_bid_sum"]
        +
        df["top20_ask_sum"]
    )

    cumulative_price_volume = (
        df.groupby("symbol")
        .apply(
            lambda g:
            (
                g["current_price"]
                *
                volume_proxy.loc[
                    g.index
                ]
            ).cumsum()
        )
        .reset_index(
            level=0,
            drop=True,
        )
    )

    cumulative_volume = (
        df.groupby("symbol")
        [volume_proxy.name if volume_proxy.name else "current_price"]
        if False
        else None
    )

    # Safer explicit calculation

    temp_pv = (
        df["current_price"]
        *
        volume_proxy
    )

    df["_cum_pv"] = (
        temp_pv.groupby(
            df["symbol"]
        ).cumsum()
    )

    df["_cum_volume"] = (
        volume_proxy.groupby(
            df["symbol"]
        ).cumsum()
    )

    df["vwap"] = (
        df["_cum_pv"]
        /
        (
            df["_cum_volume"]
            + 1e-8
        )
    )

    df["VWAP_DISTANCE"] = (
        (
            df["current_price"]
            -
            df["vwap"]
        )
        /
        (
            df["vwap"]
            + 1e-8
        )
    ) * 100

    df["VWAP_DISTANCE"] = (
        df["VWAP_DISTANCE"]
        .clip(
            -1,
            1,
        )
    )

    df["VOLATILITY"] = (
        df["realized_vol"]
        * 100
    ).clip(
        0,
        1,
    )

    return df


# ============================================================
# TARGET
# ============================================================

def build_target(
    df,
    horizon_rows=5,
    threshold=0.0005,
):

    df = df.copy()

    df["future_price"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .shift(
            -horizon_rows
        )
    )

    df["future_return"] = (
        (
            df["future_price"]
            -
            df["current_price"]
        )
        /
        (
            df["current_price"]
            + 1e-8
        )
    )

    # 1 = bullish
    # 0 = bearish/neutral

    df["target"] = (
        df["future_return"]
        >
        threshold
    ).astype(int)

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    X,
    y,
):

    n = len(X)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    X_train = X.iloc[
        :train_end
    ]

    y_train = y.iloc[
        :train_end
    ]

    X_validation = X.iloc[
        train_end:validation_end
    ]

    y_validation = y.iloc[
        train_end:validation_end
    ]

    X_test = X.iloc[
        validation_end:
    ]

    y_test = y.iloc[
        validation_end:
    ]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
    }


# ============================================================
# TRAIN INITIAL MODEL
# ============================================================

def train_initial_model():

    if not os.path.exists(
        MARKET_DATA_FILE
    ):
        print(
            "❌ market_data_log.csv not found."
        )
        return False

    df = pd.read_csv(
        MARKET_DATA_FILE
    )

    if len(df) < 200:
        print(
            "❌ Not enough data. "
            "Collect more market data first."
        )
        return False

    df = build_features(
        df
    )

    df = build_target(
        df,
        horizon_rows=5,
        threshold=0.0005,
    )

    required = (
        FEATURES
        +
        [
            "current_price",
            "future_price",
            "future_return",
            "target",
        ]
    )

    df = df.dropna(
        subset=required
    ).copy()

    X = df[
        FEATURES
    ]

    y = df[
        "target"
    ]

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = chronological_split(
        X,
        y,
    )

    model = create_model()

    print(
        "🚀 Training initial XGBoost model..."
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    metadata = {
        "version": 1,
        "features": FEATURES,
        "metrics": metrics,
        "training_samples": int(
            len(X_train)
        ),
        "test_samples": int(
            len(X_test)
        ),
    }

    with open(
        MODEL_META_FILE,
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    print(
        f"✅ Model saved: {MODEL_FILE}"
    )

    return True


# ============================================================
# RETRAIN AFTER 20 TRADES
# ============================================================

def retrain_after_20_trades():

    if not os.path.exists(
        MARKET_DATA_FILE
    ):
        return False

    df = pd.read_csv(
        MARKET_DATA_FILE
    )

    if len(df) < 200:
        return False

    df = build_features(
        df
    )

    df = build_target(
        df,
        horizon_rows=5,
        threshold=0.0005,
    )

    df = df.dropna(
        subset=FEATURES
        +
        [
            "future_price",
            "future_return",
            "target",
        ]
    ).copy()

    X = df[
        FEATURES
    ]

    y = df[
        "target"
    ]

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = chronological_split(
        X,
        y,
    )

    candidate = create_model()

    candidate.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    candidate_metrics = evaluate_model(
        candidate,
        X_test,
        y_test,
    )

    print(
        "Candidate model:",
        candidate_metrics,
    )

    # --------------------------------------------------------
    # OLD MODEL
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_FILE
    ):

        joblib.dump(
            candidate,
            MODEL_FILE,
        )

        return True

    old_model = joblib.load(
        MODEL_FILE
    )

    old_metrics = evaluate_model(
        old_model,
        X_test,
        y_test,
    )

    print(
        "Old model:",
        old_metrics,
    )

    # --------------------------------------------------------
    # ACCEPTANCE GATE
    # --------------------------------------------------------

    candidate_score = (
        candidate_metrics["f1"]
        +
        candidate_metrics["precision"]
        +
        candidate_metrics["accuracy"]
    )

    old_score = (
        old_metrics["f1"]
        +
        old_metrics["precision"]
        +
        old_metrics["accuracy"]
    )

    if (
        candidate_score
        >
        old_score
    ):

        joblib.dump(
            candidate,
            MODEL_FILE,
        )

        print(
            "✅ Candidate model is better."
        )

        print(
            "✅ New model accepted."
        )

        return True

    print(
        "⚠️ Candidate model is not better."
    )

    print(
        "🛡️ Old model kept."
    )

    return False


# ============================================================
# TRADE COUNTER
# ============================================================

def count_completed_trades():

    if not os.path.exists(
        TRADE_FEEDBACK_FILE
    ):
        return 0

    try:

        df = pd.read_csv(
            TRADE_FEEDBACK_FILE
        )

        if "status" in df.columns:

            df = df[
                df["status"]
                ==
                "CLOSED"
            ]

        return len(df)

    except Exception:
        return 0


def check_and_retrain():

    count = count_completed_trades()

    print(
        f"📊 Completed trades: {count}"
    )

    if count == 0:
        return False

    if count % RETRAIN_EVERY != 0:
        return False

    print(
        "🔄 20 completed trades reached."
    )

    return retrain_after_20_trades()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if os.path.exists(
        MODEL_FILE
    ):

        print(
            "Existing model found."
        )

        print(
            "Running retraining evaluation..."
        )

        retrain_after_20_trades()

    else:

        train_initial_model()
