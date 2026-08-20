import os
import time
import uuid
import datetime as dt

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Quant Research Trading Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, Arial, sans-serif;
}
.stApp {
    background: #080a0f;
    color: #e2e8f0;
}
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2638;
}
.card {
    background: #111622;
    border: 1px solid #1e2638;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}
.label {
    color: #8b949e;
    font-size: 12px;
}
.value {
    font-size: 21px;
    font-weight: 700;
}
.green { color: #00e676; }
.red { color: #ff5252; }
.blue { color: #38bdf8; }
.orange { color: #f59e0b; }
.small { font-size: 11px; color: #8b949e; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIG
# ============================================================
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "PEPEUSDT",
]

TIMEFRAMES = {
    "1m": ("1m", 1),
    "15m": ("15m", 15),
    "30m": ("30m", 30),
    "1h": ("1h", 60),
    "4h": ("4h", 240),
}

HISTORY_FILE = "trade_history.csv"


# ============================================================
# EXACT 12-FEATURE ENGINE + RISK ENGINE
# ============================================================
class TenPaperResearchLab:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol
        self.scaler = StandardScaler()

        base_model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )

        # Compatible with newer sklearn versions.
        try:
            self.ml_model = CalibratedClassifierCV(
                estimator=base_model,
                method="sigmoid",
                cv="prefit"
            )
        except TypeError:
            self.ml_model = CalibratedClassifierCV(
                base_estimator=base_model,
                method="sigmoid",
                cv="prefit"
            )

        self.is_model_trained = False

        self.feature_names = [
            "HAWKES", "BOOK_IMB", "TAKER_FLOW", "QUANT_IMPLY",
            "BAYESIAN", "QUANTILES", "TARGET_INV", "ADAPT_CONF",
            "FRAC_KELLY", "RMT_DOM", "CONF_CROSS", "REWARD_RISK"
        ]

        self.dynamic_weights = {
            k: 1.0 / len(self.feature_names)
            for k in self.feature_names
        }

    def extract_features(self, df, bids, asks):
        results = {}

        if len(bids) == 0 or len(asks) == 0 or df.empty or len(df) < 15:
            return {k: 0.0 for k in self.feature_names}

        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        mid_price = (bids[0, 0] + asks[0, 0]) / 2

        returns = df["Close"].pct_change().dropna()
        realized_vol = returns.std() + 1e-8

        returns_h = (
            (df["Close"].iloc[-1] - df["Close"].iloc[-5])
            / (df["Close"].iloc[-5] + 1e-8)
        )

        delta_p = df["Close"].iloc[-1] - df["Close"].iloc[-2]

        # 1. Hawkes-type intensity
        vol_changes = df["Volume"].pct_change().dropna().values

        if len(vol_changes) >= 15:
            denominator = np.mean(vol_changes[-15:]) + 1e-8
            hawkes_intensity = np.mean(vol_changes[-3:]) / denominator
        else:
            hawkes_intensity = 1.0

        results["HAWKES"] = np.clip(
            (hawkes_intensity - 1.0) * np.sign(returns_h), -1, 1
        )

        # 2. Book imbalance
        results["BOOK_IMB"] = (
            (bid_vol - ask_vol)
            / (bid_vol + ask_vol + 1e-8)
        )

        # 3. Taker-flow estimate
        last_volume = df["Volume"].iloc[-1]

        taker_buy = last_volume * (1.0 if delta_p > 0 else 0.3)
        taker_sell = last_volume * (1.0 if delta_p <= 0 else 0.3)

        results["TAKER_FLOW"] = (
            (taker_buy - taker_sell)
            / (taker_buy + taker_sell + 1e-8)
        )

        # 4. Quantity imply
        depth_skew = (
            (bids[0, 1] - asks[0, 1])
            / (bids[0, 1] + asks[0, 1] + 1e-8)
        )

        results["QUANT_IMPLY"] = np.clip(depth_skew * 1.5, -1, 1)

        # 5. Bayesian probability
        prior = 0.745
        likelihood = 1.0 if results["BOOK_IMB"] > 0 else 0.25

        posterior = (
            likelihood * prior
        ) / (
            likelihood * prior
            + (1 - likelihood) * (1 - prior)
            + 1e-8
        )

        results["BAYESIAN"] = np.clip(
            (posterior - 0.5) * 2.0, -1, 1
        )

        # 6. Quantiles
        q90 = returns.quantile(0.90) if len(returns) > 5 else 0.01
        q10 = returns.quantile(0.10) if len(returns) > 5 else -0.01

        results["QUANTILES"] = np.clip(
            ((returns_h - q10) / (q90 - q10 + 1e-8)) * 2.0 - 1.0,
            -1,
            1
        )

        # 7. Target / invalidation threshold
        target_diff = delta_p / (df["Close"].iloc[-1] + 1e-8)

        results["TARGET_INV"] = (
            1.0 if target_diff >= 0.0006
            else (-1.0 if target_diff <= -0.0006 else 0.0)
        )

        # 8. Adaptive conformal-style MA signal
        ma_fast = df["Close"].rolling(3).mean().iloc[-1]
        ma_slow = df["Close"].rolling(10).mean().iloc[-1]

        results["ADAPT_CONF"] = np.clip(
            (ma_fast - ma_slow)
            / (realized_vol * mid_price + 1e-8),
            -1,
            1
        )

        # 9. Fractional Kelly
        win_prob = 0.55 + (
            0.15 * np.sign(results["BOOK_IMB"])
        )

        kelly_fraction = win_prob - (
            (1 - win_prob) / 1.5
        )

        results["FRAC_KELLY"] = np.clip(
            kelly_fraction * 2.0 * np.sign(returns_h),
            -1,
            1
        )

        # 10. RMT market dominance
        rmt_dom = (
            abs(returns_h)
            / (realized_vol * np.sqrt(5) + 1e-8)
        ) / 3.0

        results["RMT_DOM"] = np.clip(
            rmt_dom * np.sign(returns_h),
            -1,
            1
        )

        # 11. Conformal interval
        conformal_spread = realized_vol * 1.96
        upper_b = mid_price * (1 + conformal_spread)
        lower_b = mid_price * (1 - conformal_spread)

        # The center of these symmetric bands is mid_price,
        # so this remains neutral unless the implementation is
        # replaced with an actual interval-crossing condition.
        band_center = (upper_b + lower_b) / 2

        results["CONF_CROSS"] = (
            1.0 if mid_price > band_center
            else (-1.0 if mid_price < band_center else 0.0)
        )

        # 12. Quantile reward/risk
        rr_ratio = abs(q90) / (abs(q10) + 1e-8)

        results["REWARD_RISK"] = (
            1.0 if rr_ratio >= 1.2
            else (-1.0 if rr_ratio < 0.8 else 0.0)
        )

        return results

    def calculate_all_signals(
        self,
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=None
    ):
        results = self.extract_features(df, bids, asks)

        feature_vector = np.array(
            [results[k] for k in self.feature_names]
        ).reshape(1, -1)

        try:
            scaled_features = self.scaler.partial_fit(
                feature_vector
            ).transform(feature_vector)
        except Exception:
            scaled_features = feature_vector

        # IMPORTANT:
        # Historical training is only enabled if actual saved
        # feature vectors are available in trade history.
        # We do NOT train on random fake features.
        if performance_history:
            X_train = []
            y_train = []

            for hist in performance_history[-30:]:
                features = hist.get("features")

                if (
                    isinstance(features, dict)
                    and all(k in features for k in self.feature_names)
                    and hist.get("outcome") in {"WIN", "LOSS"}
                ):
                    X_train.append([
                        float(features[k])
                        for k in self.feature_names
                    ])
                    y_train.append(
                        1 if hist["outcome"] == "WIN" else 0
                    )

            if len(X_train) >= 5 and len(set(y_train)) > 1:
                try:
                    X_arr = np.asarray(X_train, dtype=float)
                    y_arr = np.asarray(y_train, dtype=int)

                    self.scaler.fit(X_arr)
                    X_scaled = self.scaler.transform(X_arr)

                    base_clf = SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=0.0001,
                        max_iter=500,
                        random_state=42
                    )

                    base_clf.fit(X_scaled, y_arr)

                    try:
                        self.ml_model = CalibratedClassifierCV(
                            estimator=base_clf,
                            method="sigmoid",
                            cv="prefit"
                        )
                    except TypeError:
                        self.ml_model = CalibratedClassifierCV(
                            base_estimator=base_clf,
                            method="sigmoid",
                            cv="prefit"
                        )

                    # CalibratedClassifierCV with prefit model.
                    self.ml_model.fit(X_scaled, y_arr)
                    self.is_model_trained = True

                except Exception:
                    self.is_model_trained = False

        if self.is_model_trained:
            try:
                ml_prob = self.ml_model.predict_proba(
                    scaled_features
                )[0][1]

                final_score = float(
                    (ml_prob - 0.5) * 2.0
                )

            except Exception:
                weight_vector = np.array(
                    list(self.dynamic_weights.values())
                )
                final_score = float(
                    np.dot(feature_vector[0], weight_vector)
                )
        else:
            weight_vector = np.array(
                list(self.dynamic_weights.values())
            )

            final_score = float(
                np.dot(feature_vector[0], weight_vector)
            )

        return results, final_score, self.dynamic_weights


class PowerTradingRiskEngine:
    def __init__(self):
        pass

    def calculate_risk_metrics(
        self,
        liquidation_volumes,
        displayed_vol,
        cancelled_vol,
        time_exists,
        obs_window,
        open_interest,
        leverage,
        volatility
    ):
        total_ltz = (
            np.sum(liquidation_volumes)
            if len(liquidation_volumes) > 0
            else 0.0
        )

        max_ltz = (
            np.max(liquidation_volumes)
            if len(liquidation_volumes) > 0
            else 0.0
        )

        ltz_score = (
            max_ltz / (total_ltz + 1e-8)
        ) * 100

        spoof_ratio = (
            cancelled_vol
            / (displayed_vol + 1e-8)
        )

        persistence = min(
            max(time_exists / (obs_window + 1e-8), 0),
            1
        )

        spoof_score = spoof_ratio * (1 - persistence)

        squeeze_risk = (
            total_ltz
            * open_interest
            * leverage
            * volatility
        )

        market_risk = (
            ltz_score
            + spoof_score
            + squeeze_risk
        )

        return {
            "LTZ_Score": float(ltz_score),
            "Spoof_Score": float(spoof_score),
            "Squeeze_Risk": float(squeeze_risk),
            "Market_Risk": float(market_risk),
        }


# ============================================================
# DATA FUNCTIONS
# ============================================================
@st.cache_data(ttl=10)
def fetch_klines(symbol, interval, limit=150):
    url = (
        "https://data-api.binance.vision/api/v3/klines"
        f"?symbol={symbol}&interval={interval}&limit={limit}"
    )

    try:
        data = requests.get(url, timeout=5).json()

        if not isinstance(data, list) or not data:
            return pd.DataFrame()

        df = pd.DataFrame(
            data,
            columns=[
                "Open_Time", "Open", "High", "Low", "Close",
                "Volume", "Close_Time", "QAV", "NAT",
                "TBBAV", "TBQAV", "Ignore"
            ]
        )

        df["Time"] = pd.to_datetime(
            df["Open_Time"], unit="ms"
        )

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[
            ["Time", "Open", "High", "Low", "Close", "Volume"]
        ].dropna()

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3)
def fetch_order_book(symbol, limit=20):
    url = (
        "https://data-api.binance.vision/api/v3/depth"
        f"?symbol={symbol}&limit={limit}"
    )

    try:
        data = requests.get(url, timeout=5).json()

        if "bids" not in data or "asks" not in data:
            return np.empty((0, 2)), np.empty((0, 2))

        bids = np.asarray(data["bids"], dtype=float)
        asks = np.asarray(data["asks"], dtype=float)

        return bids, asks

    except Exception:
        return np.empty((0, 2)), np.empty((0, 2))


# ============================================================
# HISTORY
# ============================================================
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        df = pd.read_csv(HISTORY_FILE)

        if "outcome" not in df.columns:
            df["outcome"] = "PENDING"

        records = df.to_dict("records")

        for r in records:
            if "features" not in r:
                r["features"] = None

        return records

    except Exception:
        return []


def save_history(history):
    if not history:
        return

    rows = []

    for item in history:
        row = dict(item)

        features = row.pop("features", None)

        if isinstance(features, dict):
            for key, value in features.items():
                row[f"F_{key}"] = value

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        HISTORY_FILE,
        index=False
    )


def load_ml_features_from_history():
    """
    Reconstruct saved F_* columns so the ML model can learn
    from real historical feature values.
    """
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        df = pd.read_csv(HISTORY_FILE)

        feature_cols = [
            f"F_{x}"
            for x in TenPaperResearchLab().feature_names
        ]

        if not all(c in df.columns for c in feature_cols):
            return []

        records = []

        for _, row in df.iterrows():
            features = {
                c[2:]: float(row[c])
                for c in feature_cols
                if pd.notna(row[c])
            }

            records.append({
                "outcome": row.get("outcome", "PENDING"),
                "features": features
            })

        return records

    except Exception:
        return []


# ============================================================
# TRADE LOGIC
# ============================================================
def signal_from_score(score):
    if score >= 0.15:
        return "LONG"
    if score <= -0.15:
        return "SHORT"
    return "NEUTRAL"


def create_trade(
    symbol,
    timeframe,
    direction,
    entry,
    score,
    confidence,
    atr,
    features
):
    # Simple ATR-based paper trade levels.
    # The engine's directional score remains unchanged.
    if direction == "LONG":
        sl = entry - (1.0 * atr)
        tp1 = entry + (1.5 * atr)
        tp2 = entry + (2.0 * atr)

    elif direction == "SHORT":
        sl = entry + (1.0 * atr)
        tp1 = entry - (1.5 * atr)
        tp2 = entry - (2.0 * atr)

    else:
        return None

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "trade_id": str(uuid.uuid4())[:8].upper(),
        "timestamp": now,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "entry_price": float(entry),
        "stop_loss": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "exit_price": np.nan,
        "confidence": float(confidence),
        "final_score": float(score),
        "outcome": "PENDING",
        "pnl_percent": 0.0,
        "duration_minutes": 0.0,
        "features": features,
    }


def update_trade(trade, current_price):
    if trade["outcome"] != "PENDING":
        return trade

    direction = trade["direction"]
    entry = float(trade["entry_price"])
    sl = float(trade["stop_loss"])
    tp2 = float(trade["tp2"])

    if direction == "LONG":
        hit_sl = current_price <= sl
        hit_tp = current_price >= tp2

    else:
        hit_sl = current_price >= sl
        hit_tp = current_price <= tp2

    if hit_sl and hit_tp:
        # If both are crossed in one polling interval,
        # use conservative LOSS assumption.
        trade["outcome"] = "LOSS"
        trade["exit_price"] = float(sl)

    elif hit_sl:
        trade["outcome"] = "LOSS"
        trade["exit_price"] = float(sl)

    elif hit_tp:
        trade["outcome"] = "WIN"
        trade["exit_price"] = float(tp2)

    if trade["outcome"] in {"WIN", "LOSS"}:
        exit_price = float(trade["exit_price"])

        if direction == "LONG":
            trade["pnl_percent"] = (
                (exit_price - entry) / entry
            ) * 100

        else:
            trade["pnl_percent"] = (
                (entry - exit_price) / entry
            ) * 100

        try:
            start = dt.datetime.strptime(
                trade["timestamp"],
                "%Y-%m-%d %H:%M:%S"
            )
            trade["duration_minutes"] = (
                dt.datetime.now() - start
            ).total_seconds() / 60
        except Exception:
            trade["duration_minutes"] = 0.0

    return trade


def trade_statistics(history):
    df = pd.DataFrame(history)

    if df.empty:
        return {
            "total": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }

    wins = int((df["outcome"] == "WIN").sum())
    losses = int((df["outcome"] == "LOSS").sum())
    pending = int((df["outcome"] == "PENDING").sum())

    closed = wins + losses
    total = len(df)

    win_rate = (
        wins / closed * 100
        if closed > 0
        else 0.0
    )

    pnl = pd.to_numeric(
        df.get("pnl_percent", 0),
        errors="coerce"
    ).fillna(0)

    gross_profit = float(
        pnl[pnl > 0].sum()
    )

    gross_loss = float(
        pnl[pnl < 0].sum()
    )

    net_pnl = float(pnl.sum())

    profit_factor = (
        gross_profit / abs(gross_loss)
        if gross_loss < 0
        else 0.0
    )

    avg_win = (
        gross_profit / wins
        if wins > 0
        else 0.0
    )

    avg_loss = (
        abs(gross_loss) / losses
        if losses > 0
        else 0.0
    )

    win_prob = wins / closed if closed else 0
    loss_prob = losses / closed if closed else 0

    expectancy = (
        win_prob * avg_win
        - loss_prob * avg_loss
    )

    return {
        "total": total,
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": net_pnl,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
    }


# ============================================================
# SESSION STATE
# ============================================================
if "trade_history" not in st.session_state:
    st.session_state.trade_history = load_history()

if "lab" not in st.session_state:
    st.session_state.lab = TenPaperResearchLab()

if "last_trade_key" not in st.session_state:
    st.session_state.last_trade_key = None


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚡ Terminal Controls")

symbol = st.sidebar.selectbox(
    "Cryptocurrency",
    COINS,
    index=0
)

tf_label = st.sidebar.selectbox(
    "Timeframe",
    list(TIMEFRAMES.keys()),
    index=1
)

paper_mode = st.sidebar.toggle(
    "Paper Trading",
    value=True
)

auto_refresh = st.sidebar.toggle(
    "Auto Refresh",
    value=True
)

forecast_horizon = st.sidebar.slider(
    "Forecast Candles",
    5,
    30,
    15
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Paper trading only. This terminal does not place real orders."
)


# ============================================================
# REFRESH
# ============================================================
# No browser-level hard refresh. This avoids the black-screen flash.

# Streamlit native soft refresh (no full browser reload).
# It reruns the app without navigating away from the page.
if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every="5s")
    def _refresh_tick():
        st.empty()
    _refresh_tick()


# ============================================================
# FETCH DATA
# ============================================================
interval, tf_minutes = TIMEFRAMES[tf_label]

df = fetch_klines(
    symbol,
    interval,
    150
)

bids, asks = fetch_order_book(
    symbol,
    20
)


if df.empty or len(bids) == 0 or len(asks) == 0:
    st.error("Unable to fetch live Binance market data.")
    st.stop()


# ============================================================
# ENGINE
# ============================================================
lab = st.session_state.lab

training_history = load_ml_features_from_history()

features, score, weights = lab.calculate_all_signals(
    df,
    bids,
    asks,
    current_inventory=0,
    performance_history=training_history
)

direction = signal_from_score(score)

current_price = float(df["Close"].iloc[-1])
previous_price = float(df["Close"].iloc[-2])

price_change = (
    (current_price - previous_price)
    / previous_price
) * 100

atr = float(
    (df["High"] - df["Low"])
    .rolling(14)
    .mean()
    .iloc[-1]
)

if not np.isfinite(atr) or atr <= 0:
    atr = current_price * 0.002

confidence = min(
    max(abs(score) * 100, 15),
    95
)


# ============================================================
# RISK ENGINE
# ============================================================
risk_engine = PowerTradingRiskEngine()

liquidation_volumes = np.array([
    1000.0,
    2500.0,
    500.0
])

displayed_vol = (
    float(np.sum(asks[:, 1]))
    if len(asks)
    else 1.0
)

cancelled_vol = displayed_vol * 0.12

volatility = float(
    df["Close"].pct_change().std() + 1e-8
)

risk_metrics = risk_engine.calculate_risk_metrics(
    liquidation_volumes=liquidation_volumes,
    displayed_vol=displayed_vol,
    cancelled_vol=cancelled_vol,
    time_exists=15.0,
    obs_window=60.0,
    open_interest=150000.0,
    leverage=20.0,
    volatility=volatility
)


# ============================================================
# CREATE / UPDATE PAPER TRADE
# ============================================================
if paper_mode and direction in {"LONG", "SHORT"}:
    candle_key = (
        f"{symbol}_{tf_label}_"
        f"{df['Time'].iloc[-1]}_{direction}"
    )

    existing = [
        x for x in st.session_state.trade_history
        if x.get("trade_key") == candle_key
    ]

    if not existing:
        trade = create_trade(
            symbol=symbol,
            timeframe=tf_label,
            direction=direction,
            entry=current_price,
            score=score,
            confidence=confidence,
            atr=atr,
            features=features
        )

        if trade:
            trade["trade_key"] = candle_key
            st.session_state.trade_history.insert(
                0,
                trade
            )
            save_history(
                st.session_state.trade_history
            )


# Update every pending trade for the current symbol.
changed = False

for trade in st.session_state.trade_history:
    if (
        trade.get("symbol") == symbol
        and trade.get("outcome") == "PENDING"
    ):
        before = trade.get("outcome")
        update_trade(
            trade,
            current_price
        )

        if before != trade.get("outcome"):
            changed = True

if changed:
    save_history(
        st.session_state.trade_history
    )


# ============================================================
# HEADER
# ============================================================
direction_class = (
    "green" if direction == "LONG"
    else "red" if direction == "SHORT"
    else "blue"
)

st.markdown(
    f"""
    <div class="card">
        <div style="font-size:22px;font-weight:800;">
            ⚡ Quant Research Trading Terminal
        </div>
        <div class="small">
            Live Binance Data • 12-Feature Research Engine •
            Risk Engine • Paper Trade Tracker
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP METRICS
# ============================================================
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown(
        f"""
        <div class="card">
            <div class="label">{symbol}</div>
            <div class="value green">
                ${current_price:,.6f}
            </div>
            <div class="small">
                {price_change:+.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        f"""
        <div class="card">
            <div class="label">FINAL SCORE</div>
            <div class="value {direction_class}">
                {score:+.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        f"""
        <div class="card">
            <div class="label">SIGNAL</div>
            <div class="value {direction_class}">
                {direction}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        f"""
        <div class="card">
            <div class="label">CONFIDENCE</div>
            <div class="value blue">
                {confidence:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m5:
    st.markdown(
        f"""
        <div class="card">
            <div class="label">TIMEFRAME</div>
            <div class="value blue">
                {tf_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m6:
    st.markdown(
        f"""
        <div class="card">
            <div class="label">ML STATUS</div>
            <div class="value">
                {"TRAINED" if lab.is_model_trained else "BASE"}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TRADE SETUP
# ============================================================
st.subheader("🎯 Current Trade Setup")

if direction in {"LONG", "SHORT"}:
    if direction == "LONG":
        sl = current_price - atr
        tp1 = current_price + 1.5 * atr
        tp2 = current_price + 2.0 * atr
    else:
        sl = current_price + atr
        tp1 = current_price - 1.5 * atr
        tp2 = current_price - 2.0 * atr

    rr = 2.0

    t1, t2, t3, t4, t5 = st.columns(5)

    for col, label, value, cls in [
        (t1, "ENTRY", current_price, "blue"),
        (t2, "STOP LOSS", sl, "red"),
        (t3, "TP1", tp1, "green"),
        (t4, "TP2", tp2, "green"),
        (t5, "RISK / REWARD", rr, "blue"),
    ]:
        with col:
            suffix = "R" if label == "RISK / REWARD" else ""
            prefix = "" if label == "RISK / REWARD" else "$"

            st.markdown(
                f"""
                <div class="card">
                    <div class="label">{label}</div>
                    <div class="value {cls}">
                        {prefix}{value:,.6f}{suffix}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info(
        "Current 12-feature ensemble is NEUTRAL. "
        "No new directional paper trade is created."
    )


# ============================================================
# CHART
# ============================================================
left, right = st.columns([2.5, 1])

with left:
    st.subheader("📈 Price & Trajectory")

    future_times = [
        df["Time"].iloc[-1]
        + pd.Timedelta(minutes=tf_minutes * i)
        for i in range(1, forecast_horizon + 1)
    ]

    if direction == "LONG":
        target = current_price + 2.0 * atr
        steps = np.linspace(0, np.pi / 2, forecast_horizon)
        forecast = current_price + (
            target - current_price
        ) * np.sin(steps)

    elif direction == "SHORT":
        target = current_price - 2.0 * atr
        steps = np.linspace(0, np.pi / 2, forecast_horizon)
        forecast = current_price - (
            current_price - target
        ) * np.sin(steps)

    else:
        forecast = np.repeat(
            current_price,
            forecast_horizon
        )

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["Time"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[df["Time"].iloc[-1]] + future_times,
            y=[current_price] + list(forecast),
            mode="lines+markers",
            name="Trajectory",
            line=dict(
                width=2,
                dash="dot"
            )
        )
    )

    if direction in {"LONG", "SHORT"}:
        fig.add_hline(
            y=sl,
            line_dash="dash",
            annotation_text="SL"
        )
        fig.add_hline(
            y=tp1,
            line_dash="dash",
            annotation_text="TP1"
        )
        fig.add_hline(
            y=tp2,
            line_dash="dash",
            annotation_text="TP2"
        )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#111622",
        plot_bgcolor="#111622",
        margin=dict(l=5, r=5, t=10, b=5)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# RISK
# ============================================================
with right:
    st.subheader("🛡️ Risk Engine")

    for label, value in [
        ("LTZ Score", risk_metrics["LTZ_Score"]),
        ("Spoof Score", risk_metrics["Spoof_Score"]),
        ("Squeeze Risk", risk_metrics["Squeeze_Risk"]),
        ("Market Risk", risk_metrics["Market_Risk"]),
    ]:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">{label}</div>
                <div class="value orange">
                    {value:,.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 12 FEATURES
# ============================================================
st.markdown("---")
st.subheader("🔬 12 Research Features")

feature_rows = []

for name in lab.feature_names:
    value = float(features.get(name, 0.0))

    if value > 0.1:
        status = "PASS / BULLISH"
    elif value < -0.1:
        status = "FAIL / BEARISH"
    else:
        status = "NEUTRAL"

    feature_rows.append({
        "Feature": name,
        "Value": round(value, 4),
        "Weight": f"{weights[name] * 100:.2f}%",
        "Status": status
    })

feature_df = pd.DataFrame(feature_rows)

st.dataframe(
    feature_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ORDER BOOK
# ============================================================
st.subheader("📚 Order Book — Top 20")

bid_volume = float(np.sum(bids[:, 1]))
ask_volume = float(np.sum(asks[:, 1]))

obi = (
    (bid_volume - ask_volume)
    / (bid_volume + ask_volume + 1e-8)
)

best_bid = float(bids[0, 0])
best_ask = float(asks[0, 0])

spread = best_ask - best_bid
spread_pct = (
    spread / ((best_bid + best_ask) / 2)
) * 100

o1, o2, o3, o4, o5 = st.columns(5)

for col, label, value in [
    (o1, "Bid Volume", bid_volume),
    (o2, "Ask Volume", ask_volume),
    (o3, "OBI", obi),
    (o4, "Best Bid", best_bid),
    (o5, "Best Ask", best_ask),
]:
    with col:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">{label}</div>
                <div class="value blue">{value:,.6f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.caption(
    f"Spread: {spread:,.8f} ({spread_pct:.4f}%)"
)


# ============================================================
# PERFORMANCE
# ============================================================
st.markdown("---")
st.subheader("🏆 Win Rate & Performance")

st.caption(
    "Win Rate = Wins ÷ (Wins + Losses). Pending trades are NOT included "
    "in the denominator. Total Trades = Closed + Pending."
)

stats = trade_statistics(
    st.session_state.trade_history
)

p1, p2, p3, p4, p5, p6 = st.columns(6)

metrics = [
    ("TOTAL TRADES", stats["total"]),
    ("CLOSED", stats["closed"]),
    ("WINS", stats["wins"]),
    ("LOSSES", stats["losses"]),
    ("PENDING", stats["pending"]),
    ("WIN RATE", stats["win_rate"]),
]

for col, (label, value) in zip(
    [p1, p2, p3, p4, p5, p6],
    metrics
):
    with col:
        if label == "WIN RATE":
            display = f"{value:.2f}%"
            cls = "green"
        else:
            display = str(value)
            cls = "blue"

        st.markdown(
            f"""
            <div class="card">
                <div class="label">{label}</div>
                <div class="value {cls}">
                    {display}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


q1, q2, q3, q4 = st.columns(4)

for col, label, value in [
    (q1, "Net PnL %", stats["net_pnl"]),
    (q2, "Profit Factor", stats["profit_factor"]),
    (q3, "Average Win %", stats["avg_win"]),
    (q4, "Average Loss %", stats["avg_loss"]),
]:
    with col:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">{label}</div>
                <div class="value blue">
                    {value:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FILTERED PERFORMANCE
# ============================================================
st.subheader("🔎 Win Rate Checker")

history_df = pd.DataFrame(
    st.session_state.trade_history
)

if not history_df.empty:
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        coin_filter = st.selectbox(
            "Coin",
            ["ALL"] + sorted(
                history_df["symbol"].dropna().unique().tolist()
            )
        )

    with fc2:
        tf_filter = st.selectbox(
            "Timeframe",
            ["ALL"] + sorted(
                history_df["timeframe"].dropna().unique().tolist()
            )
        )

    with fc3:
        dir_filter = st.selectbox(
            "Direction",
            ["ALL", "LONG", "SHORT"]
        )

    filtered = history_df.copy()

    if coin_filter != "ALL":
        filtered = filtered[
            filtered["symbol"] == coin_filter
        ]

    if tf_filter != "ALL":
        filtered = filtered[
            filtered["timeframe"] == tf_filter
        ]

    if dir_filter != "ALL":
        filtered = filtered[
            filtered["direction"] == dir_filter
        ]

    fwins = int(
        (filtered["outcome"] == "WIN").sum()
    )
    flosses = int(
        (filtered["outcome"] == "LOSS").sum()
    )
    fclosed = fwins + flosses

    fwinrate = (
        fwins / fclosed * 100
        if fclosed > 0
        else 0.0
    )

    fpending = int((filtered["outcome"] == "PENDING").sum())
    ftotal = len(filtered)

    st.success(
        f"Filtered — Total: {ftotal} | Closed: {fclosed} | "
        f"Pending: {fpending} | Wins: {fwins} | "
        f"Losses: {flosses} | Win Rate: {fwinrate:.2f}%"
    )

    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    checker_metrics = [
        ("TOTAL", ftotal),
        ("CLOSED", fclosed),
        ("WINS", fwins),
        ("LOSSES", flosses),
        ("WIN RATE", f"{fwinrate:.2f}%"),
    ]
    for col, (label, value) in zip(
        [fc1, fc2, fc3, fc4, fc5], checker_metrics
    ):
        with col:
            cls = "green" if label in {"WINS", "WIN RATE"} else (
                "red" if label == "LOSSES" else "blue"
            )
            st.markdown(
                f"""
                <div class="card">
                    <div class="label">{label}</div>
                    <div class="value {cls}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# TRADE HISTORY
# ============================================================
st.subheader("📋 Trade History")

if st.session_state.trade_history:
    history_view = pd.DataFrame(
        st.session_state.trade_history
    )

    columns = [
        "trade_id",
        "timestamp",
        "symbol",
        "timeframe",
        "direction",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "exit_price",
        "confidence",
        "final_score",
        "outcome",
        "pnl_percent",
        "duration_minutes",
    ]

    columns = [
        c for c in columns
        if c in history_view.columns
    ]

    st.dataframe(
        history_view[columns],
        use_container_width=True,
        hide_index=True,
        height=360
    )
else:
    st.info("No paper trades yet.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Research/paper-trading terminal. "
    "Signals are model outputs, not guaranteed predictions. "
    "No live order execution is implemented."
)
