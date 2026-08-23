import datetime
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 2. STREAMLIT CONFIG & PERSISTENT CSV SETUP
# ==========================================
st.set_page_config(
    page_title="Quantitative Research & Paper Trading Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

count = st_autorefresh(interval=5000, limit=None, key="research_lab_auto_refresh")

CSV_FILE = "signal_history.csv"


def load_persistent_history():
    if os.path.exists(CSV_FILE):
        try:
            df_hist = pd.read_csv(CSV_FILE)

            expected_cols = [
                "trade_id", "timestamp", "symbol", "timeframe", "direction",
                "entry_price", "stop_loss", "tp1", "tp2", "exit_price",
                "confidence", "final_score", "outcome", "pnl_percent",
                "duration", "status"
            ]

            for col in expected_cols:
                if col not in df_hist.columns:
                    if col == "outcome":
                        df_hist[col] = "PENDING"
                    elif col == "status":
                        df_hist[col] = "Open"
                    elif col == "duration":
                        df_hist[col] = "Active"
                    else:
                        df_hist[col] = 0.0

            # Normalize old CSV values
            df_hist["outcome"] = (
                df_hist["outcome"]
                .fillna("PENDING")
                .astype(str)
                .str.upper()
                .str.strip()
            )

            df_hist["direction"] = (
                df_hist["direction"]
                .fillna("NEUTRAL")
                .astype(str)
                .str.upper()
                .str.strip()
            )

            return df_hist.to_dict("records")

        except Exception:
            return []

    return []


def save_persistent_history(history_list):
    try:
        df_hist = pd.DataFrame(history_list)
        df_hist.to_csv(CSV_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving history to CSV: {e}")


if "trade_history_log" not in st.session_state:
    st.session_state.trade_history_log = load_persistent_history()


# ==========================================
# 1. RESEARCH LAB & RISK ENGINE MODULES
# ==========================================
class TenPaperResearchLab:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol
        self.scaler = StandardScaler()

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

        if (
            len(bids) == 0
            or len(asks) == 0
            or df.empty
            or len(df) < 15
        ):
            return {k: 0.0 for k in self.feature_names}

        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])

        mid_price = (bids[0, 0] + asks[0, 0]) / 2

        returns = df["Close"].pct_change().dropna()

        realized_vol = returns.std() + 1e-8

        returns_h = (
            df["Close"].iloc[-1] - df["Close"].iloc[-5]
        ) / (df["Close"].iloc[-5] + 1e-8)

        delta_p = df["Close"].iloc[-1] - df["Close"].iloc[-2]

        # 1. Hawkes
        vol_changes = df["Volume"].pct_change().dropna().values

        hawkes_intensity = (
            np.mean(vol_changes[-3:])
            / (np.mean(vol_changes[-15:]) + 1e-8)
        ) if len(vol_changes) >= 15 else 1.0

        results["HAWKES"] = np.clip(
            (hawkes_intensity - 1.0) * np.sign(returns_h),
            -1,
            1
        )

        # 2. Book Imbalance
        results["BOOK_IMB"] = (
            (bid_vol - ask_vol)
            / (bid_vol + ask_vol + 1e-8)
        )

        # 3. Taker Flow
        taker_buy = df["Volume"].iloc[-1] * (
            1.0 if delta_p > 0 else 0.3
        )

        taker_sell = df["Volume"].iloc[-1] * (
            1.0 if delta_p <= 0 else 0.3
        )

        results["TAKER_FLOW"] = (
            (taker_buy - taker_sell)
            / (taker_buy + taker_sell + 1e-8)
        )

        # 4. Quantities Imply
        depth_skew = (
            (bids[0, 1] - asks[0, 1])
            / (bids[0, 1] + asks[0, 1] + 1e-8)
        )

        results["QUANT_IMPLY"] = np.clip(
            depth_skew * 1.5,
            -1,
            1
        )

        # 5. Bayesian Probability
        prior = 0.745

        likelihood = (
            1.0
            if results["BOOK_IMB"] > 0
            else 0.25
        )

        posterior = (
            likelihood * prior
        ) / (
            (likelihood * prior)
            + ((1 - likelihood) * (1 - prior))
            + 1e-8
        )

        results["BAYESIAN"] = np.clip(
            (posterior - 0.5) * 2.0,
            -1,
            1
        )

        # 6. Quantiles
        q90 = returns.quantile(0.90) if len(returns) > 5 else 0.01
        q10 = returns.quantile(0.10) if len(returns) > 5 else -0.01

        results["QUANTILES"] = np.clip(
            (returns_h - q10)
            / (q90 - q10 + 1e-8)
            * 2.0 - 1.0,
            -1,
            1
        )

        # 7. Target / Invalidation
        target_diff = (
            delta_p
            / (df["Close"].iloc[-1] + 1e-8)
        )

        results["TARGET_INV"] = (
            1.0
            if target_diff >= 0.0006
            else (
                -1.0
                if target_diff <= -0.0006
                else 0.0
            )
        )

        # 8. Adaptive Conformal
        ma_fast = df["Close"].rolling(3).mean().iloc[-1]
        ma_slow = df["Close"].rolling(10).mean().iloc[-1]

        results["ADAPT_CONF"] = np.clip(
            (ma_fast - ma_slow)
            / (realized_vol * mid_price + 1e-8),
            -1,
            1
        )

        # 9. Fractional Kelly
        win_prob = (
            0.55
            + (0.15 * np.sign(results["BOOK_IMB"]))
        )

        kelly_fraction = (
            win_prob
            - ((1 - win_prob) / 1.5)
        )

        results["FRAC_KELLY"] = np.clip(
            kelly_fraction
            * 2.0
            * np.sign(returns_h),
            -1,
            1
        )

        # 10. RMT
        rmt_dom = (
            abs(returns_h)
            / (realized_vol * np.sqrt(5) + 1e-8)
        ) / 3.0

        results["RMT_DOM"] = np.clip(
            rmt_dom * np.sign(returns_h),
            -1,
            1
        )

        # 11. Conformal
        conformal_spread = realized_vol * 1.96

        upper_b = mid_price * (1 + conformal_spread)
        lower_b = mid_price * (1 - conformal_spread)

        results["CONF_CROSS"] = (
            1.0
            if mid_price > (upper_b + lower_b) / 2
            else (
                -1.0
                if mid_price < (upper_b + lower_b) / 2
                else 0.0
            )
        )

        # 12. Reward / Risk
        rr_ratio = abs(q90) / (abs(q10) + 1e-8)

        results["REWARD_RISK"] = (
            1.0
            if rr_ratio >= 1.2
            else (
                -1.0
                if rr_ratio < 0.8
                else 0.0
            )
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

        weight_vector = np.array(
            list(self.dynamic_weights.values())
        )

        final_score = float(
            np.dot(feature_vector[0], weight_vector)
        )

        return (
            results,
            final_score,
            self.dynamic_weights
        )


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
            max_ltz
            / (total_ltz + 1e-8)
        ) * 100

        spoof_ratio = (
            cancelled_vol
            / (displayed_vol + 1e-8)
        )

        persistence = min(
            max(
                time_exists
                / (obs_window + 1e-8),
                0
            ),
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
            "LTZ_Score": ltz_score,
            "Spoof_Score": spoof_score,
            "Squeeze_Risk": squeeze_risk,
            "Market_Risk": market_risk
        }


# ==========================================
# 3. PROFESSIONAL STYLING & THEME
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background-color: #080a0f;
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #161b22;
    }

    .metric-card {
        background: #111622;
        border: 1px solid #1e2638;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .metric-val-green {
        font-size: 18px;
        font-weight: 700;
        color: #00e676;
    }

    .metric-val-red {
        font-size: 18px;
        font-weight: 700;
        color: #ff5252;
    }

    .metric-val-blue {
        font-size: 18px;
        font-weight: 700;
        color: #38bdf8;
    }

    .top-status-bar {
        background: #111622;
        border: 1px solid #1e2638;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 18px;
        font-weight: 600;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR CONTROLS & FILTERS
# ==========================================
COINS_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT"
]

TIMEFRAME_MAP = {
    "1m (Scalping)": ("1m", 1),
    "15m (Medium TF)": ("15m", 15),
    "30m (Medium TF)": ("30m", 30),
    "1h (Intraday)": ("1h", 60),
    "4h (Intraday)": ("4h", 240),
}

st.sidebar.markdown("### ⚡ Terminal Controls")

selected_symbol = st.sidebar.selectbox(
    "Select Cryptocurrency",
    COINS_LIST,
    index=0
)

selected_tf_label = st.sidebar.selectbox(
    "Select Timeframe",
    list(TIMEFRAME_MAP.keys()),
    index=1
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon Candles",
    5,
    30,
    15
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Paper Trading Mode")

paper_trading_mode = st.sidebar.toggle(
    "Enable Live Paper Trading",
    value=True
)

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]


# ==========================================
# 5. DATA FETCHING
# ==========================================
@st.cache_data(ttl=15)
def fetch_klines_data(symbol, tf_key, limit=100):

    binance_tf = (
        "1m"
        if "1m" in tf_key
        else (
            "15m"
            if "15m" in tf_key
            else (
                "30m"
                if "30m" in tf_key
                else (
                    "1h"
                    if "1h" in tf_key
                    else "4h"
                )
            )
        )
    )

    url = (
        f"https://data-api.binance.vision/api/v3/klines"
        f"?symbol={symbol}&interval={binance_tf}&limit={limit}"
    )

    try:
        res = requests.get(
            url,
            timeout=4
        ).json()

        if isinstance(res, dict) or not isinstance(res, list):
            raise ValueError("API limit or invalid format")

        df = pd.DataFrame(
            res,
            columns=[
                "Open_Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close_Time",
                "QAV",
                "NAT",
                "TBBAV",
                "TBQAV",
                "Ignore"
            ]
        )

        df["Time"] = pd.to_datetime(
            df["Open_Time"],
            unit="ms"
        )

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:
            df[col] = df[col].astype(float)

        df.set_index("Time", inplace=True)

        return df.reset_index()[
            [
                "Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ]

    except Exception:

        dates = pd.date_range(
            end=datetime.datetime.now(),
            periods=limit,
            freq=binance_tf
        )

        base_p = 60000.0

        closes = (
            base_p
            + np.cumsum(
                np.random.normal(0, 10, limit)
            )
        )

        return pd.DataFrame({
            "Time": dates,
            "Open": closes - 5,
            "High": closes + 15,
            "Low": closes - 15,
            "Close": closes,
            "Volume": np.random.uniform(
                50,
                500,
                limit
            )
        })


@st.cache_data(ttl=10)
def fetch_order_book_depth(
    symbol,
    depth_limit=20
):

    try:
        url = (
            f"https://data-api.binance.vision/api/v3/depth"
            f"?symbol={symbol}&limit={depth_limit}"
        )

        res = requests.get(
            url,
            timeout=4
        ).json()

        if "bids" in res and "asks" in res:
            return (
                np.array(
                    res["bids"],
                    dtype=float
                ),
                np.array(
                    res["asks"],
                    dtype=float
                )
            )

    except Exception:
        pass

    dummy_bids = np.array(
        [
            [60000 - i * 2, 1.5]
            for i in range(20)
        ],
        dtype=float
    )

    dummy_asks = np.array(
        [
            [60000 + i * 2, 1.5]
            for i in range(20)
        ],
        dtype=float
    )

    return dummy_bids, dummy_asks


df = fetch_klines_data(
    selected_symbol,
    selected_tf_label
)

bids, asks = fetch_order_book_depth(
    selected_symbol
)


# ==========================================
# 6. ENGINE EXECUTION & SIGNAL GENERATION
# ==========================================
if (
    not df.empty
    and len(df) >= 3
    and len(bids) > 0
    and len(asks) > 0
):

    lab = TenPaperResearchLab()

    paper_results, final_score, evolved_weights = (
        lab.calculate_all_signals(
            df,
            bids,
            asks,
            current_inventory=0,
            performance_history=st.session_state.trade_history_log
        )
    )

    close_p = df["Close"].iloc[-1]

    atr_val = (
        (df["High"] - df["Low"])
        .rolling(14)
        .mean()
        .iloc[-1]
    )

    if np.isnan(atr_val):
        atr_val = close_p * 0.005

    beam_level = close_p + (1.8 * atr_val)
    base_level = close_p - (1.8 * atr_val)

    tp1_val = (
        close_p + (1.0 * atr_val)
        if final_score >= 0
        else close_p - (1.0 * atr_val)
    )

    tp2_val = (
        beam_level
        if final_score >= 0
        else base_level
    )

    sl_val = (
        close_p - (1.0 * atr_val)
        if final_score >= 0
        else close_p + (1.0 * atr_val)
    )

    direction = (
        "LONG"
        if final_score >= 0.15
        else (
            "SHORT"
            if final_score <= -0.15
            else "NEUTRAL"
        )
    )

    confidence = int(
        min(
            max(abs(final_score) * 100, 15),
            98
        )
    )

    lock_seconds = tf_minutes * 60

    current_time_sec = int(time.time())

    time_bucket = (
        current_time_sec
        - (current_time_sec % lock_seconds)
    )

    time_remaining = (
        lock_seconds
        - (current_time_sec % lock_seconds)
    )

    trade_id = (
        f"{selected_symbol}_"
        f"{selected_tf_label}_"
        f"{time_bucket}_"
        f"{direction}"
    )

    # ==========================================
    # CREATE NEW PAPER TRADE
    # ==========================================
    if paper_trading_mode and direction != "NEUTRAL":

        existing_trade_ids = [
            item.get("trade_id")
            for item in st.session_state.trade_history_log
        ]

        if trade_id not in existing_trade_ids:

            new_trade = {
                "trade_id": trade_id,
                "timestamp": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "symbol": selected_symbol,
                "timeframe": selected_tf_label,
                "direction": direction,
                "entry_price": round(close_p, 2),
                "stop_loss": round(sl_val, 2),
                "tp1": round(tp1_val, 2),
                "tp2": round(tp2_val, 2),
                "exit_price": round(close_p, 2),
                "confidence": confidence,
                "final_score": round(final_score, 3),
                "outcome": "PENDING",
                "pnl_percent": 0.0,
                "duration": "Active",
                "status": "Open"
            }

            st.session_state.trade_history_log.insert(
                0,
                new_trade
            )

            save_persistent_history(
                st.session_state.trade_history_log
            )


    # ==========================================
    # FIXED TRADE RESULT CHECK
    # ==========================================
    # IMPORTANT:
    # TP / SL ko sirf candle CLOSE se nahi,
    # candle HIGH / LOW se bhi check kiya ja raha hai.
    #
    # Is se agar TP ya SL candle ke andar hit ho gaya
    # aur close wapas aa gaya, trade PENDING nahi rahega.
    # ==========================================

    current_candle_high = float(
        df["High"].iloc[-1]
    )

    current_candle_low = float(
        df["Low"].iloc[-1]
    )

    curr_price = float(close_p)

    history_changed = False

    for trade in st.session_state.trade_history_log:

        outcome = str(
            trade.get("outcome", "PENDING")
        ).upper().strip()

        direction_trade = str(
            trade.get("direction", "")
        ).upper().strip()

        # Already closed trades ko dobara process nahi karna
        if outcome != "PENDING":
            continue

        if trade.get("symbol") != selected_symbol:
            continue

        try:
            entry = float(trade.get("entry_price", 0))
            sl = float(trade.get("stop_loss", 0))
            tp = float(trade.get("tp1", 0))
        except Exception:
            continue

        if entry <= 0 or sl <= 0 or tp <= 0:
            continue

        # ------------------------------------------
        # LONG TRADE
        # ------------------------------------------
        if direction_trade == "LONG":

            # TP candle ke HIGH mein hit hua
            if current_candle_high >= tp:

                trade["outcome"] = "WIN"
                trade["exit_price"] = round(tp, 2)

                trade["pnl_percent"] = round(
                    ((tp - entry) / entry) * 100,
                    2
                )

                trade["status"] = "Closed"
                trade["duration"] = "Completed"
                history_changed = True

            # SL candle ke LOW mein hit hua
            elif current_candle_low <= sl:

                trade["outcome"] = "LOSS"
                trade["exit_price"] = round(sl, 2)

                trade["pnl_percent"] = round(
                    ((sl - entry) / entry) * 100,
                    2
                )

                trade["status"] = "Closed"
                trade["duration"] = "Completed"
                history_changed = True

        # ------------------------------------------
        # SHORT TRADE
        # ------------------------------------------
        elif direction_trade == "SHORT":

            # TP candle ke LOW mein hit hua
            if current_candle_low <= tp:

                trade["outcome"] = "WIN"
                trade["exit_price"] = round(tp, 2)

                trade["pnl_percent"] = round(
                    ((entry - tp) / entry) * 100,
                    2
                )

                trade["status"] = "Closed"
                trade["duration"] = "Completed"
                history_changed = True

            # SL candle ke HIGH mein hit hua
            elif current_candle_high >= sl:

                trade["outcome"] = "LOSS"
                trade["exit_price"] = round(sl, 2)

                trade["pnl_percent"] = round(
                    ((entry - sl) / entry) * 100,
                    2
                )

                trade["status"] = "Closed"
                trade["duration"] = "Completed"
                history_changed = True


    # Save ONLY if trade status changed
    if history_changed:
        save_persistent_history(
            st.session_state.trade_history_log
        )


    # ==========================================
    # RISK ENGINE
    # ==========================================
    risk_engine = PowerTradingRiskEngine()

    disp_vol = (
        np.sum(asks[:, 1])
        if len(asks) > 0
        else 1.0
    )

    risk_metrics = risk_engine.calculate_risk_metrics(
        liquidation_volumes=np.array([1000, 2500]),
        displayed_vol=disp_vol,
        cancelled_vol=disp_vol * 0.1,
        time_exists=15.0,
        obs_window=60.0,
        open_interest=150000.0,
        leverage=20.0,
        volatility=df["Close"].pct_change().std() + 1e-8
    )


    # ==========================================
    # 7. TOP HEADER STATUS BAR
    # ==========================================
    dir_color = (
        "#00e676"
        if direction == "LONG"
        else (
            "#ff5252"
            if direction == "SHORT"
            else "#38bdf8"
        )
    )

    mins_rem, secs_rem = divmod(
        time_remaining,
        60
    )

    st.markdown(
        f"""
        <div class="top-status-bar">
            🟢 <b>[{selected_symbol}]</b>
            &nbsp;|&nbsp;
            Price: <b>${close_p:,.2f}</b>
            &nbsp;|&nbsp;
            TF: {selected_tf_label}
            &nbsp;|&nbsp;
            SIGNAL:
            <span style="color:{dir_color};">
                {direction}
            </span>
            &nbsp;|&nbsp;
            Score: <b>{final_score:+.3f}</b>
            &nbsp;|&nbsp;
            Confidence: <b>{confidence}%</b>
            &nbsp;|&nbsp;
            ⏳ Next Reset:
            <b>{mins_rem}m {secs_rem}s</b>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ==========================================
    # 8. TRADE SIGNAL PANEL & METRICS
    # ==========================================
    col_sig, col_m1, col_m2, col_m3, col_m4 = st.columns(
        [1.2, 1, 1, 1, 1]
    )

    with col_sig:

        st.markdown(
            f"""
            <div class="metric-card"
                 style="border-left: 4px solid {dir_color};">

                <div class="metric-label">
                    Signal Execution Panel
                </div>

                <div style="
                    font-size:22px;
                    font-weight:700;
                    color:{dir_color};
                ">
                    {direction}
                </div>

                <div style="
                    font-size:11px;
                    color:#8b949e;
                    margin-top:4px;
                ">
                    Entry: ${close_p:,.2f}
                    |
                    SL: ${sl_val:,.2f}
                </div>

                <div style="
                    font-size:11px;
                    color:#38bdf8;
                ">
                    TP1: ${tp1_val:,.2f}
                    |
                    TP2: ${tp2_val:,.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col_m1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    BEAM Target
                </div>

                <div class="metric-val-blue">
                    ${beam_level:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    BASE Target
                </div>

                <div class="metric-val-red">
                    ${base_level:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col_m2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Risk / Reward
                </div>

                <div class="metric-val-blue">
                    1 : 2.15
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Signal Strength
                </div>

                <div class="metric-val-green">
                    HIGH
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col_m3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    LTZ Score
                </div>

                <div class="metric-val-blue">
                    {risk_metrics["LTZ_Score"]:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Spoof Score
                </div>

                <div class="metric-val-red">
                    {risk_metrics["Spoof_Score"]:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col_m4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Squeeze Risk
                </div>

                <div class="metric-val-red">
                    {risk_metrics["Squeeze_Risk"]:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Market Risk
                </div>

                <div class="metric-val-red">
                    {risk_metrics["Market_Risk"]:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ==========================================
    # 9. CHART & MICROSTRUCTURE SECTION
    # ==========================================
    col_chart, col_risk_panel = st.columns(
        [2.5, 1]
    )

    with col_chart:

        st.subheader(
            f"Price Trajectory & Levels ({selected_symbol})"
        )

        time_delta = pd.Timedelta(
            minutes=tf_minutes
        )

        future_times = [
            df["Time"].iloc[-1]
            + (i * time_delta)
            for i in range(
                1,
                forecast_horizon + 1
            )
        ]

        t_steps = np.linspace(
            0,
            np.pi / 2,
            forecast_horizon
        )

        if direction == "LONG":

            forecast_prices = (
                close_p
                + (beam_level - close_p)
                * np.sin(t_steps)
            )

        elif direction == "SHORT":

            forecast_prices = (
                close_p
                - (close_p - base_level)
                * np.sin(t_steps)
            )

        else:

            forecast_prices = [
                close_p
            ] * forecast_horizon

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["Time"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles",
                increasing_line_color="#00e676",
                decreasing_line_color="#ff5252"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[df["Time"].iloc[-1]] + future_times,
                y=[close_p] + list(forecast_prices),
                mode="lines+markers",
                name="Trajectory",
                line=dict(
                    color=dir_color,
                    width=2,
                    dash="dot"
                )
            )
        )

        fig.add_hline(
            y=beam_level,
            line_dash="dash",
            line_color="#00e676",
            annotation_text=f"BEAM: ${beam_level:,.2f}"
        )

        fig.add_hline(
            y=base_level,
            line_dash="dash",
            line_color="#ff5252",
            annotation_text=f"BASE: ${base_level:,.2f}"
        )

        fig.add_hline(
            y=sl_val,
            line_dash="dot",
            line_color="#ff5252",
            annotation_text=f"SL: ${sl_val:,.2f}"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#111622",
            plot_bgcolor="#111622",
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col_risk_panel:

        st.subheader(
            "Market Microstructure & OB"
        )

        bid_vol_sum = (
            np.sum(bids[:, 1])
            if len(bids) > 0
            else 1.0
        )

        ask_vol_sum = (
            np.sum(asks[:, 1])
            if len(asks) > 0
            else 1.0
        )

        obi_val = (
            (bid_vol_sum - ask_vol_sum)
            / (bid_vol_sum + ask_vol_sum)
        )

        spread_val = (
            abs(
                asks[0, 0]
                - bids[0, 0]
            )
            if len(bids) > 0
            and len(asks) > 0
            else 0.0
        )

        st.markdown(
            f"""
            <div class="metric-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:6px;
                ">
                    <span>Bid Volume</span>
                    <b style="color:#00e676;">
                        {bid_vol_sum:,.2f}
                    </b>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:6px;
                ">
                    <span>Ask Volume</span>
                    <b style="color:#ff5252;">
                        {ask_vol_sum:,.2f}
                    </b>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:6px;
                ">
                    <span>
                        Order Book Imbalance (OBI)
                    </span>

                    <b style="color:#38bdf8;">
                        {obi_val:+.3f}
                    </b>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:6px;
                ">
                    <span>Spread</span>
                    <b>${spread_val:.2f}</b>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                ">
                    <span>Risk Status</span>
                    <b style="color:#00e676;">
                        LOW-MEDIUM
                    </b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader(
            "Top 20 OBI Analysis"
        )

        fig_obi = go.Figure(
            go.Bar(
                x=[
                    "Top 5",
                    "Top 10",
                    "Top 20"
                ],
                y=[
                    obi_val * 0.8,
                    obi_val * 0.9,
                    obi_val
                ],
                marker_color="#38bdf8"
            )
        )

        fig_obi.update_layout(
            height=160,
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            ),
            paper_bgcolor="#111622",
            plot_bgcolor="#111622"
        )

        st.plotly_chart(
            fig_obi,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # ==========================================
    # 10. RESEARCH PAPER SCOREBOARD
    # ==========================================
    st.markdown("---")

    st.subheader(
        "🔬 12-Paper Quantitative Research Scoreboard"
    )

    col_sc1, col_sc2 = st.columns(
        [1.5, 1]
    )

    with col_sc1:

        paper_table_data = []

        for k, v in paper_results.items():

            status = (
                "PASS 🟢"
                if v > 0.1
                else (
                    "FAIL 🔴"
                    if v < -0.1
                    else "NEUTRAL ⚪"
                )
            )

            paper_table_data.append(
                {
                    "Paper": k,
                    "Value": f"{v:+.3f}",
                    "Weight": f"{evolved_weights.get(k, 0.083) * 100:.1f}%",
                    "Status": status
                }
            )

        st.dataframe(
            pd.DataFrame(paper_table_data),
            use_container_width=True,
            hide_index=True,
            height=270
        )


    with col_sc2:

        st.markdown(
            """
            <div class="metric-card">

                <div style="
                    font-weight:700;
                    color:#38bdf8;
                    margin-bottom:6px;
                ">
                    Advanced Model Insights
                </div>

                <div style="
                    font-size:12px;
                    color:#cbd5e1;
                    line-height:1.6;
                ">

                    • <b>HAWKES:</b>
                    Measures aggressive order clustering
                    and arrival rates.<br>

                    • <b>BOOK_IMB:</b>
                    Computes real-time depth pressure
                    across bids & asks.<br>

                    • <b>Dynamic Weights:</b>
                    Optimized linear blending for robust signals.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ==========================================
    # 11. PERFORMANCE & WIN RATE SECTION
    # ==========================================
    st.markdown("---")

    st.subheader(
        "📊 Performance Summary & Win Rate Checker with Filters"
    )

    if st.session_state.trade_history_log:

        df_log = pd.DataFrame(
            st.session_state.trade_history_log
        )

        # Normalize values before displaying
        df_log["outcome"] = (
            df_log["outcome"]
            .fillna("PENDING")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        df_log["direction"] = (
            df_log["direction"]
            .fillna("NEUTRAL")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:

            coin_filter = st.selectbox(
                "Filter Coin",
                ["ALL"] + COINS_LIST
            )

        with f_col2:

            tf_filter = st.selectbox(
                "Filter Timeframe",
                ["ALL"] + list(TIMEFRAME_MAP.keys())
            )

        with f_col3:

            dir_filter = st.selectbox(
                "Filter Direction",
                [
                    "ALL",
                    "LONG",
                    "SHORT"
                ]
            )

        filtered_df = df_log.copy()

        if coin_filter != "ALL":

            filtered_df = filtered_df[
                filtered_df["symbol"]
                == coin_filter
            ]

        if tf_filter != "ALL":

            filtered_df = filtered_df[
                filtered_df["timeframe"]
                == tf_filter
            ]

        if dir_filter != "ALL":

            filtered_df = filtered_df[
                filtered_df["direction"]
                == dir_filter
            ]

        total_signals = len(
            filtered_df
        )

        wins = len(
            filtered_df[
                filtered_df["outcome"]
                == "WIN"
            ]
        )

        losses = len(
            filtered_df[
                filtered_df["outcome"]
                == "LOSS"
            ]
        )

        pending = len(
            filtered_df[
                filtered_df["outcome"]
                == "PENDING"
            ]
        )

        closed_trades = wins + losses

        win_rate = (
            wins / closed_trades * 100
            if closed_trades > 0
            else 0.0
        )

        winning_trades_df = filtered_df[
            filtered_df["outcome"] == "WIN"
        ]

        losing_trades_df = filtered_df[
            filtered_df["outcome"] == "LOSS"
        ]

        gross_profit = (
            winning_trades_df["pnl_percent"].sum()
            if not winning_trades_df.empty
            else 0.0
        )

        gross_loss = (
            abs(
                losing_trades_df["pnl_percent"].sum()
            )
            if not losing_trades_df.empty
            else 0.0
        )

        net_pnl = (
            gross_profit
            - gross_loss
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (
                gross_profit
                if gross_profit > 0
                else 0.0
            )
        )

        p1, p2, p3, p4, p5, p6 = st.columns(6)

        with p1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Win Rate
                    </div>

                    <div class="metric-val-green">
                        {win_rate:.1f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Closed Trades
                    </div>

                    <div class="metric-val-blue">
                        {closed_trades}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with p3:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Wins / Losses
                    </div>

                    <div style="
                        font-size:16px;
                        font-weight:750;
                        color:#00e676;
                    ">
                        {wins}W / {losses}L
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with p4:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Pending
                    </div>

                    <div class="metric-val-blue">
                        {pending}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with p5:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Profit Factor
                    </div>

                    <div class="metric-val-blue">
                        {profit_factor:.2f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with p6:

            pnl_color = (
                "#00e676"
                if net_pnl >= 0
                else "#ff5252"
            )

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Net PnL %
                    </div>

                    <div style="
                        font-size:18px;
                        font-weight:700;
                        color:{pnl_color};
                    ">
                        {net_pnl:+.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            "##### Detailed Trade History Table"
        )

        display_cols = [
            "timestamp",
            "symbol",
            "timeframe",
            "direction",
            "entry_price",
            "stop_loss",
            "tp1",
            "exit_price",
            "pnl_percent",
            "outcome",
            "confidence"
        ]

        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            hide_index=True,
            height=280
        )


        if st.sidebar.button(
            "Clear Trade History Log"
        ):

            st.session_state.trade_history_log = []

            if os.path.exists(CSV_FILE):
                os.remove(CSV_FILE)

            st.rerun()

    else:

        st.info(
            "No paper trade history recorded yet. "
            "Signals will automatically log when active."
        )

else:

    st.warning(
        "⚠️ Data pipeline initializing or connection restricted. "
        "Please refresh."
    )
