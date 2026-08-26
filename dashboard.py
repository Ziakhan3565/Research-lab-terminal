import os
import json
import time
import uuid

import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_autorefresh import st_autorefresh

from research_engine import ResearchEngine
from signal_engine import SignalEngine


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="ZIA Quant Research Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(
    interval=5000,
    key="zia_refresh",
)

BINANCE_BASE = (
    "https://api.binance.com"
)

MODEL_FILE = (
    "xgboost_obi_model.pkl"
)

SIGNAL_FILE = (
    "signal_history.csv"
)

FEEDBACK_FILE = (
    "trade_feedback.csv"
)

CONFIG_FILE = (
    "config.json"
)


COINS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XMRUSDT",
    "XRPUSDT",
    "TAOUSDT",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: #080a0f;
}

.metric-card {
    background: #111622;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
}

.signal-box {
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    background: #111622;
}

.signal-title {
    font-size: 13px;
    color: #94a3b8;
    text-transform: uppercase;
}

.signal-value {
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.small-label {
    color: #94a3b8;
    font-size: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# BINANCE API
# ============================================================

def binance_get(
    endpoint,
    params=None,
):

    try:

        response = requests.get(
            BINANCE_BASE + endpoint,
            params=params,
            timeout=5,
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        st.error(
            f"Binance API error: {e}"
        )

        return None


# ============================================================
# PRICE
# ============================================================

def get_price(
    symbol,
):

    data = binance_get(
        "/api/v3/ticker/price",
        {
            "symbol": symbol,
        },
    )

    if not data:
        return None

    return float(
        data["price"]
    )


# ============================================================
# ORDER BOOK
# ============================================================

def get_orderbook(
    symbol,
    limit=50,
):

    data = binance_get(
        "/api/v3/depth",
        {
            "symbol": symbol,
            "limit": limit,
        },
    )

    if not data:
        return None, None

    bids = np.array(
        data["bids"],
        dtype=float,
    )

    asks = np.array(
        data["asks"],
        dtype=float,
    )

    return bids, asks


# ============================================================
# KLINES
# ============================================================

def get_klines(
    symbol,
    interval="1m",
    limit=200,
):

    data = binance_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
    )

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


# ============================================================
# BINANCE AGG TRADES
# ============================================================

def get_taker_flow(
    symbol,
    limit=100,
):

    data = binance_get(
        "/api/v3/aggTrades",
        {
            "symbol": symbol,
            "limit": limit,
        },
    )

    if not data:
        return 0.0

    buy = 0.0
    sell = 0.0

    for trade in data:

        qty = float(
            trade["q"]
        )

        # m=True means buyer is maker.
        # Therefore aggressor is seller.
        if trade["m"]:
            sell += qty
        else:
            buy += qty

    total = (
        buy
        +
        sell
        +
        1e-8
    )

    return float(
        np.clip(
            (buy - sell)
            / total,
            -1,
            1,
        )
    )


# ============================================================
# OBI
# ============================================================

def calculate_obi(
    bids,
    asks,
    levels=20,
):

    bid = float(
        bids[:levels, 1].sum()
    )

    ask = float(
        asks[:levels, 1].sum()
    )

    total = (
        bid
        +
        ask
        +
        1e-8
    )

    return (
        bid - ask
    ) / total


# ============================================================
# OFI
# ============================================================

def calculate_ofi(
    current_bid,
    current_ask,
):

    previous = st.session_state.get(
        "previous_depth"
    )

    current_bid_volume = float(
        current_bid[:20, 1].sum()
    )

    current_ask_volume = float(
        current_ask[:20, 1].sum()
    )

    if previous is None:

        st.session_state[
            "previous_depth"
        ] = (
            current_bid_volume,
            current_ask_volume,
        )

        return 0.0

    previous_bid_volume = (
        previous[0]
    )

    previous_ask_volume = (
        previous[1]
    )

    raw = (
        current_bid_volume
        -
        previous_bid_volume
        -
        (
            current_ask_volume
            -
            previous_ask_volume
        )
    )

    scale = (
        current_bid_volume
        +
        current_ask_volume
        +
        1e-8
    )

    ofi = raw / scale

    st.session_state[
        "previous_depth"
    ] = (
        current_bid_volume,
        current_ask_volume,
    )

    return float(
        np.clip(
            ofi,
            -1,
            1,
        )
    )


# ============================================================
# VWAP
# ============================================================

def calculate_vwap(
    df,
):

    if df.empty:
        return 0.0

    pv = (
        df["Close"]
        *
        df["Volume"]
    )

    volume = df[
        "Volume"
    ].sum()

    if volume <= 0:
        return float(
            df["Close"].iloc[-1]
        )

    return float(
        pv.sum()
        /
        volume
    )


# ============================================================
# ML MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(
        MODEL_FILE
    ):
        return None

    try:

        return joblib.load(
            MODEL_FILE
        )

    except Exception:
        return None


model = load_model()


# ============================================================
# MODEL FEATURE VECTOR
# ============================================================

def build_ml_vector(
    bids,
    asks,
    df,
    research,
):

    bid20 = float(
        bids[:20, 1].sum()
    )

    ask20 = float(
        asks[:20, 1].sum()
    )

    bid50 = float(
        bids[:50, 1].sum()
    )

    ask50 = float(
        asks[:50, 1].sum()
    )

    obi20 = calculate_obi(
        bids,
        asks,
        20,
    )

    obi50 = calculate_obi(
        bids,
        asks,
        50,
    )

    best_bid = float(
        bids[0, 0]
    )

    best_ask = float(
        asks[0, 0]
    )

    spread = (
        best_ask
        -
        best_bid
    )

    mid = (
        best_bid
        +
        best_ask
    ) / 2

    spread_pct = (
        spread
        /
        (
            mid
            +
            1e-8
        )
    )

    vector = [

        bid20,
        ask20,

        bid50,
        ask50,

        obi20,
        obi50,

        spread,
        spread_pct,

        research.get(
            "BOOK_IMB",
            0,
        ),

        research.get(
            "OFI",
            0,
        ),

        research.get(
            "TAKER_FLOW",
            0,
        ),

        research.get(
            "QUANT_IMPLY",
            0,
        ),

        research.get(
            "ADAPT_CONF",
            0,
        ),

        research.get(
            "BAYESIAN",
            0,
        ),

        research.get(
            "FOURIER_TREND",
            0,
        ),

        research.get(
            "EMA_TREND",
            0,
        ),

        research.get(
            "VWAP_DISTANCE",
            0,
        ),

        research.get(
            "VOLATILITY",
            0,
        ),
    ]

    return np.array(
        vector,
        dtype=float,
    ).reshape(
        1,
        -1,
    )


# ============================================================
# ML PREDICTION
# ============================================================

def predict_ml(
    vector,
):

    if model is None:

        return (
            0.5,
            0.5,
        )

    try:

        probabilities = (
            model.predict_proba(
                vector
            )[0]
        )

        short_probability = float(
            probabilities[0]
        )

        long_probability = float(
            probabilities[1]
        )

        return (
            long_probability,
            short_probability,
        )

    except Exception as e:

        st.warning(
            f"ML prediction unavailable: {e}"
        )

        return (
            0.5,
            0.5,
        )


# ============================================================
# SIGNAL FILE
# ============================================================

def append_signal(
    row,
):

    df = pd.DataFrame(
        [row]
    )

    if os.path.exists(
        SIGNAL_FILE
    ):

        df.to_csv(
            SIGNAL_FILE,
            mode="a",
            header=False,
            index=False,
        )

    else:

        df.to_csv(
            SIGNAL_FILE,
            index=False,
        )


# ============================================================
# CONFIG
# ============================================================

def save_config(
    config,
):

    with open(
        CONFIG_FILE,
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )


def load_config():

    if not os.path.exists(
        CONFIG_FILE
    ):

        return {
            "is_running": False,
            "leverage": 5,
            "trade_amount_usdt": 10,
            "selected_coins": [],
        }

    try:

        with open(
            CONFIG_FILE,
            "r",
        ) as f:

            return json.load(
                f
            )

    except Exception:

        return {
            "is_running": False,
            "leverage": 5,
            "trade_amount_usdt": 10,
            "selected_coins": [],
        }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚡ ZIA QUANT ENGINE"
)

symbol = st.sidebar.selectbox(
    "Coin",
    COINS,
)

mode = st.sidebar.selectbox(
    "Trading Mode",
    [
        "SCALPING",
        "INTRADAY",
    ],
)

leverage = st.sidebar.number_input(
    "MEXC Leverage",
    min_value=1,
    max_value=50,
    value=5,
)

trade_amount = st.sidebar.number_input(
    "Trade Amount USDT",
    min_value=1.0,
    value=10.0,
)

bot_enabled = st.sidebar.toggle(
    "Enable Trading Bot",
    value=False,
)

selected_coins = st.sidebar.multiselect(
    "Bot Coins",
    COINS,
    default=[symbol],
)

save_config(
    {
        "is_running": bot_enabled,
        "leverage": leverage,
        "trade_amount_usdt": trade_amount,
        "selected_coins": selected_coins,
    }
)


# ============================================================
# MAIN DATA
# ============================================================

st.title(
    "ZIA Quantitative Research & Trading Terminal"
)

st.caption(
    "BINANCE = Market Data / Order Book | "
    "MEXC = Trade Execution"
)


price = get_price(
    symbol
)

bids, asks = get_orderbook(
    symbol,
    50,
)

df = get_klines(
    symbol,
    "1m",
    200,
)


if (
    price is None
    or bids is None
    or asks is None
    or df.empty
):

    st.error(
        "Live Binance data unavailable."
    )

    st.stop()


# ============================================================
# RESEARCH ENGINE
# ============================================================

research_engine = (
    ResearchEngine()
)

signal_engine = (
    SignalEngine()
)


prices = df[
    "Close"
].values

volumes = df[
    "Volume"
].values

obi20 = calculate_obi(
    bids,
    asks,
    20,
)

obi50 = calculate_obi(
    bids,
    asks,
    50,
)

ofi = calculate_ofi(
    bids,
    asks,
)

taker_flow = get_taker_flow(
    symbol
)

best_bid_size = float(
    bids[0, 1]
)

best_ask_size = float(
    asks[0, 1]
)

research = (
    research_engine.calculate_all(
        prices=prices,
        volumes=volumes,
        bid20=float(
            bids[:20, 1].sum()
        ),
        ask20=float(
            asks[:20, 1].sum()
        ),
        best_bid_size=best_bid_size,
        best_ask_size=best_ask_size,
        previous_bid20=None,
        previous_ask20=None,
        buy_volume=max(
            taker_flow,
            0,
        ),
        sell_volume=max(
            -taker_flow,
            0,
        ),
    )
)

# Override OFI with live Binance depth delta
research[
    "OFI"
] = ofi


# ============================================================
# EMA / VWAP
# ============================================================

ema20 = (
    df["Close"]
    .ewm(
        span=20,
        adjust=False,
    )
    .mean()
    .iloc[-1]
)

ema50 = (
    df["Close"]
    .ewm(
        span=50,
        adjust=False,
    )
    .mean()
    .iloc[-1]
)

vwap = calculate_vwap(
    df
)


# ============================================================
# ML
# ============================================================

ml_vector = build_ml_vector(
    bids,
    asks,
    df,
    research,
)

long_probability, short_probability = (
    predict_ml(
        ml_vector
    )
)


# ============================================================
# FINAL SIGNAL
# ============================================================

result = signal_engine.generate(

    research_score=float(
        research[
            "RESEARCH_SCORE"
        ]
    ),

    long_probability=long_probability,

    short_probability=short_probability,

    price=price,

    ema20=ema20,

    ema50=ema50,

    vwap=vwap,

    obi=obi20,

    ofi=ofi,

    mode=mode,
)


signal = result[
    "signal"
]

final_score = result[
    "score"
]


# ============================================================
# STOP LOSS / TAKE PROFIT
# ============================================================

if "LONG" in signal:

    stop_loss = price * (
        1 - 0.004
    )

    take_profit = price * (
        1 + 0.006
    )

elif "SHORT" in signal:

    stop_loss = price * (
        1 + 0.004
    )

    take_profit = price * (
        1 - 0.006
    )

else:

    stop_loss = 0.0
    take_profit = 0.0


# ============================================================
# DASHBOARD
# ============================================================

st.subheader(
    f"{symbol} — Binance Live"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Price",
    f"{price:,.4f}",
)

c2.metric(
    "OBI Top 20",
    f"{obi20:.4f}",
)

c3.metric(
    "OBI Top 50",
    f"{obi50:.4f}",
)

c4.metric(
    "OFI",
    f"{ofi:.4f}",
)


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Research Score",
    f"{research['RESEARCH_SCORE']:.4f}",
)

c2.metric(
    "ML Long",
    f"{long_probability * 100:.2f}%",
)

c3.metric(
    "ML Short",
    f"{short_probability * 100:.2f}%",
)

c4.metric(
    "Final Score",
    f"{final_score:.4f}",
)


# ============================================================
# SIGNAL
# ============================================================

st.markdown(
    '<div class="signal-box">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="signal-title">'
    'FINAL TRADING SIGNAL'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="signal-value">'
    f'{signal}'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="small-label">
Research: {research['RESEARCH_SCORE']:.4f}
&nbsp;&nbsp;|&nbsp;&nbsp;
ML Long: {long_probability * 100:.1f}%
&nbsp;&nbsp;|&nbsp;&nbsp;
ML Short: {short_probability * 100:.1f}%
&nbsp;&nbsp;|&nbsp;&nbsp;
Final: {final_score:.4f}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# TRADE LEVELS
# ============================================================

if signal != "NO TRADE":

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Entry",
        f"{price:,.4f}",
    )

    c2.metric(
        "Stop Loss",
        f"{stop_loss:,.4f}",
    )

    c3.metric(
        "Take Profit",
        f"{take_profit:,.4f}",
    )

    c4.metric(
        "Validity",
        (
            "30 MIN"
            if mode == "SCALPING"
            else "8 HOURS"
        ),
    )


# ============================================================
# RESEARCH FEATURES
# ============================================================

st.subheader(
    "Research Lab Features"
)

feature_cols = st.columns(5)

features_to_show = [
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

for i, name in enumerate(
    features_to_show
):

    feature_cols[
        i % 5
    ].metric(
        name,
        f"{research.get(name, 0):.4f}",
    )


# ============================================================
# SIGNAL SAVE
# ============================================================

if (
    signal != "NO TRADE"
    and st.button(
        "💾 SAVE CURRENT SIGNAL"
    )
):

    validity = (
        1800
        if mode == "SCALPING"
        else 28800
    )

    created = time.time()

    trade_id = (
        str(uuid.uuid4())
    )

    signal_row = {

        "trade_id":
            trade_id,

        "timestamp":
            pd.Timestamp.utcnow(),

        "symbol":
            symbol,

        "mode":
            mode,

        "direction":
            signal,

        "entry_price":
            price,

        "stop_loss":
            stop_loss,

        "take_profit":
            take_profit,

        "expires_at":
            created + validity,

        "validity_seconds":
            validity,

        "research_score":
            research[
                "RESEARCH_SCORE"
            ],

        "ml_long_probability":
            long_probability,

        "ml_short_probability":
            short_probability,

        "final_score":
            final_score,

        "features_json":
            json.dumps(
                {
                    k: float(v)
                    for k, v in research.items()
                    if k != "RESEARCH_SCORE"
                }
            ),
    }

    append_signal(
        signal_row
    )

    st.success(
        f"Signal saved: {signal}"
    )


# ============================================================
# TRADE FEEDBACK
# ============================================================

st.subheader(
    "Completed Trade Learning"
)

if os.path.exists(
    FEEDBACK_FILE
):

    feedback = pd.read_csv(
        FEEDBACK_FILE
    )

    closed_count = len(
        feedback[
            feedback["status"]
            ==
            "CLOSED"
        ]
    )

    wins = len(
        feedback[
            feedback["outcome"]
            ==
            "WIN"
        ]
    )

    losses = len(
        feedback[
            feedback["outcome"]
            ==
            "LOSS"
        ]
    )

else:

    feedback = pd.DataFrame()

    closed_count = 0
    wins = 0
    losses = 0


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Closed Trades",
    closed_count,
)

c2.metric(
    "Wins",
    wins,
)

c3.metric(
    "Losses",
    losses,
)

c4.metric(
    "Next Retrain",
    (
        20
        -
        (
            closed_count
            % 20
        )
    )
    if closed_count % 20 != 0
    else 20,
)


if not feedback.empty:

    st.dataframe(
        feedback.tail(20).iloc[
            ::-1
        ],
        use_container_width=True,
    )


# ============================================================
# ACTIVE TRADE
# ============================================================

st.subheader(
    "Active MEXC Trade"
)

if os.path.exists(
    "active_trade.json"
):

    try:

        with open(
            "active_trade.json",
            "r",
        ) as f:

            active = json.load(
                f
            )

        st.json(
            active
        )

    except Exception:

        st.info(
            "Active trade file unreadable."
        )

else:

    st.info(
        "No active trade."
    )


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader(
    "Machine Learning Status"
)

if model is None:

    st.warning(
        "XGBoost model not found. "
        "Train the initial model first."
    )

else:

    st.success(
        "XGBoost model loaded."
    )

    st.write(
        f"Model file: `{MODEL_FILE}`"
    )

    if hasattr(
        model,
        "feature_importances_",
    ):

        importances = (
            model.feature_importances_
        )

        st.write(
            "Feature importance:"
        )

        st.dataframe(
            pd.DataFrame(
                {
                    "feature":
                    [
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
                    ],
                    "importance":
                        importances,
                }
            ).sort_values(
                "importance",
                ascending=False,
            ),
            use_container_width=True,
        )


# ============================================================
# DATA FLOW
# ============================================================

st.subheader(
    "System Flow"
)

st.code(
    """
BINANCE LIVE ORDER BOOK
        ↓
OBI + OFI
        ↓
Research Engine V2
        ↓
EMA + VWAP + Fourier
        ↓
Weighted Research Score
        ↓
XGBoost
        ↓
FINAL SIGNAL
        ↓
LONG / STRONG LONG
SHORT / STRONG SHORT
        ↓
signal_history.csv
        ↓
MEXC EXECUTION
        ↓
OPEN POSITION
        ↓
TP / SL / EXPIRY
        ↓
WIN / LOSS
        ↓
trade_feedback.csv
        ↓
20 CLOSED TRADES
        ↓
XGBoost CANDIDATE
        ↓
OLD MODEL vs CANDIDATE
        ↓
BETTER → ACCEPT
WORSE  → KEEP OLD
""",
)
