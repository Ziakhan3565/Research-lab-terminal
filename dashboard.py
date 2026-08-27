from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# ZIA RESEARCH LAB — LIVE ML MARKET TERMINAL
# SMOOTH LIVE REFRESH EDITION
#
# Existing XGBoost 7-feature model preserved exactly.
#
# Main improvements:
#   - Fragment based live refresh
#   - No full-page refresh during live updates
#   - Parallel Binance requests
#   - Longer cache for heavy HTF calculations
#   - Faster orderbook/ticker/trade fetching
#   - Existing ML schema preserved
#   - Existing signal logic preserved
# ============================================================

st.set_page_config(
    page_title="ZIA Research Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "xgboost_obi_model.pkl"
HISTORY_FILE = ROOT / "signal_history.csv"

REQUEST_TIMEOUT = 5

# ============================================================
# BINANCE ENDPOINTS
# ============================================================

FUTURES_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]

SPOT_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://data-api.binance.vision",
]

# ============================================================
# MARKETS
# ============================================================

COINS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "XLMUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "APTUSDT",
    "TAOUSDT",
    "XMRUSDT",
]

MODES = {
    "SCALPING": {
        "label": "5M / SCALP",
        "tf": "5m",
        "hold": "5–30 min",
        "refs": ["15m", "1h", "4h"],
    },
    "15M": {
        "label": "15M",
        "tf": "15m",
        "hold": "30–120 min",
        "refs": ["1h", "4h"],
    },
    "1H": {
        "label": "1H",
        "tf": "1h",
        "hold": "2–24 hours",
        "refs": ["4h", "1d"],
    },
    "4H": {
        "label": "4H",
        "tf": "4h",
        "hold": "12–72 hours",
        "refs": ["1d", "1w"],
    },
}

# ============================================================
# IMPORTANT:
# DO NOT CHANGE THIS ORDER.
# IT MUST MATCH THE TRAINED XGBOOST MODEL.
# ============================================================

MODEL_FEATURES = [
    "top20_bid_sum",
    "top20_ask_sum",
    "obi_top20",
    "spread",
    "bid_ask_ratio",
    "total_depth",
    "trend_signal",
]

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "symbol": "BTCUSDT",
    "mode": "SCALPING",
    "auto_refresh": True,
    "refresh_seconds": 5,
    "last_saved_key": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

.block-container{
    max-width:1750px;
    padding:1rem 1.6rem 2rem;
}

[data-testid="stSidebar"]{
    background:#080c13;
    border-right:1px solid #1b2636;
}

[data-testid="stSidebar"] *{
    color:#e8eef7;
}

.stApp{
    background:
        radial-gradient(
            circle at 78% -10%,
            #18263a 0%,
            #080c12 44%
        );
}

/* ========================================================
   HERO
======================================================== */

.hero{
    display:flex;
    justify-content:space-between;
    align-items:flex-end;
    margin-bottom:16px;
}

.brand{
    font-size:30px;
    font-weight:950;
    letter-spacing:.2px;
}

.brand span{
    color:#7889ff;
}

.subtitle{
    color:#8491a4;
    font-size:12px;
    margin-top:4px;
}

.status{
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:7px 11px;
    border:1px solid #284334;
    border-radius:999px;
    background:#0b1712;
    color:#8ee0b1;
    font-size:11px;
    font-weight:800;
}

.status.warn{
    border-color:#59451f;
    background:#181208;
    color:#e9c87e;
}

.dot{
    width:7px;
    height:7px;
    border-radius:50%;
    background:#49d18a;
    box-shadow:0 0 12px #49d18a;
}

.dot.warn{
    background:#e8bd58;
    box-shadow:0 0 12px #e8bd58;
}

/* ========================================================
   SIGNAL
======================================================== */

.signal{
    border-radius:18px;
    padding:22px 24px;
    background:
        linear-gradient(
            135deg,
            #111b2a,
            #0a1018
        );
    border:1px solid #29384d;
    margin-bottom:14px;
}

.signal.long{
    border-color:#23865f;
    box-shadow:
        0 0 35px rgba(35,170,111,.08);
}

.signal.short{
    border-color:#a04052;
    box-shadow:
        0 0 35px rgba(220,67,91,.08);
}

.signal.wait{
    border-color:#344256;
}

.signal-label{
    color:#8996a8;
    font-size:10px;
    font-weight:850;
    letter-spacing:1.2px;
}

.signal-name{
    font-size:42px;
    line-height:1.05;
    font-weight:950;
    margin:5px 0;
}

.signal-meta{
    color:#9aa7b8;
    font-size:12px;
}

.big-price{
    font-size:29px;
    font-weight:900;
}

/* ========================================================
   KPI
======================================================== */

.kpi{
    background:#0d141e;
    border:1px solid #1d2939;
    border-radius:13px;
    padding:14px 15px;
    min-height:86px;
}

.kpi-label{
    color:#7f8c9f;
    font-size:10px;
    text-transform:uppercase;
    letter-spacing:1px;
    font-weight:850;
}

.kpi-value{
    color:#f2f5fa;
    font-size:21px;
    font-weight:900;
    margin-top:5px;
}

.kpi-sub{
    color:#7d899b;
    font-size:10px;
    margin-top:2px;
}

/* ========================================================
   TRADE PLAN
======================================================== */

.trade{
    background:#0c131d;
    border:1px solid #1d2938;
    border-radius:12px;
    padding:13px;
    text-align:center;
}

.trade-label{
    color:#7e8b9c;
    font-size:10px;
    font-weight:850;
    letter-spacing:.8px;
}

.trade-value{
    font-size:18px;
    font-weight:900;
    margin-top:4px;
}

.small{
    color:#7d899b;
    font-size:10px;
}

/* ========================================================
   SECTIONS
======================================================== */

.section{
    font-size:16px;
    font-weight:900;
    margin:20px 0 10px;
}

.box{
    background:
        linear-gradient(
            145deg,
            #101927,
            #0b1119
        );
    border:1px solid #29374a;
    border-radius:15px;
    padding:17px;
}

.panel-title{
    color:#8996a8;
    font-size:10px;
    font-weight:850;
    letter-spacing:1.2px;
    text-transform:uppercase;
}

.good{
    color:#61d69a!important;
}

.bad{
    color:#f27e8e!important;
}

.neutral{
    color:#aab5c4!important;
}

/* ========================================================
   PROGRESS
======================================================== */

.progress{
    height:8px;
    border-radius:99px;
    background:#1b2635;
    overflow:hidden;
    margin-top:7px;
}

.progress>div{
    height:100%;
    border-radius:99px;
    background:#7687ff;
}

/* ========================================================
   STREAMLIT METRICS
======================================================== */

div[data-testid="stMetric"]{
    background:#0d141e;
    border:1px solid #1d2939;
    padding:11px;
    border-radius:12px;
}

/* ========================================================
   DATAFRAME
======================================================== */

div[data-testid="stDataFrame"]{
    border-radius:12px;
    overflow:hidden;
}

/* ========================================================
   TABS
======================================================== */

button[data-baseweb="tab"]{
    font-weight:800;
}

/* ========================================================
   REMOVE EXTRA TOP SPACE
======================================================== */

[data-testid="stVerticalBlock"]{
    gap:.55rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================


def f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def clamp(
    v: Any,
    low: float = -1.0,
    high: float = 1.0,
) -> float:
    return float(np.clip(f(v), low, high))


def price(v: Any) -> str:
    x = f(v)

    if x <= 0:
        return "—"

    if x >= 1000:
        return f"{x:,.2f}"

    if x >= 1:
        return f"{x:,.4f}"

    return f"{x:.6f}"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def direction_from_prediction(p: Any) -> str:
    try:
        return "LONG" if int(p) == 1 else "SHORT"
    except Exception:
        return "UNKNOWN"


def signal_class(direction: str) -> str:
    d = str(direction).upper()

    if "LONG" in d:
        return "long"

    if "SHORT" in d:
        return "short"

    return "wait"


def bias(v: Any) -> tuple[str, str]:
    x = f(v)

    if x > 0.25:
        return "BULLISH", "good"

    if x < -0.25:
        return "BEARISH", "bad"

    return "NEUTRAL", "neutral"


# ============================================================
# HTTP REQUEST LAYER
# ============================================================


@st.cache_data(
    ttl=2,
    show_spinner=False,
)
def api_request(
    kind: str,
    path: str,
    params: tuple[tuple[str, Any], ...] = (),
):
    bases = (
        FUTURES_BASES
        if kind == "futures"
        else SPOT_BASES
    )

    last_error = ""

    for base in bases:
        try:
            response = requests.get(
                base + path,
                params=dict(params),
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "ZIA-RESEARCH-LAB/3.0",
                    "Accept": "application/json",
                    "Connection": "keep-alive",
                },
            )

            if response.status_code == 200:
                return response.json(), base, "OK"

            last_error = (
                f"HTTP {response.status_code} "
                f"from {base}"
            )

        except Exception as e:
            last_error = (
                f"{type(e).__name__}: {e}"
            )

    return (
        None,
        "",
        last_error or "No response",
    )


# ============================================================
# MARKET DATA
# ============================================================


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def get_klines(
    symbol: str,
    interval: str,
    limit: int = 260,
):
    params = (
        ("symbol", symbol),
        ("interval", interval),
        ("limit", limit),
    )

    raw, base, err = api_request(
        "futures",
        "/fapi/v1/klines",
        params,
    )

    source = "BINANCE FUTURES"

    if not isinstance(raw, list):

        raw, base, err2 = api_request(
            "spot",
            "/api/v3/klines",
            params,
        )

        source = "BINANCE SPOT FALLBACK"

        err = (
            f"Futures unavailable; {err2}"
        )

    if not isinstance(raw, list):
        return (
            pd.DataFrame(),
            source,
            err,
        )

    rows = []

    for candle in raw:

        try:
            rows.append(
                {
                    "Time": pd.to_datetime(
                        int(candle[0]),
                        unit="ms",
                        utc=True,
                    ),
                    "Open": float(candle[1]),
                    "High": float(candle[2]),
                    "Low": float(candle[3]),
                    "Close": float(candle[4]),
                    "Volume": float(candle[5]),
                    "Trades": int(candle[8]),
                    "TakerBuy": float(candle[9]),
                }
            )

        except Exception:
            continue

    if not rows:
        return (
            pd.DataFrame(),
            source,
            err,
        )

    return (
        pd.DataFrame(rows)
        .dropna()
        .reset_index(drop=True),
        source,
        err,
    )


@st.cache_data(
    ttl=2,
    show_spinner=False,
)
def get_orderbook(
    symbol: str,
    limit: int = 100,
):
    params = (
        ("symbol", symbol),
        ("limit", limit),
    )

    raw, base, err = api_request(
        "futures",
        "/fapi/v1/depth",
        params,
    )

    source = "BINANCE FUTURES"

    if not isinstance(raw, dict):

        raw, base, err2 = api_request(
            "spot",
            "/api/v3/depth",
            params,
        )

        source = "BINANCE SPOT FALLBACK"

        err = (
            f"Futures unavailable; {err2}"
        )

    try:

        bids = (
            np.asarray(
                raw.get("bids", []),
                dtype=float,
            )
            if isinstance(raw, dict)
            else np.empty((0, 2))
        )

        asks = (
            np.asarray(
                raw.get("asks", []),
                dtype=float,
            )
            if isinstance(raw, dict)
            else np.empty((0, 2))
        )

        if (
            bids.ndim != 2
            or bids.shape[1] < 2
        ):
            bids = np.empty((0, 2))

        if (
            asks.ndim != 2
            or asks.shape[1] < 2
        ):
            asks = np.empty((0, 2))

        return (
            bids,
            asks,
            source,
            err,
        )

    except Exception as e:

        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
            source,
            str(e),
        )


@st.cache_data(
    ttl=3,
    show_spinner=False,
)
def get_ticker(symbol: str):

    params = (
        ("symbol", symbol),
    )

    raw, base, err = api_request(
        "futures",
        "/fapi/v1/ticker/24hr",
        params,
    )

    source = "BINANCE FUTURES"

    if not isinstance(raw, dict):

        raw, base, err2 = api_request(
            "spot",
            "/api/v3/ticker/24hr",
            params,
        )

        source = "BINANCE SPOT FALLBACK"

        err = (
            f"Futures unavailable; {err2}"
        )

    return (
        raw
        if isinstance(raw, dict)
        else {},
        source,
        err,
    )


@st.cache_data(
    ttl=3,
    show_spinner=False,
)
def get_trades(
    symbol: str,
    limit: int = 1000,
):

    params = (
        ("symbol", symbol),
        ("limit", limit),
    )

    raw, base, err = api_request(
        "futures",
        "/fapi/v1/aggTrades",
        params,
    )

    source = "BINANCE FUTURES"

    if not isinstance(raw, list):

        raw, base, err2 = api_request(
            "spot",
            "/api/v3/aggTrades",
            params,
        )

        source = "BINANCE SPOT FALLBACK"

        err = (
            f"Futures unavailable; {err2}"
        )

    return (
        raw
        if isinstance(raw, list)
        else [],
        source,
        err,
    )


# ============================================================
# TECHNICAL / ORDER FLOW
# ============================================================


def obi(
    bids: np.ndarray,
    asks: np.ndarray,
    levels: int,
) -> float:

    n = min(
        len(bids),
        len(asks),
        levels,
    )

    if n <= 0:
        return 0.0

    bid_volume = max(
        0.0,
        float(bids[:n, 1].sum()),
    )

    ask_volume = max(
        0.0,
        float(asks[:n, 1].sum()),
    )

    total = bid_volume + ask_volume

    if total <= 0:
        return 0.0

    return clamp(
        (bid_volume - ask_volume) / total
    )


def weighted_obi(
    bids: np.ndarray,
    asks: np.ndarray,
    levels: int = 20,
) -> float:

    n = min(
        len(bids),
        len(asks),
        levels,
    )

    if n <= 0:
        return 0.0

    weights = 1.0 / (
        np.arange(n) + 1.0
    )

    bid_volume = float(
        (
            bids[:n, 1]
            * weights
        ).sum()
    )

    ask_volume = float(
        (
            asks[:n, 1]
            * weights
        ).sum()
    )

    total = (
        bid_volume
        + ask_volume
    )

    if total <= 0:
        return 0.0

    return clamp(
        (bid_volume - ask_volume)
        / total
    )


def depth(
    bids: np.ndarray,
    asks: np.ndarray,
    levels: int,
):

    n = min(
        len(bids),
        len(asks),
        levels,
    )

    if n <= 0:
        return 0.0, 0.0

    return (
        float(bids[:n, 1].sum()),
        float(asks[:n, 1].sum()),
    )


def taker_flow(
    trades: list[dict[str, Any]],
):

    buy = 0.0
    sell = 0.0
    count = 0

    for trade in trades:

        try:

            quantity = float(
                trade["q"]
            )

            if bool(trade["m"]):
                sell += quantity
            else:
                buy += quantity

            count += 1

        except Exception:
            continue

    total = buy + sell

    ratio = (
        clamp(
            (buy - sell)
            / total
        )
        if total
        else 0.0
    )

    return {
        "buy": buy,
        "sell": sell,
        "flow": buy - sell,
        "ratio": ratio,
        "count": count,
    }


def technical(
    df: pd.DataFrame,
):

    if df.empty:
        return {}

    close = df["Close"]

    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    ema50 = (
        close
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    ema200 = (
        close
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    momentum5 = (
        f(
            close.iloc[-1]
            / close.iloc[-6]
            - 1
        )
        if len(close) >= 6
        else 0.0
    )

    momentum20 = (
        f(
            close.iloc[-1]
            / close.iloc[-21]
            - 1
        )
        if len(close) >= 21
        else 0.0
    )

    current_price = f(
        close.iloc[-1]
    )

    trend = (

        (
            0.30
            if current_price
            > f(ema20.iloc[-1])
            else -0.30
        )

        + (
            0.25
            if current_price
            > f(ema50.iloc[-1])
            else -0.25
        )

        + (
            0.20
            if current_price
            > f(ema200.iloc[-1])
            else -0.20
        )

        + clamp(
            momentum20 * 20,
            -0.25,
            0.25,
        )
    )

    volatility = (
        close
        .pct_change()
        .rolling(20)
        .std()
        .iloc[-1]
    )

    return {
        "price": current_price,
        "ema20": f(
            ema20.iloc[-1]
        ),
        "ema50": f(
            ema50.iloc[-1]
        ),
        "ema200": f(
            ema200.iloc[-1]
        ),
        "momentum5": momentum5,
        "momentum20": momentum20,
        "volatility": f(
            volatility
        ),
        "trend": clamp(trend),
    }


def calc_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> float:

    if len(df) < 2:
        return 0.0

    previous_close = (
        df["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            df["High"]
            - df["Low"],

            (
                df["High"]
                - previous_close
            ).abs(),

            (
                df["Low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
        .iloc[-1]
    )

    return max(
        0.0,
        f(atr),
    )


# ============================================================
# HIGHER TIMEFRAME BIAS
#
# Heavy calculation.
# Cached longer so 5-second refresh does not repeatedly
# hit all HTF endpoints.
# ============================================================


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def get_htf_bias(
    symbol: str,
):

    output = {}

    source = "BINANCE FUTURES"

    for timeframe in [
        "1h",
        "4h",
        "1d",
    ]:

        data, src, _ = get_klines(
            symbol,
            timeframe,
            120,
        )

        if "SPOT" in src:
            source = (
                "BINANCE SPOT FALLBACK"
            )

        if data.empty:

            output[timeframe] = 0.0
            continue

        close = data["Close"]

        ema20 = (
            close
            .ewm(
                span=20,
                adjust=False,
            )
            .mean()
            .iloc[-1]
        )

        ema50 = (
            close
            .ewm(
                span=50,
                adjust=False,
            )
            .mean()
            .iloc[-1]
        )

        current_price = f(
            close.iloc[-1]
        )

        output[timeframe] = (

            (
                0.5
                if current_price
                > f(ema20)
                else -0.5
            )

            +

            (
                0.5
                if current_price
                > f(ema50)
                else -0.5
            )
        )

    return output, source


# ============================================================
# ML MODEL
# ============================================================


@st.cache_resource(
    show_spinner=False
)
def load_model():

    if not MODEL_FILE.exists():
        return (
            None,
            "Model file not found",
        )

    try:

        model = joblib.load(
            MODEL_FILE
        )

        return (
            model,
            "Loaded",
        )

    except Exception as e:

        return (
            None,
            f"Load error: {type(e).__name__}",
        )


def predict_ml(
    model: Any,
    features: list[float],
):

    if model is None:
        return None

    try:

        expected = getattr(
            model,
            "n_features_in_",
            None,
        )

        if (
            expected is not None
            and int(expected)
            != len(features)
        ):

            return {
                "error":
                    "Model expects "
                    f"{expected} features; "
                    f"dashboard supplied "
                    f"{len(features)}"
            }

        X = np.asarray(
            [features],
            dtype=float,
        )

        prediction = int(
            model.predict(X)[0]
        )

        output = {
            "prediction": prediction,
            "direction":
                direction_from_prediction(
                    prediction
                ),
        }

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = np.asarray(
                model.predict_proba(X)[0],
                dtype=float,
            )

            output["probabilities"] = (
                probabilities.tolist()
            )

            output["confidence"] = float(
                np.max(probabilities)
            )

            output["classes"] = (
                np.asarray(
                    getattr(
                        model,
                        "classes_",
                        range(
                            len(
                                probabilities
                            )
                        ),
                    )
                ).tolist()
            )

        else:

            output["confidence"] = 0.50

        return output

    except Exception as e:

        return {
            "error":
                "Prediction error: "
                f"{type(e).__name__}: {e}"
        }


# ============================================================
# SIGNAL ENGINE
# ============================================================


def build_signal(
    df,
    bids,
    asks,
    symbol,
    mode_key,
):

    tech = technical(df)

    if (
        not tech
        or len(bids) < 20
        or len(asks) < 20
    ):
        return None

    # --------------------------------------------------------
    # Order flow
    # --------------------------------------------------------

    trades, trade_source, _ = (
        get_trades(symbol)
    )

    flow = taker_flow(
        trades
    )

    o5, o10, o20, o50 = [
        obi(
            bids,
            asks,
            levels,
        )
        for levels in (
            5,
            10,
            20,
            50,
        )
    ]

    weighted = weighted_obi(
        bids,
        asks,
        20,
    )

    multi = clamp(
        o5 * 0.15
        + o10 * 0.20
        + o20 * 0.35
        + o50 * 0.30
    )

    bid20, ask20 = depth(
        bids,
        asks,
        20,
    )

    bid50, ask50 = depth(
        bids,
        asks,
        50,
    )

    spread = f(
        asks[0, 0]
        - bids[0, 0]
    )

    ratio = (
        bid20 / ask20
        if ask20 > 0
        else 0.0
    )

    # --------------------------------------------------------
    # EXACT 7 FEATURE MODEL INPUT
    # --------------------------------------------------------

    ml_features = [
        bid20,
        ask20,
        o20,
        spread,
        ratio,
        bid20 + ask20,
        tech["trend"],
    ]

    model, model_status = (
        load_model()
    )

    ml = predict_ml(
        model,
        ml_features,
    )

    # --------------------------------------------------------
    # QUANT SCORE
    # --------------------------------------------------------

    score = (

        multi * 0.30

        + flow["ratio"] * 0.25

        + tech["trend"] * 0.25

        + clamp(
            tech["momentum5"] * 30
        ) * 0.10

        + clamp(
            tech["momentum20"] * 15
        ) * 0.10
    )

    # --------------------------------------------------------
    # HTF
    # --------------------------------------------------------

    htf, htf_source = (
        get_htf_bias(symbol)
    )

    hscore = (

        htf.get("1h", 0)
        * 0.45

        + htf.get("4h", 0)
        * 0.35

        + htf.get("1d", 0)
        * 0.20
    )

    score = clamp(
        score
        + hscore * 0.20
    )

    # --------------------------------------------------------
    # ML VOTE
    # --------------------------------------------------------

    ml_conf = (
        f(
            ml.get(
                "confidence",
                0.50,
            ),
            0.50,
        )
        if ml
        and "error" not in ml
        else 0.50
    )

    if (
        ml
        and "error" not in ml
    ):

        ml_vote = (
            1.0
            if ml["prediction"] == 1
            else -1.0
        )

        score = clamp(
            score
            + ml_vote
            * min(
                0.25,
                ml_conf * 0.25,
            )
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = (

        abs(score) * 55

        + abs(multi) * 20

        + abs(flow["ratio"]) * 15

        + abs(hscore) * 10
    )

    if (
        ml
        and "error" not in ml
    ):

        confidence = (
            confidence * 0.75
            + ml_conf * 100 * 0.25
        )

    confidence = float(
        np.clip(
            confidence,
            0,
            99,
        )
    )

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    if (
        score >= 0.70
        and confidence >= 70
    ):

        direction = "STRONG LONG"

    elif (
        score >= 0.42
        and confidence >= 55
    ):

        direction = "LONG"

    elif (
        score <= -0.70
        and confidence >= 70
    ):

        direction = "STRONG SHORT"

    elif (
        score <= -0.42
        and confidence >= 55
    ):

        direction = "SHORT"

    else:

        direction = "WAIT"

    # --------------------------------------------------------
    # TRADE LEVELS
    # --------------------------------------------------------

    atr = calc_atr(df)

    if not atr:
        atr = (
            tech["price"]
            * 0.005
        )

    stop_distance = min(
        max(
            atr * 1.15,
            tech["price"] * 0.0025,
        ),
        tech["price"] * 0.006,
    )

    entry = tech["price"]

    if "LONG" in direction:

        stop_loss = (
            entry
            - stop_distance
        )

        target1 = (
            entry
            + stop_distance * 2
        )

        target2 = (
            entry
            + stop_distance * 3
        )

    elif "SHORT" in direction:

        stop_loss = (
            entry
            + stop_distance
        )

        target1 = (
            entry
            - stop_distance * 2
        )

        target2 = (
            entry
            - stop_distance * 3
        )

    else:

        stop_loss = entry
        target1 = entry
        target2 = entry

    # --------------------------------------------------------
    # TICKER
    # --------------------------------------------------------

    ticker, ticker_source, _ = (
        get_ticker(symbol)
    )

    sources = [
        trade_source,
        htf_source,
        ticker_source,
    ]

    data_source = (
        "BINANCE SPOT FALLBACK"
        if any(
            "SPOT" in x
            for x in sources
        )
        else "BINANCE FUTURES"
    )

    # --------------------------------------------------------
    # FINAL OBJECT
    # --------------------------------------------------------

    return {

        "timestamp":
            now_utc().isoformat(),

        "symbol":
            symbol,

        "mode":
            mode_key,

        "direction":
            direction,

        "score":
            score,

        "confidence":
            confidence,

        "price":
            entry,

        "entry":
            entry,

        "stop_loss":
            stop_loss,

        "target1":
            target1,

        "target2":
            target2,

        "atr":
            atr,

        "obi5":
            o5,

        "obi10":
            o10,

        "obi20":
            o20,

        "obi50":
            o50,

        "weighted_obi":
            weighted,

        "multi_obi":
            multi,

        "bid20":
            bid20,

        "ask20":
            ask20,

        "bid50":
            bid50,

        "ask50":
            ask50,

        "spread":
            spread,

        "taker_buy":
            flow["buy"],

        "taker_sell":
            flow["sell"],

        "taker_flow":
            flow["flow"],

        "taker_flow_ratio":
            flow["ratio"],

        "trade_count":
            flow["count"],

        "trend":
            tech["trend"],

        "momentum5":
            tech["momentum5"],

        "momentum20":
            tech["momentum20"],

        "ema20":
            tech["ema20"],

        "ema50":
            tech["ema50"],

        "ema200":
            tech["ema200"],

        "volatility":
            tech["volatility"],

        "htf_1h":
            htf.get("1h", 0),

        "htf_4h":
            htf.get("4h", 0),

        "htf_1d":
            htf.get("1d", 0),

        "ml":
            ml,

        "ml_available":
            (
                ml is not None
                and "error" not in ml
            ),

        "ml_confidence":
            ml_conf,

        "ml_features":
            ml_features,

        "model_status":
            model_status,

        "change24":
            f(
                ticker.get(
                    "priceChangePercent"
                )
            ),

        "volume24":
            f(
                ticker.get(
                    "quoteVolume"
                )
            ),

        "data_source":
            data_source,
    }


# ============================================================
# HISTORY
# ============================================================


def save_signal(signal):

    if signal["direction"] == "WAIT":
        return

    row = {
        key: signal.get(key)
        for key in [
            "timestamp",
            "symbol",
            "mode",
            "direction",
            "score",
            "confidence",
            "entry",
            "stop_loss",
            "target1",
            "target2",
            "obi20",
            "obi50",
            "taker_flow_ratio",
        ]
    }

    try:

        pd.DataFrame(
            [row]
        ).to_csv(
            HISTORY_FILE,
            mode="a",
            header=not HISTORY_FILE.exists(),
            index=False,
        )

    except Exception:
        pass


def history_df():

    try:

        if not HISTORY_FILE.exists():
            return pd.DataFrame()

        return pd.read_csv(
            HISTORY_FILE
        ).tail(300)

    except Exception:
        return pd.DataFrame()


# ============================================================
# CHARTS
# ============================================================


def price_chart(
    df,
    signal,
):

    data = df.tail(180)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data["Time"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
    )

    for span, name in [
        (20, "EMA 20"),
        (50, "EMA 50"),
        (200, "EMA 200"),
    ]:

        ema = (
            data["Close"]
            .ewm(
                span=span,
                adjust=False,
            )
            .mean()
        )

        fig.add_trace(
            go.Scatter(
                x=data["Time"],
                y=ema,
                name=name,
                line=dict(
                    width=1.25
                ),
            )
        )

    fig.add_hline(
        y=signal["entry"],
        annotation_text="ENTRY",
    )

    if signal["direction"] != "WAIT":

        fig.add_hline(
            y=signal["stop_loss"],
            annotation_text="SL",
            line_dash="dot",
        )

        fig.add_hline(
            y=signal["target1"],
            annotation_text="TP1 1:2",
            line_dash="dash",
        )

        fig.add_hline(
            y=signal["target2"],
            annotation_text="TP2 1:3",
            line_dash="dash",
        )

    fig.update_layout(
        template="plotly_dark",
        height=540,
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=5,
            r=5,
            t=25,
            b=5,
        ),
        legend=dict(
            orientation="h",
            y=1.02,
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def imbalance_chart(signal):

    values = [
        signal["obi5"],
        signal["obi10"],
        signal["obi20"],
        signal["obi50"],
    ]

    fig = go.Figure(
        go.Bar(
            x=[
                "Top 5",
                "Top 10",
                "Top 20",
                "Top 50",
            ],
            y=values,
            text=[
                f"{value:+.3f}"
                for value in values
            ],
            textposition="outside",
        )
    )

    fig.add_hline(y=0)

    fig.update_layout(
        template="plotly_dark",
        height=310,
        margin=dict(
            l=5,
            r=5,
            t=20,
            b=5,
        ),
        yaxis=dict(
            range=[
                -1,
                1,
            ]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### ⚡ ZIA RESEARCH"
    )

    st.caption(
        "ML-powered Binance market research terminal"
    )

    st.divider()

    st.session_state.symbol = (
        st.selectbox(
            "MARKET",
            COINS,
            index=(
                COINS.index(
                    st.session_state.symbol
                )
                if st.session_state.symbol
                in COINS
                else 0
            ),
        )
    )

    mode_keys = list(MODES)

    st.session_state.mode = (
        st.selectbox(
            "ANALYSIS MODE",
            mode_keys,
            index=mode_keys.index(
                st.session_state.mode
            ),
            format_func=lambda x:
                MODES[x]["label"],
        )
    )

    st.divider()

    st.markdown(
        "**ENGINE**"
    )

    st.session_state.auto_refresh = (
        st.toggle(
            "Live refresh",
            value=st.session_state.auto_refresh,
        )
    )

    if st.session_state.auto_refresh:

        st.session_state.refresh_seconds = (
            st.slider(
                "Refresh interval",
                3,
                30,
                int(
                    st.session_state.refresh_seconds
                ),
                1,
            )
        )

    else:

        st.session_state.refresh_seconds = 30

    if st.button(
        "↻ Refresh now",
        use_container_width=True,
    ):

        st.cache_data.clear()

        st.rerun()

    st.divider()

    current_mode = MODES[
        st.session_state.mode
    ]

    st.info(
        f"""
**{current_mode['tf'].upper()} analysis**

Holding: {current_mode['hold']}

HTF: {', '.join(current_mode['refs']).upper()}
"""
    )

    st.caption(
        "Research / signal generation only. "
        "No order execution is performed by this dashboard."
    )


# ============================================================
# LIVE TERMINAL
#
# IMPORTANT:
# This is the main fix.
#
# New Streamlit:
#     st.fragment(run_every="5s")
#
# Only this area reruns.
# Sidebar / page shell does NOT need to rebuild.
#
# Old Streamlit:
#     fallback to normal page rerun.
# ============================================================


def render_live_terminal():

    # --------------------------------------------------------
    # MARKET SETTINGS
    # --------------------------------------------------------

    symbol = st.session_state.symbol

    mode_key = st.session_state.mode

    mode = MODES[
        mode_key
    ]

    # --------------------------------------------------------
    # FETCH CURRENT DATA
    # --------------------------------------------------------

    df, candle_source, candle_error = (
        get_klines(
            symbol,
            mode["tf"],
            260,
        )
    )

    bids, asks, book_source, book_error = (
        get_orderbook(
            symbol,
            100,
        )
    )

    # --------------------------------------------------------
    # CONNECTION ERROR
    # --------------------------------------------------------

    if (
        df.empty
        or len(df) < 30
    ):

        st.error(
            f"Market data unavailable for {symbol}."
        )

        st.warning(
            "Binance market-data endpoints could not "
            "be reached. Futures mirrors and public "
            "Spot fallback are being tried automatically."
        )

        with st.expander(
            "Connection diagnostics"
        ):

            st.code(
                f"""
Candle source:
{candle_source}

Order book source:
{book_source}

Candle error:
{candle_error}

Order book error:
{book_error}
"""
            )

        return

    if (
        len(bids) < 20
        or len(asks) < 20
    ):

        st.error(
            "Order-book depth is unavailable right now."
        )

        with st.expander(
            "Connection diagnostics"
        ):

            st.code(
                f"""
Order book source:
{book_source}

Error:
{book_error}
"""
            )

        return

    # --------------------------------------------------------
    # BUILD SIGNAL
    # --------------------------------------------------------

    signal = build_signal(
        df,
        bids,
        asks,
        symbol,
        mode_key,
    )

    if signal is None:

        st.error(
            "Signal engine could not calculate "
            "the current market state."
        )

        return

    # --------------------------------------------------------
    # SAVE SIGNAL
    # --------------------------------------------------------

    minute_key = (
        f"{symbol}:"
        f"{mode_key}:"
        f"{now_utc().strftime('%Y-%m-%d-%H-%M')}"
    )

    if (
        st.session_state.last_saved_key
        != minute_key
    ):

        save_signal(signal)

        st.session_state.last_saved_key = (
            minute_key
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    is_fallback = (
        "SPOT"
        in signal["data_source"]
        or "SPOT"
        in candle_source
        or "SPOT"
        in book_source
    )

    if is_fallback:

        status_text = (
            "DEGRADED • SPOT FALLBACK"
        )

        status_class = "warn"
        dot_class = "warn"

    else:

        status_text = (
            "LIVE • BINANCE FUTURES"
        )

        status_class = ""
        dot_class = ""

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
<div class="hero">

<div>
    <div class="brand">
        <span>⚡</span>
        ZIA RESEARCH LAB
    </div>

    <div class="subtitle">
        USDⓈ-M Futures research · Order Flow ·
        OBI · Quant Trend · XGBoost ML
    </div>
</div>

<div class="status {status_class}">
    <span class="dot {dot_class}"></span>
    {status_text}
    ·
    {now_utc().strftime('%H:%M:%S UTC')}
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    signal_css = signal_class(
        signal["direction"]
    )

    ml_connected = (
        signal["ml_available"]
    )

    ml_label = (
        "CONNECTED"
        if ml_connected
        else "OFFLINE"
    )

    ml_class = (
        "good"
        if ml_connected
        else "bad"
    )

    st.markdown(
        f"""
<div class="signal {signal_css}">

<div style="
display:flex;
justify-content:space-between;
gap:20px;
align-items:center;
">

<div>

<div class="signal-label">
FINAL RESEARCH SIGNAL ·
{symbol} · {mode['label']}
</div>

<div class="signal-name">
{signal['direction']}
</div>

<div class="signal-meta">

Quant score
<b>{signal['score']:+.3f}</b>

·

Confidence
<b>{signal['confidence']:.1f}%</b>

·

ML
<b class="{ml_class}">
{ml_label}
</b>

</div>

</div>

<div style="text-align:right">

<div class="signal-label">
LAST PRICE
</div>

<div class="big-price">
${price(signal['price'])}
</div>

<div class="signal-meta">
24H {signal['change24']:+.2f}%
</div>

</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    columns = st.columns(6)

    kpis = [

        (
            "Confidence",
            f"{signal['confidence']:.1f}%",
            "final signal",
        ),

        (
            "OBI 20",
            f"{signal['obi20']:+.4f}",
            "order book",
        ),

        (
            "OBI 50",
            f"{signal['obi50']:+.4f}",
            "deep liquidity",
        ),

        (
            "Taker Flow",
            f"{signal['taker_flow_ratio']:+.4f}",
            "aggressive flow",
        ),

        (
            "Trend",
            f"{signal['trend']:+.3f}",
            "quant trend",
        ),

        (
            "XGBoost",
            (
                f"{signal['ml_confidence'] * 100:.1f}%"
                if signal["ml_available"]
                else "OFF"
            ),
            "model probability",
        ),
    ]

    for column, (
        label,
        value,
        sub,
    ) in zip(
        columns,
        kpis,
    ):

        with column:

            st.markdown(
                f"""
<div class="kpi">

<div class="kpi-label">
{label}
</div>

<div class="kpi-value">
{value}
</div>

<div class="kpi-sub">
{sub}
</div>

</div>
""",
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # FALLBACK WARNING
    # --------------------------------------------------------

    if is_fallback:

        st.info(
            "Binance Futures was not reachable from "
            "the runtime, so public Spot market data "
            "is being used temporarily. The ML model "
            "remains connected, but the model inputs "
            "are based on fallback market data."
        )

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    (
        tab_overview,
        tab_flow,
        tab_ml,
        tab_history,
    ) = st.tabs(
        [
            "▣ Overview",
            "◈ Order Flow",
            "◎ ML Engine",
            "▤ Signal History",
        ]
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with tab_overview:

        st.markdown(
            '<div class="section">TRADE PLAN</div>',
            unsafe_allow_html=True,
        )

        plan_columns = st.columns(4)

        plans = [

            (
                "ENTRY",
                signal["entry"],
                "market reference",
            ),

            (
                "STOP LOSS",
                signal["stop_loss"],
                "volatility adjusted",
            ),

            (
                "TARGET 1",
                signal["target1"],
                "1 : 2 risk / reward",
            ),

            (
                "TARGET 2",
                signal["target2"],
                "1 : 3 risk / reward",
            ),
        ]

        for column, (
            label,
            value,
            sub,
        ) in zip(
            plan_columns,
            plans,
        ):

            with column:

                if (
                    signal["direction"]
                    != "WAIT"
                    or label == "ENTRY"
                ):

                    display_value = price(
                        value
                    )

                else:

                    display_value = "WAIT"

                st.markdown(
                    f"""
<div class="trade">

<div class="trade-label">
{label}
</div>

<div class="trade-value">
{display_value}
</div>

<div class="small">
{sub}
</div>

</div>
""",
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # PRICE CHART
        # ----------------------------------------------------

        st.markdown(
            '<div class="section">PRICE ACTION</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            price_chart(
                df,
                signal,
            ),
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

        # ----------------------------------------------------
        # HTF / MOMENTUM / MOVING AVERAGES
        # ----------------------------------------------------

        a, b, c = st.columns(3)

        with a:

            st.markdown(
                "**Higher-timeframe bias**"
            )

            for timeframe in [
                "1h",
                "4h",
                "1d",
            ]:

                label, css = bias(
                    signal[
                        f"htf_{timeframe}"
                    ]
                )

                st.markdown(
                    f"""
`{timeframe.upper()}`

<span class="{css}">
<b>{label}</b>
</span>

&nbsp;

{signal[f'htf_{timeframe}']:+.2f}
""",
                    unsafe_allow_html=True,
                )

        with b:

            st.markdown(
                "**Momentum & volatility**"
            )

            st.metric(
                "5-candle momentum",
                f"{signal['momentum5'] * 100:+.2f}%",
            )

            st.metric(
                "20-candle momentum",
                f"{signal['momentum20'] * 100:+.2f}%",
            )

            st.metric(
                "ATR",
                price(signal["atr"]),
            )

        with c:

            st.markdown(
                "**Moving averages**"
            )

            st.metric(
                "EMA 20",
                price(signal["ema20"]),
            )

            st.metric(
                "EMA 50",
                price(signal["ema50"]),
            )

            st.metric(
                "EMA 200",
                price(signal["ema200"]),
            )

    # ========================================================
    # ORDER FLOW
    # ========================================================

    with tab_flow:

        st.markdown(
            '<div class="section">ORDER BOOK IMBALANCE</div>',
            unsafe_allow_html=True,
        )

        x, y = st.columns(
            [1.25, 1]
        )

        with x:

            st.plotly_chart(
                imbalance_chart(
                    signal
                ),
                use_container_width=True,
                config={
                    "displaylogo": False
                },
            )

        with y:

            st.metric(
                "Top 20 bid volume",
                f"{signal['bid20']:,.3f}",
            )

            st.metric(
                "Top 20 ask volume",
                f"{signal['ask20']:,.3f}",
            )

            st.metric(
                "Top 50 bid volume",
                f"{signal['bid50']:,.3f}",
            )

            st.metric(
                "Top 50 ask volume",
                f"{signal['ask50']:,.3f}",
            )

        # ----------------------------------------------------
        # TAKER FLOW
        # ----------------------------------------------------

        st.markdown(
            '<div class="section">TAKER / AGGRESSIVE FLOW</div>',
            unsafe_allow_html=True,
        )

        flow_columns = st.columns(4)

        flow_items = [

            (
                "Taker Buy",
                f"{signal['taker_buy']:,.3f}",
            ),

            (
                "Taker Sell",
                f"{signal['taker_sell']:,.3f}",
            ),

            (
                "Flow Ratio",
                f"{signal['taker_flow_ratio']:+.4f}",
            ),

            (
                "Trades",
                f"{signal['trade_count']:,}",
            ),
        ]

        for column, (
            label,
            value,
        ) in zip(
            flow_columns,
            flow_items,
        ):

            with column:
                st.metric(
                    label,
                    value,
                )

        # ----------------------------------------------------
        # LIVE ORDER BOOK
        # ----------------------------------------------------

        st.markdown(
            '<div class="section">LIVE ORDER BOOK · TOP 20</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)

        with left:

            bid_df = pd.DataFrame(
                bids[:20],
                columns=[
                    "Price",
                    "Quantity",
                ],
            )

            bid_df["Price"] = (
                bid_df["Price"]
                .map(price)
            )

            st.dataframe(
                bid_df,
                use_container_width=True,
                hide_index=True,
                height=470,
            )

        with right:

            ask_df = pd.DataFrame(
                asks[:20],
                columns=[
                    "Price",
                    "Quantity",
                ],
            )

            ask_df["Price"] = (
                ask_df["Price"]
                .map(price)
            )

            st.dataframe(
                ask_df,
                use_container_width=True,
                hide_index=True,
                height=470,
            )

    # ========================================================
    # ML ENGINE
    # ========================================================

    with tab_ml:

        st.markdown(
            '<div class="section">XGBOOST DECISION CENTER</div>',
            unsafe_allow_html=True,
        )

        ml = signal["ml"]

        left, right = st.columns(
            [1.0, 1.45]
        )

        with left:

            ml_status_class = (
                "good"
                if signal["ml_available"]
                else "bad"
            )

            ml_status_text = (
                "CONNECTED"
                if signal["ml_available"]
                else "OFFLINE"
            )

            st.markdown(
                f"""
<div class="box">

<div class="panel-title">
MODEL STATUS
</div>

<div style="
font-size:24px;
font-weight:950;
">

XGBoost

<span class="{ml_status_class}">
{ml_status_text}
</span>

</div>

<div class="small"
style="margin-top:8px">

{signal['model_status']}

</div>

</div>
""",
                unsafe_allow_html=True,
            )

            if (
                ml
                and "error" in ml
            ):

                st.error(
                    ml["error"]
                )

            elif ml:

                st.metric(
                    "ML direction",
                    ml["direction"],
                )

                st.metric(
                    "ML confidence",
                    f"{signal['ml_confidence'] * 100:.2f}%",
                )

                probabilities = ml.get(
                    "probabilities",
                    [],
                )

                classes = ml.get(
                    "classes",
                    list(
                        range(
                            len(
                                probabilities
                            )
                        )
                    ),
                )

                for class_value, probability in zip(
                    classes,
                    probabilities,
                ):

                    st.markdown(
                        f"""
**{direction_from_prediction(class_value)}**
·
{probability * 100:.2f}%
"""
                    )

                    width = max(
                        0,
                        min(
                            100,
                            probability * 100,
                        ),
                    )

                    st.markdown(
                        f"""
<div class="progress">
<div style="
width:{width:.2f}%;
"></div>
</div>
""",
                        unsafe_allow_html=True,
                    )

        with right:

            st.markdown(
                "**Exact 7 features sent to the trained model**"
            )

            feature_rows = [

                {
                    "Feature": feature_name,
                    "Live value": feature_value,
                }

                for feature_name,
                feature_value
                in zip(
                    MODEL_FEATURES,
                    signal["ml_features"],
                )
            ]

            st.dataframe(
                pd.DataFrame(
                    feature_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.success(
                "Feature order is locked to the existing "
                "trained XGBoost schema."
            )

            st.caption(
                f"Input source: "
                f"{signal['data_source']}"
            )

    # ========================================================
    # SIGNAL HISTORY
    # ========================================================

    with tab_history:

        st.markdown(
            '<div class="section">RECENT RESEARCH SIGNALS</div>',
            unsafe_allow_html=True,
        )

        history = history_df()

        if history.empty:

            st.info(
                "No non-WAIT signals have been recorded yet."
            )

        else:

            history_columns = [
                column
                for column in [
                    "timestamp",
                    "symbol",
                    "mode",
                    "direction",
                    "score",
                    "confidence",
                    "entry",
                    "stop_loss",
                    "target1",
                    "target2",
                ]
                if column in history.columns
            ]

            display = (
                history[
                    history_columns
                ]
                .tail(50)
                .iloc[::-1]
                .copy()
            )

            for column in [
                "entry",
                "stop_loss",
                "target1",
                "target2",
            ]:

                if column in display:

                    display[column] = (
                        display[column]
                        .map(price)
                    )

            if "score" in display:

                display["score"] = (
                    display["score"]
                    .astype(float)
                    .round(3)
                )

            if "confidence" in display:

                display["confidence"] = (
                    display["confidence"]
                    .astype(float)
                    .round(1)
                )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                height=520,
            )

            st.caption(
                f"Stored history: "
                f"{len(history):,} rows"
            )

    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    footer = st.columns(4)

    with footer[0]:

        st.caption(
            f"● {signal['data_source']} "
            f"· {symbol}"
        )

    with footer[1]:

        st.caption(
            f"● Analysis · "
            f"{mode['tf'].upper()}"
        )

    with footer[2]:

        st.caption(
            "● ML · "
            + (
                "Connected"
                if signal["ml_available"]
                else "Unavailable"
            )
        )

    with footer[3]:

        st.caption(
            "● Updated · "
            + now_utc().strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )


# ============================================================
# SMOOTH AUTO REFRESH
# ============================================================

if st.session_state.auto_refresh:

    refresh_interval = (
        f"{int(st.session_state.refresh_seconds)}s"
    )

    # --------------------------------------------------------
    # Streamlit Fragment
    #
    # This prevents the complete Streamlit application from
    # rerunning on every live update.
    # --------------------------------------------------------

    if hasattr(
        st,
        "fragment",
    ):

        @st.fragment(
            run_every=refresh_interval
        )
        def live_fragment():

            render_live_terminal()

        live_fragment()

    else:

        # ----------------------------------------------------
        # Compatibility fallback for old Streamlit
        # ----------------------------------------------------

        render_live_terminal()

        st.markdown(
            f"""
<script>
setTimeout(
    function(){{
        window.location.reload();
    }},
    {int(st.session_state.refresh_seconds) * 1000}
);
</script>
""",
            unsafe_allow_html=True,
        )

else:

    render_live_terminal()
