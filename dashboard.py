import streamlit as st
import pandas as pd
import numpy as np
import time


# ============================================================
# MAIN RESEARCH LAB ENGINE
# ============================================================

from src.research_lab import IntegratedTradingEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TRI Quant Research Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0b0f14;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .signal-long {
        background: #12351f;
        border: 1px solid #1f9d55;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .signal-short {
        background: #3b1518;
        border: 1px solid #e74c3c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .signal-wait {
        background: #302b12;
        border: 1px solid #d4ac0d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .small-text {
        color: #8f9aa6;
        font-size: 13px;
    }

    .big-number {
        font-size: 30px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MAIN RESEARCH LAB
# ============================================================

@st.cache_resource
def load_engine():
    return IntegratedTradingEngine()


try:

    engine = load_engine()

except Exception as e:

    st.error("❌ Research Lab Engine load nahi ho saka.")

    st.code(
        str(e),
        language="text"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("⚡ TRI Quant Research Lab")

st.caption(
    "12-Paper Research Lab + TRI Line + Order Book + "
    "ML Ensemble + Power Risk Engine"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Engine Settings")


symbol = st.sidebar.text_input(
    "Symbol",
    value="BTCUSDT"
)


timeframe = st.sidebar.selectbox(
    "Timeframe",
    [
        "1m",
        "5m",
        "15m",
        "1h",
        "4h",
        "1D"
    ],
    index=1
)


top_levels = st.sidebar.slider(
    "Order Book Levels",
    5,
    50,
    20
)


auto_refresh = st.sidebar.checkbox(
    "Auto Refresh",
    value=False
)


refresh_seconds = st.sidebar.slider(
    "Refresh Seconds",
    1,
    60,
    5
)


# ============================================================
# DEMO MARKET DATA
# ============================================================
#
# IMPORTANT:
# Ye abhi DEMO DATA hai.
# Real Binance/MEXC websocket data ke liye baad mein
# isi section ko live collector se connect karna hoga.
#
# Research Lab ko hum change nahi kar rahe.
# ============================================================

def create_demo_market_data(levels=20):

    seed = int(time.time()) % 100000

    np.random.seed(seed)

    # --------------------------------------------------------
    # Base Price
    # --------------------------------------------------------

    base_price = (
        110000
        +
        np.random.randn() * 100
    )

    # --------------------------------------------------------
    # OHLC-style close history
    # --------------------------------------------------------

    prices = []

    current = base_price

    for _ in range(100):

        current += np.random.randn() * 25

        prices.append(current)

    df = pd.DataFrame(
        {
            "Close": prices,
            "Volume": np.random.uniform(
                100,
                1000,
                100
            )
        }
    )

    # --------------------------------------------------------
    # Order Book
    # --------------------------------------------------------

    levels = int(levels)

    bid_prices = np.array(
        [
            base_price - i * 2
            for i in range(1, levels + 1)
        ],
        dtype=float
    )

    ask_prices = np.array(
        [
            base_price + i * 2
            for i in range(1, levels + 1)
        ],
        dtype=float
    )

    bid_volume = np.random.uniform(
        1,
        100,
        levels
    )

    ask_volume = np.random.uniform(
        1,
        100,
        levels
    )

    bids = np.column_stack(
        [
            bid_prices,
            bid_volume
        ]
    )

    asks = np.column_stack(
        [
            ask_prices,
            ask_volume
        ]
    )

    # --------------------------------------------------------
    # TRI LEVELS
    # --------------------------------------------------------

    tri_data = {

        "mBody50":
            base_price - 350,

        "mUpper50":
            base_price + 800,

        "mLower50":
            base_price - 1100,

        "wBody50":
            base_price - 150,

        "wUpper50":
            base_price + 450,

        "wLower50":
            base_price - 600,

        "dBody50":
            base_price - 50,

        "dUpper50":
            base_price + 180,

        "dLower50":
            base_price - 220
    }

    return (
        df,
        bids,
        asks,
        tri_data
    )


# ============================================================
# CREATE MARKET DATA
# ============================================================

df, bids, asks, tri_data = (
    create_demo_market_data(
        top_levels
    )
)


# ============================================================
# CURRENT PRICE
# ============================================================

current_price = float(
    df["Close"].iloc[-1]
)


# ============================================================
# VOLATILITY
# ============================================================

returns = (
    df["Close"]
    .pct_change()
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna()
)


volatility = float(
    returns.std()
    if len(returns) > 0
    else 0.0
)


# ============================================================
# ORDER BOOK VOLUME
# ============================================================

bid_total = float(
    np.sum(
        bids[:, 1]
    )
)


ask_total = float(
    np.sum(
        asks[:, 1]
    )
)


displayed_volume = (
    bid_total
    +
    ask_total
)


# ============================================================
# RUN MAIN RESEARCH LAB ENGINE
# ============================================================

try:

    result = engine.analyze(

        df=df,

        bids=bids,

        asks=asks,

        tri_data=tri_data,

        liquidation_volumes=[
            1000,
            1500,
            2500,
            1200
        ],

        displayed_vol=displayed_volume,

        cancelled_vol=500,

        time_exists=15,

        obs_window=60,

        open_interest=500000,

        leverage=5,

        volatility=volatility
    )

except Exception as e:

    st.error(
        "❌ Research Lab analysis mein error."
    )

    st.exception(e)

    st.stop()


# ============================================================
# READ ENGINE RESULTS
# ============================================================

signal = result.get(
    "SIGNAL",
    "WAIT"
)


raw_signal = result.get(
    "RAW_SIGNAL",
    "WAIT"
)


score = float(
    result.get(
        "SCORE",
        0.0
    )
)


confidence = float(
    result.get(
        "CONFIDENCE",
        0.0
    )
)


features = result.get(
    "FEATURES",
    {}
)


risk = result.get(
    "RISK",
    {}
)


# ============================================================
# SAFE RISK VALUES
# ============================================================

ltz_score = float(
    risk.get(
        "LTZ_Score",
        0.0
    )
)


spoof_score = float(
    risk.get(
        "Spoof_Score",
        0.0
    )
)


squeeze_risk = float(
    risk.get(
        "Squeeze_Risk",
        0.0
    )
)


market_risk = float(
    risk.get(
        "Market_Risk",
        0.0
    )
)


risk_level = risk.get(
    "Risk_Level",
    "UNKNOWN"
)


# ============================================================
# TOP MARKET BAR
# ============================================================

st.markdown(
    "### 📡 LIVE MARKET"
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Symbol",
        symbol
    )


with col2:

    st.metric(
        "Price",
        f"${current_price:,.2f}"
    )


with col3:

    st.metric(
        "Timeframe",
        timeframe
    )


with col4:

    st.metric(
        "Research Score",
        f"{score:+.3f}"
    )


with col5:

    st.metric(
        "Confidence",
        f"{confidence:.1f}%"
    )


# ============================================================
# FINAL SIGNAL
# ============================================================

st.markdown(
    "### 🎯 FINAL TRADING SIGNAL"
)


if signal == "LONG":

    st.markdown(
        f"""
        <div class="signal-long">

            <div class="small-text">
                FINAL RESEARCH LAB SIGNAL
            </div>

            <div class="big-number">
                🟢 LONG
            </div>

            <div>
                Confidence: {confidence:.1f}%
            </div>

            <div>
                Score: {score:+.3f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


elif signal == "SHORT":

    st.markdown(
        f"""
        <div class="signal-short">

            <div class="small-text">
                FINAL RESEARCH LAB SIGNAL
            </div>

            <div class="big-number">
                🔴 SHORT
            </div>

            <div>
                Confidence: {confidence:.1f}%
            </div>

            <div>
                Score: {score:+.3f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


else:

    st.markdown(
        f"""
        <div class="signal-wait">

            <div class="small-text">
                FINAL RESEARCH LAB SIGNAL
            </div>

            <div class="big-number">
                🟡 WAIT
            </div>

            <div>
                Confidence: {confidence:.1f}%
            </div>

            <div>
                Score: {score:+.3f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIGNAL DETAILS
# ============================================================

st.markdown(
    "### 🔎 SIGNAL DETAILS"
)


sig1, sig2, sig3 = st.columns(3)


with sig1:

    st.metric(
        "Raw Research Signal",
        raw_signal
    )


with sig2:

    st.metric(
        "Final Signal",
        signal
    )


with sig3:

    st.metric(
        "Risk Level",
        risk_level
    )


# ============================================================
# TRI LINE ANALYSIS
# ============================================================

st.markdown(
    "### 📐 TRI LINE ANALYSIS"
)


tri_col1, tri_col2, tri_col3 = (
    st.columns(3)
)


def level_status(
    price,
    level
):

    if price > level:
        return "ABOVE"

    elif price < level:
        return "BELOW"

    return "AT LEVEL"


# ------------------------------------------------------------
# MONTHLY
# ------------------------------------------------------------

with tri_col1:

    st.markdown(
        "#### 🔴 MONTHLY"
    )

    m_body = tri_data.get(
        "mBody50",
        current_price
    )

    m_upper = tri_data.get(
        "mUpper50",
        current_price
    )

    m_lower = tri_data.get(
        "mLower50",
        current_price
    )

    st.metric(
        "Body 50%",
        f"{m_body:,.2f}",
        level_status(
            current_price,
            m_body
        )
    )

    st.metric(
        "Upper 50%",
        f"{m_upper:,.2f}"
    )

    st.metric(
        "Lower 50%",
        f"{m_lower:,.2f}"
    )


# ------------------------------------------------------------
# WEEKLY
# ------------------------------------------------------------

with tri_col2:

    st.markdown(
        "#### 🟢 WEEKLY"
    )

    w_body = tri_data.get(
        "wBody50",
        current_price
    )

    w_upper = tri_data.get(
        "wUpper50",
        current_price
    )

    w_lower = tri_data.get(
        "wLower50",
        current_price
    )

    st.metric(
        "Body 50%",
        f"{w_body:,.2f}",
        level_status(
            current_price,
            w_body
        )
    )

    st.metric(
        "Upper 50%",
        f"{w_upper:,.2f}"
    )

    st.metric(
        "Lower 50%",
        f"{w_lower:,.2f}"
    )


# ------------------------------------------------------------
# DAILY
# ------------------------------------------------------------

with tri_col3:

    st.markdown(
        "#### ⚫ DAILY"
    )

    d_body = tri_data.get(
        "dBody50",
        current_price
    )

    d_upper = tri_data.get(
        "dUpper50",
        current_price
    )

    d_lower = tri_data.get(
        "dLower50",
        current_price
    )

    st.metric(
        "Body 50%",
        f"{d_body:,.2f}",
        level_status(
            current_price,
            d_body
        )
    )

    st.metric(
        "Upper 50%",
        f"{d_upper:,.2f}"
    )

    st.metric(
        "Lower 50%",
        f"{d_lower:,.2f}"
    )


# ============================================================
# TRI DIRECTION
# ============================================================

tri_direction = float(
    features.get(
        "TRI_DIRECTION",
        0.0
    )
)


if tri_direction > 0.25:

    tri_signal = "LONG"


elif tri_direction < -0.25:

    tri_signal = "SHORT"


else:

    tri_signal = "WAIT"


st.info(
    f"TRI Direction: **{tri_signal}**  |  "
    f"TRI Score: **{tri_direction:+.3f}**"
)


# ============================================================
# ORDER BOOK ANALYSIS
# ============================================================

st.markdown(
    "### 📚 ORDER BOOK ANALYSIS"
)


obi = (
    bid_total
    -
    ask_total
) / (
    bid_total
    +
    ask_total
    +
    1e-8
)


if obi > 0.15:

    ob_signal = "BUY PRESSURE"


elif obi < -0.15:

    ob_signal = "SELL PRESSURE"


else:

    ob_signal = "BALANCED"


ob_col1, ob_col2, ob_col3, ob_col4 = (
    st.columns(4)
)


with ob_col1:

    st.metric(
        "Bid Volume",
        f"{bid_total:,.2f}"
    )


with ob_col2:

    st.metric(
        "Ask Volume",
        f"{ask_total:,.2f}"
    )


with ob_col3:

    st.metric(
        "OBI",
        f"{obi:+.3f}"
    )


with ob_col4:

    st.metric(
        "Order Flow",
        ob_signal
    )


# ============================================================
# ORDER BOOK TABLE
# ============================================================

book_col1, book_col2 = (
    st.columns(2)
)


# ------------------------------------------------------------
# BIDS
# ------------------------------------------------------------

with book_col1:

    st.markdown(
        "#### 🟢 BIDS"
    )

    bid_df = pd.DataFrame(
        bids,
        columns=[
            "Price",
            "Quantity"
        ]
    )

    bid_df["Value"] = (
        bid_df["Price"]
        *
        bid_df["Quantity"]
    )

    st.dataframe(
        bid_df,
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------
# ASKS
# ------------------------------------------------------------

with book_col2:

    st.markdown(
        "#### 🔴 ASKS"
    )

    ask_df = pd.DataFrame(
        asks,
        columns=[
            "Price",
            "Quantity"
        ]
    )

    ask_df["Value"] = (
        ask_df["Price"]
        *
        ask_df["Quantity"]
    )

    st.dataframe(
        ask_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 12 PAPER RESEARCH FEATURES
# ============================================================

st.markdown(
    "### 🧠 12-PAPER RESEARCH LAB"
)


research_features = [

    "HAWKES",

    "BOOK_IMB",

    "TAKER_FLOW",

    "QUANT_IMPLY",

    "BAYESIAN",

    "QUANTILES",

    "TARGET_INV",

    "ADAPT_CONF",

    "FRAC_KELLY",

    "RMT_DOM",

    "CONF_CROSS",

    "REWARD_RISK"
]


# ------------------------------------------------------------
# Research cards
# ------------------------------------------------------------

r1, r2, r3, r4 = (
    st.columns(4)
)


research_columns = [
    r1,
    r2,
    r3,
    r4
]


for index, name in enumerate(
    research_features
):

    value = float(
        features.get(
            name,
            0.0
        )
    )

    col = research_columns[
        index % 4
    ]

    with col:

        st.metric(
            name,
            f"{value:+.3f}"
        )


# ============================================================
# FULL FEATURE TABLE
# ============================================================

st.markdown(
    "### 📊 COMPLETE FEATURE MATRIX"
)


feature_rows = []


for name, value in features.items():

    try:

        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        numeric_value = 0.0


    if numeric_value > 0.05:

        direction = "BULLISH"


    elif numeric_value < -0.05:

        direction = "BEARISH"


    else:

        direction = "NEUTRAL"


    feature_rows.append(
        {
            "Feature": name,

            "Value": round(
                numeric_value,
                4
            ),

            "Direction": direction
        }
    )


feature_df = pd.DataFrame(
    feature_rows
)


st.dataframe(
    feature_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TRI FEATURE MATRIX
# ============================================================

st.markdown(
    "### 📐 TRI FEATURE MATRIX"
)


tri_features = [

    "TRI_M_BODY",

    "TRI_M_UPPER",

    "TRI_M_LOWER",

    "TRI_W_BODY",

    "TRI_W_UPPER",

    "TRI_W_LOWER",

    "TRI_D_BODY",

    "TRI_D_UPPER",

    "TRI_D_LOWER",

    "TRI_DIRECTION"
]


tri_feature_data = []


for name in tri_features:

    value = float(
        features.get(
            name,
            0.0
        )
    )

    if value > 0.05:

        direction = "BULLISH"


    elif value < -0.05:

        direction = "BEARISH"


    else:

        direction = "NEUTRAL"


    tri_feature_data.append(
        {
            "TRI Feature": name,

            "Score": round(
                value,
                4
            ),

            "Direction": direction
        }
    )


st.dataframe(
    pd.DataFrame(
        tri_feature_data
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# POWER RISK ENGINE
# ============================================================

st.markdown(
    "### ⚠️ POWER RISK ENGINE"
)


risk_col1, risk_col2, risk_col3, risk_col4 = (
    st.columns(4)
)


with risk_col1:

    st.metric(
        "LTZ Score",
        f"{ltz_score:.2f}"
    )


with risk_col2:

    st.metric(
        "Spoof Score",
        f"{spoof_score:.2f}"
    )


with risk_col3:

    st.metric(
        "Squeeze Risk",
        f"{squeeze_risk:.2f}"
    )


with risk_col4:

    st.metric(
        "Market Risk",
        f"{market_risk:.2f}"
    )


# ============================================================
# RISK LEVEL
# ============================================================

if risk_level == "LOW":

    st.success(
        f"Risk Level: **{risk_level}**"
    )


elif risk_level == "MEDIUM":

    st.info(
        f"Risk Level: **{risk_level}**"
    )


elif risk_level == "HIGH":

    st.warning(
        f"Risk Level: **{risk_level}**"
    )


else:

    st.error(
        f"Risk Level: **{risk_level}**"
    )


# ============================================================
# RISK INTERPRETATION
# ============================================================

if market_risk < 25:

    st.success(
        "LOW RISK — Market conditions relatively stable."
    )


elif market_risk < 50:

    st.info(
        "MEDIUM RISK — Monitor order flow and liquidity."
    )


elif market_risk < 75:

    st.warning(
        "HIGH RISK — Reduce exposure and wait for confirmation."
    )


else:

    st.error(
        "EXTREME RISK — Research Lab forces WAIT."
    )


# ============================================================
# DECISION MATRIX
# ============================================================

st.markdown(
    "### 🎯 DECISION MATRIX"
)


decision_df = pd.DataFrame(
    {
        "Component": [

            "TRI Direction",

            "Order Book",

            "Research Ensemble",

            "ML / Score",

            "Risk Engine",

            "Final Decision"
        ],

        "Value": [

            tri_signal,

            ob_signal,

            f"{score:+.3f}",

            f"{confidence:.1f}%",

            risk_level,

            signal
        ]
    }
)


st.dataframe(
    decision_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MARKET STATISTICS
# ============================================================

st.markdown(
    "### 📈 MARKET STATISTICS"
)


market1, market2, market3, market4 = (
    st.columns(4)
)


with market1:

    st.metric(
        "Volatility",
        f"{volatility:.6f}"
    )


with market2:

    st.metric(
        "Bid Levels",
        len(bids)
    )


with market3:

    st.metric(
        "Ask Levels",
        len(asks)
    )


with market4:

    st.metric(
        "Total Book Volume",
        f"{displayed_volume:,.2f}"
    )


# ============================================================
# ENGINE STATUS
# ============================================================

st.markdown(
    "### 🟢 ENGINE STATUS"
)


status1, status2, status3 = (
    st.columns(3)
)


with status1:

    st.success(
        "TRI Engine: ONLINE"
    )


with status2:

    st.success(
        "12-Paper Research Lab: ONLINE"
    )


with status3:

    st.success(
        "Power Risk Engine: ONLINE"
    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.markdown(
    "### 🏗️ ENGINE ARCHITECTURE"
)


st.code(
    """
MARKET DATA
     │
     ├── Price / Volume
     │
     ├── Order Book
     │
     └── TRI Levels
             │
             ▼
┌──────────────────────────────┐
│      RESEARCH LAB ENGINE     │
│                              │
│  12 Research Features        │
│  + TRI Line Features         │
│  + Ensemble Score            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      POWER RISK ENGINE       │
│                              │
│  LTZ                         │
│  Spoof Risk                  │
│  Squeeze Risk                │
│  Market Risk                 │
└──────────────┬───────────────┘
               │
               ▼
        FINAL DECISION

      🟢 LONG
      🔴 SHORT
      🟡 WAIT
    """,
    language="text"
)


# ============================================================
# RAW ENGINE JSON
# ============================================================

with st.expander(
    "🔧 Raw Research Lab Output"
):

    st.json(
        result
    )


# ============================================================
# AUTO REFRESH
# ============================================================

if auto_refresh:

    time.sleep(
        refresh_seconds
    )

    st.rerun()
