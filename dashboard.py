import streamlit as st
import pandas as pd
import numpy as np
import time
import json

from engine import IntegratedTradingEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Lab Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0b0f14;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .signal-long {
        background: #12351f;
        border: 1px solid #1f9d55;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
    }

    .signal-short {
        background: #3b1518;
        border: 1px solid #e74c3c;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
    }

    .signal-wait {
        background: #302b12;
        border: 1px solid #d4ac0d;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
    }

    .small-text {
        color: #9aa5b1;
        font-size: 13px;
    }

    .big-number {
        font-size: 32px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RESEARCH LAB MAIN ENGINE
# ============================================================

@st.cache_resource
def load_research_lab():

    return IntegratedTradingEngine()


research_lab = load_research_lab()


# ============================================================
# HEADER
# ============================================================

st.title("🧠 RESEARCH LAB TERMINAL")

st.caption(
    "Research Lab = MAIN ENGINE | "
    "TRI + 12 Research Features + Order Book + Risk"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ MARKET SETTINGS")

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

data_mode = st.sidebar.selectbox(
    "Market Data",
    [
        "Demo Mode"
    ]
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

def create_demo_market_data():

    seed = int(
        time.time() * 100
    ) % 1000000

    rng = np.random.default_rng(seed)

    base_price = (
        110000
        +
        rng.normal(0, 100)
    )

    prices = []

    current = base_price

    for _ in range(150):

        current += rng.normal(
            0,
            25
        )

        prices.append(current)

    df = pd.DataFrame(
        {
            "Close": prices,
            "Volume": rng.uniform(
                100,
                1000,
                len(prices)
            )
        }
    )

    # --------------------------------------------------------
    # ORDER BOOK TOP 20
    # --------------------------------------------------------

    bid_prices = np.array(
        [
            current - i * 2
            for i in range(1, 21)
        ],
        dtype=float
    )

    ask_prices = np.array(
        [
            current + i * 2
            for i in range(1, 21)
        ],
        dtype=float
    )

    bid_volume = rng.uniform(
        1,
        100,
        20
    )

    ask_volume = rng.uniform(
        1,
        100,
        20
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

        "mBody50": current - 350,
        "mUpper50": current + 800,
        "mLower50": current - 1100,

        "wBody50": current - 150,
        "wUpper50": current + 450,
        "wLower50": current - 600,

        "dBody50": current - 50,
        "dUpper50": current + 180,
        "dLower50": current - 220
    }

    return (
        df,
        bids,
        asks,
        tri_data
    )


# ============================================================
# LOAD DATA
# ============================================================

df, bids, asks, tri_data = (
    create_demo_market_data()
)

current_price = float(
    df["Close"].iloc[-1]
)


# ============================================================
# MARKET CALCULATIONS
# ============================================================

bid_total = float(
    np.sum(bids[:, 1])
)

ask_total = float(
    np.sum(asks[:, 1])
)

total_book_volume = (
    bid_total +
    ask_total
)

obi = (
    bid_total -
    ask_total
) / (
    total_book_volume +
    1e-8
)

volatility = float(
    df["Close"]
    .pct_change()
    .std()
)


# ============================================================
# RUN RESEARCH LAB
# ============================================================

try:

    result = research_lab.analyze(

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

        displayed_vol=(
            bid_total +
            ask_total
        ),

        cancelled_vol=500,

        time_exists=15,

        obs_window=60,

        open_interest=500000,

        leverage=5,

        volatility=volatility
    )

    engine_error = None

except Exception as e:

    result = None
    engine_error = str(e)


# ============================================================
# ERROR
# ============================================================

if engine_error:

    st.error(
        "Research Lab Engine Error"
    )

    st.code(
        engine_error
    )

    st.stop()


# ============================================================
# RESULTS
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
        0
    )
)

confidence = float(
    result.get(
        "CONFIDENCE",
        0
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
# TOP MARKET BAR
# ============================================================

st.markdown(
    "### 📡 MARKET"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:

    st.metric(
        "Symbol",
        symbol
    )

with c2:

    st.metric(
        "Price",
        f"${current_price:,.2f}"
    )

with c3:

    st.metric(
        "Timeframe",
        timeframe
    )

with c4:

    st.metric(
        "Score",
        f"{score:+.3f}"
    )

with c5:

    st.metric(
        "Confidence",
        f"{confidence:.1f}%"
    )

with c6:

    st.metric(
        "OBI",
        f"{obi:+.3f}"
    )


# ============================================================
# FINAL SIGNAL
# ============================================================

st.markdown(
    "### 🎯 FINAL RESEARCH LAB SIGNAL"
)

if signal == "LONG":

    st.markdown(
        f"""
        <div class="signal-long">
            <div class="small-text">
                RESEARCH LAB FINAL DECISION
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
                RESEARCH LAB FINAL DECISION
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
                RESEARCH LAB FINAL DECISION
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

s1, s2, s3 = st.columns(3)

with s1:

    st.metric(
        "Raw Research Signal",
        raw_signal
    )

with s2:

    st.metric(
        "Final Signal",
        signal
    )

with s3:

    st.metric(
        "Risk Level",
        risk.get(
            "Risk_Level",
            "UNKNOWN"
        )
    )


# ============================================================
# TRI LINE
# ============================================================

st.markdown(
    "### 📐 TRI LINE ANALYSIS"
)

tri1, tri2, tri3 = st.columns(3)


def level_status(
    price,
    level
):

    if price > level:
        return "ABOVE"

    if price < level:
        return "BELOW"

    return "AT LEVEL"


with tri1:

    st.markdown(
        "#### 🔴 MONTHLY"
    )

    st.metric(
        "Body 50%",
        f"{tri_data['mBody50']:,.2f}",
        level_status(
            current_price,
            tri_data["mBody50"]
        )
    )

    st.metric(
        "Upper 50%",
        f"{tri_data['mUpper50']:,.2f}"
    )

    st.metric(
        "Lower 50%",
        f"{tri_data['mLower50']:,.2f}"
    )


with tri2:

    st.markdown(
        "#### 🟢 WEEKLY"
    )

    st.metric(
        "Body 50%",
        f"{tri_data['wBody50']:,.2f}",
        level_status(
            current_price,
            tri_data["wBody50"]
        )
    )

    st.metric(
        "Upper 50%",
        f"{tri_data['wUpper50']:,.2f}"
    )

    st.metric(
        "Lower 50%",
        f"{tri_data['wLower50']:,.2f}"
    )


with tri3:

    st.markdown(
        "#### ⚫ DAILY"
    )

    st.metric(
        "Body 50%",
        f"{tri_data['dBody50']:,.2f}",
        level_status(
            current_price,
            tri_data["dBody50"]
        )
    )

    st.metric(
        "Upper 50%",
        f"{tri_data['dUpper50']:,.2f}"
    )

    st.metric(
        "Lower 50%",
        f"{tri_data['dLower50']:,.2f}"
    )


# ============================================================
# TRI DIRECTION
# ============================================================

tri_direction = float(
    features.get(
        "TRI_DIRECTION",
        0
    )
)

if tri_direction > 0.25:

    tri_signal = "LONG"

elif tri_direction < -0.25:

    tri_signal = "SHORT"

else:

    tri_signal = "WAIT"


st.info(
    f"TRI Direction: **{tri_signal}** | "
    f"Score: **{tri_direction:+.3f}**"
)


# ============================================================
# ORDER BOOK
# ============================================================

st.markdown(
    "### 📚 LEVEL-2 ORDER BOOK"
)

ob1, ob2, ob3, ob4 = st.columns(4)

with ob1:

    st.metric(
        "Bid Volume",
        f"{bid_total:,.2f}"
    )

with ob2:

    st.metric(
        "Ask Volume",
        f"{ask_total:,.2f}"
    )

with ob3:

    st.metric(
        "OBI",
        f"{obi:+.3f}"
    )

with ob4:

    if obi > 0.15:

        ob_signal = "BUY PRESSURE"

    elif obi < -0.15:

        ob_signal = "SELL PRESSURE"

    else:

        ob_signal = "BALANCED"

    st.metric(
        "Order Flow",
        ob_signal
    )


# ============================================================
# ORDER BOOK TABLES
# ============================================================

book1, book2 = st.columns(2)

with book1:

    st.markdown(
        "#### 🟢 BIDS — TOP 20"
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
        bid_df.head(20),
        use_container_width=True,
        hide_index=True
    )


with book2:

    st.markdown(
        "#### 🔴 ASKS — TOP 20"
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
        ask_df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 12 PAPER FEATURES
# ============================================================

st.markdown(
    "### 🧠 12-PAPER RESEARCH ENGINE"
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

feature_rows = []

for name in research_features:

    value = float(
        features.get(
            name,
            0
        )
    )

    if value > 0.05:

        direction = "BULLISH"

    elif value < -0.05:

        direction = "BEARISH"

    else:

        direction = "NEUTRAL"

    feature_rows.append(
        {
            "Research Feature": name,
            "Value": round(
                value,
                4
            ),
            "Direction": direction
        }
    )


st.dataframe(
    pd.DataFrame(
        feature_rows
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RESEARCH FEATURE METRICS
# ============================================================

cols = st.columns(4)

for index, name in enumerate(
    research_features
):

    with cols[index % 4]:

        value = float(
            features.get(
                name,
                0
            )
        )

        st.metric(
            name,
            f"{value:+.3f}"
        )


# ============================================================
# TRI FEATURES
# ============================================================

st.markdown(
    "### 📐 TRI FEATURES"
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

tri_rows = []

for name in tri_features:

    tri_rows.append(
        {
            "TRI Feature": name,
            "Score": round(
                float(
                    features.get(
                        name,
                        0
                    )
                ),
                4
            )
        }
    )


st.dataframe(
    pd.DataFrame(
        tri_rows
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

r1, r2, r3, r4 = st.columns(4)

with r1:

    st.metric(
        "LTZ Score",
        f"{risk.get('LTZ_Score', 0):.2f}"
    )

with r2:

    st.metric(
        "Spoof Score",
        f"{risk.get('Spoof_Score', 0):.2f}"
    )

with r3:

    st.metric(
        "Squeeze Risk",
        f"{risk.get('Squeeze_Risk', 0):.2f}"
    )

with r4:

    st.metric(
        "Market Risk",
        f"{risk.get('Market_Risk', 0):.2f}"
    )


risk_level = risk.get(
    "Risk_Level",
    "UNKNOWN"
)

if risk_level == "LOW":

    st.success(
        "Risk Level: LOW"
    )

elif risk_level == "MEDIUM":

    st.info(
        "Risk Level: MEDIUM"
    )

elif risk_level == "HIGH":

    st.warning(
        "Risk Level: HIGH"
    )

else:

    st.error(
        f"Risk Level: {risk_level}"
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
            "Confidence",
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
# ENGINE STATUS
# ============================================================

st.markdown(
    "### 🟢 ENGINE STATUS"
)

e1, e2, e3 = st.columns(3)

with e1:

    st.success(
        "Research Lab: ONLINE"
    )

with e2:

    st.success(
        "TRI Engine: ONLINE"
    )

with e3:

    st.success(
        "Risk Engine: ONLINE"
    )


# ============================================================
# RAW JSON
# ============================================================

with st.expander(
    "🔧 RAW RESEARCH LAB OUTPUT"
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
