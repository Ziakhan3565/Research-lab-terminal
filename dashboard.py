import streamlit as st
import pandas as pd
import numpy as np
import time
import json

# ============================================================
# IMPORT ENGINE
# ============================================================

from engine import IntegratedTradingEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TRI Quant Trading Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0b0f14;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.metric-card {
    background: #111820;
    border: 1px solid #26313d;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
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
    font-size: 28px;
    font-weight: bold;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# ENGINE
# ============================================================

@st.cache_resource
def load_engine():
    return IntegratedTradingEngine()


engine = load_engine()


# ============================================================
# HEADER
# ============================================================

st.title("⚡ TRI Quant Trading Engine")

st.caption(
    "TRI Line + 12 Research Features + Order Book + ML + Risk Engine"
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
# DEMO DATA FUNCTION
# ============================================================

def create_demo_market_data():

    np.random.seed(int(time.time()) % 100000)

    price = 110000 + np.random.randn() * 100

    prices = []

    current = price

    for _ in range(100):

        current += np.random.randn() * 25

        prices.append(current)

    df = pd.DataFrame({
        "Close": prices,
        "Volume": np.random.uniform(
            100,
            1000,
            100
        )
    })

    # --------------------------------------------------------
    # Order Book
    # --------------------------------------------------------

    bid_prices = np.array([
        price - i * 2
        for i in range(1, 21)
    ])

    ask_prices = np.array([
        price + i * 2
        for i in range(1, 21)
    ])

    bid_volume = np.random.uniform(
        1,
        100,
        20
    )

    ask_volume = np.random.uniform(
        1,
        100,
        20
    )

    bids = np.column_stack([
        bid_prices,
        bid_volume
    ])

    asks = np.column_stack([
        ask_prices,
        ask_volume
    ])

    # --------------------------------------------------------
    # TRI Levels
    # --------------------------------------------------------

    tri_data = {

        "mBody50": price - 350,
        "mUpper50": price + 800,
        "mLower50": price - 1100,

        "wBody50": price - 150,
        "wUpper50": price + 450,
        "wLower50": price - 600,

        "dBody50": price - 50,
        "dUpper50": price + 180,
        "dLower50": price - 220
    }

    return (
        df,
        bids,
        asks,
        tri_data
    )


# ============================================================
# LOAD MARKET DATA
# ============================================================

df, bids, asks, tri_data = (
    create_demo_market_data()
)


current_price = float(
    df["Close"].iloc[-1]
)


# ============================================================
# RUN ENGINE
# ============================================================

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

    displayed_vol=float(
        np.sum(bids[:, 1]) +
        np.sum(asks[:, 1])
    ),

    cancelled_vol=500,

    time_exists=15,

    obs_window=60,

    open_interest=500000,

    leverage=5,

    volatility=float(
        df["Close"]
        .pct_change()
        .std()
    )
)


# ============================================================
# GET RESULTS
# ============================================================

signal = result["SIGNAL"]

raw_signal = result["RAW_SIGNAL"]

score = result["SCORE"]

confidence = result["CONFIDENCE"]

features = result["FEATURES"]

risk = result["RISK"]


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
        "ML / Ensemble Score",
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
            <div class="small-text">FINAL SIGNAL</div>
            <div class="big-number">🟢 LONG</div>
            <div>Confidence: {confidence:.1f}%</div>
            <div>Score: {score:+.3f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

elif signal == "SHORT":

    st.markdown(
        f"""
        <div class="signal-short">
            <div class="small-text">FINAL SIGNAL</div>
            <div class="big-number">🔴 SHORT</div>
            <div>Confidence: {confidence:.1f}%</div>
            <div>Score: {score:+.3f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        f"""
        <div class="signal-wait">
            <div class="small-text">FINAL SIGNAL</div>
            <div class="big-number">🟡 WAIT</div>
            <div>Confidence: {confidence:.1f}%</div>
            <div>Score: {score:+.3f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TRI LINE ANALYSIS
# ============================================================

st.markdown(
    "### 📐 TRI LINE ANALYSIS"
)


tri_col1, tri_col2, tri_col3 = st.columns(3)


def level_status(price, level):

    if price > level:
        return "ABOVE"

    if price < level:
        return "BELOW"

    return "AT LEVEL"


with tri_col1:

    st.markdown("#### 🔴 MONTHLY")

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


with tri_col2:

    st.markdown("#### 🟢 WEEKLY")

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


with tri_col3:

    st.markdown("#### ⚫ DAILY")

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

tri_direction = features.get(
    "TRI_DIRECTION",
    0
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
    "### 📚 ORDER BOOK ANALYSIS"
)


bid_total = np.sum(
    bids[:, 1]
)

ask_total = np.sum(
    asks[:, 1]
)

obi = (
    bid_total -
    ask_total
) / (
    bid_total +
    ask_total +
    1e-8
)


ob_col1, ob_col2, ob_col3, ob_col4 = st.columns(4)


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
# ORDER BOOK TABLE
# ============================================================

book_col1, book_col2 = st.columns(2)


with book_col1:

    st.markdown("#### 🟢 BIDS")

    bid_df = pd.DataFrame(
        bids,
        columns=[
            "Price",
            "Quantity"
        ]
    )

    bid_df["Value"] = (
        bid_df["Price"] *
        bid_df["Quantity"]
    )

    st.dataframe(
        bid_df.head(20),
        use_container_width=True,
        hide_index=True
    )


with book_col2:

    st.markdown("#### 🔴 ASKS")

    ask_df = pd.DataFrame(
        asks,
        columns=[
            "Price",
            "Quantity"
        ]
    )

    ask_df["Value"] = (
        ask_df["Price"] *
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
    "### 🧠 12-PAPER + TRI FEATURE ENGINE"
)


feature_rows = []


for name, value in features.items():

    feature_rows.append({

        "Feature": name,

        "Value": round(
            float(value),
            4
        ),

        "Direction":
            "BULLISH"
            if value > 0.05
            else
            "BEARISH"
            if value < -0.05
            else
            "NEUTRAL"
    })


feature_df = pd.DataFrame(
    feature_rows
)


st.dataframe(
    feature_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RESEARCH FEATURE COLUMNS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


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


for index, name in enumerate(
    research_features
):

    col = [
        c1,
        c2,
        c3,
        c4
    ][index % 4]

    with col:

        value = features.get(
            name,
            0
        )

        st.metric(
            name,
            f"{value:+.3f}"
        )


# ============================================================
# TRI FEATURE COLUMNS
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


tri_feature_data = []


for name in tri_features:

    tri_feature_data.append({

        "TRI Feature": name,

        "Score": round(
            features.get(
                name,
                0
            ),
            4
        )

    })


st.dataframe(
    pd.DataFrame(tri_feature_data),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RISK ENGINE
# ============================================================

st.markdown(
    "### ⚠️ POWER RISK ENGINE"
)


risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)


with risk_col1:

    st.metric(
        "LTZ Score",
        f"{risk['LTZ_Score']:.2f}"
    )


with risk_col2:

    st.metric(
        "Spoof Score",
        f"{risk['Spoof_Score']:.2f}"
    )


with risk_col3:

    st.metric(
        "Squeeze Risk",
        f"{risk['Squeeze_Risk']:.2f}"
    )


with risk_col4:

    st.metric(
        "Market Risk",
        f"{risk['Market_Risk']:.2f}"
    )


st.warning(
    f"Risk Level: **{risk['Risk_Level']}**"
)


# ============================================================
# RISK INTERPRETATION
# ============================================================

risk_score = risk["Market_Risk"]


if risk_score < 25:

    st.success(
        "LOW RISK — Market conditions relatively stable."
    )

elif risk_score < 50:

    st.info(
        "MEDIUM RISK — Monitor order flow and liquidity."
    )

elif risk_score < 75:

    st.warning(
        "HIGH RISK — Reduce exposure and wait for confirmation."
    )

else:

    st.error(
        "EXTREME RISK — Engine forces WAIT."
    )


# ============================================================
# DECISION MATRIX
# ============================================================

st.markdown(
    "### 🎯 DECISION MATRIX"
)


decision_df = pd.DataFrame({

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

        risk["Risk_Level"],

        signal
    ]
})


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


status1, status2, status3 = st.columns(3)


with status1:

    st.success(
        "TRI Engine: ONLINE"
    )


with status2:

    st.success(
        "Research Engine: ONLINE"
    )


with status3:

    st.success(
        "Risk Engine: ONLINE"
    )


# ============================================================
# JSON OUTPUT
# ============================================================

with st.expander(
    "🔧 Raw Engine JSON"
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
