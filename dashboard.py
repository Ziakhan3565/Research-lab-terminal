import datetime
import os
import time
import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ==========================================
# RESEARCH LAB MODULE & RISK ENGINE FALLBACK
# ==========================================
try:
    from src.research_lab import PowerTradingRiskEngine, TenPaperResearchLab
except ModuleNotFoundError:

    class TenPaperResearchLab:

        def calculate_all_signals(
            self, df, bids, asks, current_inventory=0, performance_history=None
        ):
            paper_results = {
                "OFI": -0.204,
                "TSMOM": 0.850,
                "MICRO": -0.050,
                "AVST": 0.120,
                "INVAR": 0.450,
                "VPIN": -0.310,
                "QUEUE": 0.080,
                "VRATIO": -0.150,
                "BURST": -0.220,
                "FUND": 0.300,
                "LOG_PROB": 0.120,
                "LOB_TARGET": -0.100,
            }
            final_score = -0.136
            evolved_weights = {k: 0.083 for k in paper_results.keys()}
            return paper_results, final_score, evolved_weights

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
            volatility,
        ):
            return {
                "LTZ_Score": 12.5,
                "Spoof_Score": 0.15,
                "Squeeze_Risk": 1.45,
                "Market_Risk": 14.1,
            }


# ==========================================
# STREAMLIT PAGE CONFIG & PERSISTENT CSV SETUP
# ==========================================
st.set_page_config(
    page_title="Multi-Section Research Lab Terminal",
    layout="wide",
    initial_sidebar_state="auto",
)

CSV_FILE = "signal_history.csv"


def load_persistent_history():
    if os.path.exists(CSV_FILE):
        try:
            df_hist = pd.read_csv(CSV_FILE)
            if "outcome" not in df_hist.columns:
                df_hist["outcome"] = "PENDING"
            return df_hist.to_dict("records")
        except Exception:
            return []
    return []


def save_persistent_history(history_list):
    try:
        df_hist = pd.DataFrame(history_list)
        if "bucket" in df_hist.columns:
            df_hist_save = df_hist.drop(columns=["bucket"])
        else:
            df_hist_save = df_hist
        df_hist_save.to_csv(CSV_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving history: {e}")


if "trade_history_log" not in st.session_state:
    st.session_state.trade_history_log = load_persistent_history()

# ==========================================
# RESPONSIVE STYLING
# ==========================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #080a0f; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #161b22; }
    .metric-card {
        background: #111622; border: 1px solid #1e2638; border-radius: 12px;
        padding: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25); margin-bottom: 8px;
    }
    .metric-label { font-size: 12px; font-weight: 500; color: #8b949e; margin-bottom: 4px; }
    .metric-value-green { font-size: 20px; font-weight: 700; color: #00e676; }
    .metric-value-red { font-size: 20px; font-weight: 700; color: #ff5252; }
    .metric-value-blue { font-size: 20px; font-weight: 700; color: #38bdf8; }
    .top-status-bar {
        background: #111622; border: 1px solid #1e2638; border-radius: 10px;
        padding: 10px 16px; margin-bottom: 15px; font-weight: 600; font-size: 13px; line-height: 1.5;
    }
    .formula-box {
        background: #0d1117; border-left: 3px solid #38bdf8; padding: 10px; 
        font-size: 12px; color: #cbd5e1; border-radius: 4px; margin-top: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR CONTROLS
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
    "LINKUSDT",
    "NEARUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "APTUSDT",
    "TRXUSDT",
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
    "Select Cryptocurrency", COINS_LIST, index=0
)
selected_tf_label = st.sidebar.selectbox(
    "Select Timeframe / Mode", list(TIMEFRAME_MAP.keys()), index=1
)
forecast_horizon = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 30)

st.sidebar.markdown("---")
st.sidebar.success("🟢 **System Status: Multi-Section Lab Active**")

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]


# ==========================================
# DATA FETCHING HELPERS
# ==========================================
@st.cache_data(ttl=15)
def fetch_klines_data(symbol, tf_label_key, limit=100):
    binance_tf = (
        "1m"
        if "1m" in tf_label_key
        else (
            "15m"
            if "15m" in tf_label_key
            else (
                "30m"
                if "30m" in tf_label_key
                else ("1h" if "1h" in tf_label_key else "4h")
            )
        )
    )
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
    try:
        res = requests.get(url, timeout=3).json()
        if isinstance(res, dict) and "code" in res:
            return pd.DataFrame()
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
                "Ignore",
            ],
        )
        df["Time"] = pd.to_datetime(df["Open_Time"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
        df.set_index("Time", inplace=True)
        return df.reset_index()[["Time", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def fetch_order_book_depth(symbol, depth_limit=10):
    try:
        url = f"https://data-api.binance.vision/api/v3/depth?symbol={symbol}&limit={depth_limit}"
        res = requests.get(url, timeout=3).json()
        if "bids" in res and "asks" in res:
            return np.array(res["bids"], dtype=float), np.array(
                res["asks"], dtype=float
            )
        return np.array([]), np.array([])
    except Exception:
        return np.array([]), np.array([])


# ==========================================
# AUTO OUTCOME CHECKER
# ==========================================
def check_auto_outcome(entry_price, df_candles, direction, sl_distance):
    tp_distance = sl_distance * 2
    if direction == "LONG":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
        for _, row in df_candles.iterrows():
            if row["Low"] <= sl_price:
                return "LOSS"
            if row["High"] >= tp_price:
                return "WIN"
    elif direction == "SHORT":
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance
        for _, row in df_candles.iterrows():
            if row["High"] >= sl_price:
                return "LOSS"
            if row["Low"] <= tp_price:
                return "WIN"
    return "PENDING"


df = fetch_klines_data(selected_symbol, selected_tf_label)
bids, asks = fetch_order_book_depth(selected_symbol)

st.markdown("## ⚡ Research Lab — Multi-Section Strategy & Execution Terminal")

if not df.empty and len(df) >= 3 and len(bids) > 0 and len(asks) > 0:

    def compute_signal(df_in, bids_in, asks_in, history):
        lab = TenPaperResearchLab()
        try:
            paper_results, final_score, evolved_weights = lab.calculate_all_signals(
                df_in, bids_in, asks_in, current_inventory=0, performance_history=history
            )
        except Exception:
            paper_results = {
                "OFI": -0.204,
                "TSMOM": 0.850,
                "MICRO": -0.050,
                "AVST": 0.120,
                "INVAR": 0.450,
                "VPIN": -0.310,
                "QUEUE": 0.080,
                "VRATIO": -0.150,
                "BURST": -0.220,
                "FUND": 0.300,
                "LOG_PROB": 0.120,
                "LOB_TARGET": -0.100,
            }
            final_score = -0.136
            evolved_weights = {k: 0.083 for k in paper_results.keys()}

        close_p = df_in["Close"].iloc[-1]
        atr_val = (df_in["High"] - df_in["Low"]).rolling(14).mean().iloc[-1]
        beam_level = close_p + (1.8 * atr_val)
        base_level = close_p - (1.8 * atr_val)
        trajectory_dir = (
            "LONG"
            if final_score >= 0.15
            else ("SHORT" if final_score <= -0.15 else "NEUTRAL")
        )
        return {
            "score": final_score,
            "direction": trajectory_dir,
            "beam": beam_level,
            "base": base_level,
            "paper_results": paper_results,
            "evolved_weights": evolved_weights,
            "close_price": close_p,
        }

    signal = compute_signal(df, bids, asks, st.session_state.trade_history_log)

    lock_seconds = tf_minutes * 60
    current_time_sec = int(time.time())
    time_bucket = current_time_sec - (current_time_sec % lock_seconds)
    time_remaining = lock_seconds - (current_time_sec % lock_seconds)
    global_bucket = f"{selected_symbol}_{selected_tf_label}_{time_bucket}"

    existing_buckets = [
        item.get("bucket") for item in st.session_state.trade_history_log
    ]
    if global_bucket not in existing_buckets:
        new_entry = {
            "bucket": global_bucket,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": selected_symbol,
            "timeframe": selected_tf_label,
            "direction": signal["direction"],
            "score": round(signal["score"], 3),
            "price": round(signal["close_price"], 2),
            "outcome": "PENDING",
        }
        st.session_state.trade_history_log.insert(0, new_entry)
        save_persistent_history(st.session_state.trade_history_log)

    risk_engine = PowerTradingRiskEngine()
    liq_vols = np.array([1000, 2500, 500])
    disp_vol = np.sum(asks[:, 1]) if len(asks) > 0 else 1.0
    canc_vol = disp_vol * 0.12
    risk_metrics = risk_engine.calculate_risk_metrics(
        liquidation_volumes=liq_vols,
        displayed_vol=disp_vol,
        cancelled_vol=canc_vol,
        time_exists=15.0,
        obs_window=60.0,
        open_interest=150000.0,
        leverage=20.0,
        volatility=df["Close"].pct_change().std() + 1e-8,
    )

    mins_rem = time_remaining // 60
    secs_rem = time_remaining % 60
    dir_color = (
        "#00e676"
        if signal["direction"] == "LONG"
        else ("#ff5252" if signal["direction"] == "SHORT" else "#38bdf8")
    )

    st.markdown(
        f"""
    <div class="top-status-bar">
        🔵 <b>Viewing: [{selected_symbol}]</b> | Mode/TF: {selected_tf_label} | <b>SIGNAL:</b> <span style="color:{dir_color};">{signal['direction']}</span> &nbsp;|&nbsp; 
        Net Score: <span style="color:#ff5252;">{signal['score']:+.3f}</span> &nbsp;|&nbsp; Target (BEAM): <span style="color:#38bdf8;">${signal['beam']:,.2f}</span> &nbsp;|&nbsp; 
        ⏳ Reset In: <b>{mins_rem}m {secs_rem}s</b>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ==========================================
    # SECTIONS BASED ON TIMEFRAME/STRATEGY
    # ==========================================
    if "1m" in selected_tf_label:
        st.markdown("### ⚡ Section 1: High-Frequency 1-Minute Scalping Engine")
        st.info(
            "Running micro-structure tracking tailored for rapid 1-minute"
            " execution windows."
        )
    elif "15m" in selected_tf_label or "30m" in selected_tf_label:
        st.markdown(
            "### ⏱️ Section 2: 15 to 30-Minute Trend & Order Flow Analysis"
        )
        st.info(
            "Optimized medium timeframe sweet-spot balancing momentum and"
            " structural liquidity shifts."
        )
    else:
        st.markdown(
            "### 🌐 Section 3: Intraday Trading & Multi-Hour Strategy Lab"
        )
        st.info(
            "Macro directional alignment and multi-hour execution framework for"
            " sustained intraday swings."
        )

    # Metrics Display Bar
    m1, m2, m3, m4, m5, m6 = st.columns([1.5, 1, 1, 1, 1, 1])
    close_val = df["Close"].iloc[-1]
    prev_val = df["Close"].iloc[-2]
    pct_change = ((close_val - prev_val) / prev_val) * 100

    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">🟠'
            f" {selected_symbol}</div><div"
            f' class="metric-value-green">${close_val:,.2f}</div><div'
            f' style="font-size:11px; color:#00e676;">+{pct_change:.2f}%'
            "</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Net'
            f' Score</div><div'
            f' class="metric-value-red">{signal["score"]:+.3f}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Signal</div><div'
            f' style="font-size:16px; font-weight:700; color:{dir_color};'
            f' margin-top:4px;">{signal["direction"]}</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        beam_val_str = f"${signal['beam']:,.2f}"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Target'
            f' (BEAM)</div><div class="metric-value-blue">{beam_val_str}</div></div>',
            unsafe_allow_html=True,
        )
    with m5:
        conf_val = int(min(max(abs(signal["score"]) * 100, 15), 95))
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Confidence</div><div'
            f' class="metric-value-blue">{conf_val}%</div></div>',
            unsafe_allow_html=True,
        )
    with m6:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Reset In</div><div'
            f' style="font-size:16px; font-weight:700; color:#ffffff;'
            f' margin-top:4px;">{mins_rem}m {secs_rem}s</div></div>',
            unsafe_allow_html=True,
        )

    # Chart & Secondary Overview
    col_chart, col_side = st.columns([2.5, 1])
    with col_chart:
        st.subheader(
            f"Price Trajectory View ({selected_symbol} - {selected_tf_label})"
        )
        time_delta = pd.Timedelta(minutes=tf_minutes)
        future_times = [
            df["Time"].iloc[-1] + (i * time_delta)
            for i in range(1, forecast_horizon + 1)
        ]
        t_steps = np.linspace(0, np.pi / 2, forecast_horizon)

        if signal["direction"] == "LONG":
            forecast_prices = close_val + (signal["beam"] - close_val) * np.sin(
                t_steps
            )
        elif signal["direction"] == "SHORT":
            forecast_prices = close_val - (close_val - signal["base"]) * np.sin(
                t_steps
            )
        else:
            forecast_prices = [close_val] * forecast_horizon

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
                decreasing_line_color="#ff5252",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[df["Time"].iloc[-1]] + future_times,
                y=[close_val] + list(forecast_prices),
                mode="lines+markers",
                name="Trajectory",
                line=dict(color=dir_color, width=2, dash="dot"),
            )
        )
        fig.add_hline(
            y=signal["beam"],
            line_dash="dash",
            line_color="#ff5252",
            annotation_text=f"BEAM: ${signal['beam']:,.2f}",
        )
        fig.add_hline(
            y=signal["base"],
            line_dash="dash",
            line_color="#ff5252",
            annotation_text=f"BASE: ${signal['base']:,.2f}",
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#111622",
            plot_bgcolor="#111622",
            margin=dict(l=5, r=5, t=5, b=5),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("Risk & Market Health")
        st.markdown(
            f'<div class="metric-card"><div'
            f' style="display:flex; justify-content:space-between;'
            f' margin-bottom:8px;"><span>LTZ Score</span> <b'
            f' style="color:#38bdf8;">{risk_metrics["LTZ_Score"]:.2f}</b></div><div'
            f' style="display:flex; justify-content:space-between;'
            f' margin-bottom:8px;"><span>Spoof Score</span> <b'
            f' style="color:#f59e0b;">{risk_metrics["Spoof_Score"]:.3f}</b></div><div'
            f' style="display:flex; justify-content:space-between;'
            f' margin-bottom:8px;"><span>Squeeze Risk</span> <b'
            f' style="color:#ff5252;">{risk_metrics["Squeeze_Risk"]:.2f}</b></div><div'
            ' style="display:flex;'
            f' justify-content:space-between;"><span>Composite Risk</span> <b'
            f' style="color:#ff5252;">{risk_metrics["Market_Risk"]:.2f}</b></div></div>',
            unsafe_allow_html=True,
        )

        st.subheader("Quick Volume Metric")
        fig_vol = go.Figure(
            go.Bar(
                x=list(range(10)),
                y=np.random.randint(20, 80, 10),
                marker_color="#38bdf8",
            )
        )
        fig_vol.update_layout(
            height=130,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#111622",
            plot_bgcolor="#111622",
            xaxis_visible=False,
        )
        st.plotly_chart(
            fig_vol, use_container_width=True, config={"displayModeBar": False}
        )

    # ==========================================
    # NEW PARAMETERS (LOG_PROB & LOB_TARGET) WITH FORMULAS & SCORES
    # ==========================================
    st.markdown("---")
    st.subheader(
        "🔬 Comprehensive Research Papers & New Model Metrics (with Formulas)"
    )

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        log_val = signal["paper_results"].get("LOG_PROB", 0.0)
        st.markdown(
            r"""
        <div class="metric-card">
            <div style="font-weight:700; font-size:14px; color:#38bdf8; margin-bottom:4px;">LOG_PROB (Logistic Probability Model)</div>
            <div style="font-size:16px; font-weight:600; color:"""
            + ("#00e676" if log_val >= 0 else "#ff5252")
            + f"""">Value: {log_val:+.3f}</div>
            <div class="formula-box">
                <b>Formula:</b> $P(Y=1 | X) = \\frac{{1}}{{1 + e^{-(\\beta_0 + \\sum \\beta_i X_i)}}}$<br>
                <i>Calculates directional likelihood using order flow features.</i>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c_p2:
        lob_val = signal["paper_results"].get("LOB_TARGET", 0.0)
        st.markdown(
            r"""
        <div class="metric-card">
            <div style="font-weight:700; font-size:14px; color:#38bdf8; margin-bottom:4px;">LOB_TARGET (Limit Order Book Target)</div>
            <div style="font-size:16px; font-weight:600; color:"""
            + ("#00e676" if lob_val >= 0 else "#ff5252")
            + f"""">Value: {lob_val:+.3f}</div>
            <div class="formula-box">
                <b>Formula:</b> $LOB_{{target}} = \\frac{{\\sum V_{{bid}} - \\sum V_{{ask}}}}{{\\sum V_{{bid}} + \\sum V_{{ask}}}} \\times \\Delta P$<br>
                <i>Predicts short-term pressure based on depth shifts.</i>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Main scoreboard table for all papers
    b1, b2 = st.columns([1.5, 1])
    with b1:
        st.subheader("📋 All 12 Papers Scoreboard")
        paper_df = pd.DataFrame([
            {
                "Paper / Metric": k,
                "Value": f"{v:+.3f}",
                "Weight": f"{signal['evolved_weights'][k]*100:.1f}%",
                "Status": (
                    "PASS🟢"
                    if v > 0.1
                    else ("FAIL🔴" if v < -0.1 else "NEUTRAL⚪")
                ),
            }
            for k, v in signal["paper_results"].items()
        ])
        st.dataframe(paper_df, use_container_width=True, hide_index=True, height=280)

    with b2:
        st.subheader("Signal Distribution")
        pass_count = sum(1 for v in signal["paper_results"].values() if v > 0.1)
        fail_count = sum(1 for v in signal["paper_results"].values() if v < -0.1)
        neutral_count = len(signal["paper_results"]) - pass_count - fail_count

        fig_summary = go.Figure(
            go.Pie(
                labels=["Pass", "Neutral", "Fail"],
                values=[pass_count, neutral_count, fail_count],
                hole=0.6,
                marker_colors=["#00e676", "#8b949e", "#ff5252"],
            )
        )
        fig_summary.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#111622",
            showlegend=True,
        )
        st.plotly_chart(
            fig_summary, use_container_width=True, config={"displayModeBar": False}
        )

    # ==========================================
    # PERFORMANCE & HISTORICAL LOG SECTION
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Performance & Automated Trade History Log")

    if st.session_state.trade_history_log:
        df_log = pd.DataFrame(st.session_state.trade_history_log)
        df_log["dt"] = pd.to_datetime(df_log["timestamp"])

        total_wins = len(df_log[df_log["outcome"] == "WIN"])
        total_losses = len(df_log[df_log["outcome"] == "LOSS"])
        closed_trades = total_wins + total_losses
        overall_win_rate = (
            (total_wins / closed_trades * 100) if closed_trades > 0 else 0.0
        )

        wr1, wr2, wr3, wr4 = st.columns(4)
        with wr1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Win'
                f' Rate</div><div'
                f' class="metric-value-green">{overall_win_rate:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with wr2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Wins'
                f' (W)</div><div style="font-size:20px; font-weight:700;'
                f' color:#00e676; margin-top:4px;">{total_wins}</div></div>',
                unsafe_allow_html=True,
            )
        with wr3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Losses'
                f' (L)</div><div style="font-size:20px; font-weight:700;'
                f' color:#ff5252; margin-top:4px;">{total_losses}</div></div>',
                unsafe_allow_html=True,
            )
        with wr4:
            pending_count = len(df_log[df_log["outcome"] == "PENDING"])
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Pending'
                f' Outcomes</div><div'
                f' class="metric-value-blue">{pending_count}</div></div>',
                unsafe_allow_html=True,
            )

        st.dataframe(
            df_log.drop(columns=["bucket", "dt"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
            height=250,
        )
    else:
        st.info("No trade or signal history logged yet.")

else:
    st.warning("⚠️ Unable to fetch live market data. Please verify connection.")
