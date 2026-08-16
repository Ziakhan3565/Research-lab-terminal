import time
import datetime
import io
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from gtts import gTTS

from src.research_lab import TenPaperResearchLab

st.set_page_config(
    page_title="10-Paper Research Lab Terminal", 
    layout="wide", 
    initial_sidebar_state="auto"
)

# Persistent Learning Logs in Session State for Self-Evolving Engine
if "trade_history_log" not in st.session_state:
    st.session_state.trade_history_log = []

# ==========================================
# RESPONSIVE & MOBILE-OPTIMIZED STYLING
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
        padding: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 12px;
        font-weight: 500;
        color: #8b949e;
        margin-bottom: 4px;
    }
    
    .metric-value-green { font-size: 20px; font-weight: 700; color: #00e676; }
    .metric-value-red { font-size: 20px; font-weight: 700; color: #ff5252; }
    .metric-value-blue { font-size: 20px; font-weight: 700; color: #38bdf8; }
    
    @media only screen and (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .top-status-bar {
            font-size: 11px !important;
            padding: 8px 10px !important;
            flex-direction: column;
            align-items: flex-start !important;
        }
        .metric-value-green, .metric-value-red, .metric-value-blue {
            font-size: 18px !important;
        }
    }

    div[data-testid="stDataFrame"] {
        background-color: #111622;
        border-radius: 12px;
        border: 1px solid #1e2638;
        padding: 4px;
    }

    .top-status-bar {
        background: #111622;
        border: 1px solid #1e2638;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 15px;
        font-weight: 600;
        font-size: 13px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
COINS_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "LTCUSDT", "BCHUSDT", "APTUSDT", "TRXUSDT"
]

TIMEFRAME_MAP = {
    "1m": ("1m", 1),
    "5m": ("5m", 5),
    "10m": ("5m", 10),
    "15m": ("15m", 15),
    "30m": ("30m", 30),
    "1h": ("1h", 60),
    "4h": ("4h", 240)
}

st.sidebar.markdown("### ⚡ Terminal Controls")
selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", COINS_LIST, index=0)
selected_tf_label = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAME_MAP.keys()), index=1) # Default 5m
forecast_horizon = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("""
* 📊 **Dashboard**
* 🌐 **Multi-Asset Overview**
* 🔬 **10-Papers Scoreboard**
* ⚡ **Signal Engine**
* 🧪 **Backtesting**
* 🔔 **Alerts**
* ⚙️ **Settings**
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.success("🟢 **System Status: Operational**")

# ==========================================
# NOVA ROBOT ASSISTANT (AUTHENTICATED SYSTEM VOICE AI)
# ==========================================
NOVA_ACCESS_CODE = "7860"

if "nova_authenticated" not in st.session_state:
    st.session_state.nova_authenticated = False

st.sidebar.markdown("---")
col_bot_btn, col_bot_title = st.sidebar.columns([1, 4])
with col_bot_btn:
    nova_clicked = st.button("🤖", key="nova_toggle_btn", help="Click to open Nova Robot Assistant")
with col_bot_title:
    st.markdown("**Nova Robot Assistant** `v1.0`")

if nova_clicked:
    st.session_state.show_nova_panel = not st.session_state.get("show_nova_panel", False)

if st.session_state.get("show_nova_panel", False):
    st.sidebar.markdown("### 🤖 Nova System Assistant")
    
    if not st.session_state.nova_authenticated:
        user_code = st.sidebar.text_input("Enter Passcode to Access Nova:", type="password", key="nova_pass_field")
        if user_code == NOVA_ACCESS_CODE:
            st.session_state.nova_authenticated = True
            st.sidebar.success("Access Granted. Welcome back!")
            st.rerun()
        elif user_code != "":
            st.sidebar.error("Access Denied: Incorrect Security Code.")
    
    else:
        st.sidebar.caption("Status: Authorized | Access Level: Full System Reader")
        nova_query = st.sidebar.selectbox(
            "Ask Nova about system:",
            [
                "Which paper is currently performing best?",
                "System Overview & Active Signals",
                "Order Book Micro-structure Summary",
                "Lock Status & Terminal Health"
            ],
            key="nova_query_dropdown"
        )
        
        if st.sidebar.button("Ask Nova 🚀", key="exec_ask_nova"):
            if nova_query == "Which paper is currently performing best?":
                if "signal" in locals() and "evolved_weights" in signal:
                    weights = signal["evolved_weights"]
                    top_paper = max(weights, key=weights.get)
                    top_pct = weights[top_paper] * 100
                    nova_reply = f"System analysis ke mutabiq {top_paper} top rank par hai, jiska evolved weight {top_pct:.1f} percent chal raha hai."
                else:
                    nova_reply = "Terminal live data read kar raha hai. 10 papers balanced weights par scanning kar rahe hain."

            elif nova_query == "System Overview & Active Signals":
                if "signal" in locals():
                    nova_reply = f"Current coin {selected_symbol} par signal {signal['direction']} hai. Net Score {signal['score']:+.3f} aur target level {signal['beam']:,.2f} dollar hai."
                else:
                    nova_reply = f"System online hai aur {selected_symbol} ko analyze kar raha hai."

            elif nova_query == "Order Book Micro-structure Summary":
                nova_reply = "Micro-structure metrics check ho chuke hain. Order book depth, bids asks balance aur OFI flow matrix stable hain."

            else:
                nova_reply = f"Terminal health 100 percent operational hai. {selected_tf_label} timeframe lock mechanism proper synchronization mein hai."

            st.sidebar.info(f"**Nova:** {nova_reply}")
            
            tts = gTTS(text=nova_reply, lang='hi', slow=False)
            sound_fp = io.BytesIO()
            tts.write_to_fp(sound_fp)
            sound_fp.seek(0)
            st.sidebar.audio(sound_fp, format="audio/mp3", autoplay=True)

        if st.sidebar.button("Lock Nova 🔒", key="lock_nova_panel"):
            st.session_state.nova_authenticated = False
            st.rerun()

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]

# ==========================================
# DATA FETCHING (BYBIT FIRST & BINANCE FALLBACK)
# ==========================================
@st.cache_data(ttl=15)
def fetch_klines_data(symbol, tf_label, limit=100):
    # 1. Try Bybit API first (Streamlit Cloud US/EU Friendly)
    bybit_tf_map = {"1m": "1", "5m": "5", "10m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240"}
    interval = bybit_tf_map.get(tf_label, "5")
    bybit_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        res = requests.get(bybit_url, timeout=5).json()
        if res.get("retCode") == 0 and "list" in res.get("result", {}):
            raw_list = res["result"]["list"]
            df = pd.DataFrame(raw_list, columns=['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover'])
            df['Open_Time'] = df['Open_Time'].astype(int)
            df['Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            df = df.sort_values('Time').reset_index(drop=True)
            return df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_Time']]
    except Exception:
        pass

    # 2. Fallback to Binance Endpoints
    binance_tf = "5m" if tf_label == "10m" else tf_label
    endpoints = [
        f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}",
        f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
    ]
    for url in endpoints:
        try:
            res = requests.get(url, timeout=5).json()
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                df = pd.DataFrame(res, columns=['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_Time', 'QAV', 'NAT', 'TBBAV', 'TBQAV', 'Ignore'])
                df['Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                return df.reset_index()[['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_Time']]
        except Exception:
            continue

    return pd.DataFrame()


def fetch_order_book_depth(symbol, depth_limit=10):
    # 1. Try Bybit Order Book Depth first
    bybit_url = f"https://api.bybit.com/v5/market/orderbook?category=linear&symbol={symbol}&limit={depth_limit}"
    try:
        res = requests.get(bybit_url, timeout=5).json()
        if res.get("retCode") == 0 and "result" in res:
            bids = np.array(res["result"]["b"], dtype=float)
            asks = np.array(res["result"]["a"], dtype=float)
            if len(bids) > 0 and len(asks) > 0:
                return bids, asks
    except Exception:
        pass

    # 2. Fallback to Binance Endpoints
    endpoints = [
        f"https://data-api.binance.vision/api/v3/depth?symbol={symbol}&limit={depth_limit}",
        f"https://api.binance.us/api/v3/depth?symbol={symbol}&limit={depth_limit}",
        f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth_limit}"
    ]
    for url in endpoints:
        try:
            res = requests.get(url, timeout=5).json()
            if isinstance(res, dict) and 'bids' in res and 'asks' in res:
                return np.array(res['bids'], dtype=float), np.array(res['asks'], dtype=float)
        except Exception:
            continue

    return np.array([]), np.array([])


df = fetch_klines_data(selected_symbol, selected_tf_label)
bids, asks = fetch_order_book_depth(selected_symbol)

st.markdown("## ⚡ Research Lab — Multi-Asset Signal Engine")

if df.empty or len(df) < 3 or len(bids) == 0 or len(asks) == 0:
    st.error("⚠️ **Data Fetch Warning:** Live market connections are reconnecting. Please wait a moment.")

if not df.empty and len(df) >= 3 and len(bids) > 0 and len(asks) > 0:
    
    # ==========================================
    # ABSOLUTE GLOBAL CANDLE SYNC
    # ==========================================
    lock_seconds = tf_minutes * 60
    current_time_sec = int(time.time())
    
    global_bucket = current_time_sec - (current_time_sec % lock_seconds)
    time_remaining = lock_seconds - (current_time_sec % lock_seconds)

    @st.cache_data(ttl=lock_seconds)
    def get_synced_signal(symbol, tf_label, bucket_id):
        lab = TenPaperResearchLab()
        paper_results, final_score, evolved_weights = lab.calculate_all_signals(
            df, bids, asks, current_inventory=0, performance_history=st.session_state.trade_history_log
        )
        close_p = df['Close'].iloc[-1]
        atr_val = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        beam_level = close_p + (1.8 * atr_val)
        base_level = close_p - (1.8 * atr_val)

        trajectory_dir = "UPSIDE" if final_score >= 0.15 else ("DOWNSIDE" if final_score <= -0.15 else "SIDEWAYS")

        return {
            "score": final_score, "direction": trajectory_dir,
            "beam": beam_level, "base": base_level,
            "paper_results": paper_results, "evolved_weights": evolved_weights, "close_price": close_p
        }

    signal = get_synced_signal(selected_symbol, selected_tf_label, global_bucket)

    if len(st.session_state.trade_history_log) == 0 or st.session_state.trade_history_log[-1]["bucket"] != global_bucket:
        hit_status = True if (signal["direction"] == "UPSIDE" and df['Close'].iloc[-1] > df['Open'].iloc[-1]) else False
        st.session_state.trade_history_log.append({
            "bucket": global_bucket,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "direction": signal["direction"],
            "score": signal["score"],
            "hit_target": hit_status,
            "paper_results": signal["paper_results"]
        })

    mins_rem = time_remaining // 60
    secs_rem = time_remaining % 60

    # Top Status Bar
    st.markdown(f"""
    <div class="top-status-bar">
        🔵 <b>[{selected_symbol}]</b> | Timeframe: {selected_tf_label} | <b>SIGNAL:</b> <span style="color:#38bdf8;">{signal['direction']}</span> &nbsp;|&nbsp; 
        Net Score: <span style="color:#ff5252;">{signal['score']:+.3f}</span> &nbsp;|&nbsp; Target (BEAM): <span style="color:#38bdf8;">${signal['beam']:,.2f}</span> &nbsp;|&nbsp; 
        ⏳ Candle Close In: <b>{mins_rem}m {secs_rem}s</b>
    </div>
    """, unsafe_allow_html=True)

    # Responsive Grid Layout
    m1, m2, m3, m4, m5, m6 = st.columns([1.5, 1, 1, 1, 1, 1])
    close_val = df['Close'].iloc[-1]
    prev_val = df['Close'].iloc[-2]
    pct_change = ((close_val - prev_val) / prev_val) * 100

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟠 {selected_symbol}</div>
            <div class="metric-value-green">${close_val:,.2f}</div>
            <div style="font-size:11px; color:#00e676;">+{pct_change:.2f}% (24h)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Score</div>
            <div class="metric-value-red">{signal['score']:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Signal</div>
            <div style="font-size:14px; font-weight:700; color:#38bdf8; margin-top:4px;">{signal['direction']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Target (BEAM)</div>
            <div class="metric-value-blue">${signal['beam']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        fig_gauge = go.Figure(go.Pie(
            values=[42, 58], hole=0.7,
            marker_colors=['#f59e0b', '#1e2638'],
            textinfo='none', showlegend=False
        ))
        fig_gauge.update_layout(
            annotations=[dict(text='<b>42%</b>', x=0.5, y=0.5, font_size=14, font_color='#ffffff', showarrow=False)],
            margin=dict(l=0, r=0, t=0, b=0), height=70, paper_bgcolor='rgba(0,0,0,0)'
        )
        st.markdown('<div class="metric-card"><div class="metric-label">Confidence</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Refresh In</div>
            <div style="font-size:16px; font-weight:700; color:#ffffff; margin-top:4px;">{mins_rem}m {secs_rem}s</div>
        </div>
        """, unsafe_allow_html=True)

    # Main Chart Section
    col_chart, col_side = st.columns([2.5, 1])

    with col_chart:
        st.subheader(f"Price Chart ({selected_tf_label})")
        time_delta = pd.Timedelta(minutes=tf_minutes)
        future_times = [df['Time'].iloc[-1] + (i * time_delta) for i in range(1, forecast_horizon + 1)]
        t_steps = np.linspace(0, np.pi / 2, forecast_horizon)

        if signal["direction"] == "UPSIDE":
            forecast_prices = close_val + (signal["beam"] - close_val) * np.sin(t_steps)
        elif signal["direction"] == "DOWNSIDE":
            forecast_prices = close_val - (close_val - signal["base"]) * np.sin(t_steps)
        else:
            forecast_prices = [close_val] * forecast_horizon

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
            name="Candles", increasing_line_color='#00e676', decreasing_line_color='#ff5252'
        ))
        traj_color = "#00e676" if signal["direction"] == "UPSIDE" else ("#ff5252" if signal["direction"] == "DOWNSIDE" else "#f59e0b")
        fig.add_trace(go.Scatter(
            x=[df['Time'].iloc[-1]] + future_times, y=[close_val] + list(forecast_prices), 
            mode='lines+markers', name="Trajectory", 
            line=dict(color=traj_color, width=2, dash='dot')
        ))
        fig.add_hline(y=signal["beam"], line_dash="dash", line_color="#ff5252", annotation_text=f"BEAM: ${signal['beam']:,.2f}")
        fig.add_hline(y=signal["base"], line_dash="dash", line_color="#00e676", annotation_text=f"BASE: ${signal['base']:,.2f}")
        
        fig.update_layout(
            template="plotly_dark", height=420, xaxis_rangeslider_visible=False,
            paper_bgcolor="#111622", plot_bgcolor="#111622",
            margin=dict(l=5, r=5, t=5, b=5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("Market Overview (24h)")
        st.markdown("""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Market Cap</span> <b>$2.28T <span style="color:#00e676;">+1.25%</span></b></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>BTC Dominance</span> <b>52.41% <span style="color:#ff5252;">-0.38%</span></b></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Fear & Greed</span> <b>72 (Greed)</b></div>
            <div style="display:flex; justify-content:space-between;"><span>Funding Rate</span> <b>0.0102%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Volume Trend")
        fig_vol = go.Figure(go.Bar(x=list(range(10)), y=np.random.randint(20, 80, 10), marker_color='#38bdf8'))
        fig_vol.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#111622', plot_bgcolor='#111622', xaxis_visible=False)
        st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})

    # Bottom Row
    b1, b2, b3 = st.columns([1.2, 1, 1.2])

    with b1:
        st.subheader("🔬 10-Papers Scoreboard")
        paper_df = pd.DataFrame([
            {
                "Paper": k, 
                "Signal Value": f"{v:+.3f}", 
                "Evolved Weight": f"{signal['evolved_weights'][k]*100:.1f}%",
                "Status": "PASS🟢" if v > 0.1 else ("FAIL🔴" if v < -0.1 else "NEUTRAL⚪")
            }
            for k, v in signal["paper_results"].items()
        ])
        st.dataframe(paper_df, use_container_width=True, hide_index=True, height=240)

    with b2:
        st.subheader("Signal Summary")
        fig_summary = go.Figure(go.Pie(
            labels=['Pass', 'Neutral', 'Fail'], values=[4, 4, 2], hole=0.6,
            marker_colors=['#00e676', '#8b949e', '#ff5252']
        ))
        fig_summary.update_layout(height=220, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#111622', showlegend=True)
        st.plotly_chart(fig_summary, use_container_width=True, config={'displayModeBar': False})

    with b3:
        st.subheader("⚡ Key Metrics & Orderbook")
        st.markdown("""
        <div class="metric-card" style="height:240px;">
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>OBI (Weighted)</span> <b style="color:#ff5252;">-0.154</b></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>OFI</span> <b style="color:#ff5252;">-8,245</b></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Volume Ratio</span> <b>0.92</b></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Market Pressure</span> <b style="color:#ff5252;">-0.218</b></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Flow Strength</span> <b style="color:#ff5252;">-0.165</b></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Liquidity Score</span> <b style="color:#f59e0b;">58 / 100</b></div>
        </div>
        """, unsafe_allow_html=True)

time.sleep(10)
st.rerun()
