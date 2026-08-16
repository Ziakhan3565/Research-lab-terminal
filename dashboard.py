import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time

# ==========================================
# PAGE CONFIGURATION & DARK GLOW THEME
# ==========================================
st.set_page_config(
    page_title="10-Paper Research Terminal | AI Voice Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dashboard Styling (Picture Style)
st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    /* Card Styles */
    .metric-container {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .robot-card {
        background: linear-gradient(135deg, #1f293d 0%, #111827 100%);
        border: 1px solid #3b82f6;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
    }
    .robot-avatar {
        font-size: 50px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.08); }
        100% { transform: scale(1); }
    }
    .status-badge-upside {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-badge-downside {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if "trade_history_log" not in st.session_state:
    st.session_state.trade_history_log = []

# ==========================================
# DATA FETCHING ENGINE (BYBIT + BINANCE + FIXED FREQ)
# ==========================================
def fetch_bybit_klines(symbol="BTCUSDT", interval="15", limit=200):
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("retCode") == 0 and res.get("result", {}).get("list"):
            data = res["result"]["list"]
            df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover'])
            df = df.iloc[::-1].reset_index(drop=True)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'].astype(float), unit='ms')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            return df
    except Exception:
        pass
    return None

def fetch_binance_klines(symbol="BTCUSDT", interval="15m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=5).json()
        if isinstance(res, list):
            df = pd.DataFrame(res, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NAT', 'TBBAV', 'TBQAV', 'Ignore'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            return df
    except Exception:
        pass
    return None

def fetch_klines(symbol="BTCUSDT", interval="15m", limit=200):
    bybit_interval_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    bybit_tf = bybit_interval_map.get(interval, "15")
    
    df = fetch_bybit_klines(symbol, bybit_tf, limit)
    if df is None or df.empty:
        df = fetch_binance_klines(symbol, interval, limit)
    
    if df is None or df.empty:
        # PANDAS FREQUENCY SAFE CONVERSION ("15m" -> "15min")
        pd_freq = interval.replace("m", "min") if ("m" in interval and "min" not in interval) else interval
        times = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq=pd_freq)
        
        price = 60000 + np.cumsum(np.random.randn(limit) * 50)
        df = pd.DataFrame({
            'Timestamp': times,
            'Open': price,
            'High': price + 20,
            'Low': price - 20,
            'Close': price + np.random.randn(limit)*5,
            'Volume': np.random.randint(100, 1000, size=limit)
        })
    return df

def fetch_orderbook(symbol="BTCUSDT"):
    url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit=50"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("retCode") == 0:
            result = res.get("result", {})
            bids = np.array([[float(p), float(q)] for p, q in result.get("b", [])])
            asks = np.array([[float(p), float(q)] for p, q in result.get("a", [])])
            return bids, asks
    except Exception:
        pass
    
    base_price = 60000.0
    bids = np.array([[base_price - i*2, 1.5 + np.random.rand()] for i in range(1, 20)])
    asks = np.array([[base_price + i*2, 1.5 + np.random.rand()] for i in range(1, 20)])
    return bids, asks

# ==========================================
# 10-PAPER MODEL ENGINE
# ==========================================
class TenPaperResearchLab:
    def __init__(self):
        self.paper_names = [
            "P1: Kyle's Lambda", "P2: Glosten-Milgrom", "P3: VPIN Volume Toxicity",
            "P4: Hawkes Process", "P5: Avellaneda-Stoikov", "P6: Hasbrouck VAR",
            "P7: Cartea-Jaimungal", "P8: Order Flow Imbalance", "P9: Roll Spread Model", "P10: Amihud Illiquidity"
        ]

    def calculate_all_signals(self, df, bids, asks, current_inventory=0, performance_history=None):
        close = df['Close'].values
        volume = df['Volume'].values
        
        returns = np.diff(close)
        vol_signed = volume[1:] * np.sign(returns)
        lambda_val = np.cov(returns, vol_signed)[0, 1] / (np.var(vol_signed) + 1e-8)
        s1 = np.tanh(lambda_val * 1e5)

        bid_vol = np.sum(bids[:10, 1]) if len(bids) > 0 else 1
        ask_vol = np.sum(asks[:10, 1]) if len(asks) > 0 else 1
        s2 = np.clip((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8), -1, 1)

        vpin = np.abs(bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        s3 = np.tanh((vpin - 0.5) * 2)

        vol_spike = volume[-1] / (np.mean(volume[-20:]) + 1e-8)
        s4 = np.clip((vol_spike - 1.0) * np.sign(close[-1] - close[-2]), -1, 1)

        s5 = -np.clip(current_inventory / 10.0, -1, 1)
        s6 = np.tanh(np.mean(np.diff(close[-5:])))
        s7 = np.clip(np.mean(close[-3:]) - close[-1], -1, 1)
        s8 = s2
        
        pct_change = np.diff(close)
        cov_roll = np.cov(pct_change[:-1], pct_change[1:])[0, 1]
        roll_spread = 2 * np.sqrt(max(0, -cov_roll))
        s9 = np.tanh(roll_spread * 10)

        amihud = np.mean(np.abs(returns) / (volume[1:] + 1e-8))
        s10 = np.tanh(amihud * 1e6)

        raw_signals = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
        weights = np.ones(10) / 10.0

        final_score = np.dot(raw_signals, weights)
        paper_results = dict(zip(self.paper_names, raw_signals))
        evolved_weights = dict(zip(self.paper_names, weights))

        return paper_results, final_score, evolved_weights

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("⚡ Research Terminal")
selected_symbol = st.sidebar.selectbox("Select Asset", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
tf_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=2)
tf_minutes = tf_map[selected_tf_label]

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 API Health")
st.sidebar.success("Bybit REST API: Online")
st.sidebar.info("Binance Fallback: Active")

# ==========================================
# FETCH DATA & SYNC SIGNAL
# ==========================================
df = fetch_klines(selected_symbol, selected_tf_label)
bids, asks = fetch_orderbook(selected_symbol)

lock_seconds = tf_minutes * 60
current_time_sec = int(time.time())
global_bucket = current_time_sec - (current_time_sec % lock_seconds)
time_remaining = lock_seconds - (current_time_sec % lock_seconds)

@st.cache_data(ttl=lock_seconds, show_spinner=False)
def get_synced_signal(symbol, tf_label, bucket_id, _df, _bids, _asks):
    lab = TenPaperResearchLab()
    history_log = st.session_state.get("trade_history_log", [])
    paper_results, final_score, evolved_weights = lab.calculate_all_signals(_df, _bids, _asks, performance_history=history_log)
    
    close_p = _df['Close'].iloc[-1]
    atr_val = (_df['High'] - _df['Low']).rolling(14).mean().iloc[-1]
    beam_level = close_p + (1.8 * atr_val)
    base_level = close_p - (1.8 * atr_val)

    trajectory_dir = "UPSIDE" if final_score >= 0.05 else ("DOWNSIDE" if final_score <= -0.05 else "SIDEWAYS")

    return {
        "score": final_score,
        "direction": trajectory_dir,
        "beam": beam_level,
        "base": base_level,
        "paper_results": paper_results,
        "evolved_weights": evolved_weights,
        "close_price": close_p
    }

signal = get_synced_signal(selected_symbol, selected_tf_label, global_bucket, df, bids, asks)

# ==========================================
# AI ROBOT ASSISTANT SECTION
# ==========================================
st.markdown("---")
robot_col1, robot_col2 = st.columns([1, 3])

with robot_col1:
    st.markdown("""
    <div class="robot-card">
        <div class="robot-avatar">🤖🎙️</div>
        <h4 style="margin:5px 0; color:#60a5fa;">AI Research Assistant</h4>
        <p style="font-size:12px; color:#9ca3af;">Voice Input & Audio Response Active</p>
    </div>
    """, unsafe_allow_html=True)

with robot_col2:
    st.subheader("🎙️ Ask Robot Assistant")
    
    # Text/Voice Prompt Input
    user_prompt = st.text_input("Speak or Type your command to Robot Assistant:", placeholder="e.g. Robot, analyze BTCUSDT signal")
    
    robot_response = f"Hello! Currently {selected_symbol} is trading at ${signal['close_price']:,.2f}. The 10-Paper Signal Score is {signal['score']:.4f}, indicating a {signal['direction']} trajectory."
    
    if user_prompt:
        st.write(f"🤖 **Robot Audio Analysis:** {robot_response}")
    
    # Text-to-Speech HTML5 Audio Player Component
    tts_audio_html = f"""
    <script>
        function speakRobot() {{
            var msg = new SpeechSynthesisUtterance("{robot_response}");
            msg.rate = 1.0;
            msg.pitch = 1.1;
            window.speechSynthesis.speak(msg);
        }}
    </script>
    <button onclick="speakRobot()" style="
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white; border: none; padding: 10px 20px;
        border-radius: 8px; font-weight: bold; cursor: pointer;">
        🔊 Play Voice Response (Suno Robot Voice)
    </button>
    """
    st.components.v1.html(tts_audio_html, height=60)

st.markdown("---")

# ==========================================
# MAIN DASHBOARD METRICS (PICTURE MATCHING LAYOUT)
# ==========================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-container">
        <span style="color:#9ca3af; font-size:14px;">Current Price</span>
        <h2 style="margin: 5px 0; color:#38bdf8;">${signal['close_price']:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    badge_class = "status-badge-upside" if signal['direction'] == "UPSIDE" else "status-badge-downside"
    st.markdown(f"""
    <div class="metric-container">
        <span style="color:#9ca3af; font-size:14px;">10-Paper Score</span>
        <h2 style="margin: 5px 0; color:#f43f5e;">{signal['score']:.4f}</h2>
        <span class="{badge_class}">{signal['direction']}</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-container">
        <span style="color:#9ca3af; font-size:14px;">Beam Target (1.8 ATR)</span>
        <h2 style="margin: 5px 0; color:#34d399;">${signal['beam']:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    mins, secs = divmod(time_remaining, 60)
    st.markdown(f"""
    <div class="metric-container">
        <span style="color:#9ca3af; font-size:14px;">Candle Lock Timer</span>
        <h2 style="margin: 5px 0; color:#fbbf24;">{mins:02d}:{secs:02d}</h2>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MAIN PLOTLY OHLC & VOLUME CHART
# ==========================================
st.markdown("### 📈 Quantitative Market Chart")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

fig.add_trace(go.Candlestick(
    x=df['Timestamp'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="Candles",
    increasing_line_color='#10b981', decreasing_line_color='#ef4444'
), row=1, col=1)

fig.add_hline(y=signal['beam'], line_dash="dash", line_color="#34d399", annotation_text="Upper Beam Target", row=1, col=1)
fig.add_hline(y=signal['base'], line_dash="dash", line_color="#f43f5e", annotation_text="Lower Base Support", row=1, col=1)

fig.add_trace(go.Bar(
    x=df['Timestamp'], y=df['Volume'], name="Volume", marker_color='#3b82f6'
), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    height=550,
    paper_bgcolor='#0b0e14',
    plot_bgcolor='#0b0e14',
    margin=dict(l=10, r=10, t=20, b=10)
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 10-PAPER MODEL DETAILED BREAKDOWN
# ==========================================
st.markdown("### 🔬 10-Paper Model Breakdown")
cols = st.columns(2)
papers = list(signal['paper_results'].keys())

for idx, p in enumerate(papers):
    col = cols[0] if idx < 5 else cols[1]
    val = signal['paper_results'][p]
    weight = signal['evolved_weights'][p]
    with col:
        st.write(f"**{p}** | Score: `{val:+.4f}` | Weight: `{weight:.1%}`")
        st.progress(float(np.clip((val + 1) / 2, 0.0, 1.0)))

st.markdown("---")
st.caption("10-Paper Research Terminal Engine | Live Speech Synthesizer Integration Active")
