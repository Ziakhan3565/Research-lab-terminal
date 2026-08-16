import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Research Lab — Multi-Asset & Balanced Signal Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to match the uploaded UI exactly
st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    
    /* Top Header Bar */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 10px;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 15px;
    }
    
    /* Grid Card System */
    .card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 15px;
        height: 100%;
    }
    
    .metric-title {
        color: #9ca3af;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 4px;
    }
    
    .metric-value-green {
        color: #10b981;
        font-size: 24px;
        font-weight: 700;
    }
    
    .metric-value-blue {
        color: #3b82f6;
        font-size: 24px;
        font-weight: 700;
    }
    
    .metric-value-red {
        color: #ef4444;
        font-size: 24px;
        font-weight: 700;
    }
    
    .status-pass { color: #10b981; font-weight: bold; }
    .status-fail { color: #ef4444; font-weight: bold; }
    .status-neutral { color: #9ca3af; font-weight: bold; }

    /* Robot Widget Box */
    .robot-box {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA FETCHING ENGINE
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
        pd_freq = interval.replace("m", "min") if ("m" in interval and "min" not in interval) else interval
        times = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq=pd_freq)
        price = 63000 + np.cumsum(np.random.randn(limit) * 40)
        df = pd.DataFrame({
            'Timestamp': times, 'Open': price, 'High': price + 15,
            'Low': price - 15, 'Close': price + np.random.randn(limit)*4,
            'Volume': np.random.randint(100, 1000, size=limit)
        })
    return df

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## ⚡ RESEARCH LAB")
st.sidebar.caption("Signal Engine")

st.sidebar.markdown("---")
st.sidebar.markdown("### CONTROLS")
selected_symbol = st.sidebar.selectbox("Select Coin", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
tf_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=2)
tf_minutes = tf_map[selected_tf]

forecast_candles = st.sidebar.slider("Forecast Candles", 5, 30, 15)

st.sidebar.markdown("---")
st.sidebar.button("📊 Dashboard", use_container_width=True)
st.sidebar.button("🎯 Scoreboard", use_container_width=True)
st.sidebar.button("⚡ Signal Engine", use_container_width=True)
st.sidebar.button("📈 Backtesting", use_container_width=True)
st.sidebar.button("🔔 Alerts", use_container_width=True)
st.sidebar.button("⚙️ Settings", use_container_width=True)

# ==========================================
# DATA PROCESSING & METRICS
# ==========================================
df = fetch_klines(selected_symbol, selected_tf)
close_price = df['Close'].iloc[-1]
price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
pct_change = (price_change / df['Close'].iloc[-2]) * 100

atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
beam_target = close_price + (1.2 * atr)
base_target = close_price - (1.2 * atr)

net_score = -0.136
signal_text = "NEUTRAL / STAND ASIDE"
confidence = 42

lock_seconds = tf_minutes * 60
current_time_sec = int(time.time())
time_remaining = lock_seconds - (current_time_sec % lock_seconds)
mins, secs = divmod(time_remaining, 60)

# ==========================================
# URDU AI ROBOT ASSISTANT (LISTEN & SPEAK URDU)
# ==========================================
st.markdown("""
<div class="robot-box">
    <div style="display:flex; align-items:center; gap:15px;">
        <span style="font-size:36px;">🤖🎙️</span>
        <div>
            <h4 style="margin:0; color:#60a5fa;">Urdu Voice AI Assistant</h4>
            <p style="margin:0; font-size:12px; color:#9ca3af;">Urdu mein baat sune aur Urdu mein reply karega (Sawal sunte hi jawab de kar chup ho jayega)</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

robot_prompt = st.text_input("🤖 Urdu Voice / Text Input:", placeholder="Jaise: 'Mujhe BTC ki signal update do'")

urdu_bot_reply = f"Assalam-o-Alaikum! Is waqt {selected_symbol} ki keemat {close_price:,.2f} dollar hai. Net score {net_score} hai aur signal Neutral hai."

if robot_prompt:
    st.info(f"🤖 **Robot ka Urdu Jawab:** {urdu_bot_reply}")

# Audio TTS & Speech-to-Text HTML/JS Block
st.components.v1.html(f"""
<div style="font-family:sans-serif; color:white;">
    <button id="start-btn" style="background:#2563eb; color:white; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-weight:bold;">
        🎙️ Urdu Mein Boliye (Mic)
    </button>
    <button id="speak-btn" style="background:#10b981; color:white; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-weight:bold; margin-left:10px;">
        🔊 Jawab Suniye (Urdu Speech)
    </button>
    <p id="speech-status" style="font-size:12px; color:#9ca3af; margin-top:5px;"></p>
</div>

<script>
    const urduReply = "{urdu_bot_reply}";

    // Speech Output (Speak Urdu and Auto Stop)
    document.getElementById('speak-btn').onclick = function() {{
        window.speechSynthesis.cancel(); // Reset previous voice
        var msg = new SpeechSynthesisUtterance(urduReply);
        msg.lang = 'ur-PK';
        msg.rate = 0.9;
        msg.onend = function() {{
            document.getElementById('speech-status').innerText = "Robot jawab de kar chup ho gaya hai.";
        }};
        window.speechSynthesis.speak(msg);
        document.getElementById('speech-status').innerText = "Robot bol raha hai...";
    }};

    // Speech Input (Listen Urdu)
    const startBtn = document.getElementById('start-btn');
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = 'ur-PK';
        
        startBtn.onclick = function() {{
            recognition.start();
            document.getElementById('speech-status').innerText = "Suntay hain... Boliye!";
        }};

        recognition.onresult = function(event) {{
            const transcript = event.results[0][0].transcript;
            document.getElementById('speech-status').innerText = "Aap ne kaha: " + transcript;
            
            // Speak reply automatically after listening
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance(urduReply);
            msg.lang = 'ur-PK';
            msg.onend = function() {{
                document.getElementById('speech-status').innerText = "Robot jawab de kar chup ho gaya hai.";
            }};
            window.speechSynthesis.speak(msg);
        }};
    }} else {{
        startBtn.style.display = 'none';
    }}
</script>
""", height=90)

# ==========================================
# MAIN DASHBOARD HEADER & TOP CARDS
# ==========================================
st.markdown("## ⚡ Research Lab — Multi-Asset & Balanced Signal Engine")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">{selected_symbol} Price</div>
        <div class="metric-value-green">${close_price:,.2f}</div>
        <div style="color:#10b981; font-size:12px;">+{price_change:.2f} (+{pct_change:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Net Score</div>
        <div class="metric-value-red">{net_score}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Signal</div>
        <div style="color:#3b82f6; font-size:16px; font-weight:bold; margin-top:8px;">{signal_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Target (BEAM)</div>
        <div class="metric-value-blue">${beam_target:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Confidence</div>
        <div class="metric-value-green">{confidence}%</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Refresh In</div>
        <div style="font-size:22px; font-weight:bold; color:white;">{mins}m {secs}s</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# MIDDLE SECTION: CHART + MARKET OVERVIEW
# ==========================================
m_col1, m_col2 = st.columns([2.6, 1])

with m_col1:
    st.markdown(f"### Price Chart ({selected_tf})")
    
    fig = make_subplots(rows=1, cols=1)
    
    # Balanced & Proportional Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df['Timestamp'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="OHLC",
        increasing_line_color='#10b981',
        decreasing_line_color='#ef4444',
        increasing_fillcolor='#10b981',
        decreasing_fillcolor='#ef4444'
    ))
    
    # Target Lines
    fig.add_hline(y=beam_target, line_dash="dash", line_color="#ef4444", annotation_text=f"BEAM: ${beam_target:,.2f}")
    fig.add_hline(y=base_target, line_dash="dash", line_color="#10b981", annotation_text=f"BASE: ${base_target:,.2f}")

    # Layout & Scaling Fixes for Balanced Candles
    fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor='#111827',
        plot_bgcolor='#111827',
        xaxis_rangeslider_visible=False,  # Range slider removed to fix squeeze
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(type="date", gridcolor="#1f2937"),
        yaxis=dict(autorange=True, fixedrange=False, gridcolor="#1f2937")
    )
    
    st.plotly_chart(fig, use_container_width=True)

with m_col2:
    st.markdown("### Market Overview (24h)")
    st.markdown("""
    <div class="card">
        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
            <span style="color:#9ca3af;">Market Cap</span>
            <span style="color:white; font-weight:bold;">$2.28T <span style="color:#10b981;">+1.25%</span></span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
            <span style="color:#9ca3af;">BTC Dominance</span>
            <span style="color:white; font-weight:bold;">52.41% <span style="color:#ef4444;">-0.38%</span></span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
            <span style="color:#9ca3af;">Fear & Greed Index</span>
            <span style="color:white; font-weight:bold;">72 (Greed)</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:15px;">
            <span style="color:#9ca3af;">Funding Rate</span>
            <span style="color:white; font-weight:bold;">0.0102%</span>
        </div>
        <hr style="border-color:#1f2937;">
        <div style="color:#9ca3af; font-size:12px;">Volume (24h)</div>
        <div style="font-size:18px; font-weight:bold; color:white;">24.68B USDT <span style="color:#ef4444; font-size:12px;">-5.21%</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# BOTTOM SECTION: SCOREBOARD + SUMMARY + RECENT SIGNALS
# ==========================================
b_col1, b_col2, b_col3 = st.columns([1.2, 1, 1])

with b_col1:
    st.markdown("### 10-Papers Scoreboard")
    scoreboard_data = {
        "Paper": ["OFI", "TSMOM", "MICRO", "AVST", "INVAR"],
        "Signal Value": ["-0.204", "+1.000", "+0.000", "+0.000", "+1.000"],
        "Status": ["FAIL 🔴", "PASS 🟢", "NEUTRAL ⚪", "NEUTRAL ⚪", "PASS 🟢"]
    }
    st.table(pd.DataFrame(scoreboard_data))

with b_col2:
    st.markdown("### Signal Summary")
    st.markdown("""
    <div class="card" style="text-align:center;">
        <h2 style="color:#10b981; font-size:36px; margin:10px 0;">42%</h2>
        <div style="color:#9ca3af; font-size:14px;">Confidence Score</div>
        <hr style="border-color:#1f2937; margin:15px 0;">
        <div style="display:flex; justify-content:space-around;">
            <div><span style="color:#9ca3af;">Total Score</span><br><b style="color:#ef4444;">-0.136</b></div>
            <div><span style="color:#9ca3af;">Pass Rate</span><br><b style="color:#10b981;">40%</b></div>
            <div><span style="color:#9ca3af;">Fail Rate</span><br><b style="color:#ef4444;">20%</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with b_col3:
    st.markdown("### Recent Signals")
    signals_data = {
        "Time": ["11:30 AM", "11:15 AM", "11:00 AM", "10:45 AM"],
        "Coin": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
        "Signal": ["NEUTRAL", "LONG", "NEUTRAL", "SHORT"],
        "Score": ["-0.136", "+0.186", "-0.028", "-0.212"]
    }
    st.table(pd.DataFrame(signals_data))
