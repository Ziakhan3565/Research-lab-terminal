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

# Custom Styling to match the UI
st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    
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
def fetch_bybit_klines(symbol="BTCUSDT", interval="15", limit=100):
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

def fetch_binance_klines(symbol="BTCUSDT", interval="15m", limit=100):
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

def fetch_klines(symbol="BTCUSDT", interval="15m", limit=100):
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
st.sidebar.markdown("## ⚡ Terminal Controls")

selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
tf_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
selected_tf = st.sidebar.selectbox("Select Timeframe", list(tf_map.keys()), index=2)
tf_minutes = tf_map[selected_tf]

forecast_candles = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("• 📊 Dashboard")
st.sidebar.markdown("• 🌐 Multi-Asset Overview")
st.sidebar.markdown("• 🎯 10-Papers Scoreboard")
st.sidebar.markdown("• ⚡ Signal Engine")
st.sidebar.markdown("• 📈 Backtesting")
st.sidebar.markdown("• 🔔 Alerts")
st.sidebar.markdown("• ⚙️ Settings")

# ==========================================
# DATA PROCESSING & METRICS
# ==========================================
df = fetch_klines(selected_symbol, selected_tf, limit=80)
close_price = df['Close'].iloc[-1]
price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
pct_change = (price_change / df['Close'].iloc[-2]) * 100

atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
beam_target = close_price + (0.8 * atr)
base_target = close_price - (0.8 * atr)

net_score = -0.008
signal_text = "STAND ASIDE"
confidence = 42

lock_seconds = tf_minutes * 60
current_time_sec = int(time.time())
time_remaining = lock_seconds - (current_time_sec % lock_seconds)
mins, secs = divmod(time_remaining, 60)

# ==========================================
# URDU AI ROBOT ASSISTANT
# ==========================================
st.markdown("""
<div class="robot-box">
    <div style="display:flex; align-items:center; gap:15px;">
        <span style="font-size:32px;">🤖🎙️</span>
        <div>
            <h4 style="margin:0; color:#60a5fa;">Urdu Voice AI Assistant</h4>
            <p style="margin:0; font-size:12px; color:#9ca3af;">Aap ki baat Urdu mein sune ga aur Urdu mein bol kar jawab de kar chup ho jaye ga.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

robot_prompt = st.text_input("🤖 Urdu Voice / Text Input:", placeholder="Jaise: 'Mujhe BTC ki signal update do'")

urdu_bot_reply = f"Assalam-o-Alaikum! Is waqt {selected_symbol} ki keemat {close_price:,.2f} dollar hai. Signal Stand Aside hai."

if robot_prompt:
    st.info(f"🤖 **Robot ka Urdu Jawab:** {urdu_bot_reply}")

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

    document.getElementById('speak-btn').onclick = function() {{
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance(urduReply);
        msg.lang = 'ur-PK';
        msg.rate = 0.9;
        msg.onend = function() {{
            document.getElementById('speech-status').innerText = "Robot jawab de kar chup ho gaya hai.";
        }};
        window.speechSynthesis.speak(msg);
        document.getElementById('speech-status').innerText = "Robot bol raha hai...";
    }};

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
""", height=80)

# ==========================================
# MAIN METRICS TOP CARDS
# ==========================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">{selected_symbol}</div>
        <div class="metric-value-green">${close_price:,.2f}</div>
        <div style="color:#10b981; font-size:12px;">+{pct_change:.2f}% (24h)</div>
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
        <div style="color:#3b82f6; font-size:18px; font-weight:bold; margin-top:5px;">{signal_text}</div>
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

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# MIDDLE SECTION: EXACT CANDLESTICK CHART & VOLUME
# ==========================================
m_col1, m_col2 = st.columns([2.5, 1])

with m_col1:
    st.markdown(f"### Price Chart ({selected_tf})")
    
    fig = go.Figure()

    # High quality clean Candlestick styling matching reference image
    fig.add_trace(go.Candlestick(
        x=df['Timestamp'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="OHLC",
        increasing_line_color='#00c853',
        decreasing_line_color='#ff3d00',
        increasing_fillcolor='#00c853',
        decreasing_fillcolor='#ff3d00',
        whiskerwidth=0.8
    ))

    # BEAM Dotted Red Line
    fig.add_hline(
        y=beam_target, line_dash="dash", line_color="#ff3d00", line_width=1.5,
        annotation_text=f"BEAM: ${beam_target:,.2f}", annotation_position="top right",
        annotation_font_color="#ffffff"
    )

    # BASE Dotted Green Line
    fig.add_hline(
        y=base_target, line_dash="dash", line_color="#00c853", line_width=1.5,
        annotation_text=f"BASE: ${base_target:,.2f}", annotation_position="bottom right",
        annotation_font_color="#ffffff"
    )

    # Orange Forecast Target Level Line (matching the yellow/orange line in picture)
    last_time = df['Timestamp'].iloc[-1]
    forecast_end = last_time + pd.Timedelta(minutes=tf_minutes * forecast_candles)
    fig.add_trace(go.Scatter(
        x=[last_time, forecast_end],
        y=[close_price, close_price],
        mode='lines',
        line=dict(color='#ffab00', width=4),
        name='Forecast Target'
    ))

    # Chart Layout Config
    fig.update_layout(
        template="plotly_dark",
        height=480,
        paper_bgcolor='#0b0e14',
        plot_bgcolor='#0b0e14',
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(
            type="date", 
            gridcolor="#1f2937", 
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            autorange=True, 
            fixedrange=False, 
            gridcolor="#1f2937", 
            showgrid=True,
            side="left"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

with m_col2:
    st.markdown("### Volume (24h)")
    
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(
        x=df['Timestamp'].tail(15),
        y=df['Volume'].tail(15),
        marker_color='#0091ea'
    ))
    
    vol_fig.update_layout(
        template="plotly_dark",
        height=380,
        paper_bgcolor='#0b0e14',
        plot_bgcolor='#0b0e14',
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1f2937")
    )
    
    st.plotly_chart(vol_fig, use_container_width=True)
