import time
import datetime
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

from src.research_lab import TenPaperResearchLab

st.set_page_config(page_title="10-Paper Research Lab Terminal (Balanced Signals)", layout="wide")
st.title("⚡ Research Lab — Multi-Asset & Balanced Signal Engine")

COINS_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "LTCUSDT", "BCHUSDT", "APTUSDT", "TRXUSDT",
    "SHIBUSDT", "UNIUSDT", "ATOMUSDT", "SUIUSDT", "INJUSDT", "ICPUSDT"
]

TIMEFRAME_MAP = {
    "1m": ("1m", 1),
    "10m": ("5m", 10),
    "15m": ("15m", 15),
    "30m": ("30m", 30),
    "1h": ("1h", 60),
    "4h": ("4h", 240)
}

st.sidebar.header("📊 Terminal Controls")
selected_symbol = st.sidebar.selectbox("Select Cryptocurrency (20+ Coins)", COINS_LIST, index=0)
selected_tf_label = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAME_MAP.keys()), index=2)
forecast_horizon = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 15)

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]

if "locked_signal" not in st.session_state:
    st.session_state.locked_signal = None
if "lock_timestamp" not in st.session_state:
    st.session_state.lock_timestamp = 0
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = selected_symbol
if "last_tf" not in st.session_state:
    st.session_state.last_tf = selected_tf_label

# Reset on change of coin or timeframe
if (st.session_state.last_symbol != selected_symbol) or (st.session_state.last_tf != selected_tf_label):
    st.session_state.locked_signal = None
    st.session_state.lock_timestamp = 0
    st.session_state.last_symbol = selected_symbol
    st.session_state.last_tf = selected_tf_label

@st.cache_data(ttl=60)
def fetch_klines_data(symbol, tf_label, limit=100):
    binance_tf = "5m" if tf_label == "10m" else tf_label
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
    res = requests.get(url).json()
    
    if isinstance(res, dict) and "code" in res:
        st.error(f"Binance API Error: {res.get('msg', 'Failed to fetch data')}")
        return pd.DataFrame()
        
    df = pd.DataFrame(res, columns=['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_Time', 'QAV', 'NAT', 'TBBAV', 'TBQAV', 'Ignore'])
    df['Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    df.set_index('Time', inplace=True)
    
    if tf_label == "10m":
        df_resampled = df.resample('10min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna().reset_index()
    else:
        df_resampled = df.reset_index()[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    return df_resampled

def fetch_order_book_depth(symbol, depth_limit=10):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth_limit}"
    res = requests.get(url).json()
    bids = np.array(res['bids'], dtype=float)
    asks = np.array(res['asks'], dtype=float)
    return bids, asks

df = fetch_klines_data(selected_symbol, selected_tf_label)
bids, asks = fetch_order_book_depth(selected_symbol)

if not df.empty and len(df) >= 3:
    current_time = time.time()
    lock_duration = 15 * 60

    if st.session_state.locked_signal is None or (current_time - st.session_state.lock_timestamp) >= lock_duration:
        lab = TenPaperResearchLab()
        paper_results, final_score = lab.calculate_all_signals(df, bids, asks)

        close_p = df['Close'].iloc[-1]
        atr_val = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        beam_level = close_p + (1.8 * atr_val)
        base_level = close_p - (1.8 * atr_val)

        # ADJUSTED BALANCED THRESHOLDS
        if final_score >= 0.15:
            trajectory_dir = "UPSIDE"
        elif final_score <= -0.15:
            trajectory_dir = "DOWNSIDE"
        else:
            trajectory_dir = "SIDEWAYS"

        st.session_state.locked_signal = {
            "score": final_score,
            "direction": trajectory_dir,
            "beam": beam_level,
            "base": base_level,
            "paper_results": paper_results,
            "close_price": close_p
        }
        st.session_state.lock_timestamp = current_time

    signal = st.session_state.locked_signal
    time_remaining = int(lock_duration - (current_time - st.session_state.lock_timestamp))
    mins_rem = time_remaining // 60
    secs_rem = time_remaining % 60

    if signal["direction"] == "UPSIDE":
        st.success(f"🟢 **[{selected_symbol} | {selected_tf_label}] BEST SIGNAL: STRONG LONG** | Net Score: {signal['score']:+.3f} | Target (BEAM): ${signal['beam']:,.2f} | ⏳ Refresh in: {mins_rem}m {secs_rem}s")
    elif signal["direction"] == "DOWNSIDE":
        st.error(f"🔴 **[{selected_symbol} | {selected_tf_label}] BEST SIGNAL: STRONG SHORT** | Net Score: {signal['score']:+.3f} | Target (BASE): ${signal['base']:,.2f} | ⏳ Refresh in: {mins_rem}m {secs_rem}s")
    else:
        st.info(f"⚪ **[{selected_symbol} | {selected_tf_label}] SIGNAL: NEUTRAL / STAND ASIDE** | Net Score: {signal['score']:+.3f} | ⏳ Refresh in: {mins_rem}m {secs_rem}s")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🔬 10-Papers Scoreboard")
        paper_df = pd.DataFrame([
            {"Paper": k, "Signal Value": f"{v:+.3f}", "Status": "PASS🟢" if v > 0.1 else ("FAIL🔴" if v < -0.1 else "NEUTRAL⚪")}
            for k, v in signal["paper_results"].items()
        ])
        st.dataframe(paper_df, use_container_width=True, hide_index=True)

    with col_right:
        time_delta = pd.Timedelta(minutes=tf_minutes)
        future_times = [df['Time'].iloc[-1] + (i * time_delta) for i in range(1, forecast_horizon + 1)]
        t_steps = np.linspace(0, np.pi / 2, forecast_horizon)

        close_p = df['Close'].iloc[-1]
        if signal["direction"] == "UPSIDE":
            forecast_prices = close_p + (signal["beam"] - close_p) * np.sin(t_steps)
        elif signal["direction"] == "DOWNSIDE":
            forecast_prices = close_p - (close_p - signal["base"]) * np.sin(t_steps)
        else:
            forecast_prices = [close_p] * forecast_horizon

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
            name=f"{selected_tf_label} Candles", increasing_line_color='#00F0FF', decreasing_line_color='#FF0055'
        ))
        
        traj_color = "#00FF66" if signal["direction"] == "UPSIDE" else ("#FF0055" if signal["direction"] == "DOWNSIDE" else "#FFFF00")
        fig.add_trace(go.Scatter(
            x=[df['Time'].iloc[-1]] + future_times, y=[close_p] + list(forecast_prices), 
            mode='lines+markers', name="Locked Trajectory", 
            line=dict(color=traj_color, width=3, dash='dot')
        ))
        
        fig.add_hline(y=signal["beam"], line_dash="dash", line_color="#FF0055", annotation_text=f"BEAM: ${signal['beam']:,.2f}")
        fig.add_hline(y=signal["base"], line_dash="dash", line_color="#00FFCC", annotation_text=f"BASE: ${signal['base']:,.2f}")
        fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


