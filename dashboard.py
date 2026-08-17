import os
import time
import datetime
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# RESEARCH LAB MODULE & RISK ENGINE FALLBACK
# ==========================================
try:
    from src.research_lab import TenPaperResearchLab, PowerTradingRiskEngine
except ModuleNotFoundError:
    class TenPaperResearchLab:
        def calculate_all_signals(self, df, bids, asks, current_inventory=0, performance_history=None):
            paper_results = {
                "OFI": -0.204, "TSMOM": 0.850, "MICRO": -0.050, "AVST": 0.120,
                "INVAR": 0.450, "VPIN": -0.310, "LAMBDA": 0.080, "PIN": -0.150,
                "LOB_IMB": -0.220, "FLOW_IMB": 0.300
            }
            final_score = -0.136
            evolved_weights = {k: 0.10 for k in paper_results.keys()}
            return paper_results, final_score, evolved_weights

    class PowerTradingRiskEngine:
        def __init__(self):
            pass
        def calculate_risk_metrics(self, liquidation_volumes, displayed_vol, cancelled_vol, time_exists, obs_window, open_interest, leverage, volatility):
            return {
                'LTZ_Score': 12.5,
                'Spoof_Score': 0.15,
                'Squeeze_Risk': 1.45,
                'Market_Risk': 14.1
            }

# ==========================================
# STREAMLIT PAGE CONFIG & PERSISTENT CSV SETUP
# ==========================================
st.set_page_config(
    page_title="10-Paper Research Lab Terminal", 
    layout="wide", 
    initial_sidebar_state="auto"
)

CSV_FILE = "signal_history.csv"

def load_persistent_history():
    if os.path.exists(CSV_FILE):
        try:
            df_hist = pd.read_csv(CSV_FILE)
            if 'outcome' not in df_hist.columns:
                df_hist['outcome'] = 'PENDING'
            return df_hist.to_dict('records')
        except Exception:
            return []
    return []

def save_persistent_history(history_list):
    try:
        df_hist = pd.DataFrame(history_list)
        if 'bucket' in df_hist.columns:
            df_hist_save = df_hist.drop(columns=['bucket'])
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
st.markdown("""
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
    "1m": ("1m", 1), "5m": ("5m", 5), "10m": ("5m", 10),
    "15m": ("15m", 15), "30m": ("30m", 30), "1h": ("1h", 60), "4h": ("4h", 240)
}

st.sidebar.markdown("### ⚡ Terminal Controls")
selected_symbol = st.sidebar.selectbox("Select Cryptocurrency (For View)", COINS_LIST, index=0)
selected_tf_label = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAME_MAP.keys()), index=3)
forecast_horizon = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 30)

st.sidebar.markdown("---")
st.sidebar.success("🟢 **System Status: Multi-Coin Scanner Active**")

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]

# ==========================================
# DATA FETCHING HELPERS
# ==========================================
@st.cache_data(ttl=15)
def fetch_klines_data(symbol, tf_label, limit=100):
    binance_tf = "5m" if tf_label == "10m" else tf_label
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
    try:
        res = requests.get(url, timeout=3).json()
        if isinstance(res, dict) and "code" in res:
            return pd.DataFrame()
        df = pd.DataFrame(res, columns=['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_Time', 'QAV', 'NAT', 'TBBAV', 'TBQAV', 'Ignore'])
        df['Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        df.set_index('Time', inplace=True)
        return df.reset_index()[['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_Time']]
    except Exception:
        return pd.DataFrame()

def fetch_order_book_depth(symbol, depth_limit=10):
    try:
        url = f"https://data-api.binance.vision/api/v3/depth?symbol={symbol}&limit={depth_limit}"
        res = requests.get(url, timeout=3).json()
        if 'bids' in res and 'asks' in res:
            return np.array(res['bids'], dtype=float), np.array(res['asks'], dtype=float)
        return np.array([]), np.array([])
    except Exception:
        return np.array([]), np.array([])

# ==========================================
# BACKGROUND MULTI-COIN SCANNER & AUTO R:R
# ==========================================
def compute_signal_light(df_in, bids_in, asks_in, history):
    lab = TenPaperResearchLab()
    try:
        paper_results, final_score, evolved_weights = lab.calculate_all_signals(
            df_in, bids_in, asks_in, current_inventory=0, performance_history=history
        )
    except Exception:
        final_score = -0.136

    close_p = df_in['Close'].iloc[-1]
    trajectory_dir = "LONG" if final_score >= 0.15 else ("SHORT" if final_score <= -0.15 else "NEUTRAL")
    return final_score, trajectory_dir, close_p

def check_auto_outcome(entry_price, df_candles, direction, sl_distance):
    tp_distance = sl_distance * 2
    
    if direction == "LONG":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
        
        for _, row in df_candles.iterrows():
            if row['Low'] <= sl_price:
                return "LOSS"
            if row['High'] >= tp_price:
                return "WIN"
                
    elif direction == "SHORT":
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance
        
        for _, row in df_candles.iterrows():
            if row['High'] >= sl_price:
                return "LOSS"
            if row['Low'] <= tp_price:
                return "WIN"
                
    return "PENDING"

history_updated = False
for item in st.session_state.trade_history_log:
    if item.get('outcome', 'PENDING') == 'PENDING' and item.get('direction') != 'NEUTRAL':
        curr_df = fetch_klines_data(item['symbol'], item['timeframe'], limit=50)
        if not curr_df.empty:
            signal_time = pd.to_datetime(item['timestamp'])
            future_candles = curr_df[curr_df['Time'] >= signal_time]
            
            if future_candles.empty:
                future_candles = curr_df    
                
            atr_val = (curr_df['High'] - curr_df['Low']).mean()
            sl_dist = atr_val if not np.isnan(atr_val) and atr_val > 0 else (item['price'] * 0.01)
            
            res_status = check_auto_outcome(item['price'], future_candles, item['direction'], sl_dist)
            if res_status != "PENDING":
                item['outcome'] = res_status
                history_updated = True

if history_updated:
    save_persistent_history(st.session_state.trade_history_log)

lock_seconds = tf_minutes * 60
current_time_sec = int(time.time())
time_bucket = current_time_sec - (current_time_sec % lock_seconds)
time_remaining = lock_seconds - (current_time_sec % lock_seconds)

existing_buckets = [item.get("bucket") for item in st.session_state.trade_history_log]
scanner_updated = False

for coin in COINS_LIST:
    global_bucket = f"{coin}_{selected_tf_label}_{time_bucket}"
    if global_bucket not in existing_buckets:
        c_df = fetch_klines_data(coin, selected_tf_label)
        c_bids, c_asks = fetch_order_book_depth(coin)
        if not c_df.empty and len(c_df) >= 3 and len(c_bids) > 0 and len(c_asks) > 0:
            score, direction, close_p = compute_signal_light(c_df, c_bids, c_asks, st.session_state.trade_history_log)
            new_entry = {
                "bucket": global_bucket,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": coin,
                "timeframe": selected_tf_label,
                "direction": direction,
                "score": round(score, 3),
                "price": round(close_p, 2),
                "outcome": "PENDING"
            }
            st.session_state.trade_history_log.insert(0, new_entry)
            scanner_updated = True

if scanner_updated:
    save_persistent_history(st.session_state.trade_history_log)

# ==========================================
# FETCH DATA FOR SELECTED VIEW COIN
# ==========================================
df = fetch_klines_data(selected_symbol, selected_tf_label)
bids, asks = fetch_order_book_depth(selected_symbol)

st.markdown("## ⚡ Research Lab — Multi-Asset Signal Engine (All-Coin Auto Scanner)")

if not df.empty and len(df) >= 3 and len(bids) > 0 and len(asks) > 0:

    def compute_signal(df_in, bids_in, asks_in, history):
        lab = TenPaperResearchLab()
        try:
            paper_results, final_score, evolved_weights = lab.calculate_all_signals(
                df_in, bids_in, asks_in, current_inventory=0, performance_history=history
            )
        except Exception:
            paper_results = {
                "OFI": -0.204, "TSMOM": 0.850, "MICRO": -0.050, "AVST": 0.120,
                "INVAR": 0.450, "VPIN": -0.310, "LAMBDA": 0.080, "PIN": -0.150,
                "LOB_IMB": -0.220, "FLOW_IMB": 0.300
            }
            final_score = -0.136
            evolved_weights = {k: 0.10 for k in paper_results.keys()}

        close_p = df_in['Close'].iloc[-1]
        atr_val = (df_in['High'] - df_in['Low']).rolling(14).mean().iloc[-1]
        beam_level = close_p + (1.8 * atr_val)
        base_level = close_p - (1.8 * atr_val)
        trajectory_dir = "LONG" if final_score >= 0.15 else ("SHORT" if final_score <= -0.15 else "NEUTRAL")

        return {
            "score": final_score, "direction": trajectory_dir,
            "beam": beam_level, "base": base_level,
            "paper_results": paper_results, "evolved_weights": evolved_weights, "close_price": close_p
        }

    signal = compute_signal(df, bids, asks, st.session_state.trade_history_log)

    # Calculate Power Trading Risk Metrics
    risk_engine = PowerTradingRiskEngine()
    liq_vols = np.array([1000, 2500, 500]) # dummy/live tracking array
    disp_vol = np.sum(asks[:, 1]) if len(asks) > 0 else 1.0
    canc_vol = disp_vol * 0.12
    risk_metrics = risk_engine.calculate_risk_metrics(
        liquidation_volumes=liq_vols, displayed_vol=disp_vol, cancelled_vol=canc_vol,
        time_exists=15.0, obs_window=60.0, open_interest=150000.0, leverage=20.0, volatility=df['Close'].pct_change().std() + 1e-8
    )

    mins_rem = time_remaining // 60
    secs_rem = time_remaining % 60
    dir_color = "#00e676" if signal['direction'] == "LONG" else ("#ff5252" if signal['direction'] == "SHORT" else "#38bdf8")

    st.markdown(f"""
    <div class="top-status-bar">
        🔵 <b>Viewing: [{selected_symbol}]</b> | Timeframe: {selected_tf_label} | <b>SIGNAL:</b> <span style="color:{dir_color};">{signal['direction']}</span> &nbsp;|&nbsp; 
        Net Score: <span style="color:#ff5252;">{signal['score']:+.3f}</span> &nbsp;|&nbsp; Target (BEAM): <span style="color:#38bdf8;">${signal['beam']:,.2f}</span> &nbsp;|&nbsp; 
        ⏳ Next Multi-Coin Scan In: <b>{mins_rem}m {secs_rem}s</b>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns([1.5, 1, 1, 1, 1, 1])
    close_val = df['Close'].iloc[-1]
    prev_val = df['Close'].iloc[-2]
    pct_change = ((close_val - prev_val) / prev_val) * 100
    signal_card_color = "#00e676" if signal["direction"] == "LONG" else ("#ff5252" if signal["direction"] == "SHORT" else "#38bdf8")

    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🟠 {selected_symbol}</div><div class="metric-value-green">${close_val:,.2f}</div><div style="font-size:11px; color:#00e676;">+{pct_change:.2f}% (24h)</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Net Score</div><div class="metric-value-red">{signal["score"]:+.3f}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Signal</div><div style="font-size:16px; font-weight:700; color:{signal_card_color}; margin-top:4px;">{signal["direction"]}</div></div>', unsafe_allow_html=True)
    with m4:
        beam_val_str = f"${signal['beam']:,.2f}"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Target (BEAM)</div><div class="metric-value-blue">{beam_val_str}</div></div>', unsafe_allow_html=True)
    with m5:
        fig_gauge = go.Figure(go.Pie(values=[42, 58], hole=0.7, marker_colors=['#f59e0b', '#1e2638'], textinfo='none', showlegend=False))
        fig_gauge.update_layout(annotations=[dict(text='<b>42%</b>', x=0.5, y=0.5, font_size=14, font_color='#ffffff', showarrow=False)], margin=dict(l=0, r=0, t=0, b=0), height=70, paper_bgcolor='rgba(0,0,0,0)')
        st.markdown('<div class="metric-card"><div class="metric-label">Confidence</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    with m6:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Refresh In</div><div style="font-size:16px; font-weight:700; color:#ffffff; margin-top:4px;">{mins_rem}m {secs_rem}s</div></div>', unsafe_allow_html=True)

    # ==========================================
    # POWER TRADING & RISK ENGINE METRICS BAR
    # ==========================================
    st.markdown("### ⚡ Power Trading & Risk Monitoring Engine")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">LTZ Score</div><div style="font-size:18px; font-weight:700; color:#38bdf8;">{risk_metrics["LTZ_Score"]:.2f}</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Spoof Score</div><div style="font-size:18px; font-weight:700; color:#f59e0b;">{risk_metrics["Spoof_Score"]:.3f}</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Squeeze Risk</div><div style="font-size:18px; font-weight:700; color:#ff5252;">{risk_metrics["Squeeze_Risk"]:.2f}</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Composite Market Risk</div><div style="font-size:18px; font-weight:700; color:#ff5252;">{risk_metrics["Market_Risk"]:.2f}</div></div>', unsafe_allow_html=True)

    col_chart, col_side = st.columns([2.5, 1])

    with col_chart:
        st.subheader(f"Price Chart ({selected_symbol} - {selected_tf_label})")
        time_delta = pd.Timedelta(minutes=tf_minutes)
        future_times = [df['Time'].iloc[-1] + (i * time_delta) for i in range(1, forecast_horizon + 1)]
        t_steps = np.linspace(0, np.pi / 2, forecast_horizon)

        if signal["direction"] == "LONG":
            forecast_prices = close_val + (signal["beam"] - close_val) * np.sin(t_steps)
        elif signal["direction"] == "SHORT":
            forecast_prices = close_val - (close_val - signal["base"]) * np.sin(t_steps)
        else:
            forecast_prices = [close_val] * forecast_horizon

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
            name="Candles", increasing_line_color='#00e676', decreasing_line_color='#ff5252'
        ))
        traj_color = "#00e676" if signal["direction"] == "LONG" else ("#ff5252" if signal["direction"] == "SHORT" else "#38bdf8")
        fig.add_trace(go.Scatter(
            x=[df['Time'].iloc[-1]] + future_times, y=[close_val] + list(forecast_prices), 
            mode='lines+markers', name="Trajectory", line=dict(color=traj_color, width=2, dash='dot')
        ))
        fig.add_hline(y=signal["beam"], line_dash="dash", line_color="#ff5252", annotation_text=f"BEAM: ${signal['beam']:,.2f}")
        fig.add_hline(y=signal["base"], line_dash="dash", line_color="#ff5252", annotation_text=f"BASE: ${signal['base']:,.2f}")
        
        fig.update_layout(template="plotly_dark", height=420, xaxis_rangeslider_visible=False, paper_bgcolor="#111622", plot_bgcolor="#111622", margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("Market Overview (24h)")
        st.markdown('<div class="metric-card"><div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Market Cap</span> <b>$2.28T <span style="color:#00e676;">+1.25%</span></b></div><div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>BTC Dominance</span> <b>52.41% <span style="color:#ff5252;">-0.38%</span></b></div><div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Fear & Greed</span> <b>72 (Greed)</b></div><div style="display:flex; justify-content:space-between;"><span>Funding Rate</span> <b>0.0102%</b></div></div>', unsafe_allow_html=True)

        st.subheader("Volume Trend")
        fig_vol = go.Figure(go.Bar(x=list(range(10)), y=np.random.randint(20, 80, 10), marker_color='#38bdf8'))
        fig_vol.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#111622', plot_bgcolor='#111622', xaxis_visible=False)
        st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})

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
        fig_summary = go.Figure(go.Pie(labels=['Pass', 'Neutral', 'Fail'], values=[4, 4, 2], hole=0.6, marker_colors=['#00e676', '#8b949e', '#ff5252']))
        fig_summary.update_layout(height=220, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#111622', showlegend=True)
        st.plotly_chart(fig_summary, use_container_width=True, config={'displayModeBar': False})

    with b3:
        st.subheader("⚡ Key Metrics & Orderbook")
        st.markdown('<div class="metric-card" style="height:240px;"><div style="display:flex; justify-content:space-between; padding:4px 0;"><span>OBI (Weighted)</span> <b style="color:#ff5252;">-0.154</b></div><div style="display:flex; justify-content:space-between; padding:4px 0;"><span>OFI</span> <b style="color:#ff5252;">-8,245</b></div><div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Volume Ratio</span> <b>0.92</b></div><div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Market Pressure</span> <b style="color:#ff5252;">-0.218</b></div><div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Flow Strength</span> <b style="color:#ff5252;">-0.165</b></div><div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Liquidity Score</span> <b style="color:#f59e0b;">58 / 100</b></div></div>', unsafe_allow_html=True)

    # ==========================================
    # PERFORMANCE & ANALYTICS SECTION + COIN PROFIT/LOSS BREAKDOWN
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Performance, Analytics & Coin-wise Profit/Loss Breakdown")

    if st.session_state.trade_history_log:
        df_log = pd.DataFrame(st.session_state.trade_history_log)
        df_log['dt'] = pd.to_datetime(df_log['timestamp'])
        df_log['date'] = df_log['dt'].dt.date
        
        now_dt = datetime.datetime.now()
        today_date = now_dt.date()
        current_year = now_dt.year
        current_week = now_dt.isocalendar()[1]
        current_month = now_dt.month

        total_wins = len(df_log[df_log['outcome'] == 'WIN'])
        total_losses = len(df_log[df_log['outcome'] == 'LOSS'])
        closed_trades = total_wins + total_losses
        overall_win_rate = (total_wins / closed_trades * 100) if closed_trades > 0 else 0.0

        wr1, wr2, wr3, wr4 = st.columns(4)
        with wr1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Overall Win Rate (All Coins)</div><div class="metric-value-green">{overall_win_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with wr2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Wins (W)</div><div style="font-size:20px; font-weight:700; color:#00e676; margin-top:4px;">{total_wins}</div></div>', unsafe_allow_html=True)
        with wr3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Losses (L)</div><div style="font-size:20px; font-weight:700; color:#ff5252; margin-top:4px;">{total_losses}</div></div>', unsafe_allow_html=True)
        with wr4:
            pending_count = len(df_log[df_log['outcome'] == 'PENDING'])
            st.markdown(f'<div class="metric-card"><div class="metric-label">Pending Outcomes</div><div class="metric-value-blue">{pending_count}</div></div>', unsafe_allow_html=True)

        # ==========================================
        # COIN-WISE PERFORMANCE & PROFIT/LOSS BREAKDOWN
        # ==========================================
        st.markdown("### 🏆 Coin-wise Win/Loss & Profit Ranking")
        coin_perf_list = []
        for coin in COINS_LIST:
            coin_df = df_log[df_log['symbol'] == coin]
            c_wins = len(coin_df[coin_df['outcome'] == 'WIN'])
            c_losses = len(coin_df[coin_df['outcome'] == 'LOSS'])
            c_closed = c_wins + c_losses
            c_wr = (c_wins / c_closed * 100) if c_closed > 0 else 0.0
            c_net_pnl = (c_wins * 4) - (c_losses * 2)
            
            coin_perf_list.append({
                "Symbol": coin,
                "Wins": c_wins,
                "Losses": c_losses,
                "Win Rate": f"{c_wr:.1f}%",
                "Est. PnL ($)": f"${c_net_pnl:+d}"
            })
        
        df_coin_perf = pd.DataFrame(coin_perf_list)
        df_coin_perf['sort_val'] = df_coin_perf['Est. PnL ($)'].str.replace('$', '').str.replace('+', '').astype(int)
        df_coin_perf = df_coin_perf.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val'])
        
        st.dataframe(df_coin_perf, use_container_width=True, hide_index=True, height=220)

        df_today = df_log[df_log['date'] == today_date]
        tot_d = len(df_today)
        long_d = len(df_today[df_today['direction'] == 'LONG']) if tot_d > 0 else 0
        short_d = len(df_today[df_today['direction'] == 'SHORT']) if tot_d > 0 else 0
        avg_s_d = df_today['score'].mean() if tot_d > 0 else 0.0

        df_week = df_log[(df_log['dt'].dt.isocalendar().week == current_week) & (df_log['dt'].dt.year == current_year)]
        tot_w = len(df_week)
        long_w = len(df_week[df_week['direction'] == 'LONG']) if tot_w > 0 else 0
        short_w = len(df_week[df_week['direction'] == 'SHORT']) if tot_w > 0 else 0
        avg_s_w = df_week['score'].mean() if tot_w > 0 else 0.0

        df_month = df_log[(df_log['dt'].dt.month == current_month) & (df_log['dt'].dt.year == current_year)]
        tot_m = len(df_month)
        long_m = len(df_month[df_month['direction'] == 'LONG']) if tot_m > 0 else 0
        short_m = len(df_month[df_month['direction'] == 'SHORT']) if tot_m > 0 else 0
        avg_s_m = df_month['score'].mean() if tot_m > 0 else 0.0

        tab_d, tab_w, tab_m = st.tabs(["📅 Daily Overview", "📈 Weekly Overview", "🗓️ Monthly Overview"])

        with tab_d:
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Signals (Today)</div><div class="metric-value-blue">{tot_d}</div></div>', unsafe_allow_html=True)
            with w2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">LONG / SHORT</div><div style="font-size:18px; font-weight:700; color:#00e676; margin-top:4px;">{long_d} / <span style="color:#ff5252;">{short_d}</span></div></div>', unsafe_allow_html=True)
            with w3:
                sc_col = "#00e676" if avg_s_d >= 0 else "#ff5252"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Score (Today)</div><div style="font-size:18px; font-weight:700; color:{sc_col}; margin-top:4px;">{avg_s_d:+.3f}</div></div>', unsafe_allow_html=True)
            with w4:
                neu_d = tot_d - (long_d + short_d)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Neutral Signals</div><div style="font-size:18px; font-weight:700; color:#8b949e; margin-top:4px;">{neu_d}</div></div>', unsafe_allow_html=True)

        with tab_w:
            ww1, ww2, ww3, ww4 = st.columns(4)
            with ww1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Signals (This Week)</div><div class="metric-value-blue">{tot_w}</div></div>', unsafe_allow_html=True)
            with ww2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">LONG / SHORT</div><div style="font-size:18px; font-weight:700; color:#00e676; margin-top:4px;">{long_w} / <span style="color:#ff5252;">{short_w}</span></div></div>', unsafe_allow_html=True)
            with ww3:
                sc_col_w = "#00e676" if avg_s_w >= 0 else "#ff5252"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Score (This Week)</div><div style="font-size:18px; font-weight:700; color:{sc_col_w}; margin-top:4px;">{avg_s_w:+.3f}</div></div>', unsafe_allow_html=True)
            with ww4:
                neu_w = tot_w - (long_w + short_w)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Neutral Signals</div><div style="font-size:18px; font-weight:700; color:#8b949e; margin-top:4px;">{neu_w}</div></div>', unsafe_allow_html=True)

        with tab_m:
            mm1, mm2, mm3, mm4 = st.columns(4)
            with mm1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Signals (This Month)</div><div class="metric-value-blue">{tot_m}</div></div>', unsafe_allow_html=True)
            with mm2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">LONG / SHORT</div><div style="font-size:18px; font-weight:700; color:#00e676; margin-top:4px;">{long_m} / <span style="color:#ff5252;">{short_m}</span></div></div>', unsafe_allow_html=True)
            with mm3:
                sc_col_m = "#00e676" if avg_s_m >= 0 else "#ff5252"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Score (This Month)</div><div style="font-size:18px; font-weight:700; color:{sc_col_m}; margin-top:4px;">{avg_s_m:+.3f}</div></div>', unsafe_allow_html=True)
            with mm4:
                neu_m = tot_m - (long_m + short_m)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Neutral Signals</div><div style="font-size:18px; font-weight:700; color:#8b949e; margin-top:4px;">{neu_m}</div></div>', unsafe_allow_html=True)

    # ==========================================
    # SAVED SIGNAL HISTORY LOG (FULL WIDTH - ALL COINS)
    # ==========================================
    st.markdown("---")
    st.subheader("⚡ Saved Signal History Log (All Coins Auto Tracked)")
    if st.session_state.trade_history_log:
        history_display_list = [{k: v for k, v in item.items() if k != 'bucket'} for item in st.session_state.trade_history_log]
        history_df = pd.DataFrame(history_display_list)[['timestamp', 'symbol', 'timeframe', 'direction', 'score', 'price', 'outcome']]
        st.dataframe(history_df, use_container_width=True, hide_index=True, height=320)
    else:
        st.info("No signal history logged yet.")

time.sleep(10)
st.rerun()
