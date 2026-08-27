import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import joblib
import os

# ==========================================
# PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="ZIA RESEARCH LAB | Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    .stMetric label { color: #8b949e !important; font-weight: 600; }
    .metric-card { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; }
    h1, h2, h3 { color: #f0f6fc; }
    .signal-strong-long { color: #3fb950; font-weight: bold; font-size: 1.5rem; }
    .signal-long { color: #56d364; font-weight: bold; font-size: 1.5rem; }
    .signal-wait { color: #d29922; font-weight: bold; font-size: 1.5rem; }
    .signal-short { color: #f85149; font-weight: bold; font-size: 1.5rem; }
    .signal-strong-short { color: #da3633; font-weight: bold; font-size: 1.5rem; }
    div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE & CONSTANTS
# ==========================================
MODEL_FEATURES = [
    "top20_bid_sum",
    "top20_ask_sum",
    "obi_top20",
    "spread",
    "bid_ask_ratio",
    "total_depth",
    "trend_signal",
]

if "refresh_seconds" not in st.session_state:
    st.session_state.refresh_seconds = 5

if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

# ==========================================
# DATA FETCHING & API MODULE (CACHED)
# ==========================================
@st.cache_resource
def load_xgboost_model():
    model_path = "xgboost_obi_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

@st.cache_data(ttl=2)
def fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=300):
    # Try Futures API first, fallback to Spot if unavailable
    endpoints = [
        f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    ]
    for url in endpoints:
        try:
            response = requests.get(url, timeout=4)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']:
                    df[col] = df[col].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=1)
def fetch_binance_orderbook(symbol="BTCUSDT", limit=100):
    endpoints = [
        f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={limit}",
        f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    ]
    for url in endpoints:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return response.json()
        except Exception:
            continue
    return {"bids": [], "asks": []}

# ==========================================
# ANALYTICS & INDICATORS
# ==========================================
def calculate_indicators(df):
    if df.empty or len(df) < 200:
        return df
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # ATR Calculation
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    return df

def get_htf_trilines(symbol, interval_str, lookback=10):
    df_htf = fetch_binance_klines(symbol=symbol, interval=interval_str, limit=lookback + 5)
    if df_htf.empty or len(df_htf) < lookback + 1:
        return None, None, None
    
    # Ignore the currently forming candle, use completed historical lookback
    completed_df = df_htf.iloc[-(lookback+1):-1]
    h_high = completed_df['high'].max()
    h_low = completed_df['low'].min()
    h_mid = (h_high + h_low) / 2.0
    return h_high, h_mid, h_low

def process_order_book(ob_data):
    bids = ob_data.get("bids", [])
    asks = ob_data.get("asks", [])
    
    bids_arr = np.array(bids, dtype=float) if bids else np.zeros((1, 2))
    asks_arr = np.array(asks, dtype=float) if asks else np.zeros((1, 2))
    
    def get_depth_sums(arr, levels):
        if len(arr) == 0 or arr.size == 1:
            return 0.0, 0.0
        n = min(levels, len(arr))
        vol_sum = np.sum(arr[:n, 1])
        weighted_sum = np.sum(arr[:n, 0] * arr[:n, 1])
        return vol_sum, weighted_sum

    b5_vol, _ = get_depth_sums(bids_arr, 5)
    a5_vol, _ = get_depth_sums(asks_arr, 5)
    
    b10_vol, _ = get_depth_sums(bids_arr, 10)
    a10_vol, _ = get_depth_sums(asks_arr, 10)
    
    b20_vol, _ = get_depth_sums(bids_arr, 20)
    a20_vol, _ = get_depth_sums(asks_arr, 20)
    
    b50_vol, _ = get_depth_sums(bids_arr, 50)
    a50_vol, _ = get_depth_sums(asks_arr, 50)
    
    total_bids = np.sum(bids_arr[:, 1]) if len(bids_arr) > 0 else 0.0
    total_asks = np.sum(asks_arr[:, 1]) if len(asks_arr) > 0 else 0.0
    total_depth = total_bids + total_asks
    
    best_bid = bids_arr[0, 0] if len(bids_arr) > 0 else 0.0
    best_ask = asks_arr[0, 0] if len(asks_arr) > 0 else 0.0
    spread = best_ask - best_bid if (best_ask > 0 and best_bid > 0) else 0.0
    
    obi_20 = (b20_vol - a20_vol) / (b20_vol + a20_vol) if (b20_vol + a20_vol) > 0 else 0.0
    bid_ask_ratio = b20_vol / a20_vol if a20_vol > 0 else 1.0
    
    return {
        "top5_bid": b5_vol, "top5_ask": a5_vol,
        "top10_bid": b10_vol, "top10_ask": a10_vol,
        "top20_bid": b20_vol, "top20_ask": a20_vol,
        "top50_bid": b50_vol, "top50_ask": a50_vol,
        "total_bids": total_bids, "total_asks": total_asks,
        "total_depth": total_depth, "spread": spread,
        "obi_top20": obi_20, "bid_ask_ratio": bid_ask_ratio,
        "best_bid": best_bid, "best_ask": best_ask
    }

# ==========================================
# TRADINGVIEW-STYLE CHART FUNCTION
# ==========================================
def price_chart(df, symbol, levels_dict):
    fig = go.Figure()
    
    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles"
    ))
    
    # 2. EMAs
    if 'ema_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], mode='lines', name='EMA 20', line=dict(color='#2962ff', width=1.5)))
    if 'ema_50' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_50'], mode='lines', name='EMA 50', line=dict(color='#ff9800', width=1.5)))
    if 'ema_200' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_200'], mode='lines', name='EMA 200', line=dict(color='#e91e63', width=1.5)))
        
    # Horizontal Trading & Tri-Line Levels across visible chart
    shapes = []
    annotations = []
    
    def add_horizontal_line(price, color, text):
        if price and not np.isnan(price):
            shapes.append(dict(
                type="line", x0=df['timestamp'].iloc[0], x1=df['timestamp'].iloc[-1],
                y0=price, y1=price, line=dict(color=color, width=1.2, dash="dot")
            ))
            annotations.append(dict(
                x=df['timestamp'].iloc[-1], y=price, text=f" {text}: {price:.2f}",
                showarrow=False, xanchor="left", font=dict(color=color, size=11)
            ))

    add_horizontal_line(levels_dict.get('entry'), '#00bcd4', 'ENTRY')
    add_horizontal_line(levels_dict.get('sl'), '#f85149', 'STOP LOSS')
    add_horizontal_line(levels_dict.get('tp1'), '#3fb950', 'TARGET 1')
    add_horizontal_line(levels_dict.get('tp2'), '#2ea043', 'TARGET 2')
    
    add_horizontal_line(levels_dict.get('h1_high'), '#8a2be2', '1H HIGH')
    add_horizontal_line(levels_dict.get('h1_mid'), '#8a2be2', '1H 50%')
    add_horizontal_line(levels_dict.get('h1_low'), '#8a2be2', '1H LOW')
    
    add_horizontal_line(levels_dict.get('h4_high'), '#ff4500', '4H HIGH')
    add_horizontal_line(levels_dict.get('h4_mid'), '#ff4500', '4H 50%')
    add_horizontal_line(levels_dict.get('h4_low'), '#ff4500', '4H LOW')

    fig.update_layout(shapes=shapes, annotations=annotations)

    # Right side future space configuration (15 candles padding)
    if len(df) > 1:
        candle_delta = df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2]
    else:
        candle_delta = timedelta(hours=1)
        
    chart_start = df['timestamp'].iloc[max(0, len(df) - 100)]
    chart_end = df['timestamp'].iloc[-1] + (candle_delta * 15)

    # TradingView-style layout config without duplicate keyword arguments
    fig.update_layout(
        title=f"<b>PRICE ACTION · 1H / 4H TRI-LINE — {symbol}</b>",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=620,
        margin=dict(l=10, r=90, t=40, b=10),
        xaxis=dict(
            type="date",
            range=[chart_start, chart_end],
            autorange=False,
            rangeslider=dict(
                visible=True,
                thickness=0.07,
            ),
            showgrid=True,
            fixedrange=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
        ),
        yaxis=dict(
            showgrid=True,
            autorange=True,
            fixedrange=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
        ),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified"
    )

    fig.update_config(
        displaylogo=False,
        responsive=True,
        scrollZoom=True,
        displayModeBar=True,
        modeBarButtonsToRemove=["lasso2d", "select2d"],
    )
    return fig

# ==========================================
# MAIN DASHBOARD TERMINAL
# ==========================================
def render_dashboard():
    st.sidebar.title("⚡ ZIA RESEARCH LAB")
    symbol = st.sidebar.selectbox("Market Asset", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"], index=0)
    timeframe = st.sidebar.selectbox("Chart Timeframe", ["15m", "1h", "4h"], index=1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Terminal Control")
    st.session_state.refresh_seconds = st.sidebar.slider("Live Refresh Rate (s)", 2, 30, 5)
    
    # Fetch Core Data
    df_candles = fetch_binance_klines(symbol=symbol, interval=timeframe, limit=250)
    df_candles = calculate_indicators(df_candles)
    
    ob_raw = fetch_binance_orderbook(symbol=symbol, limit=100)
    obi_metrics = process_order_book(ob_raw)
    
    # Fetch HTF Tri-lines
    h1_high, h1_mid, h1_low = get_htf_trilines(symbol, "1h", lookback=10)
    h4_high, h4_mid, h4_low = get_htf_trilines(symbol, "4h", lookback=10)
    
    if df_candles.empty:
        st.error(f"Failed to fetch market feeds for {symbol}. Verify network status or Binance connectivity.")
        return

    current_price = df_candles['close'].iloc[-1]
    atr_val = df_candles['atr'].iloc[-1] if 'atr' in df_candles.columns and not np.isnan(df_candles['atr'].iloc[-1]) else current_price * 0.01

    # Trend calculation
    ema20 = df_candles['ema_20'].iloc[-1] if 'ema_20' in df_candles.columns else current_price
    ema50 = df_candles['ema_50'].iloc[-1] if 'ema_50' in df_candles.columns else current_price
    trend_signal = 1 if ema20 > ema50 else -1

    # XGBoost Prediction Feed
    model = load_xgboost_model()
    ml_prob = 0.50
    if model is not None:
        try:
            features_vector = np.array([[
                obi_metrics["top20_bid"],
                obi_metrics["top20_ask"],
                obi_metrics["obi_top20"],
                obi_metrics["spread"],
                obi_metrics["bid_ask_ratio"],
                obi_metrics["total_depth"],
                trend_signal
            ]])
            preds = model.predict_proba(features_vector)
            ml_prob = float(preds[0][1])
        except Exception:
            ml_prob = 0.50

    # Signal Logic
    if ml_prob > 0.65 and trend_signal > 0:
        signal = "STRONG LONG"
    elif ml_prob > 0.55:
        signal = "LONG"
    elif ml_prob < 0.35 and trend_signal < 0:
        signal = "STRONG SHORT"
    elif ml_prob < 0.45:
        signal = "SHORT"
    else:
        signal = "WAIT"

    # Trade Plan Calculation (1:2 and 1:3 RR)
    if "LONG" in signal:
        entry = current_price
        sl = current_price - (1.5 * atr_val)
        risk = entry - sl
        tp1 = entry + (2.0 * risk)
        tp2 = entry + (3.0 * risk)
    elif "SHORT" in signal:
        entry = current_price
        sl = current_price + (1.5 * atr_val)
        risk = sl - entry
        tp1 = entry - (2.0 * risk)
        tp2 = entry - (3.0 * risk)
    else:
        entry, sl, tp1, tp2 = current_price, current_price * 0.99, current_price * 1.02, current_price * 1.03

    levels_dict = {
        'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
        'h1_high': h1_high, 'h1_mid': h1_mid, 'h1_low': h1_low,
        'h4_high': h4_high, 'h4_mid': h4_mid, 'h4_low': h4_low
    }

    # Header Metrics View
    st.title(f"ZIA RESEARCH LAB — {symbol}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Last Price", f"${current_price:,.2f}")
    col2.metric("Order Book OBI (Top 20)", f"{obi_metrics['obi_top20']:.3f}")
    col3.metric("ML Probability", f"{ml_prob*100:.1f}%")
    col4.metric("Spread", f"${obi_metrics['spread']:.2f}")
    col5.metric("ATR (14)", f"${atr_val:.2f}")

    # Signal Display Banner
    sig_class = "signal-wait"
    if "STRONG LONG" in signal: sig_class = "signal-strong-long"
    elif "LONG" in signal: sig_class = "signal-long"
    elif "STRONG SHORT" in signal: sig_class = "signal-strong-short"
    elif "SHORT" in signal: sig_class = "signal-short"

    st.markdown(f"""
    <div class="metric-card" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
            <span style="color: #8b949e; font-size: 0.9rem;">GENERATED ENGINE SIGNAL</span><br>
            <span class="{sig_class}">{signal}</span>
        </div>
        <div>
            <span style="color: #8b949e; font-size: 0.9rem;">TRADE SETUP PLAN</span><br>
            <b>Entry:</b> ${entry:,.2f} | <b>SL:</b> ${sl:,.2f} | <b>TP1 (1:2):</b> ${tp1:,.2f} | <b>TP2 (1:3):</b> ${tp2:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs Layout
    tab_overview, tab_orderflow, tab_ml, tab_history = st.tabs(["Overview & Chart", "Order Flow Depth", "ML Diagnostics", "Signal History"])

    with tab_overview:
        fig = price_chart(df_candles, symbol, levels_dict)
        st.plotly_chart(fig, use_container_width=True)

    with tab_orderflow:
        st.subheader("Order Book Depth & Liquidity Analysis")
        oc1, oc2 = st.columns(2)
        with oc1:
            st.metric("Top 5 Bid Volume", f"{obi_metrics['top5_bid']:,.2f}")
            st.metric("Top 10 Bid Volume", f"{obi_metrics['top10_bid']:,.2f}")
            st.metric("Top 20 Bid Volume", f"{obi_metrics['top20_bid']:,.2f}")
            st.metric("Top 50 Bid Volume", f"{obi_metrics['top50_bid']:,.2f}")
        with oc2:
            st.metric("Top 5 Ask Volume", f"{obi_metrics['top5_ask']:,.2f}")
            st.metric("Top 10 Ask Volume", f"{obi_metrics['top10_ask']:,.2f}")
            st.metric("Top 20 Ask Volume", f"{obi_metrics['top20_ask']:,.2f}")
            st.metric("Top 50 Ask Volume", f"{obi_metrics['top50_ask']:,.2f}")

    with tab_ml:
        st.subheader("XGBoost Inference Engine")
        st.write("Current 7-Feature Model Contract:")
        feat_df = pd.DataFrame({
            "Feature Name": MODEL_FEATURES,
            "Live Value": [
                obi_metrics["top20_bid"],
                obi_metrics["top20_ask"],
                obi_metrics["obi_top20"],
                obi_metrics["spread"],
                obi_metrics["bid_ask_ratio"],
                obi_metrics["total_depth"],
                trend_signal
            ]
        })
        st.dataframe(feat_df, use_container_width=True)
        st.info(f"Model File Loaded Status: {'Active' if model is not None else 'Fallback Rule Engine Active (File missing)'}")

    with tab_history:
        st.subheader("Session Signal History Ledger")
        if st.session_state.signal_history:
            hist_df = pd.DataFrame(st.session_state.signal_history)
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.write("Accumulating live session events...")

# ==========================================
# EXECUTION ENTRYPOINT WITH FRAGMENT REFRESH
# ==========================================
try:
    @st.fragment(run_every=st.session_state.refresh_seconds)
    def live_terminal():
        render_dashboard()

    live_terminal()
except AttributeError:
    render_dashboard()
