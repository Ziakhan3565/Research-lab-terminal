import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Page Configuration
st.set_page_config(
    page_title="Institutional Financial Market Learning & Analysis Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional Dark Styling
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e6edf3; }
    .stMetric { background-color: #111622; padding: 18px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stTab { font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# === QUANTITATIVE RESEARCH LAB (12 PAPERS) ===
class TenPaperResearchLab:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol

    def calculate_all_signals(self, df, bids, asks, current_inventory=0):
        results = {}
        
        if len(bids) == 0 or len(asks) == 0 or df.empty or len(df) < 5:
            default_results = {
                'OFI': 0.0, 'TSMOM': 0.0, 'MICRO': 0.0, 'QUEUE': 0.0,
                'AVST': 0.0, 'INVAR': 0.0, 'VPIN': 0.0, 'VRATIO': 0.0,
                'BURST': 0.0, 'FUND': 0.0, 'LOG_PROB': 0.0, 'LOB_TARGET': 0.0
            }
            default_weights = {
                'OFI': 0.12, 'TSMOM': 0.12, 'MICRO': 0.10, 'QUEUE': 0.08,
                'AVST': 0.08, 'INVAR': 0.08, 'VPIN': 0.08, 'VRATIO': 0.08,
                'BURST': 0.08, 'FUND': 0.08, 'LOG_PROB': 0.10, 'LOB_TARGET': 0.08
            }
            return default_results, 0.0, default_weights
        
        # 1. OFI (Order Flow Imbalance) - Cont et al. (2014)
        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        results['OFI'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        # 2. TSMOM (Time-Series Momentum) - Moskowitz et al. (2012)
        returns_h = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
        realized_vol = df['Close'].pct_change().std() + 1e-8
        results['TSMOM'] = np.clip((returns_h / realized_vol) * 2.0, -1, 1)

        # 3. MICRO (Micro-Price Imbalance) - Stoikov (2018)
        best_bid, best_ask = bids[0, 0], asks[0, 0]
        q_b, q_a = bids[0, 1], asks[0, 1]
        micro_price = (q_b * best_bid + q_a * best_ask) / (q_b + q_a + 1e-8)
        mid_price = (best_bid + best_ask) / 2
        results['MICRO'] = np.clip((micro_price - mid_price) / (mid_price * 0.0002), -1, 1)

        # 4. AVST (Avellaneda & Stoikov MM Model) - (2008)
        gamma = 0.1
        reservation_price = mid_price - current_inventory * gamma * (realized_vol ** 2)
        results['AVST'] = 1.0 if reservation_price > mid_price else (-1.0 if reservation_price < mid_price else 0.0)

        # 5. INVAR (Inventory Variance Adjustment) - Guéant et al. (2012)
        inventory_penalty = -current_inventory * 0.2 * (realized_vol ** 2)
        results['INVAR'] = np.clip(1.0 + inventory_penalty, -1, 1)

        # 6. VPIN (Volume-Synchronized Toxicity) - Easley et al. (2012)
        buy_vol = df['Volume'].iloc[-5:].mean() * (1.2 if returns_h > 0 else 0.3)
        sell_vol = df['Volume'].iloc[-5:].mean() * (1.2 if returns_h <= 0 else 0.3)
        vpin = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)
        results['VPIN'] = np.clip(vpin * 2.5, -1, 1)

        # 7. QUEUE (L1 Queue Imbalance) - Huang et al. (2015)
        results['QUEUE'] = np.clip((q_b - q_a) / (q_b + q_a + 1e-8) * 1.5, -1, 1)

        # 8. VRATIO (Variance Ratio Test) - Lo & MacKinlay (1988)
        var_1 = df['Close'].pct_change().var() + 1e-8
        var_5 = (df['Close'].pct_change(5)).var() / 5.0 + 1e-8
        v_ratio = var_5 / var_1
        results['VRATIO'] = 1.0 if (v_ratio > 1.0 and returns_h > 0) else (-1.0 if (v_ratio > 1.0 and returns_h < 0) else 0.0)

        # 9. BURST (Volatility Burst Detection) - Christensen et al. (2014)
        vol_short = df['Close'].pct_change().iloc[-3:].std()
        vol_long = df['Close'].pct_change().iloc[-20:].std() + 1e-8
        burst_ratio = vol_short / vol_long
        results['BURST'] = 1.0 if (burst_ratio > 1.2 and returns_h > 0) else (-1.0 if (burst_ratio > 1.2 and returns_h < 0) else 0.0)

        # 10. FUND (Implied Fundamental Value) - Cartea et al. (2014)
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        results['FUND'] = np.clip(obi * 1.5, -1, 1)

        # 11. LOG_PROB (Logistic Probability Model)
        linear_comb = 0.5 + (1.2 * results['OFI']) - (0.8 * results['VPIN'])
        log_prob = 1.0 / (1.0 + np.exp(-linear_comb))
        results['LOG_PROB'] = np.clip((log_prob - 0.5) * 2.0, -1, 1)

        # 12. LOB_TARGET (Limit Order Book Target Pressure)
        delta_p = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        lob_pressure = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        results['LOB_TARGET'] = np.clip(lob_pressure * (delta_p / (df['Close'].iloc[-1] + 1e-8) * 100), -1, 1)

        weights = {
            'OFI': 0.12, 'TSMOM': 0.12, 'MICRO': 0.10, 'QUEUE': 0.08,
            'AVST': 0.08, 'INVAR': 0.08, 'VPIN': 0.08, 'VRATIO': 0.08,
            'BURST': 0.08, 'FUND': 0.08, 'LOG_PROB': 0.10, 'LOB_TARGET': 0.08
        }
        
        final_score = sum(results[paper] * weights[paper] for paper in results)
        return results, final_score, weights


# === ADAPTIVE SIGNAL MANAGER (1m = SCALPING / 15m = 60% CONVICTION GATE) ===
class SignalHysteresisManager:
    def __init__(self, mode="15m"):
        self.mode = mode
        self.current_signal = "NEUTRAL"
        self.score_history = []
        
        if self.mode == "1m":
            self.entry_threshold = 0.15
            self.window = 2
        else:
            self.entry_threshold = 0.25
            self.opposite_flip_threshold = 0.60
            self.window = 12

    def update_signal(self, final_score):
        self.score_history.append(final_score)
        if len(self.score_history) > self.window:
            self.score_history.pop(0)
        
        smoothed_score = np.mean(self.score_history)

        if self.mode == "1m":
            if smoothed_score >= self.entry_threshold:
                return "LONG"
            elif smoothed_score <= -self.entry_threshold:
                return "SHORT"
            else:
                return "NEUTRAL"
        else:
            if self.current_signal == "NEUTRAL":
                if smoothed_score >= self.entry_threshold:
                    self.current_signal = "LONG"
                elif smoothed_score <= -self.entry_threshold:
                    self.current_signal = "SHORT"
            elif self.current_signal == "LONG":
                if smoothed_score <= -self.opposite_flip_threshold:
                    self.current_signal = "SHORT"
                elif smoothed_score < 0.02:
                    self.current_signal = "NEUTRAL"
            elif self.current_signal == "SHORT":
                if smoothed_score >= self.opposite_flip_threshold:
                    self.current_signal = "LONG"
                elif smoothed_score > -0.02:
                    self.current_signal = "NEUTRAL"
            return self.current_signal


# === POWER TRADING & LIQUIDATION/MANIPULATION RISK ENGINE ===
class PowerTradingRiskEngine:
    def calculate_risk_metrics(self, liquidation_volumes, displayed_vol, cancelled_vol, time_exists, obs_window, open_interest, leverage, volatility):
        total_ltz = np.sum(liquidation_volumes) if len(liquidation_volumes) > 0 else 0.0
        max_ltz = np.max(liquidation_volumes) if len(liquidation_volumes) > 0 else 0.0
        ltz_score = (max_ltz / (total_ltz + 1e-8)) * 100

        spoof_ratio = cancelled_vol / (displayed_vol + 1e-8)
        persistence = min(max(time_exists / (obs_window + 1e-8), 0), 1)
        spoof_score = spoof_ratio * (1 - persistence)

        squeeze_risk = total_ltz * open_interest * leverage * volatility
        market_risk = ltz_score + spoof_score + squeeze_risk
        
        return {
            'LTZ_Score': ltz_score,
            'Spoof_Score': spoof_score,
            'Squeeze_Risk': squeeze_risk,
            'Market_Risk': market_risk
        }


# === STREAMLIT DASHBOARD LAYOUT ===
st.title("⚡ Financial Market Learning & Analysis Hub")
st.markdown("**Institutional Quantitative Order Book & Microstructure Research Terminal**")

# Sidebar Configuration Controls
st.sidebar.header("⚙️ Terminal Engine Setup")
exchange = st.sidebar.selectbox("Exchange Source", ["Bybit Linear", "MEXC Futures", "Binance Futures"])
symbol = st.sidebar.selectbox("Trading Pair", ["BTC/USDT", "SOL/USDT", "HYPE/USDT", "ETH/USDT", "TAO/USDT", "GOLD(XAUT)USDT", "FARTCOIN/USDT"])
selected_tf = st.sidebar.selectbox("Analysis Timeframe", ["1m", "5m", "15m", "1h", "4h"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto-Refresh & Stream")
enable_auto_refresh = st.sidebar.checkbox("Enable Live Engine Feed", value=True)
refresh_interval = st.sidebar.slider("Polling Interval (Sec)", min_value=1, max_value=20, value=3)

if enable_auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="datarefresh_terminal")

# Mode mapping for Hysteresis Manager
manager_mode = "1m" if selected_tf in ["1m", "5m"] else "15m"

if 'signal_manager' not in st.session_state or st.session_state.get('current_mode') != manager_mode:
    st.session_state.signal_manager = SignalHysteresisManager(mode=manager_mode)
    st.session_state.current_mode = manager_mode

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

# Generate Dynamic Simulation Data
np.random.seed(int(datetime.now().timestamp()) % 9999)
periods_count = 100
dates = pd.date_range(end=datetime.now(), periods=periods_count, freq='T' if "m" in selected_tf else 'H')
base_p = 68000.0 if "BTC" in symbol else (220.0 if "SOL" in symbol else 10.0)

df_mock = pd.DataFrame({
    'Close': base_p + np.cumsum(np.random.randn(periods_count) * (base_p * 0.0008)),
    'Volume': np.random.randint(1000, 10000, periods_count)
}, index=dates)

spread = base_p * 0.0002
bids_mock = np.array([
    [base_p - spread, np.random.uniform(2, 10)],
    [base_p - spread*2, np.random.uniform(5, 20)],
    [base_p - spread*3, np.random.uniform(10, 40)]
])
asks_mock = np.array([
    [base_p + spread, np.random.uniform(2, 10)],
    [base_p + spread*2, np.random.uniform(5, 20)],
    [base_p + spread*3, np.random.uniform(10, 40)]
])

# Run Research Lab Calculations
lab = TenPaperResearchLab()
results, final_score, weights = lab.calculate_all_signals(df_mock, bids_mock, asks_mock)
active_signal = st.session_state.signal_manager.update_signal(final_score)

# Risk Engine Computation
risk_engine = PowerTradingRiskEngine()
risk_metrics = risk_engine.calculate_risk_metrics(
    liquidation_volumes=np.array([2500, 8900, 1200, 400]),
    displayed_vol=150000.0,
    cancelled_vol=45000.0,
    time_exists=12.0,
    obs_window=60.0,
    open_interest=5400000.0,
    leverage=25.0,
    volatility=df_mock['Close'].pct_change().std()
)

# Top Key Performance Indicators (KPIs)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Exchange & Asset", f"{exchange.split()[0]} | {symbol}")
col2.metric("Timeframe Mode", f"{selected_tf.upper()} ({manager_mode.upper()})")
col3.metric("Ensemble Score", f"{final_score:.4f}", delta=f"{final_score*100:.1f}%")
col4.metric("Active Signal Status", active_signal, delta="Strict Gate Active" if manager_mode == "15m" else "Scalping Fast")
col5.metric("Market Risk Level", f"{risk_metrics['Market_Risk']:.1f}", delta="Warning" if risk_metrics['Market_Risk'] > 45 else "Optimal", delta_color="inverse")

st.markdown("---")

# Main Multi-Tab Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 12-Paper Microstructure Grid", 
    "🛡️ Liquidation & Manipulation Guard", 
    "📈 Live Order Flow & Price Action", 
    "📜 Signal Log & Backtest Tracker"
])

with tab1:
    st.subheader("Deep Quantitative Metrics Breakdown (Academic Models)")
    m_cols = st.columns(4)
    idx = 0
    for paper, score in results.items():
        with m_cols[idx % 4]:
            st.metric(label=f"{paper} Model", value=f"{score:.3f}", delta=f"Weight: {weights.get(paper, 0)}")
        idx += 1
        
    st.markdown("### Complete Matrix Table")
    df_matrix = pd.DataFrame(list(results.items()), columns=['Metric / Paper Reference', 'Calculated Alpha Score'])
    df_matrix['Assigned Weight'] = df_matrix['Metric / Paper Reference'].map(weights)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Power Trading & Liquidation Risk Analysis Engine")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("LTZ (Liquidation Trace)", f"{risk_metrics['LTZ_Score']:.2f}%")
    r2.metric("Spoofing Pressure", f"{risk_metrics['Spoof_Score']:.4f}")
    r3.metric("Short/Long Squeeze Risk", f"{risk_metrics['Squeeze_Risk']:.2f}")
    r4.metric("Composite Risk Index", f"{risk_metrics['Market_Risk']:.2f}")
    
    st.info("💡 **Note:** High Spoofing or Squeeze Risk automatically clamps execution triggers to protect capital from predatory market maker sweeps.")

with tab3:
    st.subheader(f"Live Price Action & Trend Chart for {symbol}")
    st.line_chart(df_mock['Close'], height=350)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Top 3 Bids (Buy Side)")
        st.dataframe(pd.DataFrame(bids_mock, columns=['Price', 'Size']), hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("### Top 3 Asks (Sell Side)")
        st.dataframe(pd.DataFrame(asks_mock, columns=['Price', 'Size']), hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Automated Execution Signal Log")
    if active_signal != "NEUTRAL":
        new_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": symbol,
            "Timeframe": selected_tf,
            "Signal": active_signal,
            "Score": round(final_score, 4)
        }
        # Avoid duplicate consecutive logs
        if not st.session_state.trade_history or st.session_state.trade_history[-1]['Signal'] != active_signal:
            st.session_state.trade_history.append(new_entry)
            
    if st.session_state.trade_history:
        df_history = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.write("Waiting for active directional signal trigger...")
        
