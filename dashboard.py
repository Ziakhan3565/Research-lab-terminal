import streamlit as st
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Page Configuration
st.set_page_config(
    page_title="Financial Market Learning & Analysis Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Professional Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

# === QUANTITATIVE RESEARCH LAB (12 PAPERS) ===
class TenPaperResearchLab:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol

    def calculate_all_signals(self, df, bids, asks, current_inventory=0, performance_history=None):
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
    def __init__(self):
        pass

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


# === STREAMLIT DASHBOARD UI SETUP ===
st.title("⚡ Financial Market Learning & Analysis Hub")
st.markdown("### Real-Time Microstructure Research & Quantitative Terminal")

# Sidebar Controls & Auto-Refresh Setup
st.sidebar.header("⚙️ Terminal Controls")
symbol = st.sidebar.selectbox("Trading Pair", ["BTC/USDT", "SOL/USDT", "HYPE/USDT", "ETH/USDT", "TAO/USDT"])
selected_tf = st.sidebar.selectbox("Select Timeframe", ["1m", "15m", "1h", "4h"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto-Refresh Configuration")
enable_auto_refresh = st.sidebar.checkbox("Enable Live Auto-Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (Seconds)", min_value=2, max_value=30, value=5)

if enable_auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

# Mode Selection
manager_mode = "1m" if selected_tf == "1m" else "15m"

if 'signal_manager' not in st.session_state or st.session_state.get('current_mode') != manager_mode:
    st.session_state.signal_manager = SignalHysteresisManager(mode=manager_mode)
    st.session_state.current_mode = manager_mode

# Dynamic/Simulated Live Market Data Feed
np.random.seed(int(pd.Timestamp.now().timestamp()) % 1000)
dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='T' if selected_tf == '1m' else '15T')
base_price = 65000.0 if "BTC" in symbol else 180.0
df_mock = pd.DataFrame({
    'Close': base_price + np.cumsum(np.random.randn(60) * (base_price * 0.0005)),
    'Volume': np.random.randint(500, 5000, 60)
}, index=dates)

spread = base_price * 0.0001
bids_mock = np.array([[base_price - spread, np.random.uniform(1, 5)], [base_price - spread*2, np.random.uniform(2, 8)]])
asks_mock = np.array([[base_price + spread, np.random.uniform(1, 5)], [base_price + spread*2, np.random.uniform(2, 8)]])

# Run Computations
lab = TenPaperResearchLab()
results, final_score, weights = lab.calculate_all_signals(df_mock, bids_mock, asks_mock)
active_signal = st.session_state.signal_manager.update_signal(final_score)

risk_engine = PowerTradingRiskEngine()
risk_metrics = risk_engine.calculate_risk_metrics(
    liquidation_volumes=np.array([1200, 4500, 300]),
    displayed_vol=50000.0,
    cancelled_vol=12000.0,
    time_exists=15.0,
    obs_window=60.0,
    open_interest=1500000.0,
    leverage=20.0,
    volatility=0.02
)

# Top Dashboard Metrics Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Asset Pair", symbol)
col2.metric("Timeframe Mode", f"{selected_tf} ({manager_mode.upper()})")
col3.metric("Ensemble Score", f"{final_score:.4f}")
col4.metric("Stable Signal Status", active_signal, delta="Live Sync Active" if enable_auto_refresh else "Paused")

st.markdown("---")

# Main Dashboard Layout tabs
tab1, tab2, tab3 = st.tabs(["📊 12-Paper Metrics", "🛡️ Risk & Liquidation Terminal", "📈 Price & Order Flow"])

with tab1:
    st.subheader("Quantitative Metrics Breakdown (12 Academic Papers)")
    
    # Create columns for nice layout of metrics
    metrics_cols = st.columns(3)
    idx = 0
    for metric_name, val in results.items():
        with metrics_cols[idx % 3]:
            st.metric(label=f"{metric_name} Score", value=f"{val:.3f}", delta=f"Weight: {weights.get(metric_name, 0)}")
        idx += 1
        
    st.markdown("### Metrics Heatmap Table")
    df_results = pd.DataFrame(list(results.items()), columns=['Paper/Metric', 'Score'])
    st.dataframe(df_results, use_container_width=True)

with tab2:
    st.subheader("Power Trading & Manipulation Risk Engine")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("LTZ Score", f"{risk_metrics['LTZ_Score']:.2f}%")
    r_col2.metric("Spoofing Risk", f"{risk_metrics['Spoof_Score']:.4f}")
    r_col3.metric("Squeeze Risk", f"{risk_metrics['Squeeze_Risk']:.2f}")
    r_col4.metric("Composite Risk Index", f"{risk_metrics['Market_Risk']:.2f}", delta="Caution" if risk_metrics['Market_Risk'] > 50 else "Normal")

with tab3:
    st.subheader("Live Price Action & Trend Chart")
    st.line_chart(df_mock['Close'])
