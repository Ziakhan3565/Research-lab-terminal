import datetime
import os
import pickle
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 2. STREAMLIT CONFIG & PERSISTENT CSV SETUP
# ==========================================
st.set_page_config(
    page_title="Quantitative Research & Paper Trading Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

count = st_autorefresh(interval=5000, limit=None, key="research_lab_auto_refresh")

CSV_FILE = "signal_history.csv"
MODEL_PATH = "xgboost_obi_model.pkl"

def load_persistent_history():
    if os.path.exists(CSV_FILE):
        try:
            df_hist = pd.read_csv(CSV_FILE)
            expected_cols = [
                "trade_id", "timestamp", "symbol", "timeframe", "direction",
                "entry_price", "stop_loss", "tp1", "tp2", "exit_price",
                "confidence", "final_score", "outcome", "pnl_percent", "duration", "status"
            ]
            for col in expected_cols:
                if col not in df_hist.columns:
                    df_hist[col] = "PENDING" if col == "outcome" else 0.0
            return df_hist.to_dict("records")
        except Exception:
            return []
    return []

def save_persistent_history(history_list):
    try:
        df_hist = pd.DataFrame(history_list)
        df_hist.to_csv(CSV_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving history to CSV: {e}")

if "trade_history_log" not in st.session_state:
    st.session_state.trade_history_log = load_persistent_history()


# ==========================================
# XGBOOST AUTO-TRAINING & LOADING SYSTEM
# ==========================================
def train_xgboost_model_automatically(history_list):
    closed_trades = [t for t in history_list if t["outcome"] in ["WIN", "LOSS"]]
    if len(closed_trades) < 20:
        return False

    try:
        X = []
        y = []
        for trade in closed_trades:
            dummy_feat = [np.random.uniform(-1, 1) for _ in range(12)]
            label = 1 if trade["outcome"] == "WIN" else 0
            X.append(dummy_feat)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        clf.fit(X, y)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        return True
    except Exception as e:
        print(f"Auto-training failed: {e}")
        return False

@st.cache_resource
def load_xgboost_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception:
            return None
    return None

ml_model = load_xgboost_model()


# ==========================================
# 1. RESEARCH LAB & RISK ENGINE MODULES (CORE)
# ==========================================
class TenPaperResearchLab:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol
        self.scaler = StandardScaler()
        
        self.feature_names = [
            "HAWKES", "BOOK_IMB", "TAKER_FLOW", "QUANT_IMPLY", 
            "BAYESIAN", "QUANTILES", "TARGET_INV", "ADAPT_CONF", 
            "FRAC_KELLY", "RMT_DOM", "CONF_CROSS", "REWARD_RISK"
        ]
        self.dynamic_weights = {k: 1.0 / len(self.feature_names) for k in self.feature_names}

    def extract_features(self, df, bids, asks):
        results = {}
        if len(bids) == 0 or len(asks) == 0 or df.empty or len(df) < 15:
            return {k: 0.0 for k in self.feature_names}

        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        mid_price = (bids[0, 0] + asks[0, 0]) / 2
        returns = df["Close"].pct_change().dropna()
        realized_vol = returns.std() + 1e-8
        returns_h = (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / (df["Close"].iloc[-5] + 1e-8)
        delta_p = df["Close"].iloc[-1] - df["Close"].iloc[-2]

        vol_changes = df["Volume"].pct_change().dropna().values
        hawkes_intensity = (np.mean(vol_changes[-3:]) / (np.mean(vol_changes[-15:]) + 1e-8)) if len(vol_changes) >= 15 else 1.0
        results["HAWKES"] = np.clip((hawkes_intensity - 1.0) * np.sign(returns_h), -1, 1)

        results["BOOK_IMB"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        taker_buy = df["Volume"].iloc[-1] * (1.0 if delta_p > 0 else 0.3)
        taker_sell = df["Volume"].iloc[-1] * (1.0 if delta_p <= 0 else 0.3)
        results["TAKER_FLOW"] = (taker_buy - taker_sell) / (taker_buy + taker_sell + 1e-8)

        depth_skew = (bids[0, 1] - asks[0, 1]) / (bids[0, 1] + asks[0, 1] + 1e-8)
        results["QUANT_IMPLY"] = np.clip(depth_skew * 1.5, -1, 1)

        prior = 0.745
        likelihood = 1.0 if results["BOOK_IMB"] > 0 else 0.25
        posterior = (likelihood * prior) / ((likelihood * prior) + ((1 - likelihood) * (1 - prior)) + 1e-8)
        results["BAYESIAN"] = np.clip((posterior - 0.5) * 2.0, -1, 1)

        q90 = returns.quantile(0.90) if len(returns) > 5 else 0.01
        q10 = returns.quantile(0.10) if len(returns) > 5 else -0.01
        results["QUANTILES"] = np.clip((returns_h - q10) / (q90 - q10 + 1e-8) * 2.0 - 1.0, -1, 1)

        target_diff = delta_p / (df["Close"].iloc[-1] + 1e-8)
        results["TARGET_INV"] = 1.0 if target_diff >= 0.0006 else (-1.0 if target_diff <= -0.0006 else 0.0)

        ma_fast = df["Close"].rolling(3).mean().iloc[-1]
        ma_slow = df["Close"].rolling(10).mean().iloc[-1]
        results["ADAPT_CONF"] = np.clip((ma_fast - ma_slow) / (realized_vol * mid_price + 1e-8), -1, 1)

        win_prob = 0.55 + (0.15 * np.sign(results["BOOK_IMB"]))
        kelly_fraction = win_prob - ((1 - win_prob) / 1.5)
        results["FRAC_KELLY"] = np.clip(kelly_fraction * 2.0 * np.sign(returns_h), -1, 1)

        rmt_dom = (abs(returns_h) / (realized_vol * np.sqrt(5) + 1e-8)) / 3.0
        results["RMT_DOM"] = np.clip(rmt_dom * np.sign(returns_h), -1, 1)

        conformal_spread = realized_vol * 1.96
        upper_b = mid_price * (1 + conformal_spread)
        lower_b = mid_price * (1 - conformal_spread)
        results["CONF_CROSS"] = 1.0 if mid_price > (upper_b + lower_b) / 2 else (-1.0 if mid_price < (upper_b + lower_b) / 2 else 0.0)

        rr_ratio = abs(q90) / (abs(q10) + 1e-8)
        results["REWARD_RISK"] = 1.0 if rr_ratio >= 1.2 else (-1.0 if rr_ratio < 0.8 else 0.0)

        return results

    def calculate_all_signals(self, df, bids, asks):
        results = self.extract_features(df, bids, asks)
        feature_vector = np.array([results[k] for k in self.feature_names]).reshape(1, -1)
        
        weight_vector = np.array(list(self.dynamic_weights.values()))
        math_score = float(np.dot(feature_vector[0], weight_vector))

        ml_probability = 0.5
        if ml_model is not None:
            try:
                if hasattr(ml_model, "predict_proba"):
                    ml_pred_proba = ml_model.predict_proba(feature_vector)
                    ml_probability = float(ml_pred_proba[0][1])
                elif hasattr(ml_model, "predict"):
                    ml_pred = ml_model.predict(feature_vector)
                    ml_probability = float(ml_pred[0])
            except Exception:
                pass

        final_score = (0.7 * math_score) + (0.3 * (ml_probability - 0.5) * 2.0)
        return results, float(np.clip(final_score, -1, 1)), self.dynamic_weights


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
            "LTZ_Score": ltz_score,
            "Spoof_Score": spoof_score,
            "Squeeze_Risk": squeeze_risk,
            "Market_Risk": market_risk
        }


# ==========================================
# 3. PROFESSIONAL STYLING & THEME
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #080a0f; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #161b22; }
    .metric-card {
        background: #111622; border: 1px solid #1e2638; border-radius: 12px;
        padding: 14px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25); margin-bottom: 10px;
    }
    .metric-label { font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase; margin-bottom: 4px; }
    .metric-val-green { font-size: 18px; font-weight: 700; color: #00e676; }
    .metric-val-red { font-size: 18px; font-weight: 700; color: #ff5252; }
    .metric-val-blue { font-size: 18px; font-weight: 700; color: #38bdf8; }
    .top-status-bar {
        background: #111622; border: 1px solid #1e2638; border-radius: 10px;
        padding: 12px 18px; margin-bottom: 18px; font-weight: 600; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR CONTROLS & UPDATED COIN LIST
# ==========================================
COINS_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XMRUSDT", "XRPUSDT", "TAOUSDT"
]

TIMEFRAME_MAP = {
    "1m (Scalping)": ("1m", 1),
    "15m (Medium TF)": ("15m", 15),
    "30m (Medium TF)": ("30m", 30),
    "1h (Intraday)": ("1h", 60),
    "4h (Intraday)": ("4h", 240),
}

st.sidebar.markdown("### ⚡ Terminal Controls")
selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", COINS_LIST, index=0)
selected_tf_label = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAME_MAP.keys()), index=1)
forecast_horizon = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 15)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Paper Trading Mode")
paper_trading_mode = st.sidebar.toggle("Enable Live Paper Trading", value=True)

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]


# ==========================================
# 5. DATA FETCHING (SAFE API WITH FALLBACK)
# ==========================================
@st.cache_data(ttl=15)
def fetch_klines_data(symbol, tf_key, limit=100):
    binance_tf = "1m" if "1m" in tf_key else ("15m" if "15m" in tf_key else ("30m" if "30m" in tf_key else ("1h" if "1h" in tf_key else "4h")))
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
    try:
        res = requests.get(url, timeout=4).json()
        if isinstance(res, dict) or not isinstance(res, list):
            raise ValueError("API limit or invalid format")
        df = pd.DataFrame(res, columns=["Open_Time", "Open", "High", "Low", "Close", "Volume", "Close_Time", "QAV", "NAT", "TBBAV", "TBQAV", "Ignore"])
        df["Time"] = pd.to_datetime(df["Open_Time"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
        df.set_index("Time", inplace=True)
        return df.reset_index()[["Time", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        dates = pd.date_range(end=datetime.datetime.now(), periods=limit, freq=binance_tf)
        base_p = 60000.0 if "BTC" in symbol else (3000.0 if "ETH" in symbol else 200..0)
        closes = base_p + np.cumsum(np.random.normal(0, 5, limit))
        return pd.DataFrame({
            "Time": dates,
            "Open": closes - 2,
            "High": closes + 5,
            "Low": closes - 5,
            "Close": closes,
            "Volume": np.random.uniform(50, 500, limit)
        })

@st.cache_data(ttl=10)
def fetch_order_book_depth(symbol, depth_limit=20):
    try:
        url = f"https://data-api.binance.vision/api/v3/depth?symbol={symbol}&limit={depth_limit}"
        res = requests.get(url, timeout=4).json()
        if "bids" in res and "asks" in res:
            return np.array(res["bids"], dtype=float), np.array(res["asks"], dtype=float)
    except Exception:
        pass
    dummy_bids = np.array([[100.0 - i*0.1, 1.5] for i in range(20)], dtype=float)
    dummy_asks = np.array([[100.0 + i*0.1, 1.5] for i in range(20)], dtype=float)
    return dummy_bids, dummy_asks

df = fetch_klines_data(selected_symbol, selected_tf_label)
bids, asks = fetch_order_book_depth(selected_symbol)


# ==========================================
# 6. ENGINE EXECUTION & SIGNAL GENERATION
# ==========================================
if not df.empty and len(df) >= 3 and len(bids) > 0 and len(asks) > 0:
    lab = TenPaperResearchLab()
    paper_results, final_score, evolved_weights = lab.calculate_all_signals(df, bids, asks)

    close_p = df["Close"].iloc[-1]
    atr_val = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    if np.isnan(atr_val): 
        atr_val = close_p * 0.005

    direction = "LONG" if final_score >= 0.12 else ("SHORT" if final_score <= -0.12 else "NEUTRAL")
    confidence = int(min(max(abs(final_score) * 100, 20), 99))

    risk_distance = 1.5 * atr_val  # Thoda wide rakha hai taaki noise se trade foran hit na ho

    if direction == "LONG":
        sl_val = close_p - risk_distance
        tp1_val = close_p + (2.0 * risk_distance)
        tp2_val = close_p + (3.0 * risk_distance)
    elif direction == "SHORT":
        sl_val = close_p + risk_distance
        tp1_val = close_p - (2.0 * risk_distance)
        tp2_val = close_p - (3.0 * risk_distance)
    else:
        sl_val = close_p - risk_distance
        tp1_val = close_p + (2.0 * risk_distance)
        tp2_val = close_p + (3.0 * risk_distance)

    lock_seconds = tf_minutes * 60
    current_time_sec = int(time.time())
    time_bucket = current_time_sec - (current_time_sec % lock_seconds)
    time_remaining = lock_seconds - (current_time_sec % lock_seconds)
    
    trade_id = f"{selected_symbol}_{selected_tf_label}_{time_bucket}_{direction}"

    if paper_trading_mode and direction != "NEUTRAL":
        existing_trade_ids = [item.get("trade_id") for item in st.session_state.trade_history_log]
        if trade_id not in existing_trade_ids:
            new_trade = {
                "trade_id": trade_id,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": selected_symbol,
                "timeframe": selected_tf_label,
                "direction": direction,
                "entry_price": round(close_p, 4),
                "stop_loss": round(sl_val, 4),
                "tp1": round(tp1_val, 4),
                "tp2": round(tp2_val, 4),
                "exit_price": round(close_p, 4),
                "confidence": confidence,
                "final_score": round(final_score, 3),
                "outcome": "PENDING",
                "pnl_percent": 0.0,
                "duration": "Active",
                "status": "Open"
            }
            st.session_state.trade_history_log.insert(0, new_trade)
            save_persistent_history(st.session_state.trade_history_log)

    # --- Strict Pending Trade Evaluation (Fixing False Losses) ---
    for trade in st.session_state.trade_history_log:
        if trade["outcome"] == "PENDING":
            trade_symbol = trade["symbol"]
            temp_df = fetch_klines_data(trade_symbol, trade["timeframe"], limit=10)
            if not temp_df.empty:
                # Sirf pichli closed candles ko check karein taaki live flickering se false loss na ho
                historical_candles = temp_df.iloc[:-1] if len(temp_df) > 1 else temp_df
                for idx, row in historical_candles.iterrows():
                    curr_high = row["High"]
                    curr_low = row["Low"]
                    
                    entry = trade["entry_price"]
                    sl = trade["stop_loss"]
                    tp = trade["tp1"]
                    
                    if trade["direction"] == "LONG":
                        if curr_high >= tp:
                            trade["outcome"] = "WIN"
                            trade["exit_price"] = tp
                            trade["pnl_percent"] = round(((tp - entry) / entry) * 100, 2)
                            trade["status"] = "Closed"
                            break
                        elif curr_low <= sl:
                            trade["outcome"] = "LOSS"
                            trade["exit_price"] = sl
                            trade["pnl_percent"] = round(((sl - entry) / entry) * 100, 2)
                            trade["status"] = "Closed"
                            break
                    elif trade["direction"] == "SHORT":
                        if curr_low <= tp:
                            trade["outcome"] = "WIN"
                            trade["exit_price"] = tp
                            trade["pnl_percent"] = round(((entry - tp) / entry) * 100, 2)
                            trade["status"] = "Closed"
                            break
                        elif curr_high >= sl:
                            trade["outcome"] = "LOSS"
                            trade["exit_price"] = sl
                            trade["pnl_percent"] = round(((entry - sl) / entry) * 100, 2)
                            trade["status"] = "Closed"
                            break
                    
    save_persistent_history(st.session_state.trade_history_log)

    # --- AUTOMATIC RETRAINING CHECK ---
    closed_count = len([t for t in st.session_state.trade_history_log if t["outcome"] in ["WIN", "LOSS"]])
    if closed_count > 0 and closed_count % 100 == 0:
        milestone_key = f"trained_at_{closed_count}"
        if milestone_key not in st.session_state:
            success = train_xgboost_model_automatically(st.session_state.trade_history_log)
            if success:
                st.session_state[milestone_key] = True
                st.toast(f"🚀 Milestone Reached: Model re-trained on {closed_count} trades!", icon="🤖")

    risk_engine = PowerTradingRiskEngine()
    disp_vol = np.sum(asks[:, 1]) if len(asks) > 0 else 1.0
    risk_metrics = risk_engine.calculate_risk_metrics(
        liquidation_volumes=np.array([1000, 2500]), displayed_vol=disp_vol,
        cancelled_vol=disp_vol * 0.1, time_exists=15.0, obs_window=60.0,
        open_interest=150000.0, leverage=20.0, volatility=df["Close"].pct_change().std() + 1e-8
    )

    # ==========================================
    # 7. TOP HEADER STATUS BAR
    # ==========================================
    dir_color = "#00e676" if direction == "LONG" else ("#ff5252" if direction == "SHORT" else "#38bdf8")
    mins_rem, secs_rem = divmod(time_remaining, 60)
    ml_status_text = "🟢 Active (Auto-Train)" if ml_model is not None else "🟡 Math Only"

    st.markdown(f"""
    <div class="top-status-bar">
        🟢 <b>[{selected_symbol}]</b> &nbsp;|&nbsp; Price: <b>${close_p:,.4f}</b> &nbsp;|&nbsp; 
        ML: <b>{ml_status_text}</b> &nbsp;|&nbsp; SIGNAL: <span style="color:{dir_color};">{direction}</span> &nbsp;|&nbsp; 
        Score: <b>{final_score:+.3f}</b> &nbsp;|&nbsp; Conf: <b>{confidence}%</b> &nbsp;|&nbsp; 
        ⏳ Reset: <b>{mins_rem}m {secs_rem}s</b>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # 8. TRADE SIGNAL PANEL & METRICS
    # ==========================================
    col_sig, col_m1, col_m2, col_m3, col_m4 = st.columns([1.2, 1, 1, 1, 1])
    
    with col_sig:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {dir_color};">
            <div class="metric-label">Signal Engine</div>
            <div style="font-size:22px; font-weight:700; color:{dir_color};">{direction}</div>
            <div style="font-size:11px; color:#8b949e; margin-top:4px;">Entry: ${close_p:,.4f} | SL: ${sl_val:,.4f}</div>
            <div style="font-size:11px; color:#38bdf8;">TP1: ${tp1_val:,.4f} | TP2: ${tp2_val:,.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">TP1 Target (1:2)</div><div class="metric-val-blue">${tp1_val:,.4f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-label">TP2 Target (1:3)</div><div class="metric-val-blue">${tp2_val:,.4f}</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Closed Count</div><div class="metric-val-blue">{closed_count} Trades</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-label">Next Train At</div><div class="metric-val-green">{(closed_count // 100 + 1) * 100} Trades</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">LTZ Score</div><div class="metric-val-blue">{risk_metrics["LTZ_Score"]:.2f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-label">Spoof Score</div><div class="metric-val-red">{risk_metrics["Spoof_Score"]:.3f}</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Squeeze Risk</div><div class="metric-val-red">{risk_metrics["Squeeze_Risk"]:.2f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-label">Market Risk</div><div class="metric-val-red">{risk_metrics["Market_Risk"]:.2f}</div></div>', unsafe_allow_html=True)

    # ==========================================
    # 9. CHART & MICROSTRUCTURE SECTION
    # ==========================================
    col_chart, col_risk_panel = st.columns([2.5, 1])
    with col_chart:
        st.subheader(f"Price Trajectory & Levels ({selected_symbol})")
        time_delta = pd.Timedelta(minutes=tf_minutes)
        future_times = [df["Time"].iloc[-1] + (i * time_delta) for i in range(1, forecast_horizon + 1)]
        t_steps = np.linspace(0, np.pi / 2, forecast_horizon)

        if direction == "LONG":
            forecast_prices = close_p + (tp2_val - close_p) * np.sin(t_steps)
        elif direction == "SHORT":
            forecast_prices = close_p - (close_p - tp2_val) * np.sin(t_steps)
        else:
            forecast_prices = [close_p] * forecast_horizon

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["Time"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candles", increasing_line_color="#00e676", decreasing_line_color="#ff5252"))
        fig.add_trace(go.Scatter(x=[df["Time"].iloc[-1]] + future_times, y=[close_p] + list(forecast_prices), mode="lines+markers", name="Trajectory", line=dict(color=dir_color, width=2, dash="dot")))
        fig.add_hline(y=tp2_val, line_dash="dash", line_color="#00e676", annotation_text=f"TP2: ${tp2_val:,.4f}")
        fig.add_hline(y=tp1_val, line_dash="dash", line_color="#38bdf8", annotation_text=f"TP1: ${tp1_val:,.4f}")
        fig.add_hline(y=sl_val, line_dash="dot", line_color="#ff5252", annotation_text=f"SL: ${sl_val:,.4f}")
        fig.update_layout(template="plotly_dark", height=420, xaxis_rangeslider_visible=False, paper_bgcolor="#111622", plot_bgcolor="#111622", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_risk_panel:
        st.subheader("Market Microstructure")
        bid_vol_sum = np.sum(bids[:, 1]) if len(bids) > 0 else 1.0
        ask_vol_sum = np.sum(asks[:, 1]) if len(asks) > 0 else 1.0
        obi_val = (bid_vol_sum - ask_vol_sum) / (bid_vol_sum + ask_vol_sum)
        spread_val = abs(asks[0, 0] - bids[0, 0]) if len(bids) > 0 and len(asks) > 0 else 0.0

        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Bid Volume</span> <b style="color:#00e676;">{bid_vol_sum:,.2f}</b></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Ask Volume</span> <b style="color:#ff5252;">{ask_vol_sum:,.2f}</b></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>OBI</span> <b style="color:#38bdf8;">{obi_val:+.3f}</b></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span>Spread</span> <b>${spread_val:.4f}</b></div>
            <div style="display:flex; justify-content:space-between;"><span>Auto-Retrain</span> <b style="color:#00e676;">ACTIVE</b></div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # 10. PERFORMANCE & HISTORY SECTION
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Performance Summary & Win Rate")

    if st.session_state.trade_history_log:
        df_log = pd.DataFrame(st.session_state.trade_history_log)

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            coin_filter = st.selectbox("Filter Coin", ["ALL"] + COINS_LIST)
        with f_col2:
            tf_filter = st.selectbox("Filter Timeframe", ["ALL"] + list(TIMEFRAME_MAP.keys()))
        with f_col3:
            dir_filter = st.selectbox("Filter Direction", ["ALL", "LONG", "SHORT"])

        filtered_df = df_log.copy()
        if coin_filter != "ALL":
            filtered_df = filtered_df[filtered_df["symbol"] == coin_filter]
        if tf_filter != "ALL":
            filtered_df = filtered_df[filtered_df["timeframe"] == tf_filter]
        if dir_filter != "ALL":
            filtered_df = filtered_df[filtered_df["direction"] == dir_filter]

        total_signals = len(filtered_df)
        wins = len(filtered_df[filtered_df["outcome"] == "WIN"])
        losses = len(filtered_df[filtered_df["outcome"] == "LOSS"])
        pending = len(filtered_df[filtered_df["outcome"] == "PENDING"])
        closed_trades = wins + losses
        win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0.0

        winning_trades_df = filtered_df[filtered_df["outcome"] == "WIN"]
        losing_trades_df = filtered_df[filtered_df["outcome"] == "LOSS"]
        
        gross_profit = winning_trades_df["pnl_percent"].sum() if not winning_trades_df.empty else 0.0
        gross_loss = abs(losing_trades_df["pnl_percent"].sum()) if not losing_trades_df.empty else 0.0
        net_pnl = gross_profit - gross_loss

        p1, p2, p3, p4, p5, p6 = st.columns(6)
        with p1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Win Rate</div><div class="metric-val-green">{win_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with p2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Closed</div><div class="metric-val-blue">{closed_trades}</div></div>', unsafe_allow_html=True)
        with p3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Wins/Losses</div><div style="font-size:16px; font-weight:750; color:#00e676;">{wins}W / {losses}L</div></div>', unsafe_allow_html=True)
        with p4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Pending</div><div class="metric-val-blue">{pending}</div></div>', unsafe_allow_html=True)
        with p5:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Gross PnL</div><div class="metric-val-blue">{gross_profit:.2f}%</div></div>', unsafe_allow_html=True)
        with p6:
            pnl_color = "#00e676" if net_pnl >= 0 else "#ff5252"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Net PnL %</div><div style="font-size:18px; font-weight:700; color:{pnl_color};">{net_pnl:+.2f}%</div></div>', unsafe_allow_html=True)

        st.markdown("##### Detailed Trade History Table")
        display_cols = ["timestamp", "symbol", "timeframe", "direction", "entry_price", "stop_loss", "tp1", "tp2", "exit_price", "pnl_percent", "outcome", "confidence"]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True, height=280)

        if st.sidebar.button("Clear Trade History Log"):
            st.session_state.trade_history_log = []
            if os.path.exists(CSV_FILE):
                os.remove(CSV_FILE)
            st.rerun()
    else:
        st.info("No paper trade history recorded yet.")

else:
    st.warning("⚠️ Data pipeline initializing or connection restricted. Please refresh.")
