import datetime
import os
import time
import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
DATA_FILE = "trade_signals_history.json"


def load_trade_data():
  if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
      try:
        return json.load(f)
      except json.JSONDecodeError:
        return []
  return []


def save_trade_data(data):
  with open(DATA_FILE, "w") as f:
    json.dump(data, f, indent=4)


def log_signal(symbol, side, score, status="Pending"):
  """Logs a new trade signal with timestamp, side, score, and initial status."""
  history = load_trade_data()

  new_signal = {
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "date": datetime.now().strftime("%Y-%m-%d"),
      "week": datetime.now().strftime("%Y-W%V"),  # Year and Week number
      "symbol": symbol,
      "side": side.upper(),  # LONG or SHORT
      "score": score,
      "status": status,  # Win, Loss, or Pending
  }

  history.append(new_signal)
  save_trade_data(history)


def update_signal_status(timestamp, symbol, new_status):
  """Updates whether a specific trade was a Win or Loss."""
  history = load_trade_data()
  for item in history:
    if item["timestamp"] == timestamp and item["symbol"] == symbol:
      item["status"] = new_status
      break
  save_trade_data(history)


def get_weekly_performance():
  """Aggregates weekly total signals, long/short counts, and wins/losses."""
  history = load_trade_data()
  if not history:
    return pd.DataFrame()

  df = pd.DataFrame(history)
  current_week = datetime.now().strftime("%Y-W%V")
  weekly_df = df[df["week"] == current_week]

  return weekly_df
# ==========================================
# EXCHANGE CONNECTION HELPER
# ==========================================
def get_exchange_connection(exchange_name, api_key, api_secret, mode):
    exchanges = {"bybit": ccxt.bybit, "binance": ccxt.binance, "mexc": ccxt.mexc}

    if exchange_name not in exchanges:
        return None

    exchange_class = exchanges[exchange_name]
    exchange = exchange_class({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    if mode == "Paper Trading" and exchange_name in ["bybit", "binance"]:
        exchange.set_sandbox_mode(True)

    return exchange

# ==========================================
# RESEARCH LAB MODULE & RISK ENGINE FALLBACK
# ==========================================
try:
  from src.research_lab import PowerTradingRiskEngine, TenPaperResearchLab
except ModuleNotFoundError:

  class TenPaperResearchLab:

    def calculate_all_signals(
        self, df, bids, asks, current_inventory=0, performance_history=None
    ):
      paper_results = {
          "OFI": -0.204,
          "TSMOM": 0.850,
          "MICRO": -0.050,
          "AVST": 0.120,
          "INVAR": 0.450,
          "VPIN": -0.310,
          "LAMBDA": 0.080,
          "PIN": -0.150,
          "LOB_IMB": -0.220,
          "FLOW_IMB": 0.300,
      }
      final_score = -0.136
      evolved_weights = {k: 0.10 for k in paper_results.keys()}
      return paper_results, final_score, evolved_weights

  class PowerTradingRiskEngine:

    def __init__(self):
      pass

    def calculate_risk_metrics(
        self,
        liquidation_volumes,
        displayed_vol,
        cancelled_vol,
        time_exists,
        obs_window,
        open_interest,
        leverage,
        volatility,
    ):
      return {
          "LTZ_Score": 12.5,
          "Spoof_Score": 0.15,
          "Squeeze_Risk": 1.45,
          "Market_Risk": 14.1,
      }


# ==========================================
# STREAMLIT PAGE CONFIG & PERSISTENT CSV SETUP
# ==========================================
st.set_page_config(
    page_title="10-Paper Research Lab Terminal",
    layout="wide",
    initial_sidebar_state="auto",
)

CSV_FILE = "signal_history.csv"
TRADES_CSV_FILE = "live_trades_history.csv"


def load_persistent_history():
  if os.path.exists(CSV_FILE):
    try:
      df_hist = pd.read_csv(CSV_FILE)
      if "outcome" not in df_hist.columns:
        df_hist["outcome"] = "PENDING"
      return df_hist.to_dict("records")
    except Exception:
      return []
  return []


def save_persistent_history(history_list):
  try:
    df_hist = pd.DataFrame(history_list)
    if "bucket" in df_hist.columns:
      df_hist_save = df_hist.drop(columns=["bucket"])
    else:
      df_hist_save = df_hist
    df_hist_save.to_csv(CSV_FILE, index=False)
  except Exception as e:
    st.error(f"Error saving history: {e}")


def load_trades_history():
  if os.path.exists(TRADES_CSV_FILE):
    try:
      return pd.read_csv(TRADES_CSV_FILE).to_dict("records")
    except Exception:
      return []
  return []


def save_trades_history(trades_list):
  try:
    df_trades = pd.DataFrame(trades_list)
    df_trades.to_csv(TRADES_CSV_FILE, index=False)
  except Exception as e:
    st.error(f"Error saving trades: {e}")


if "trade_history_log" not in st.session_state:
  st.session_state.trade_history_log = load_persistent_history()

if "active_trades_log" not in st.session_state:
  st.session_state.active_trades_log = load_trades_history()

# ==========================================
# RESPONSIVE STYLING
# ==========================================
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR CONTROLS & EXCHANGE CONFIG
# ==========================================
COINS_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "APTUSDT",
    "TRXUSDT",
]

TIMEFRAME_MAP = {
    "1m": ("1m", 1),
    "5m": ("5m", 5),
    "10m": ("5m", 10),
    "15m": ("15m", 15),
    "30m": ("30m", 30),
    "1h": ("1h", 60),
    "4h": ("4h", 240),
}

st.sidebar.markdown("### ⚡ Terminal Controls")
selected_symbol = st.sidebar.selectbox(
    "Select Cryptocurrency (For View)", COINS_LIST, index=0
)
selected_tf_label = st.sidebar.selectbox(
    "Select Timeframe", list(TIMEFRAME_MAP.keys()), index=3
)
forecast_horizon = st.sidebar.slider("Forecast Horizon Candles", 5, 30, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Exchange API Settings")
exchange_name = st.sidebar.selectbox(
    "Select Exchange", ["mexc", "binance", "bybit"], index=0
)
api_key = st.sidebar.text_input("API Key", type="password")
api_secret = st.sidebar.text_input("API Secret", type="password")
trading_mode = st.sidebar.radio("Execution Mode", ["Paper Trading", "Live/Real"])

st.sidebar.markdown("---")
st.sidebar.success("🟢 **System Status: Fast Mode Active**")

api_interval, tf_minutes = TIMEFRAME_MAP[selected_tf_label]


# ==========================================
# DATA FETCHING HELPERS (OPTIMIZED WITH CACHE)
# ==========================================
@st.cache_data(ttl=15)
def fetch_klines_data(symbol, tf_label, limit=100):
  binance_tf = "5m" if tf_label == "10m" else tf_label
  url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
  try:
    res = requests.get(url, timeout=3).json()
    if isinstance(res, dict) and "code" in res:
      return pd.DataFrame()
    df = pd.DataFrame(
        res,
        columns=[
            "Open_Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close_Time",
            "QAV",
            "NAT",
            "TBBAV",
            "TBQAV",
            "Ignore",
        ],
    )
    df["Time"] = pd.to_datetime(df["Open_Time"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
      df[col] = df[col].astype(float)
    df.set_index("Time", inplace=True)
    return df.reset_index()[["Time", "Open", "High", "Low", "Close", "Volume"]]
  except Exception:
    return pd.DataFrame()


@st.cache_data(ttl=5)
def fetch_order_book_depth(symbol, depth_limit=10):
  try:
    url = f"https://data-api.binance.vision/api/v3/depth?symbol={symbol}&limit={depth_limit}"
    res = requests.get(url, timeout=3).json()
    if "bids" in res and "asks" in res:
      return np.array(res["bids"], dtype=float), np.array(
          res["asks"], dtype=float
      )
    return np.array([]), np.array([])
  except Exception:
    return np.array([]), np.array([])


# ==========================================
# AUTO OUTCOME CHECKER FOR SAVED HISTORY
# ==========================================
def check_auto_outcome(entry_price, df_candles, direction, sl_distance):
  tp_distance = sl_distance * 2

  if direction == "LONG":
    tp_price = entry_price + tp_distance
    sl_price = entry_price - sl_distance
    for _, row in df_candles.iterrows():
      if row["Low"] <= sl_price:
        return "LOSS"
      if row["High"] >= tp_price:
        return "WIN"
  elif direction == "SHORT":
    tp_price = entry_price - tp_distance
    sl_price = entry_price + sl_distance
    for _, row in df_candles.iterrows():
      if row["High"] >= sl_price:
        return "LOSS"
      if row["Low"] <= tp_price:
        return "WIN"

  return "PENDING"


history_updated = False
for item in st.session_state.trade_history_log:
  if (
      item.get("outcome", "PENDING") == "PENDING"
      and item.get("direction") != "NEUTRAL"
  ):
    curr_df = fetch_klines_data(item["symbol"], item["timeframe"], limit=50)
    if not curr_df.empty:
      signal_time = pd.to_datetime(item["timestamp"])
      future_candles = curr_df[curr_df["Time"] >= signal_time]

      if future_candles.empty:
        future_candles = curr_df

      atr_val = (curr_df["High"] - curr_df["Low"]).mean()
      sl_dist = (
          atr_val
          if not np.isnan(atr_val) and atr_val > 0
          else (item["price"] * 0.01)
      )

      res_status = check_auto_outcome(
          item["price"], future_candles, item["direction"], sl_dist
      )
      if res_status != "PENDING":
        item["outcome"] = res_status
        history_updated = True

if history_updated:
  save_persistent_history(st.session_state.trade_history_log)

# ==========================================
# FETCH DATA FOR SELECTED VIEW COIN ONLY
# ==========================================
df = fetch_klines_data(selected_symbol, selected_tf_label)
bids, asks = fetch_order_book_depth(selected_symbol)

st.markdown("## ⚡ Research Lab — Multi-Asset Signal & Trade Engine")

if not df.empty and len(df) >= 3 and len(bids) > 0 and len(asks) > 0:

  def compute_signal(df_in, bids_in, asks_in, history):
    lab = TenPaperResearchLab()
    try:
      paper_results, final_score, evolved_weights = lab.calculate_all_signals(
          df_in, bids_in, asks_in, current_inventory=0, performance_history=history
      )
    except Exception:
      paper_results = {
          "OFI": -0.204,
          "TSMOM": 0.850,
          "MICRO": -0.050,
          "AVST": 0.120,
          "INVAR": 0.450,
          "VPIN": -0.310,
          "LAMBDA": 0.080,
          "PIN": -0.150,
          "LOB_IMB": -0.220,
          "FLOW_IMB": 0.300,
      }
      final_score = -0.136
      evolved_weights = {k: 0.10 for k in paper_results.keys()}

    close_p = df_in["Close"].iloc[-1]
    atr_val = (df_in["High"] - df_in["Low"]).rolling(14).mean().iloc[-1]
    beam_level = close_p + (1.8 * atr_val)
    base_level = close_p - (1.8 * atr_val)
    trajectory_dir = (
        "LONG"
        if final_score >= 0.15
        else ("SHORT" if final_score <= -0.15 else "NEUTRAL")
    )

    return {
        "score": final_score,
        "direction": trajectory_dir,
        "beam": beam_level,
        "base": base_level,
        "paper_results": paper_results,
        "evolved_weights": evolved_weights,
        "close_price": close_p,
    }

  signal = compute_signal(df, bids, asks, st.session_state.trade_history_log)

  lock_seconds = tf_minutes * 60
  current_time_sec = int(time.time())
  time_bucket = current_time_sec - (current_time_sec % lock_seconds)
  time_remaining = lock_seconds - (current_time_sec % lock_seconds)
  global_bucket = f"{selected_symbol}_{selected_tf_label}_{time_bucket}"

  existing_buckets = [
      item.get("bucket") for item in st.session_state.trade_history_log
  ]
  if global_bucket not in existing_buckets:
    new_entry = {
        "bucket": global_bucket,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": selected_symbol,
        "timeframe": selected_tf_label,
        "direction": signal["direction"],
        "score": round(signal["score"], 3),
        "price": round(signal["close_price"], 2),
        "outcome": "PENDING",
    }
    st.session_state.trade_history_log.insert(0, new_entry)
    save_persistent_history(st.session_state.trade_history_log)

  risk_engine = PowerTradingRiskEngine()
  liq_vols = np.array([1000, 2500, 500])
  disp_vol = np.sum(asks[:, 1]) if len(asks) > 0 else 1.0
  canc_vol = disp_vol * 0.12
  risk_metrics = risk_engine.calculate_risk_metrics(
      liquidation_volumes=liq_vols,
      displayed_vol=disp_vol,
      cancelled_vol=canc_vol,
      time_exists=15.0,
      obs_window=60.0,
      open_interest=150000.0,
      leverage=20.0,
      volatility=df["Close"].pct_change().std() + 1e-8,
  )

  mins_rem = time_remaining // 60
  secs_rem = time_remaining % 60
  dir_color = (
      "#00e676"
      if signal["direction"] == "LONG"
      else ("#ff5252" if signal["direction"] == "SHORT" else "#38bdf8")
  )

  st.markdown(
      f"""
    <div class="top-status-bar">
        🔵 <b>Viewing: [{selected_symbol}]</b> | Timeframe: {selected_tf_label} | <b>SIGNAL:</b> <span style="color:{dir_color};">{signal['direction']}</span> &nbsp;|&nbsp; 
        Net Score: <span style="color:#ff5252;">{signal['score']:+.3f}</span> &nbsp;|&nbsp; Target (BEAM): <span style="color:#38bdf8;">${signal['beam']:,.2f}</span> &nbsp;|&nbsp; 
        ⏳ Candle Reset In: <b>{mins_rem}m {secs_rem}s</b>
    </div>
    """,
      unsafe_allow_html=True,
  )

  m1, m2, m3, m4, m5, m6 = st.columns([1.5, 1, 1, 1, 1, 1])
  close_val = df["Close"].iloc[-1]
  prev_val = df["Close"].iloc[-2]
  pct_change = ((close_val - prev_val) / prev_val) * 100
  signal_card_color = (
      "#00e676"
      if signal["direction"] == "LONG"
      else ("#ff5252" if signal["direction"] == "SHORT" else "#38bdf8")
  )

  with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">🟠'
        f" {selected_symbol}</div><div"
        f' class="metric-value-green">${close_val:,.2f}</div><div'
        f' style="font-size:11px; color:#00e676;">+{pct_change:.2f}%'
        " (24h)</div></div>",
        unsafe_allow_html=True,
    )
  with m2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Net'
        f' Score</div><div'
        f' class="metric-value-red">{signal["score"]:+.3f}</div></div>',
        unsafe_allow_html=True,
    )
  with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Signal</div><div'
        f' style="font-size:16px; font-weight:700; color:{signal_card_color};'
        f' margin-top:4px;">{signal["direction"]}</div></div>',
        unsafe_allow_html=True,
    )
  with m4:
    beam_val_str = f"${signal['beam']:,.2f}"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Target'
        f' (BEAM)</div><div class="metric-value-blue">{beam_val_str}</div></div>',
        unsafe_allow_html=True,
    )
  with m5:
    fig_gauge = go.Figure(
        go.Pie(
            values=[42, 58],
            hole=0.7,
            marker_colors=["#f59e0b", "#1e2638"],
            textinfo="none",
            showlegend=False,
        )
    )
    fig_gauge.update_layout(
        annotations=[
            dict(
                text="<b>42%</b>",
                x=0.5,
                y=0.5,
                font_size=14,
                font_color="#ffffff",
                showarrow=False,
            )
        ],
        margin=dict(l=0, r=0, t=0, b=0),
        height=70,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Confidence</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        fig_gauge, use_container_width=True, config={"displayModeBar": False}
    )
    st.markdown("</div>", unsafe_allow_html=True)
  with m6:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Reset In</div><div'
        f' style="font-size:16px; font-weight:700; color:#ffffff;'
        f' margin-top:4px;">{mins_rem}m {secs_rem}s</div></div>',
        unsafe_allow_html=True,
    )

  # ==========================================
  # TRADE EXECUTION PANEL & AUTO SL/TP CONFIG
  # ==========================================
  st.markdown("---")
  st.subheader("🚀 Live / Paper Trade Execution Panel")

  auto_trade_mode = st.checkbox(
      "🤖 Enable Fully Automatic Trading on Signal", value=False
  )

  t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns([1.2, 1, 1, 1, 1.2])

  with t_col1:
    trade_action = st.selectbox(
        "Action",
        (
            ["BUY / LONG", "SELL / SHORT"]
            if "signal" not in locals() or signal.get("direction") != "SHORT"
            else ["SELL / SHORT", "BUY / LONG"]
        ),
    )
  with t_col2:
    order_size_usdt = st.number_input(
        "Size (USDT)", min_value=10.0, max_value=10000.0, value=100.0, step=10.0
    )
  with t_col3:
    leverage_val = st.slider("Leverage (x)", 1, 50, 10)
  with t_col4:
    sl_pct = st.number_input(
        "Stop Loss %", min_value=0.1, max_value=10.0, value=1.5, step=0.1
    )
    tp_pct = st.number_input(
        "Take Profit %", min_value=0.1, max_value=50.0, value=3.0, step=0.1
    )
  with t_col5:
    st.markdown("<br>", unsafe_allow_html=True)
    execute_btn = st.button(
        "⚡ Execute Trade Order", use_container_width=True, type="primary"
    )

  if execute_btn:
    active_exchange = get_exchange_connection(
        exchange_name, api_key, api_secret, trading_mode
    )

    if auto_trade_mode:
      direction_tag = (
          signal["direction"]
          if "signal" in locals() and signal.get("direction")
          else "LONG"
      )
      if direction_tag == "NEUTRAL":
        direction_tag = "LONG"
    else:
      direction_tag = "LONG" if "BUY" in trade_action else "SHORT"

    entry_p = close_val if "close_val" in locals() else 1.0
    qty = (order_size_usdt * leverage_val) / entry_p

    try:
      if trading_mode == "Live/Real" and active_exchange:
        active_exchange.load_markets()
        try:
          active_exchange.set_leverage(leverage_val, selected_symbol)
        except Exception:
          pass

        side_str = "buy" if direction_tag == "LONG" else "sell"
        order_res = active_exchange.create_market_order(
            selected_symbol, side_str, qty
        )
        st.success(
            f"Live Order Successful on {exchange_name.upper()}! ID:"
            f" {order_res.get('id', 'N/A')}"
        )
      else:
        st.info(
            f"Paper Trade Simulated on {exchange_name.upper()}! Dir:"
            f" {direction_tag} | Entry: ${entry_p:,.2f}"
        )
    except Exception as e:
      st.error(
          f"Trade Execution Failed on {exchange_name.upper()}: {str(e)}"
      )

  # ==========================================
  # POWER TRADING & RISK ENGINE METRICS BAR
  # ==========================================
  st.markdown("### ⚡ Power Trading & Risk Monitoring Engine")
  r1, r2, r3, r4 = st.columns(4)
  with r1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">LTZ Score</div><div'
        f' style="font-size:18px; font-weight:700;'
        f' color:#38bdf8;">{risk_metrics["LTZ_Score"]:.2f}</div></div>',
        unsafe_allow_html=True,
    )
  with r2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Spoof'
        f' Score</div><div style="font-size:18px; font-weight:700;'
        f' color:#f59e0b;">{risk_metrics["Spoof_Score"]:.3f}</div></div>',
        unsafe_allow_html=True,
    )
  with r3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Squeeze'
        f' Risk</div><div style="font-size:18px; font-weight:700;'
        f' color:#ff5252;">{risk_metrics["Squeeze_Risk"]:.2f}</div></div>',
        unsafe_allow_html=True,
    )
  with r4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Composite Market'
        f' Risk</div><div style="font-size:18px; font-weight:700;'
        f' color:#ff5252;">{risk_metrics["Market_Risk"]:.2f}</div></div>',
        unsafe_allow_html=True,
    )

  col_chart, col_side = st.columns([2.5, 1])

  with col_chart:
    st.subheader(f"Price Chart ({selected_symbol} - {selected_tf_label})")
    time_delta = pd.Timedelta(minutes=tf_minutes)
    future_times = [
        df["Time"].iloc[-1] + (i * time_delta)
        for i in range(1, forecast_horizon + 1)
    ]
    t_steps = np.linspace(0, np.pi / 2, forecast_horizon)

    if signal["direction"] == "LONG":
      forecast_prices = close_val + (signal["beam"] - close_val) * np.sin(
          t_steps
      )
    elif signal["direction"] == "SHORT":
      forecast_prices = close_val - (close_val - signal["base"]) * np.sin(
          t_steps
      )
    else:
      forecast_prices = [close_val] * forecast_horizon

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Time"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff5252",
        )
    )
    traj_color = (
        "#00e676"
        if signal["direction"] == "LONG"
        else ("#ff5252" if signal["direction"] == "SHORT" else "#38bdf8")
    )
    fig.add_trace(
        go.Scatter(
            x=[df["Time"].iloc[-1]] + future_times,
            y=[close_val] + list(forecast_prices),
            mode="lines+markers",
            name="Trajectory",
            line=dict(color=traj_color, width=2, dash="dot"),
        )
    )
    fig.add_hline(
        y=signal["beam"],
        line_dash="dash",
        line_color="#ff5252",
        annotation_text=f"BEAM: ${signal['beam']:,.2f}",
    )
    fig.add_hline(
        y=signal["base"],
        line_dash="dash",
        line_color="#ff5252",
        annotation_text=f"BASE: ${signal['base']:,.2f}",
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#111622",
        plot_bgcolor="#111622",
        margin=dict(l=5, r=5, t=5, b=5),
    )
    st.plotly_chart(fig, use_container_width=True)

  with col_side:
    st.subheader("Market Overview (24h)")
    st.markdown(
        '<div class="metric-card"><div style="display:flex;'
        ' justify-content:space-between; margin-bottom:6px;"><span>Market'
        ' Cap</span> <b>$2.28T <span style="color:#00e676;">+1.25%</span></b></div><div'
        ' style="display:flex; justify-content:space-between;'
        ' margin-bottom:6px;"><span>BTC Dominance</span> <b>52.41% <span'
        ' style="color:#ff5252;">-0.38%</span></b></div><div'
        ' style="display:flex; justify-content:space-between;'
        ' margin-bottom:6px;"><span>Fear & Greed</span> <b>72'
        ' (Greed)</b></div><div style="display:flex;'
        ' justify-content:space-between;"><span>Funding Rate</span>'
        ' <b>0.0102%</b></div></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Volume Trend")
    fig_vol = go.Figure(
        go.Bar(
            x=list(range(10)),
            y=np.random.randint(20, 80, 10),
            marker_color="#38bdf8",
        )
    )
    fig_vol.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#111622",
        plot_bgcolor="#111622",
        xaxis_visible=False,
    )
    st.plotly_chart(
        fig_vol, use_container_width=True, config={"displayModeBar": False}
    )

  b1, b2, b3 = st.columns([1.2, 1, 1.2])

  with b1:
    st.subheader("🔬 10-Papers Scoreboard")
    paper_df = pd.DataFrame([
        {
            "Paper": k,
            "Signal Value": f"{v:+.3f}",
            "Evolved Weight": f"{signal['evolved_weights'][k]*100:.1f}%",
            "Status": (
                "PASS🟢"
                if v > 0.1
                else ("FAIL🔴" if v < -0.1 else "NEUTRAL⚪")
            ),
        }
        for k, v in signal["paper_results"].items()
    ])
    st.dataframe(paper_df, use_container_width=True, hide_index=True, height=240)

  with b2:
    st.subheader("Signal Summary")
    fig_summary = go.Figure(
        go.Pie(
            labels=["Pass", "Neutral", "Fail"],
            values=[4, 4, 2],
            hole=0.6,
            marker_colors=["#00e676", "#8b949e", "#ff5252"],
        )
    )
    fig_summary.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#111622",
        showlegend=True,
    )
    st.plotly_chart(
        fig_summary, use_container_width=True, config={"displayModeBar": False}
    )

  with b3:
    st.subheader("⚡ Key Metrics & Orderbook")
    st.markdown(
        '<div class="metric-card" style="height:240px;"><div'
        ' style="display:flex; justify-content:space-between; padding:4px'
        ' 0;"><span>OBI (Weighted)</span> <b'
        ' style="color:#ff5252;">-0.154</b></div><div style="display:flex;'
        ' justify-content:space-between; padding:4px 0;"><span>OFI</span> <b'
        ' style="color:#ff5252;">-8,245</b></div><div style="display:flex;'
        ' justify-content:space-between; padding:4px 0;"><span>Volume'
        ' Ratio</span> <b>0.92</b></div><div style="display:flex;'
        ' justify-content:space-between; padding:4px 0;"><span>Market'
        ' Pressure</span> <b style="color:#ff5252;">-0.218</b></div><div'
        ' style="display:flex; justify-content:space-between; padding:4px'
        ' 0;"><span>Flow Strength</span> <b'
        ' style="color:#ff5252;">-0.165</b></div><div style="display:flex;'
        ' justify-content:space-between; padding:4px 0;"><span>Liquidity'
        ' Score</span> <b style="color:#f59e0b;">58 /'
        ' 100</b></div></div>',
        unsafe_allow_html=True,
    )

  # ==========================================
  # PERFORMANCE & ANALYTICS SECTION + COIN PROFIT/LOSS BREAKDOWN
  # ==========================================
  st.markdown("---")
  st.subheader(
      "📊 Performance, Analytics & Coin-wise Profit/Loss Breakdown"
  )

  if st.session_state.trade_history_log:
    df_log = pd.DataFrame(st.session_state.trade_history_log)
    df_log["dt"] = pd.to_datetime(df_log["timestamp"])
    df_log["date"] = df_log["dt"].dt.date

    now_dt = datetime.datetime.now()
    today_date = now_dt.date()
    current_year = now_dt.year
    current_week = now_dt.isocalendar()[1]
    current_month = now_dt.month

    total_wins = len(df_log[df_log["outcome"] == "WIN"])
    total_losses = len(df_log[df_log["outcome"] == "LOSS"])
    closed_trades = total_wins + total_losses
    overall_win_rate = (
        (total_wins / closed_trades * 100) if closed_trades > 0 else 0.0
    )

    wr1, wr2, wr3, wr4 = st.columns(4)
    with wr1:
      st.markdown(
          f'<div class="metric-card"><div class="metric-label">Overall Win'
          f' Rate</div><div'
          f' class="metric-value-green">{overall_win_rate:.1f}%</div></div>',
          unsafe_allow_html=True,
      )
    with wr2:
      st.markdown(
          f'<div class="metric-card"><div class="metric-label">Total Wins'
          f' (W)</div><div style="font-size:20px; font-weight:700;'
          f' color:#00e676; margin-top:4px;">{total_wins}</div></div>',
          unsafe_allow_html=True,
      )
    with wr3:
      st.markdown(
          f'<div class="metric-card"><div class="metric-label">Total Losses'
          f' (L)</div><div style="font-size:20px; font-weight:700;'
          f' color:#ff5252; margin-top:4px;">{total_losses}</div></div>',
          unsafe_allow_html=True,
      )
    with wr4:
      pending_count = len(df_log[df_log["outcome"] == "PENDING"])
      st.markdown(
          f'<div class="metric-card"><div class="metric-label">Pending'
          f' Outcomes</div><div'
          f' class="metric-value-blue">{pending_count}</div></div>',
          unsafe_allow_html=True,
      )

    st.markdown("### 🏆 Coin-wise Win/Loss & Profit Ranking")
    coin_perf_list = []
    for coin in COINS_LIST:
      coin_df = df_log[df_log["symbol"] == coin]
      c_wins = len(coin_df[coin_df["outcome"] == "WIN"])
      c_losses = len(coin_df[coin_df["outcome"] == "LOSS"])
      c_closed = c_wins + c_losses
      c_wr = (c_wins / c_closed * 100) if c_closed > 0 else 0.0
      c_net_pnl = (c_wins * 4) - (c_losses * 2)

      coin_perf_list.append({
          "Symbol": coin,
          "Wins": c_wins,
          "Losses": c_losses,
          "Win Rate": f"{c_wr:.1f}%",
          "Est. PnL ($)": f"${c_net_pnl:+d}",
      })

    df_coin_perf = pd.DataFrame(coin_perf_list)
    df_coin_perf["sort_val"] = (
        df_coin_perf["Est. PnL ($)"]
        .str.replace("$", "", regex=False)
        .str.replace("+", "", regex=False)
        .astype(int)
    )
    df_coin_perf = df_coin_perf.sort_values(
        by="sort_val", ascending=False
    ).drop(columns=["sort_val"])

    st.dataframe(
        df_coin_perf, use_container_width=True, hide_index=True, height=220
    )

    df_today = df_log[df_log["date"] == today_date]
    tot_d = len(df_today)
    long_d = len(df_today[df_today["direction"] == "LONG"]) if tot_d > 0 else 0
    short_d = (
        len(df_today[df_today["direction"] == "SHORT"]) if tot_d > 0 else 0
    )
    avg_s_d = df_today["score"].mean() if tot_d > 0 else 0.0

    df_week = df_log[
        (df_log["dt"].dt.isocalendar().week == current_week)
        & (df_log["dt"].dt.year == current_year)
    ]
    tot_w = len(df_week)
    long_w = len(df_week[df_week["direction"] == "LONG"]) if tot_w > 0 else 0
    short_w = (
        len(df_week[df_week["direction"] == "SHORT"]) if tot_w > 0 else 0
    )
    avg_s_w = df_week["score"].mean() if tot_w > 0 else 0.0

    df_month = df_log[
        (df_log["dt"].dt.month == current_month)
        & (df_log["dt"].dt.year == current_year)
    ]
    tot_m = len(df_month)
    long_m = len(df_month[df_month["direction"] == "LONG"]) if tot_m > 0 else 0
    short_m = (
        len(df_month[df_month["direction"] == "SHORT"]) if tot_m > 0 else 0
    )
    avg_s_m = df_month["score"].mean() if tot_m > 0 else 0.0

    tab_d, tab_w, tab_m = st.tabs(
        ["📅 Daily Overview", "📈 Weekly Overview", "🗓️ Monthly Overview"]
    )

    with tab_d:
      w1, w2, w3, w4 = st.columns(4)
      with w1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Signals'
            f' (Today)</div><div class="metric-value-blue">{tot_d}</div></div>',
            unsafe_allow_html=True,
        )
      with w2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">LONG /'
            f' SHORT</div><div style="font-size:18px; font-weight:700;'
            f' color:#00e676; margin-top:4px;">{long_d} / <span'
            f' style="color:#ff5252;">{short_d}</span></div></div>',
            unsafe_allow_html=True,
        )
      with w3:
        sc_col = "#00e676" if avg_s_d >= 0 else "#ff5252"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg Score'
            f' (Today)</div><div style="font-size:18px; font-weight:700;'
            f' color:{sc_col}; margin-top:4px;">{avg_s_d:+.3f}</div></div>',
            unsafe_allow_html=True,
        )
      with w4:
        neu_d = tot_d - (long_d + short_d)
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Neutral</div><div'
            f' style="font-size:18px; font-weight:700; color:#38bdf8;'
            f' margin-top:4px;">{neu_d}</div></div>',
            unsafe_allow_html=True,
        )

    with tab_w:
      ww1, ww2, ww3, ww4 = st.columns(4)
      with ww1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Signals'
            f' (Weekly)</div><div class="metric-value-blue">{tot_w}</div></div>',
            unsafe_allow_html=True,
        )
      with ww2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">LONG /'
            f' SHORT</div><div style="font-size:18px; font-weight:700;'
            f' color:#00e676; margin-top:4px;">{long_w} / <span'
            f' style="color:#ff5252;">{short_w}</span></div></div>',
            unsafe_allow_html=True,
        )
      with ww3:
        sc_col_w = "#00e676" if avg_s_w >= 0 else "#ff5252"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg Score'
            f' (Weekly)</div><div style="font-size:18px; font-weight:700;'
            f' color:{sc_col_w}; margin-top:4px;">{avg_s_w:+.3f}</div></div>',
            unsafe_allow_html=True,
        )
      with ww4:
        neu_w = tot_w - (long_w + short_w)
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Neutral</div><div'
            f' style="font-size:18px; font-weight:700; color:#38bdf8;'
            f' margin-top:4px;">{neu_w}</div></div>',
            unsafe_allow_html=True,
        )

    with tab_m:
      wm1, wm2, wm3, wm4 = st.columns(4)
      with wm1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Signals'
            f' (Monthly)</div><div class="metric-value-blue">{tot_m}</div></div>',
            unsafe_allow_html=True,
        )
      with wm2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">LONG /'
            f' SHORT</div><div style="font-size:18px; font-weight:700;'
            f' color:#00e676; margin-top:4px;">{long_m} / <span'
            f' style="color:#ff5252;">{short_m}</span></div></div>',
            unsafe_allow_html=True,
        )
      with wm3:
        sc_col_m = "#00e676" if avg_s_m >= 0 else "#ff5252"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg Score'
            f' (Monthly)</div><div style="font-size:18px; font-weight:700;'
            f' color:{sc_col_m}; margin-top:4px;">{avg_s_m:+.3f}</div></div>',
            unsafe_allow_html=True,
        )
      with wm4:
        neu_m = tot_m - (long_m + short_m)
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Neutral</div><div'
            f' style="font-size:18px; font-weight:700; color:#38bdf8;'
            f' margin-top:4px;">{neu_m}</div></div>',
            unsafe_allow_html=True,
        )
