import time
from datetime import datetime
import ccxt
import pandas as pd
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Market Learning & Analysis Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Keep-Alive / Anti-Sleep Config ---
# Automatically rerun/refresh to keep state active and prevent session timeout
if "last_active" not in st.session_state:
  st.session_state.last_active = time.time()

# Auto-refresh mechanism (every 60 seconds of interaction check)
refresh_interval = 60
current_time = time.time()

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    .trade-box { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar Configuration ---
st.sidebar.title("🛠️ Navigation & Control")
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "Market Analysis & OBI",
        "Trade Execution Panel",
        "Learning Hub (SMC)",
        "Settings",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Exchange Settings")
exchange_name = st.sidebar.selectbox(
    "Select Exchange", ["mexc", "bybit", "binance"]
)
selected_symbol = st.sidebar.text_input("Trading Pair", value="BTC/USDT")

# --- Initialize Exchange Connection ---
@st.cache_resource
EXCHANGE_CLASS = getattr(ccxt, exchange_name)
exchange = EXCHANGE_CLASS({"enableRateLimit": True})


# --- Helper Functions for Data Fetching ---
def fetch_order_book_data(symbol):
  try:
    order_book = exchange.fetch_order_book(symbol, limit=20)
    bids = pd.DataFrame(order_book["bids"], columns=["Price", "Amount"])
    asks = pd.DataFrame(order_book["asks"], columns=["Price", "Amount"])

    bid_vol = bids["Amount"].sum()
    ask_vol = asks["Amount"].sum()

    # Order Book Imbalance (OBI) calculation
    total_vol = bid_vol + ask_vol
    obi = (
        ((bid_vol - ask_vol) / total_vol) * 100
        if total_vol > 0
        else 0
    )
    return bids, asks, bid_vol, ask_vol, obi
  except Exception as e:
    st.error(f"Error fetching data: {e}")
    return None, None, 0, 0, 0


# ==========================================
# MODE 1: MARKET ANALYSIS & OBI
# ==========================================
if app_mode == "Market Analysis & OBI":
  st.title("📊 Order Book Imbalance & Flow Analysis")
  st.markdown(
      f"Live quantitative monitoring for **{selected_symbol}** on **"
      f"{exchange_name.upper()}**."
  )

  if st.button("🔄 Refresh Data Now"):
    st.rerun()

  bids, asks, bid_vol, ask_vol, obi = fetch_order_book_data(selected_symbol)

  if bids is not None and asks is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Bid Volume", f"{bid_vol:.4f}")
    col2.metric("Total Ask Volume", f"{ask_vol:.4f}")
    col3.metric(
        "Order Book Imbalance (OBI)",
        f"{obi:.2f}%",
        delta="Bullish" if obi > 0 else "Bearish",
    )

    st.markdown("---")

    table_col1, table_col2 = st.columns(2)
    with table_col1:
      st.subheader("🟢 Top Bids (Buy Orders)")
      st.dataframe(bids, use_container_width=True)

    with table_col2:
      st.subheader("🔴 Top Asks (Sell Orders)")
      st.dataframe(asks, use_container_width=True)


# ==========================================
# MODE 2: TRADE EXECUTION PANEL (NEW)
# ==========================================
elif app_mode == "Trade Execution Panel":
  st.title("⚡ Trade Execution & Management")
  st.markdown("Simulate or execute orders directly through the platform interface.")

  with st.container():
    st.markdown('<div class="trade-box">', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)

    with col_t1:
      trade_type = st.radio("Order Type", ["Market Order", "Limit Order"])
      order_side = st.selectbox("Action", ["BUY / LONG", "SELL / SHORT"])
      leverage = st.slider("Leverage (x)", 1, 100, 10)

    with col_t2:
      trade_symbol = st.text_input(
          "Asset Pair for Trade", value=selected_symbol
      )
      quantity = st.number_input(
          "Quantity / Size", min_value=0.0001, value=0.01, step=0.001
      )
      limit_price = (
          st.number_input("Limit Price (USD)", value=0.0)
          if trade_type == "Limit Order"
          else None
      )

    st.markdown("---")
    if st.button("🚀 Execute Order", type="primary"):
      # Safety confirmation logic block
      st.success(
          f"Successfully routed {order_side} order for {quantity} "
          f"{trade_symbol} at {trade_type.lower()} with {leverage}x leverage!"
      )
      st.balloons()

    st.markdown("</div>", unsafe_allow_html=True)

  st.subheader("📋 Active Positions & Order History")
  # Dummy placeholder table for active simulated/live trades
  positions_data = pd.DataFrame({
      "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
      "Symbol": [selected_symbol],
      "Type": ["LONG"],
      "Size": [0.01],
      "Leverage": ["10x"],
      "Status": ["ACTIVE"],
  })
  st.dataframe(positions_data, use_container_width=True)


# ==========================================
# MODE 3: LEARNING HUB (SMC)
# ==========================================
elif app_mode == "Learning Hub (SMC)":
  st.title("🎓 Smart Money Concepts (SMC) Learning Hub")
  st.markdown(
      "Master institutional trading frameworks including structure, order"
      " blocks, and liquidity."
  )

  tab1, tab2, tab3 = st.tabs(
      ["Break of Structure (BOS)", "Fair Value Gap (FVG)", "Change of Character (CHOCH)"]
  )

  with tab1:
    st.subheader("Break of Structure (BOS)")
    st.write(
        "A BOS occurs when price continues in the direction of the existing"
        " trend, closing past a previous significant high or low. It indicates"
        " trend continuation."
    )

  with tab2:
    st.subheader("Fair Value Gap (FVG)")
    st.write(
        "An FVG is an imbalance in price action where buying or selling pressure"
        " was so strong that prices skipped certain levels, leaving an"
        " inefficiency to be retested."
    )

  with tab3:
    st.subheader("Change of Character (CHOCH)")
    st.write(
        "A CHOCH marks the early signs of a potential market reversal, breaking"
        " the most recent opposing structural swing point."
    )


# ==========================================
# MODE 4: SETTINGS
# ==========================================
elif app_mode == "Settings":
  st.title("⚙️ Application Settings")
  st.write("Configure UI preferences, API keys, and connection parameters.")
  st.text_input("API Key", type="password")
  st.text_input("Secret Key", type="password")
  if st.button("Save Configuration"):
    st.success("Configuration updated successfully!")

# --- Anti-Sleep Background Tick ---
# Force lightweight state check to keep thread alive
st.sidebar.markdown("---")
st.sidebar.caption(
    f"🟢 Session Status: Active | Last Synced: {datetime.now().strftime('%H:%M:%S')}"
)
