import json
import os
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.feature_pipeline import (
    calculate_orderbook_values,
    normalized_ofi,
    calculate_research_features,
    make_feature_vector,
    FEATURES,
)
from signal_engine import SignalEngine

st.set_page_config(
    page_title="ZIA Quant Research Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)
st_autorefresh(interval=5000, key="zia_refresh")

BINANCE_BASE = "https://api.binance.com"
MODEL_FILE = "xgboost_obi_model.pkl"
SIGNAL_FILE = "signal_history.csv"
FEEDBACK_FILE = "trade_feedback.csv"
CONFIG_FILE = "config.json"
SIGNAL_STATE_FILE = "last_signal_state.json"

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XMRUSDT", "XRPUSDT", "TAOUSDT"]


def binance_get(endpoint, params=None):
    try:
        r = requests.get(BINANCE_BASE + endpoint, params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Binance API error: {exc}")
        return None


def get_price(symbol):
    data = binance_get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"]) if data else None


def get_orderbook(symbol, limit=50):
    data = binance_get("/api/v3/depth", {"symbol": symbol, "limit": limit})
    if not data:
        return None, None
    return np.asarray(data["bids"], dtype=float), np.asarray(data["asks"], dtype=float)


def get_klines(symbol, interval="1m", limit=200):
    data = binance_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not data:
        return pd.DataFrame()
    columns = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_base",
        "taker_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=columns)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_taker_flow(symbol, limit=100):
    data = binance_get("/api/v3/aggTrades", {"symbol": symbol, "limit": limit})
    if not data:
        return 0.0
    buy = sum(float(x["q"]) for x in data if not x["m"])
    sell = sum(float(x["q"]) for x in data if x["m"])
    return float(np.clip((buy - sell) / (buy + sell + 1e-8), -1, 1))


def calculate_vwap(df):
    if df.empty:
        return 0.0
    volume = df["Volume"].sum()
    if volume <= 0:
        return float(df["Close"].iloc[-1])
    return float((df["Close"] * df["Volume"]).sum() / volume)


def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return None


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def append_signal(row):
    pd.DataFrame([row]).to_csv(
        SIGNAL_FILE,
        mode="a",
        header=not os.path.exists(SIGNAL_FILE),
        index=False,
    )


def auto_save_signal(row):
    # Prevent Streamlit's 5-second rerun from creating the same trade repeatedly.
    state = load_json(SIGNAL_STATE_FILE, {})
    fingerprint = "|".join([
        str(row["symbol"]),
        str(row["mode"]),
        str(row["direction"]),
        f"{float(row['entry_price']):.8f}",
        f"{float(row['final_score']):.6f}",
    ])

    if state.get("fingerprint") == fingerprint and time.time() < float(state.get("expires_at", 0)):
        return False

    append_signal(row)
    save_json(
        SIGNAL_STATE_FILE,
        {
            "fingerprint": fingerprint,
            "trade_id": row["trade_id"],
            "expires_at": row["expires_at"],
        },
    )
    return True


st.sidebar.title("⚡ ZIA QUANT ENGINE")
symbol = st.sidebar.selectbox("Coin", COINS)
mode = st.sidebar.selectbox("Trading Mode", ["SCALPING", "INTRADAY"])
leverage = st.sidebar.number_input("MEXC Leverage", 1, 50, 5)
trade_amount = st.sidebar.number_input("Trade Amount USDT", min_value=1.0, value=10.0)
bot_enabled = st.sidebar.toggle("Enable Trading Bot", value=False)
selected_coins = st.sidebar.multiselect("Bot Coins", COINS, default=[symbol])

save_json(CONFIG_FILE, {
    "is_running": bot_enabled,
    "leverage": int(leverage),
    "trade_amount_usdt": float(trade_amount),
    "selected_coins": selected_coins,
})

st.title("ZIA Quantitative Research & Trading Terminal")
st.caption("BINANCE = Market Data / Order Book / Research | MEXC = Trade Execution")

price = get_price(symbol)
bids, asks = get_orderbook(symbol, 50)
df = get_klines(symbol, "1m", 200)

if price is None or bids is None or asks is None or df.empty:
    st.error("Live Binance data unavailable.")
    st.stop()

ob = calculate_orderbook_values(bids, asks)

previous = st.session_state.get("previous_depth")
ofi = normalized_ofi(
    previous[0] if previous else None,
    previous[1] if previous else None,
    ob["top20_bid_sum"],
    ob["top20_ask_sum"],
)
st.session_state["previous_depth"] = (
    ob["top20_bid_sum"],
    ob["top20_ask_sum"],
)

taker_flow = get_taker_flow(symbol)

research = calculate_research_features(
    df["Close"].values,
    df["Volume"].values,
    ob["top20_bid_sum"],
    ob["top20_ask_sum"],
    float(bids[0, 1]),
    float(asks[0, 1]),
    ofi,
    taker_flow,
)

# Research score: all research formulas contribute; volatility is a regime/quality
# feature and is therefore not allowed to create direction by itself.
research_weights = {
    "BOOK_IMB": 0.20,
    "OFI": 0.20,
    "TAKER_FLOW": 0.10,
    "QUANT_IMPLY": 0.08,
    "ADAPT_CONF": 0.10,
    "BAYESIAN": 0.05,
    "FOURIER_TREND": 0.08,
    "EMA_TREND": 0.07,
    "VWAP_DISTANCE": 0.05,
}
weight_total = sum(research_weights.values())
research_score = sum(
    research[name] * weight / weight_total
    for name, weight in research_weights.items()
)
research["RESEARCH_SCORE"] = float(np.clip(research_score, -1, 1))

orderbook_for_ml = ob
ml_vector = make_feature_vector(orderbook_for_ml, research)
model = load_model()

if model is None:
    long_probability = short_probability = 0.5
    ml_available = False
else:
    try:
        probs = model.predict_proba(ml_vector)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        p0 = float(probs[classes.index(0)]) if 0 in classes else 0.5
        p1 = float(probs[classes.index(1)]) if 1 in classes else 0.5
        short_probability, long_probability = p0, p1
        ml_available = True
    except Exception as exc:
        st.warning(f"ML prediction unavailable: {exc}")
        long_probability = short_probability = 0.5
        ml_available = False

ema20 = df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
ema50 = df["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
vwap = calculate_vwap(df)

result = SignalEngine().generate(
    research_score=research["RESEARCH_SCORE"],
    long_probability=long_probability,
    short_probability=short_probability,
    price=price,
    ema20=ema20,
    ema50=ema50,
    vwap=vwap,
    obi=ob["obi_top20"],
    ofi=ofi,
    mode=mode,
    ml_available=ml_available,
)

signal = result["signal"]
final_score = result["score"]

if "LONG" in signal:
    stop_loss = price * 0.996
    take_profit = price * 1.006
elif "SHORT" in signal:
    stop_loss = price * 1.004
    take_profit = price * 0.994
else:
    stop_loss = take_profit = 0.0

st.subheader(f"{symbol} — Binance Live")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"{price:,.4f}")
c2.metric("OBI Top 20", f"{ob['obi_top20']:.4f}")
c3.metric("OBI Top 50", f"{ob['obi_top50']:.4f}")
c4.metric("OFI", f"{ofi:.4f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Research Score", f"{research['RESEARCH_SCORE']:.4f}")
c2.metric("ML Long", f"{long_probability*100:.2f}%")
c3.metric("ML Short", f"{short_probability*100:.2f}%")
c4.metric("Final Score", f"{final_score:.4f}")

st.markdown(f"## FINAL SIGNAL: **{signal}**")
st.write(
    f"Research={research['RESEARCH_SCORE']:.4f} | "
    f"ML Long={long_probability*100:.1f}% | "
    f"ML Short={short_probability*100:.1f}% | "
    f"Final={final_score:.4f}"
)

if signal != "NO TRADE":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry", f"{price:,.4f}")
    c2.metric("Stop Loss", f"{stop_loss:,.4f}")
    c3.metric("Take Profit", f"{take_profit:,.4f}")
    c4.metric("Validity", "30 MIN" if mode == "SCALPING" else "8 HOURS")

    signal_row = {
        "trade_id": str(uuid.uuid4()),
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": symbol,
        "mode": mode,
        "direction": signal,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "expires_at": time.time() + result["validity_seconds"],
        "validity_seconds": result["validity_seconds"],
        "research_score": research["RESEARCH_SCORE"],
        "ml_long_probability": long_probability,
        "ml_short_probability": short_probability,
        "final_score": final_score,
        "features_json": json.dumps({
            **{k: float(v) for k, v in ob.items() if k in FEATURES},
            **{k: float(v) for k, v in research.items() if k in FEATURES},
        }),
    }

    if auto_save_signal(signal_row):
        st.success(f"Signal automatically saved: {signal}")

st.subheader("Research Lab Features")
cols = st.columns(5)
for i, name in enumerate([
    "BOOK_IMB", "OFI", "TAKER_FLOW", "QUANT_IMPLY", "ADAPT_CONF",
    "BAYESIAN", "FOURIER_TREND", "EMA_TREND", "VWAP_DISTANCE", "VOLATILITY",
]):
    cols[i % 5].metric(name, f"{research[name]:.4f}")

st.subheader("Completed Trade Learning")
if os.path.exists(FEEDBACK_FILE):
    feedback = pd.read_csv(FEEDBACK_FILE)
    closed = feedback[feedback.get("status", "") == "CLOSED"]
else:
    feedback = pd.DataFrame()
    closed = feedback

wins = int((closed.get("outcome", pd.Series(dtype=str)) == "WIN").sum())
losses = int((closed.get("outcome", pd.Series(dtype=str)) == "LOSS").sum())
count = len(closed)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Closed Trades", count)
c2.metric("Wins", wins)
c3.metric("Losses", losses)
c4.metric("Next Retrain", 20 - (count % 20) if count % 20 else 20)

if not feedback.empty:
    st.dataframe(feedback.tail(20).iloc[::-1], use_container_width=True)

st.subheader("Active MEXC Trade")
active = load_json("active_trade.json", None)
if active:
    st.json(active)
else:
    st.info("No active trade.")

st.subheader("Machine Learning Status")
if model is None:
    st.warning("XGBoost model not found. Run: python train_model.py")
else:
    st.success("XGBoost model loaded.")
    metadata = load_json("model_metadata.json", {})
    if metadata:
        st.json(metadata)

st.subheader("System Flow")
st.code(
"""BINANCE LIVE ORDER BOOK + TRADES
        ↓
OBI20/50 + OFI + Taker Flow
        ↓
Research Lab formulas
        ↓
Weighted Research Score
        ↓
XGBoost (same 18 features)
        ↓
EMA + VWAP confirmation
        ↓
FINAL MULTI-FACTOR SIGNAL
        ↓
AUTO SAVE signal_history.csv
        ↓
MEXC execution
        ↓
TP / SL / 30m or 8h expiry
        ↓
WIN / LOSS + full feature snapshot
        ↓
20 CLOSED TRADES
        ↓
XGBoost candidate retraining
        ↓
candidate vs old model
        ↓
better → accept | worse → keep old"""
)
