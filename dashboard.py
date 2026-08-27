from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================================
# ZIA RESEARCH LAB — LIVE ML MARKET TERMINAL
# Dashboard-only upgrade. Existing model/research files untouched.
# The trained XGBoost 7-feature interface is preserved exactly.
# ============================================================

st.set_page_config(
    page_title="ZIA Research Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "xgboost_obi_model.pkl"
HISTORY_FILE = ROOT / "signal_history.csv"
REQUEST_TIMEOUT = 8

# Binance USD-M Futures has several public hostnames. Cloud deployments
# can fail against one hostname/region, so the dashboard rotates them.
FUTURES_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]

# Public Spot market-data fallbacks. These are used only when Futures
# market data cannot be reached from the Streamlit runtime.
SPOT_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://data-api.binance.vision",
]

COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "TRXUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT", "XLMUSDT",
    "NEARUSDT", "UNIUSDT", "APTUSDT", "TAOUSDT", "XMRUSDT",
]

MODES = {
    "SCALPING": {"label": "5M / SCALP", "tf": "5m", "hold": "5–30 min", "refs": ["15m", "1h", "4h"]},
    "15M": {"label": "15M", "tf": "15m", "hold": "30–120 min", "refs": ["1h", "4h"]},
    "1H": {"label": "1H", "tf": "1h", "hold": "2–24 hours", "refs": ["4h", "1d"]},
    "4H": {"label": "4H", "tf": "4h", "hold": "12–72 hours", "refs": ["1d", "1w"]},
}

# DO NOT change this order: it matches the existing trained model.
MODEL_FEATURES = [
    "top20_bid_sum",
    "top20_ask_sum",
    "obi_top20",
    "spread",
    "bid_ask_ratio",
    "total_depth",
    "trend_signal",
]

for k, v in {
    "symbol": "BTCUSDT",
    "mode": "SCALPING",
    "auto_refresh": True,
    "refresh_seconds": 5,
    "last_saved_key": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# UI
# ============================================================

st.markdown(
    """
<style>
.block-container{max-width:1750px;padding:1rem 1.6rem 2rem}
[data-testid="stSidebar"]{background:#080c13;border-right:1px solid #1b2636}
[data-testid="stSidebar"] *{color:#e8eef7}
.stApp{background:radial-gradient(circle at 78% -10%,#18263a 0%,#080c12 44%)}
.hero{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:16px}
.brand{font-size:30px;font-weight:950;letter-spacing:.2px}.brand span{color:#7889ff}
.subtitle{color:#8491a4;font-size:12px;margin-top:4px}
.status{display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border:1px solid #284334;border-radius:999px;background:#0b1712;color:#8ee0b1;font-size:11px;font-weight:800}
.status.warn{border-color:#59451f;background:#181208;color:#e9c87e}.dot{width:7px;height:7px;border-radius:50%;background:#49d18a;box-shadow:0 0 12px #49d18a}
.dot.warn{background:#e8bd58;box-shadow:0 0 12px #e8bd58}
.signal{border-radius:18px;padding:22px 24px;background:linear-gradient(135deg,#111b2a,#0a1018);border:1px solid #29384d;margin-bottom:14px}
.signal.long{border-color:#23865f;box-shadow:0 0 35px rgba(35,170,111,.08)}
.signal.short{border-color:#a04052;box-shadow:0 0 35px rgba(220,67,91,.08)}
.signal.wait{border-color:#344256}
.signal-label{color:#8996a8;font-size:10px;font-weight:850;letter-spacing:1.2px}.signal-name{font-size:42px;line-height:1.05;font-weight:950;margin:5px 0}.signal-meta{color:#9aa7b8;font-size:12px}.big-price{font-size:29px;font-weight:900}
.kpi{background:#0d141e;border:1px solid #1d2939;border-radius:13px;padding:14px 15px;min-height:86px}.kpi-label{color:#7f8c9f;font-size:10px;text-transform:uppercase;letter-spacing:1px;font-weight:850}.kpi-value{color:#f2f5fa;font-size:21px;font-weight:900;margin-top:5px}.kpi-sub{color:#7d899b;font-size:10px;margin-top:2px}
.trade{background:#0c131d;border:1px solid #1d2938;border-radius:12px;padding:13px;text-align:center}.trade-label{color:#7e8b9c;font-size:10px;font-weight:850;letter-spacing:.8px}.trade-value{font-size:18px;font-weight:900;margin-top:4px}.small{color:#7d899b;font-size:10px}
.section{font-size:16px;font-weight:900;margin:20px 0 10px}.box{background:linear-gradient(145deg,#101927,#0b1119);border:1px solid #29374a;border-radius:15px;padding:17px}.panel-title{color:#8996a8;font-size:10px;font-weight:850;letter-spacing:1.2px;text-transform:uppercase}
.good{color:#61d69a!important}.bad{color:#f27e8e!important}.neutral{color:#aab5c4!important}
.progress{height:8px;border-radius:99px;background:#1b2635;overflow:hidden;margin-top:7px}.progress>div{height:100%;border-radius:99px;background:#7687ff}
div[data-testid="stMetric"]{background:#0d141e;border:1px solid #1d2939;padding:11px;border-radius:12px}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================

def f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def clamp(v: Any, low: float = -1.0, high: float = 1.0) -> float:
    return float(np.clip(f(v), low, high))


def price(v: Any) -> str:
    x = f(v)
    if x <= 0:
        return "—"
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:.6f}"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def direction_from_prediction(p: Any) -> str:
    try:
        return "LONG" if int(p) == 1 else "SHORT"
    except Exception:
        return "UNKNOWN"


def signal_class(direction: str) -> str:
    d = str(direction).upper()
    if "LONG" in d:
        return "long"
    if "SHORT" in d:
        return "short"
    return "wait"


def bias(v: Any) -> tuple[str, str]:
    x = f(v)
    if x > .25:
        return "BULLISH", "good"
    if x < -.25:
        return "BEARISH", "bad"
    return "NEUTRAL", "neutral"

# ============================================================
# Robust Binance data layer
# ============================================================

@st.cache_data(ttl=3, show_spinner=False)
def api_request(kind: str, path: str, params: tuple[tuple[str, Any], ...] = ()):
    bases = FUTURES_BASES if kind == "futures" else SPOT_BASES
    last_error = ""
    for base in bases:
        try:
            r = requests.get(
                base + path,
                params=dict(params),
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "ZIA-RESEARCH-LAB/2.0",
                    "Accept": "application/json",
                },
            )
            if r.status_code == 200:
                return r.json(), base, "OK"
            last_error = f"HTTP {r.status_code} from {base}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
    return None, "", last_error or "No response"


@st.cache_data(ttl=7, show_spinner=False)
def get_klines(symbol: str, interval: str, limit: int = 260):
    p = (("symbol", symbol), ("interval", interval), ("limit", limit))
    raw, base, err = api_request("futures", "/fapi/v1/klines", p)
    source = "BINANCE FUTURES"
    if not isinstance(raw, list):
        raw, base, err2 = api_request("spot", "/api/v3/klines", p)
        err = f"Futures unavailable; {err2}"
        source = "BINANCE SPOT FALLBACK"
    if not isinstance(raw, list):
        return pd.DataFrame(), source, err
    rows = []
    for c in raw:
        try:
            rows.append({
                "Time": pd.to_datetime(int(c[0]), unit="ms", utc=True),
                "Open": float(c[1]), "High": float(c[2]), "Low": float(c[3]),
                "Close": float(c[4]), "Volume": float(c[5]),
                "Trades": int(c[8]), "TakerBuy": float(c[9]),
            })
        except Exception:
            continue
    return pd.DataFrame(rows).dropna().reset_index(drop=True) if rows else pd.DataFrame(), source, err


@st.cache_data(ttl=2, show_spinner=False)
def get_orderbook(symbol: str, limit: int = 100):
    p = (("symbol", symbol), ("limit", limit))
    raw, base, err = api_request("futures", "/fapi/v1/depth", p)
    source = "BINANCE FUTURES"
    if not isinstance(raw, dict):
        raw, base, err2 = api_request("spot", "/api/v3/depth", p)
        err = f"Futures unavailable; {err2}"
        source = "BINANCE SPOT FALLBACK"
    try:
        bids = np.asarray(raw.get("bids", []), dtype=float) if isinstance(raw, dict) else np.empty((0, 2))
        asks = np.asarray(raw.get("asks", []), dtype=float) if isinstance(raw, dict) else np.empty((0, 2))
        if bids.ndim != 2 or bids.shape[1] < 2: bids = np.empty((0, 2))
        if asks.ndim != 2 or asks.shape[1] < 2: asks = np.empty((0, 2))
        return bids, asks, source, err
    except Exception as e:
        return np.empty((0, 2)), np.empty((0, 2)), source, str(e)


@st.cache_data(ttl=4, show_spinner=False)
def get_ticker(symbol: str):
    p = (("symbol", symbol),)
    raw, base, err = api_request("futures", "/fapi/v1/ticker/24hr", p)
    source = "BINANCE FUTURES"
    if not isinstance(raw, dict):
        raw, base, err2 = api_request("spot", "/api/v3/ticker/24hr", p)
        source = "BINANCE SPOT FALLBACK"
        err = f"Futures unavailable; {err2}"
    return raw if isinstance(raw, dict) else {}, source, err


@st.cache_data(ttl=3, show_spinner=False)
def get_trades(symbol: str, limit: int = 1000):
    p = (("symbol", symbol), ("limit", limit))
    raw, base, err = api_request("futures", "/fapi/v1/aggTrades", p)
    source = "BINANCE FUTURES"
    if not isinstance(raw, list):
        raw, base, err2 = api_request("spot", "/api/v3/aggTrades", p)
        source = "BINANCE SPOT FALLBACK"
        err = f"Futures unavailable; {err2}"
    return raw if isinstance(raw, list) else [], source, err

# ============================================================
# Research / signal math
# ============================================================

def obi(bids: np.ndarray, asks: np.ndarray, levels: int) -> float:
    n = min(len(bids), len(asks), levels)
    if n <= 0:
        return 0.0
    bv = max(0.0, float(bids[:n, 1].sum()))
    av = max(0.0, float(asks[:n, 1].sum()))
    return clamp((bv - av) / (bv + av)) if bv + av else 0.0


def weighted_obi(bids: np.ndarray, asks: np.ndarray, levels: int = 20) -> float:
    n = min(len(bids), len(asks), levels)
    if n <= 0:
        return 0.0
    w = 1.0 / (np.arange(n) + 1.0)
    bv = float((bids[:n, 1] * w).sum())
    av = float((asks[:n, 1] * w).sum())
    return clamp((bv - av) / (bv + av)) if bv + av else 0.0


def depth(bids: np.ndarray, asks: np.ndarray, levels: int):
    n = min(len(bids), len(asks), levels)
    return (float(bids[:n, 1].sum()), float(asks[:n, 1].sum())) if n else (0.0, 0.0)


def taker_flow(trades: list[dict[str, Any]]):
    buy = sell = 0.0
    count = 0
    for t in trades:
        try:
            q = float(t["q"])
            if bool(t["m"]):
                sell += q
            else:
                buy += q
            count += 1
        except Exception:
            continue
    total = buy + sell
    return {
        "buy": buy,
        "sell": sell,
        "flow": buy - sell,
        "ratio": clamp((buy - sell) / total) if total else 0.0,
        "count": count,
    }


def technical(df: pd.DataFrame):
    if df.empty:
        return {}
    c = df["Close"]
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    m5 = f(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else 0.0
    m20 = f(c.iloc[-1] / c.iloc[-21] - 1) if len(c) >= 21 else 0.0
    p = f(c.iloc[-1])
    trend = (
        (.30 if p > f(e20.iloc[-1]) else -.30)
        + (.25 if p > f(e50.iloc[-1]) else -.25)
        + (.20 if p > f(e200.iloc[-1]) else -.20)
        + clamp(m20 * 20, -.25, .25)
    )
    return {
        "price": p, "ema20": f(e20.iloc[-1]), "ema50": f(e50.iloc[-1]),
        "ema200": f(e200.iloc[-1]), "momentum5": m5, "momentum20": m20,
        "volatility": f(c.pct_change().rolling(20).std().iloc[-1]),
        "trend": clamp(trend),
    }


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < 2:
        return 0.0
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return max(0.0, f(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]))


@st.cache_data(ttl=15, show_spinner=False)
def get_htf_bias(symbol: str):
    out = {}
    source = "BINANCE FUTURES"
    for tf in ["1h", "4h", "1d"]:
        d, src, _ = get_klines(symbol, tf, 120)
        if "SPOT" in src:
            source = "BINANCE SPOT FALLBACK"
        if d.empty:
            out[tf] = 0.0
            continue
        c = d["Close"]
        e20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        p = f(c.iloc[-1])
        out[tf] = (.5 if p > f(e20) else -.5) + (.5 if p > f(e50) else -.5)
    return out, source

# ============================================================
# ML connection — exact existing 7-feature model
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_FILE.exists():
        return None, "Model file not found"
    try:
        return joblib.load(MODEL_FILE), "Loaded"
    except Exception as e:
        return None, f"Load error: {type(e).__name__}"


def predict_ml(model: Any, features: list[float]):
    if model is None:
        return None
    try:
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and int(expected) != len(features):
            return {"error": f"Model expects {expected} features; dashboard supplied {len(features)}"}
        X = np.asarray([features], dtype=float)
        pred = int(model.predict(X)[0])
        out = {"prediction": pred, "direction": direction_from_prediction(pred)}
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(X)[0], dtype=float)
            out["probabilities"] = probs.tolist()
            out["confidence"] = float(np.max(probs))
            out["classes"] = np.asarray(getattr(model, "classes_", range(len(probs)))).tolist()
        else:
            out["confidence"] = .50
        return out
    except Exception as e:
        return {"error": f"Prediction error: {type(e).__name__}: {e}"}


def build_signal(df, bids, asks, symbol, mode_key):
    tech = technical(df)
    if not tech or len(bids) < 20 or len(asks) < 20:
        return None

    trades, trade_source, _ = get_trades(symbol)
    flow = taker_flow(trades)
    o5, o10, o20, o50 = [obi(bids, asks, n) for n in (5, 10, 20, 50)]
    wobi = weighted_obi(bids, asks, 20)
    multi = clamp(o5 * .15 + o10 * .20 + o20 * .35 + o50 * .30)
    bid20, ask20 = depth(bids, asks, 20)
    bid50, ask50 = depth(bids, asks, 50)
    spread = f(asks[0, 0] - bids[0, 0])
    ratio = bid20 / ask20 if ask20 > 0 else 0.0

    # EXACT order used by the existing trained XGBoost model.
    ml_features = [bid20, ask20, o20, spread, ratio, bid20 + ask20, tech["trend"]]
    model, model_status = load_model()
    ml = predict_ml(model, ml_features)

    score = (
        multi * .30
        + flow["ratio"] * .25
        + tech["trend"] * .25
        + clamp(tech["momentum5"] * 30) * .10
        + clamp(tech["momentum20"] * 15) * .10
    )

    htf, htf_source = get_htf_bias(symbol)
    hscore = htf.get("1h", 0) * .45 + htf.get("4h", 0) * .35 + htf.get("1d", 0) * .20
    score = clamp(score + hscore * .20)

    ml_conf = f(ml.get("confidence", .50), .50) if ml and "error" not in ml else .50
    if ml and "error" not in ml:
        ml_vote = 1.0 if ml["prediction"] == 1 else -1.0
        score = clamp(score + ml_vote * min(.25, ml_conf * .25))

    conf = abs(score) * 55 + abs(multi) * 20 + abs(flow["ratio"]) * 15 + abs(hscore) * 10
    if ml and "error" not in ml:
        conf = conf * .75 + ml_conf * 100 * .25
    confidence = float(np.clip(conf, 0, 99))

    if score >= .70 and confidence >= 70:
        direction = "STRONG LONG"
    elif score >= .42 and confidence >= 55:
        direction = "LONG"
    elif score <= -.70 and confidence >= 70:
        direction = "STRONG SHORT"
    elif score <= -.42 and confidence >= 55:
        direction = "SHORT"
    else:
        direction = "WAIT"

    a = calc_atr(df) or tech["price"] * .005
    sd = min(max(a * 1.15, tech["price"] * .0025), tech["price"] * .006)
    entry = tech["price"]
    if "LONG" in direction:
        sl, tp1, tp2 = entry - sd, entry + sd * 2, entry + sd * 3
    elif "SHORT" in direction:
        sl, tp1, tp2 = entry + sd, entry - sd * 2, entry - sd * 3
    else:
        sl = tp1 = tp2 = entry

    tk, ticker_source, _ = get_ticker(symbol)
    sources = [trade_source, htf_source, ticker_source]
    source = "BINANCE SPOT FALLBACK" if any("SPOT" in s for s in sources) else "BINANCE FUTURES"

    return {
        "timestamp": now_utc().isoformat(), "symbol": symbol, "mode": mode_key,
        "direction": direction, "score": score, "confidence": confidence,
        "price": entry, "entry": entry, "stop_loss": sl, "target1": tp1, "target2": tp2,
        "atr": a, "obi5": o5, "obi10": o10, "obi20": o20, "obi50": o50,
        "weighted_obi": wobi, "multi_obi": multi, "bid20": bid20, "ask20": ask20,
        "bid50": bid50, "ask50": ask50, "spread": spread,
        "taker_buy": flow["buy"], "taker_sell": flow["sell"], "taker_flow": flow["flow"],
        "taker_flow_ratio": flow["ratio"], "trade_count": flow["count"],
        "trend": tech["trend"], "momentum5": tech["momentum5"], "momentum20": tech["momentum20"],
        "ema20": tech["ema20"], "ema50": tech["ema50"], "ema200": tech["ema200"],
        "volatility": tech["volatility"], "htf_1h": htf.get("1h", 0), "htf_4h": htf.get("4h", 0),
        "htf_1d": htf.get("1d", 0), "ml": ml,
        "ml_available": ml is not None and "error" not in ml,
        "ml_confidence": ml_conf, "ml_features": ml_features,
        "model_status": model_status, "change24": f(tk.get("priceChangePercent")),
        "volume24": f(tk.get("quoteVolume")), "data_source": source,
    }

# ============================================================
# Persistence
# ============================================================

def save_signal(s):
    if s["direction"] == "WAIT":
        return
    row = {k: s.get(k) for k in [
        "timestamp", "symbol", "mode", "direction", "score", "confidence",
        "entry", "stop_loss", "target1", "target2", "obi20", "obi50", "taker_flow_ratio",
    ]}
    try:
        pd.DataFrame([row]).to_csv(HISTORY_FILE, mode="a", header=not HISTORY_FILE.exists(), index=False)
    except Exception:
        pass


def history_df():
    try:
        return pd.read_csv(HISTORY_FILE).tail(300) if HISTORY_FILE.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ============================================================
# Charts
# ============================================================

def price_chart(df, s):
    d = df.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d["Time"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"))
    for span, name in [(20, "EMA 20"), (50, "EMA 50"), (200, "EMA 200")]:
        fig.add_trace(go.Scatter(x=d["Time"], y=d["Close"].ewm(span=span, adjust=False).mean(), name=name, line=dict(width=1.25)))
    fig.add_hline(y=s["entry"], annotation_text="ENTRY")
    if s["direction"] != "WAIT":
        fig.add_hline(y=s["stop_loss"], annotation_text="SL", line_dash="dot")
        fig.add_hline(y=s["target1"], annotation_text="TP1 1:2", line_dash="dash")
        fig.add_hline(y=s["target2"], annotation_text="TP2 1:3", line_dash="dash")
    fig.update_layout(template="plotly_dark", height=540, xaxis_rangeslider_visible=False, margin=dict(l=5,r=5,t=25,b=5), legend=dict(orientation="h", y=1.02), hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def imbalance_chart(s):
    vals = [s["obi5"], s["obi10"], s["obi20"], s["obi50"]]
    fig = go.Figure(go.Bar(x=["Top 5", "Top 10", "Top 20", "Top 50"], y=vals, text=[f"{v:+.3f}" for v in vals], textposition="outside"))
    fig.add_hline(y=0)
    fig.update_layout(template="plotly_dark", height=310, margin=dict(l=5,r=5,t=20,b=5), yaxis=dict(range=[-1,1]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### ⚡ ZIA RESEARCH")
    st.caption("ML-powered Binance market research terminal")
    st.divider()
    st.session_state.symbol = st.selectbox("MARKET", COINS, index=COINS.index(st.session_state.symbol) if st.session_state.symbol in COINS else 0)
    keys = list(MODES)
    st.session_state.mode = st.selectbox("ANALYSIS MODE", keys, index=keys.index(st.session_state.mode), format_func=lambda x: MODES[x]["label"])
    st.divider()
    st.markdown("**ENGINE**")
    st.session_state.auto_refresh = st.toggle("Live refresh", value=st.session_state.auto_refresh)
    if st.session_state.auto_refresh:
        st.session_state.refresh_seconds = st.slider("Refresh interval", 3, 30, int(st.session_state.refresh_seconds), 1)
    else:
        st.session_state.refresh_seconds = 30
    if st.button("↻ Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    m = MODES[st.session_state.mode]
    st.info(f"**{m['tf'].upper()}** analysis\n\nHolding: {m['hold']}\n\nHTF: {', '.join(m['refs']).upper()}")
    st.caption("Research / signal generation only. No order execution is performed by this dashboard.")

if st.session_state.auto_refresh:
    st_autorefresh(interval=int(st.session_state.refresh_seconds * 1000), key="zia_live_refresh")

# ============================================================
# Load market
# ============================================================

symbol = st.session_state.symbol
mode_key = st.session_state.mode
mode = MODES[mode_key]

df, candle_source, candle_error = get_klines(symbol, mode["tf"], 260)
bids, asks, book_source, book_error = get_orderbook(symbol, 100)

if df.empty or len(df) < 30:
    st.error(f"Market data unavailable for {symbol}.")
    st.warning("The Streamlit server could not reach Binance market-data endpoints. The dashboard now tries Futures mirrors and Spot public-data fallbacks automatically.")
    with st.expander("Connection diagnostics"):
        st.code(f"Candle source: {candle_source}\nOrder book source: {book_source}\nCandle error: {candle_error}\nOrder book error: {book_error}")
    st.stop()

if len(bids) < 20 or len(asks) < 20:
    st.error("Order-book depth is unavailable right now.")
    with st.expander("Connection diagnostics"):
        st.code(f"Order book source: {book_source}\n{book_error}")
    st.stop()

s = build_signal(df, bids, asks, symbol, mode_key)
if s is None:
    st.error("Signal engine could not calculate the current market state.")
    st.stop()

if st.session_state.last_saved_key != f"{symbol}:{mode_key}:{now_utc().strftime('%Y-%m-%d-%H-%M')}" :
    save_signal(s)
    st.session_state.last_saved_key = f"{symbol}:{mode_key}:{now_utc().strftime('%Y-%m-%d-%H-%M')}"

# ============================================================
# Header
# ============================================================

is_fallback = "SPOT" in s["data_source"] or "SPOT" in candle_source or "SPOT" in book_source
status_text = "LIVE • BINANCE FUTURES" if not is_fallback else "DEGRADED • SPOT FALLBACK"
status_cls = "" if not is_fallback else "warn"
dot_cls = "" if not is_fallback else "warn"

st.markdown(
    f"""
<div class="hero">
  <div><div class="brand"><span>⚡</span> ZIA RESEARCH LAB</div>
  <div class="subtitle">USDⓈ-M Futures research · Order Flow · OBI · Quant Trend · XGBoost ML</div></div>
  <div class="status {status_cls}"><span class="dot {dot_cls}"></span>{status_text} · {now_utc().strftime('%H:%M:%S UTC')}</div>
</div>
""",
    unsafe_allow_html=True,
)

sc = signal_class(s["direction"])
ml_label = "CONNECTED" if s["ml_available"] else "OFFLINE"
ml_cls = "good" if s["ml_available"] else "bad"

st.markdown(
    f"""
<div class="signal {sc}">
  <div style="display:flex;justify-content:space-between;gap:20px;align-items:center;">
    <div>
      <div class="signal-label">FINAL RESEARCH SIGNAL · {symbol} · {mode['label']}</div>
      <div class="signal-name">{s['direction']}</div>
      <div class="signal-meta">Quant score <b>{s['score']:+.3f}</b> · Confidence <b>{s['confidence']:.1f}%</b> · ML <b class="{ml_cls}">{ml_label}</b></div>
    </div>
    <div style="text-align:right"><div class="signal-label">LAST PRICE</div><div class="big-price">${price(s['price'])}</div><div class="signal-meta">24H {s['change24']:+.2f}%</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

cols = st.columns(6)
kpis = [
    ("Confidence", f"{s['confidence']:.1f}%", "final signal"),
    ("OBI 20", f"{s['obi20']:+.4f}", "order book"),
    ("OBI 50", f"{s['obi50']:+.4f}", "deep liquidity"),
    ("Taker Flow", f"{s['taker_flow_ratio']:+.4f}", "aggressive flow"),
    ("Trend", f"{s['trend']:+.3f}", "quant trend"),
    ("XGBoost", f"{s['ml_confidence']*100:.1f}%" if s['ml_available'] else "OFF", "model probability"),
]
for c, (lab, val, sub) in zip(cols, kpis):
    with c:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>{lab}</div><div class='kpi-value'>{val}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

if is_fallback:
    st.info("Binance Futures was not reachable from the cloud runtime, so public Spot market data is being used as a temporary fallback. The ML model remains connected, but its inputs are marked as fallback market data.")

# ============================================================
# Tabs
# ============================================================

tab_overview, tab_flow, tab_ml, tab_history = st.tabs(["▣ Overview", "◈ Order Flow", "◎ ML Engine", "▤ Signal History"])

with tab_overview:
    st.markdown('<div class="section">TRADE PLAN</div>', unsafe_allow_html=True)
    pc = st.columns(4)
    plans = [
        ("ENTRY", s["entry"], "market reference"),
        ("STOP LOSS", s["stop_loss"], "volatility adjusted"),
        ("TARGET 1", s["target1"], "1 : 2 risk / reward"),
        ("TARGET 2", s["target2"], "1 : 3 risk / reward"),
    ]
    for c, (lab, val, sub) in zip(pc, plans):
        with c:
            display = price(val) if s["direction"] != "WAIT" or lab == "ENTRY" else "WAIT"
            st.markdown(f"<div class='trade'><div class='trade-label'>{lab}</div><div class='trade-value'>{display}</div><div class='small'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section">PRICE ACTION</div>', unsafe_allow_html=True)
    st.plotly_chart(price_chart(df, s), use_container_width=True, config={"displaylogo": False, "responsive": True})

    a, b, c = st.columns(3)
    with a:
        st.markdown("**Higher-timeframe bias**")
        for tf in ["1h", "4h", "1d"]:
            label, cls = bias(s[f"htf_{tf}"])
            st.markdown(f"`{tf.upper()}` &nbsp; <span class='{cls}'><b>{label}</b></span> &nbsp; {s[f'htf_{tf}']:+.2f}", unsafe_allow_html=True)
    with b:
        st.markdown("**Momentum & volatility**")
        st.metric("5-candle momentum", f"{s['momentum5']*100:+.2f}%")
        st.metric("20-candle momentum", f"{s['momentum20']*100:+.2f}%")
        st.metric("ATR", price(s["atr"]))
    with c:
        st.markdown("**Moving averages**")
        st.metric("EMA 20", price(s["ema20"]))
        st.metric("EMA 50", price(s["ema50"]))
        st.metric("EMA 200", price(s["ema200"]))

with tab_flow:
    st.markdown('<div class="section">ORDER BOOK IMBALANCE</div>', unsafe_allow_html=True)
    x, y = st.columns([1.25, 1])
    with x:
        st.plotly_chart(imbalance_chart(s), use_container_width=True, config={"displaylogo": False})
    with y:
        st.metric("Top 20 bid volume", f"{s['bid20']:,.3f}")
        st.metric("Top 20 ask volume", f"{s['ask20']:,.3f}")
        st.metric("Top 50 bid volume", f"{s['bid50']:,.3f}")
        st.metric("Top 50 ask volume", f"{s['ask50']:,.3f}")

    st.markdown('<div class="section">TAKER / AGGRESSIVE FLOW</div>', unsafe_allow_html=True)
    fc = st.columns(4)
    for c, lab, val in zip(fc, ["Taker Buy", "Taker Sell", "Flow Ratio", "Trades"], [f"{s['taker_buy']:,.3f}", f"{s['taker_sell']:,.3f}", f"{s['taker_flow_ratio']:+.4f}", f"{s['trade_count']:,}"]):
        with c: st.metric(lab, val)

    st.markdown('<div class="section">LIVE ORDER BOOK · TOP 20</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        bd = pd.DataFrame(bids[:20], columns=["Price", "Quantity"])
        bd["Price"] = bd["Price"].map(price)
        st.dataframe(bd, use_container_width=True, hide_index=True, height=470)
    with right:
        ad = pd.DataFrame(asks[:20], columns=["Price", "Quantity"])
        ad["Price"] = ad["Price"].map(price)
        st.dataframe(ad, use_container_width=True, hide_index=True, height=470)

with tab_ml:
    st.markdown('<div class="section">XGBOOST DECISION CENTER</div>', unsafe_allow_html=True)
    ml = s["ml"]
    left, right = st.columns([1.0, 1.45])
    with left:
        st.markdown(f"<div class='box'><div class='panel-title'>MODEL STATUS</div><div style='font-size:24px;font-weight:950'>XGBoost <span class='{'good' if s['ml_available'] else 'bad'}'>{'CONNECTED' if s['ml_available'] else 'OFFLINE'}</span></div><div class='small' style='margin-top:8px'>{s['model_status']}</div></div>", unsafe_allow_html=True)
        if ml and "error" in ml:
            st.error(ml["error"])
        elif ml:
            st.metric("ML direction", ml["direction"])
            st.metric("ML confidence", f"{s['ml_confidence']*100:.2f}%")
            probs = ml.get("probabilities", [])
            classes = ml.get("classes", list(range(len(probs))))
            for cl, p in zip(classes, probs):
                st.markdown(f"**{direction_from_prediction(cl)}** · {p*100:.2f}%")
                st.markdown(f"<div class='progress'><div style='width:{max(0,min(100,p*100)):.2f}%'></div></div>", unsafe_allow_html=True)
    with right:
        st.markdown("**Exact 7 features sent to the trained model**")
        rows = [{"Feature": n, "Live value": v} for n, v in zip(MODEL_FEATURES, s["ml_features"])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.success("Feature order is locked to the existing trained XGBoost schema.")
        st.caption(f"Input source: {s['data_source']}")

with tab_history:
    st.markdown('<div class="section">RECENT RESEARCH SIGNALS</div>', unsafe_allow_html=True)
    h = history_df()
    if h.empty:
        st.info("No non-WAIT signals have been recorded yet.")
    else:
        cols = [c for c in ["timestamp","symbol","mode","direction","score","confidence","entry","stop_loss","target1","target2"] if c in h.columns]
        display = h[cols].tail(50).iloc[::-1].copy()
        for c in ["entry","stop_loss","target1","target2"]:
            if c in display:
                display[c] = display[c].map(price)
        if "score" in display: display["score"] = display["score"].astype(float).round(3)
        if "confidence" in display: display["confidence"] = display["confidence"].astype(float).round(1)
        st.dataframe(display, use_container_width=True, hide_index=True, height=520)
        st.caption(f"Stored history: {len(h):,} rows")

st.divider()
foot = st.columns(4)
with foot[0]: st.caption(f"● {s['data_source']} · {symbol}")
with foot[1]: st.caption(f"● Analysis · {mode['tf'].upper()}")
with foot[2]: st.caption(f"● ML · {'Connected' if s['ml_available'] else 'Unavailable'}")
with foot[3]: st.caption(f"● Updated · {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
