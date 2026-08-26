import json
import os
import time
import traceback
import uuid

import ccxt
import pandas as pd

from train_model import check_and_retrain

SIGNAL_FILE = "signal_history.csv"
FEEDBACK_FILE = "trade_feedback.csv"
ACTIVE_FILE = "active_trade.json"
POLL_SECONDS = 5

# Safety: REAL orders require MEXC_DRY_RUN=false.
DRY_RUN = os.getenv("MEXC_DRY_RUN", "true").lower() == "true"

DEFAULT_LEVERAGE = 5
DEFAULT_TRADE_USDT = 10.0
TAKE_PROFIT_DEFAULT = 0.006
STOP_LOSS_DEFAULT = 0.004

mexc = ccxt.mexc({
    "apiKey": os.getenv("MEXC_API_KEY", ""),
    "secret": os.getenv("MEXC_SECRET_KEY", ""),
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},
})


def now_ts():
    return time.time()


def iso_now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def normalize_mexc_symbol(symbol):
    symbol = str(symbol).upper().strip()
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT:USDT"
    return symbol


def load_config():
    if not os.path.exists("config.json"):
        return {
            "is_running": False,
            "leverage": DEFAULT_LEVERAGE,
            "trade_amount_usdt": DEFAULT_TRADE_USDT,
            "selected_coins": [],
        }
    try:
        return json.loads(open("config.json").read())
    except Exception:
        return {
            "is_running": False,
            "leverage": DEFAULT_LEVERAGE,
            "trade_amount_usdt": DEFAULT_TRADE_USDT,
            "selected_coins": [],
        }


def append_csv(path, row):
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


def load_signals():
    if not os.path.exists(SIGNAL_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(SIGNAL_FILE)
        return df.sort_values("timestamp", ascending=False)
    except Exception:
        return pd.DataFrame()


def save_active_trade(trade):
    with open(ACTIVE_FILE, "w") as f:
        json.dump(trade, f, indent=2)


def load_active_trade():
    if not os.path.exists(ACTIVE_FILE):
        return None
    try:
        return json.loads(open(ACTIVE_FILE).read())
    except Exception:
        return None


def delete_active_trade():
    if os.path.exists(ACTIVE_FILE):
        os.remove(ACTIVE_FILE)


def configure_symbol(symbol, leverage):
    try:
        mexc.set_margin_mode("isolated", symbol)
    except Exception as exc:
        print("Margin mode warning:", exc)
    try:
        mexc.set_leverage(leverage, symbol)
    except Exception as exc:
        print("Leverage warning:", exc)


def get_mexc_price(symbol):
    try:
        ticker = mexc.fetch_ticker(normalize_mexc_symbol(symbol))
        return float(ticker["last"])
    except Exception as exc:
        print("MEXC price error:", exc)
        return None


def calculate_amount(symbol, usdt_amount, price):
    symbol = normalize_mexc_symbol(symbol)
    amount = float(usdt_amount) / max(float(price), 1e-8)
    try:
        return float(mexc.amount_to_precision(symbol, amount))
    except Exception:
        return amount


def extract_fill_price(order, fallback):
    for key in ("average", "price"):
        value = order.get(key)
        if value:
            return float(value)
    return fallback


def open_trade(signal, cfg):
    direction = str(signal["direction"]).upper()
    if direction not in {"LONG", "STRONG LONG", "SHORT", "STRONG SHORT"}:
        return None

    symbol = str(signal["symbol"])
    mexc_symbol = normalize_mexc_symbol(symbol)
    leverage = int(cfg.get("leverage", DEFAULT_LEVERAGE))
    usdt_amount = float(cfg.get("trade_amount_usdt", DEFAULT_TRADE_USDT))

    configure_symbol(mexc_symbol, leverage)

    execution_price = get_mexc_price(symbol)
    if execution_price is None:
        return None

    amount = calculate_amount(mexc_symbol, usdt_amount, execution_price)
    side = "buy" if "LONG" in direction else "sell"

    if DRY_RUN:
        order = {
            "id": "DRY-" + uuid.uuid4().hex[:12],
            "average": execution_price,
        }
    else:
        try:
            order = mexc.create_order(
                symbol=mexc_symbol,
                type="market",
                side=side,
                amount=amount,
                params={},
            )
        except Exception as exc:
            print("MEXC order error:", exc)
            return None

    entry = extract_fill_price(order, execution_price)

    if "LONG" in direction:
        sl = entry * (1 - STOP_LOSS_DEFAULT)
        tp = entry * (1 + TAKE_PROFIT_DEFAULT)
    else:
        sl = entry * (1 + STOP_LOSS_DEFAULT)
        tp = entry * (1 - TAKE_PROFIT_DEFAULT)

    return {
        "order_id": order.get("id"),
        "symbol": symbol,
        "mexc_symbol": mexc_symbol,
        "side": side,
        "amount": amount,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
    }


def close_real_position(trade):
    if DRY_RUN:
        return True
    try:
        side = "sell" if "LONG" in trade["direction"] else "buy"
        mexc.create_order(
            symbol=normalize_mexc_symbol(trade["symbol"]),
            type="market",
            side=side,
            amount=float(trade["amount"]),
            params={"reduceOnly": True},
        )
        return True
    except Exception as exc:
        print("Close position error:", exc)
        return False


def check_exit(trade, current_price):
    direction = trade["direction"].upper()
    if "LONG" in direction:
        if current_price <= float(trade["stop_loss"]):
            return "STOP_LOSS"
        if current_price >= float(trade["take_profit"]):
            return "TAKE_PROFIT"
    else:
        if current_price >= float(trade["stop_loss"]):
            return "STOP_LOSS"
        if current_price <= float(trade["take_profit"]):
            return "TAKE_PROFIT"

    if now_ts() >= float(trade["expires_at"]):
        return "SIGNAL_EXPIRY"
    return None


def calculate_pnl_percent(direction, entry, exit_price):
    if "LONG" in direction.upper():
        return ((exit_price - entry) / entry) * 100
    return ((entry - exit_price) / entry) * 100


def save_completed_trade(trade, exit_price, reason):
    pnl = calculate_pnl_percent(
        trade["direction"],
        float(trade["entry_price"]),
        float(exit_price),
    )
    outcome = "WIN" if pnl > 0 else "LOSS"

    features = trade.get("features", {})
    row = {
        "trade_id": trade["trade_id"],
        "timestamp": trade["timestamp"],
        "symbol": trade["symbol"],
        "direction": trade["direction"],
        "mode": trade.get("mode", "SCALPING"),
        "entry_price": trade["entry_price"],
        "exit_price": exit_price,
        "stop_loss": trade["stop_loss"],
        "take_profit": trade["take_profit"],
        "pnl_percent": pnl,
        "outcome": outcome,
        "exit_reason": reason,
        "duration_seconds": now_ts() - float(trade["opened_at"]),
        "status": "CLOSED",
        "status_source": "MEXC_EXECUTION",
        **{name: features.get(name, 0.0) for name in [
            "top20_bid_sum", "top20_ask_sum", "top50_bid_sum", "top50_ask_sum",
            "obi_top20", "obi_top50", "spread", "spread_pct",
            "BOOK_IMB", "OFI", "TAKER_FLOW", "QUANT_IMPLY", "ADAPT_CONF",
            "BAYESIAN", "FOURIER_TREND", "EMA_TREND", "VWAP_DISTANCE", "VOLATILITY"
        ]},
        "research_score": trade.get("research_score", 0),
        "ml_long_probability": trade.get("ml_long_probability", 0.5),
        "ml_short_probability": trade.get("ml_short_probability", 0.5),
        "final_score": trade.get("final_score", 0),
    }
    append_csv(FEEDBACK_FILE, row)
    print(f"CLOSED {trade['symbol']} {outcome} pnl={pnl:.4f}%")
    return row


def process_new_signal(cfg, processed):
    if load_active_trade() is not None:
        return processed

    df = load_signals()
    if df.empty:
        return processed

    selected = set(cfg.get("selected_coins", []))

    for _, row in df.head(20).iterrows():
        trade_id = str(row.get("trade_id", ""))
        if not trade_id or trade_id in processed:
            continue

        direction = str(row.get("direction", "NO TRADE")).upper()
        if direction == "NO TRADE":
            processed.add(trade_id)
            continue

        symbol = str(row.get("symbol", ""))
        if selected and symbol not in selected:
            continue

        expires_at = float(row.get("expires_at", 0))
        if expires_at and now_ts() > expires_at:
            processed.add(trade_id)
            continue

        try:
            features = json.loads(str(row.get("features_json", "{}")))
        except Exception:
            features = {}

        signal = {
            "trade_id": trade_id,
            "timestamp": str(row.get("timestamp", iso_now())),
            "symbol": symbol,
            "direction": direction,
            "mode": str(row.get("mode", "SCALPING")),
            "expires_at": expires_at,
            "features": features,
            "research_score": float(row.get("research_score", 0)),
            "ml_long_probability": float(row.get("ml_long_probability", 0.5)),
            "ml_short_probability": float(row.get("ml_short_probability", 0.5)),
            "final_score": float(row.get("final_score", 0)),
            "opened_at": now_ts(),
        }

        execution = open_trade(signal, cfg)
        if execution is None:
            continue

        signal.update(execution)
        save_active_trade(signal)
        processed.add(trade_id)
        print("ACTIVE:", direction, symbol)
        break

    return processed


def monitor_active_trade():
    trade = load_active_trade()
    if not trade:
        return

    price = get_mexc_price(trade["symbol"])
    if price is None:
        return

    reason = check_exit(trade, price)
    if reason is None:
        print(
            f"ACTIVE {trade['symbol']} {trade['direction']} "
            f"price={price} entry={trade['entry_price']}"
        )
        return

    print("EXIT:", trade["symbol"], reason, price)
    if not close_real_position(trade):
        return

    save_completed_trade(trade, price, reason)
    delete_active_trade()

    # Exactly every 20 CLOSED trades, candidate model is evaluated.
    try:
        check_and_retrain()
    except Exception:
        traceback.print_exc()


def run_bot():
    print("MEXC EXECUTION ENGINE")
    print("BINANCE = research/order book | MEXC = execution")
    print("DRY_RUN =", DRY_RUN)

    processed = set()

    while True:
        try:
            cfg = load_config()
            if not cfg.get("is_running", False):
                time.sleep(POLL_SECONDS)
                continue

            if load_active_trade():
                monitor_active_trade()
            else:
                processed = process_new_signal(cfg, processed)

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_bot()
