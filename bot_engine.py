import os
import json
import time
import uuid
import traceback

import ccxt
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SIGNAL_FILE = "signal_history.csv"
FEEDBACK_FILE = "trade_feedback.csv"

POLL_SECONDS = 5

# IMPORTANT:
# True = bot orders place nahi karega.
# False = REAL MEXC orders.
DRY_RUN = os.getenv(
    "MEXC_DRY_RUN",
    "true"
).lower() == "true"

DEFAULT_LEVERAGE = 5
DEFAULT_TRADE_USDT = 10.0

SCALPING_VALIDITY = 1800       # 30 minutes
INTRADAY_VALIDITY = 28800      # 8 hours

TAKE_PROFIT_DEFAULT = 0.006
STOP_LOSS_DEFAULT = 0.004


# ============================================================
# MEXC
# ============================================================

mexc = ccxt.mexc({
    "apiKey": os.getenv(
        "MEXC_API_KEY",
        ""
    ),
    "secret": os.getenv(
        "MEXC_SECRET_KEY",
        ""
    ),
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap",
    },
})


# ============================================================
# HELPERS
# ============================================================

def now_ts():
    return time.time()


def iso_now():
    return time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.gmtime()
    )


def normalize_mexc_symbol(symbol):
    """
    BTCUSDT -> BTC/USDT:USDT
    ETHUSDT -> ETH/USDT:USDT
    """

    symbol = str(symbol).upper().strip()

    if "/" in symbol:
        return symbol

    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"

    return symbol


def load_config():

    if not os.path.exists(
        "config.json"
    ):
        return {
            "is_running": False,
            "leverage": DEFAULT_LEVERAGE,
            "trade_amount_usdt": DEFAULT_TRADE_USDT,
            "selected_coins": [],
        }

    try:
        with open(
            "config.json",
            "r",
        ) as f:
            return json.load(f)

    except Exception:
        return {
            "is_running": False,
            "leverage": DEFAULT_LEVERAGE,
            "trade_amount_usdt": DEFAULT_TRADE_USDT,
            "selected_coins": [],
        }


# ============================================================
# CSV
# ============================================================

def append_csv(
    file_path,
    row,
):

    df = pd.DataFrame([row])

    if os.path.exists(
        file_path
    ):
        df.to_csv(
            file_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        df.to_csv(
            file_path,
            index=False,
        )


def load_signals():

    if not os.path.exists(
        SIGNAL_FILE
    ):
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            SIGNAL_FILE
        )

        if df.empty:
            return df

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

        return df.sort_values(
            "timestamp",
            ascending=False,
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# LEVERAGE
# ============================================================

def configure_symbol(
    symbol,
    leverage,
):

    symbol = normalize_mexc_symbol(
        symbol
    )

    try:

        mexc.set_margin_mode(
            "isolated",
            symbol,
        )

    except Exception as e:
        print(
            f"⚠️ Margin mode: {e}"
        )

    try:

        mexc.set_leverage(
            leverage,
            symbol,
        )

    except Exception as e:
        print(
            f"⚠️ Leverage: {e}"
        )


# ============================================================
# MARKET PRICE
# ============================================================

def get_market_price(
    symbol,
):

    symbol = normalize_mexc_symbol(
        symbol
    )

    try:

        ticker = mexc.fetch_ticker(
            symbol
        )

        return float(
            ticker["last"]
        )

    except Exception as e:

        print(
            f"⚠️ Price error {symbol}: {e}"
        )

        return None


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_amount(
    symbol,
    usdt_amount,
    price,
):

    symbol = normalize_mexc_symbol(
        symbol
    )

    try:

        amount = (
            float(usdt_amount)
            /
            float(price)
        )

        return float(
            mexc.amount_to_precision(
                symbol,
                amount,
            )
        )

    except Exception:

        return float(
            usdt_amount
            /
            max(price, 1e-8)
        )


# ============================================================
# OPEN ORDER
# ============================================================

def open_trade(
    signal,
    cfg,
):

    symbol = str(
        signal["symbol"]
    )

    intent = str(
        signal["direction"]
    ).upper()

    if intent not in (
        "LONG",
        "STRONG LONG",
        "SHORT",
        "STRONG SHORT",
    ):
        return None

    price = float(
        signal["entry_price"]
    )

    leverage = int(
        cfg.get(
            "leverage",
            DEFAULT_LEVERAGE,
        )
    )

    usdt_amount = float(
        cfg.get(
            "trade_amount_usdt",
            DEFAULT_TRADE_USDT,
        )
    )

    mexc_symbol = normalize_mexc_symbol(
        symbol
    )

    side = (
        "buy"
        if "LONG" in intent
        else "sell"
    )

    amount = calculate_amount(
        mexc_symbol,
        usdt_amount,
        price,
    )

    configure_symbol(
        mexc_symbol,
        leverage,
    )

    print(
        f"🚀 {intent} | "
        f"{mexc_symbol} | "
        f"entry={price} | "
        f"amount={amount}"
    )

    # --------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------

    if DRY_RUN:

        order_id = (
            "DRY-"
            +
            uuid.uuid4().hex[:12]
        )

        print(
            f"🧪 DRY RUN ORDER: "
            f"{order_id}"
        )

        return {
            "order_id": order_id,
            "symbol": mexc_symbol,
            "side": side,
            "amount": amount,
            "entry_price": price,
        }

    # --------------------------------------------------------
    # REAL ORDER
    # --------------------------------------------------------

    try:

        order = mexc.create_order(
            symbol=mexc_symbol,
            type="market",
            side=side,
            amount=amount,
            params={},
        )

        return {
            "order_id": order.get(
                "id"
            ),
            "symbol": mexc_symbol,
            "side": side,
            "amount": amount,
            "entry_price": price,
        }

    except Exception as e:

        print(
            f"❌ MEXC order error: {e}"
        )

        return None


# ============================================================
# TP / SL
# ============================================================

def calculate_exit_levels(
    entry,
    direction,
    stop_loss=None,
    take_profit=None,
):

    direction = direction.upper()

    if stop_loss is not None:
        sl = float(stop_loss)
    else:
        if "LONG" in direction:
            sl = entry * (
                1 - STOP_LOSS_DEFAULT
            )
        else:
            sl = entry * (
                1 + STOP_LOSS_DEFAULT
            )

    if take_profit is not None:
        tp = float(take_profit)
    else:
        if "LONG" in direction:
            tp = entry * (
                1 + TAKE_PROFIT_DEFAULT
            )
        else:
            tp = entry * (
                1 - TAKE_PROFIT_DEFAULT
            )

    return sl, tp


# ============================================================
# EXIT REASON
# ============================================================

def check_exit(
    trade,
    current_price,
):

    direction = trade[
        "direction"
    ].upper()

    entry = float(
        trade["entry_price"]
    )

    sl = float(
        trade["stop_loss"]
    )

    tp = float(
        trade["take_profit"]
    )

    expiry = float(
        trade["expires_at"]
    )

    current_time = now_ts()

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if "LONG" in direction:

        if current_price <= sl:
            return "STOP_LOSS"

        if current_price >= tp:
            return "TAKE_PROFIT"

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    elif "SHORT" in direction:

        if current_price >= sl:
            return "STOP_LOSS"

        if current_price <= tp:
            return "TAKE_PROFIT"

    # --------------------------------------------------------
    # EXPIRY
    # --------------------------------------------------------

    if current_time >= expiry:
        return "SIGNAL_EXPIRY"

    return None


# ============================================================
# CLOSE MEXC POSITION
# ============================================================

def close_real_position(
    trade,
):

    if DRY_RUN:
        return True

    try:

        symbol = normalize_mexc_symbol(
            trade["symbol"]
        )

        side = (
            "sell"
            if "LONG" in trade["direction"]
            else "buy"
        )

        amount = float(
            trade["amount"]
        )

        mexc.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params={
                "reduceOnly": True,
            },
        )

        return True

    except Exception as e:

        print(
            f"❌ Close position error: {e}"
        )

        return False


# ============================================================
# PNL
# ============================================================

def calculate_pnl(
    direction,
    entry,
    exit_price,
):

    direction = direction.upper()

    if "LONG" in direction:

        return (
            (
                exit_price
                -
                entry
            )
            /
            entry
        ) * 100

    return (
        (
            entry
            -
            exit_price
        )
        /
        entry
    ) * 100


# ============================================================
# TRADE FEEDBACK
# ============================================================

def save_completed_trade(
    trade,
    exit_price,
    reason,
):

    pnl = calculate_pnl(
        trade["direction"],
        float(trade["entry_price"]),
        float(exit_price),
    )

    outcome = (
        "WIN"
        if pnl > 0
        else "LOSS"
    )

    duration = (
        now_ts()
        -
        float(
            trade["opened_at"]
        )
    )

    feature_snapshot = trade.get(
        "features",
        {}
    )

    row = {
        "trade_id": trade[
            "trade_id"
        ],

        "timestamp": trade[
            "timestamp"
        ],

        "symbol": trade[
            "symbol"
        ],

        "direction": trade[
            "direction"
        ],

        "mode": trade.get(
            "mode",
            "SCALPING",
        ),

        "entry_price": trade[
            "entry_price"
        ],

        "exit_price": exit_price,

        "stop_loss": trade[
            "stop_loss"
        ],

        "take_profit": trade[
            "take_profit"
        ],

        "pnl_percent": pnl,

        "outcome": outcome,

        "exit_reason": reason,

        "duration_seconds": duration,

        "status": "CLOSED",

        # ----------------------------------------------------
        # COMPLETE RESEARCH SNAPSHOT
        # ----------------------------------------------------

        "BOOK_IMB": feature_snapshot.get(
            "BOOK_IMB",
            0,
        ),

        "OFI": feature_snapshot.get(
            "OFI",
            0,
        ),

        "TAKER_FLOW": feature_snapshot.get(
            "TAKER_FLOW",
            0,
        ),

        "QUANT_IMPLY": feature_snapshot.get(
            "QUANT_IMPLY",
            0,
        ),

        "ADAPT_CONF": feature_snapshot.get(
            "ADAPT_CONF",
            0,
        ),

        "BAYESIAN": feature_snapshot.get(
            "BAYESIAN",
            0,
        ),

        "FOURIER_TREND": feature_snapshot.get(
            "FOURIER_TREND",
            0,
        ),

        "EMA_TREND": feature_snapshot.get(
            "EMA_TREND",
            0,
        ),

        "VWAP_DISTANCE": feature_snapshot.get(
            "VWAP_DISTANCE",
            0,
        ),

        "VOLATILITY": feature_snapshot.get(
            "VOLATILITY",
            0,
        ),

        "research_score": trade.get(
            "research_score",
            0,
        ),

        "ml_long_probability": trade.get(
            "ml_long_probability",
            0.5,
        ),

        "ml_short_probability": trade.get(
            "ml_short_probability",
            0.5,
        ),

        "final_score": trade.get(
            "final_score",
            0,
        ),

        "status_source": "MEXC_EXECUTION",
    }

    append_csv(
        FEEDBACK_FILE,
        row,
    )

    print(
        f"💾 CLOSED TRADE SAVED | "
        f"{trade['symbol']} | "
        f"{outcome} | "
        f"PnL={pnl:.4f}%"
    )

    return row


# ============================================================
# RETRAIN CHECK
# ============================================================

def trigger_retraining_if_needed():

    if not os.path.exists(
        FEEDBACK_FILE
    ):
        return

    try:

        df = pd.read_csv(
            FEEDBACK_FILE
        )

        closed = df[
            df["status"]
            ==
            "CLOSED"
        ]

        count = len(
            closed
        )

        print(
            f"🧠 ML feedback count: "
            f"{count}"
        )

        if count > 0 and count % 20 == 0:

            print(
                "🔄 20 completed trades reached."
            )

            try:

                from train_model import (
                    retrain_after_20_trades
                )

                retrain_after_20_trades()

            except Exception as e:

                print(
                    f"⚠️ Retraining error: {e}"
                )

    except Exception as e:

        print(
            f"⚠️ Feedback read error: {e}"
        )


# ============================================================
# ACTIVE TRADE STORAGE
# ============================================================

ACTIVE_FILE = "active_trade.json"


def load_active_trade():

    if not os.path.exists(
        ACTIVE_FILE
    ):
        return None

    try:

        with open(
            ACTIVE_FILE,
            "r",
        ) as f:
            return json.load(f)

    except Exception:
        return None


def save_active_trade(
    trade,
):

    with open(
        ACTIVE_FILE,
        "w",
    ) as f:

        json.dump(
            trade,
            f,
            indent=4,
        )


def delete_active_trade():

    if os.path.exists(
        ACTIVE_FILE
    ):
        os.remove(
            ACTIVE_FILE
        )


# ============================================================
# PROCESS NEW SIGNAL
# ============================================================

def process_new_signal(
    cfg,
    processed_signals,
):

    df = load_signals()

    if df.empty:
        return processed_signals

    selected = cfg.get(
        "selected_coins",
        [],
    )

    for _, row in df.head(10).iterrows():

        signal_id = str(
            row.get(
                "trade_id",
                ""
            )
        )

        if not signal_id:
            signal_id = (
                str(row.get("timestamp"))
                +
                "_"
                +
                str(row.get("symbol"))
            )

        if signal_id in processed_signals:
            continue

        direction = str(
            row.get(
                "direction",
                "NO TRADE",
            )
        ).upper()

        if direction == "NO TRADE":
            processed_signals.add(
                signal_id
            )
            continue

        symbol = str(
            row.get(
                "symbol",
                "",
            )
        )

        if selected and symbol not in selected:
            continue

        expires_at = float(
            row.get(
                "expires_at",
                0,
            )
        )

        if expires_at > 0 and now_ts() > expires_at:

            processed_signals.add(
                signal_id
            )

            continue

        trade = {
            "trade_id": signal_id,

            "timestamp": str(
                row.get(
                    "timestamp",
                    iso_now(),
                )
            ),

            "symbol": symbol,

            "direction": direction,

            "mode": str(
                row.get(
                    "mode",
                    "SCALPING",
                )
            ),

            "entry_price": float(
                row.get(
                    "entry_price",
                    0,
                )
            ),

            "stop_loss": float(
                row.get(
                    "stop_loss",
                    0,
                )
            ),

            "take_profit": float(
                row.get(
                    "take_profit",
                    0,
                )
            ),

            "expires_at": expires_at,

            "opened_at": now_ts(),

            "features": json.loads(
                row.get(
                    "features_json",
                    "{}",
                )
            )
            if isinstance(
                row.get(
                    "features_json",
                    "{}",
                ),
                str
            )
            else {},

            "research_score": float(
                row.get(
                    "research_score",
                    0,
                )
            ),

            "ml_long_probability": float(
                row.get(
                    "ml_long_probability",
                    0.5,
                )
            ),

            "ml_short_probability": float(
                row.get(
                    "ml_short_probability",
                    0.5,
                )
            ),

            "final_score": float(
                row.get(
                    "final_score",
                    0,
                )
            ),
        }

        execution = open_trade(
            trade,
            cfg,
        )

        if execution is None:
            continue

        trade[
            "order_id"
        ] = execution[
            "order_id"
        ]

        trade[
            "amount"
        ] = execution[
            "amount"
        ]

        save_active_trade(
            trade
        )

        processed_signals.add(
            signal_id
        )

        print(
            f"✅ Active trade created: "
            f"{direction} {symbol}"
        )

        break

    return processed_signals


# ============================================================
# MONITOR ACTIVE TRADE
# ============================================================

def monitor_active_trade():

    trade = load_active_trade()

    if trade is None:
        return

    symbol = trade[
        "symbol"
    ]

    current_price = get_market_price(
        symbol
    )

    if current_price is None:
        return

    reason = check_exit(
        trade,
        current_price,
    )

    if reason is None:

        print(
            f"📈 ACTIVE | "
            f"{symbol} | "
            f"{trade['direction']} | "
            f"price={current_price}"
        )

        return

    print(
        f"🛑 EXIT | "
        f"{symbol} | "
        f"reason={reason} | "
        f"price={current_price}"
    )

    if not close_real_position(
        trade
    ):
        return

    save_completed_trade(
        trade,
        current_price,
        reason,
    )

    delete_active_trade()

    trigger_retraining_if_needed()


# ============================================================
# MAIN BOT
# ============================================================

def run_bot():

    print("=" * 70)
    print("MEXC EXECUTION ENGINE")
    print("=" * 70)

    print(
        f"DRY RUN = {DRY_RUN}"
    )

    print(
        "Market data = BINANCE"
    )

    print(
        "Execution = MEXC"
    )

    print(
        "Scalping validity = 1800 sec"
    )

    print(
        "Intraday validity = 28800 sec"
    )

    print("=" * 70)

    processed_signals = set()

    while True:

        try:

            cfg = load_config()

            if not cfg.get(
                "is_running",
                False,
            ):

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # Existing active trade
            # ------------------------------------------------

            if load_active_trade() is not None:

                monitor_active_trade()

            else:

                processed_signals = (
                    process_new_signal(
                        cfg,
                        processed_signals,
                    )
                )

            time.sleep(
                POLL_SECONDS
            )

        except KeyboardInterrupt:

            print(
                "🛑 Bot stopped."
            )

            break

        except Exception as e:

            print(
                f"❌ BOT ERROR: {e}"
            )

            traceback.print_exc()

            time.sleep(
                POLL_SECONDS
            )


if __name__ == "__main__":
    run_bot()
