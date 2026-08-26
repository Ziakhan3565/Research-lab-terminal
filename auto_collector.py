import time
import datetime
import os
import numpy as np
import pandas as pd
import requests


# ============================================================
# BINANCE MARKET DATA CONFIGURATION
# ============================================================

COINS_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XMRUSDT",
    "XRPUSDT",
    "TAOUSDT",
]

BINANCE_BASE_URL = "https://api.binance.com"

DEPTH_LIMIT = 50
COLLECTION_INTERVAL = 5

MARKET_DATA_FILE = "market_data_log.csv"


# ============================================================
# BINANCE ORDER BOOK
# ============================================================

def fetch_binance_order_book(symbol, depth_limit=DEPTH_LIMIT):
    """
    Binance Spot live order book.
    MEXC yahan use nahi hota.
    """

    try:
        url = f"{BINANCE_BASE_URL}/api/v3/depth"

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "limit": depth_limit,
            },
            timeout=5,
        )

        response.raise_for_status()

        data = response.json()

        bids = np.array(data.get("bids", []), dtype=float)
        asks = np.array(data.get("asks", []), dtype=float)

        if len(bids) == 0 or len(asks) == 0:
            return None, None

        return bids, asks

    except Exception as e:
        print(f"❌ Binance order book error [{symbol}]: {e}")
        return None, None


# ============================================================
# BINANCE PRICE
# ============================================================

def fetch_binance_price(symbol):
    try:
        url = f"{BINANCE_BASE_URL}/api/v3/ticker/price"

        response = requests.get(
            url,
            params={"symbol": symbol},
            timeout=5,
        )

        response.raise_for_status()

        data = response.json()

        return float(data["price"])

    except Exception as e:
        print(f"❌ Binance price error [{symbol}]: {e}")
        return None


# ============================================================
# ORDER BOOK FEATURES
# ============================================================

def calculate_orderbook_features(bids, asks):
    if bids is None or asks is None:
        return {}

    if len(bids) == 0 or len(asks) == 0:
        return {}

    top20_bids = bids[:20]
    top20_asks = asks[:20]

    top50_bids = bids[:50]
    top50_asks = asks[:50]

    bid20 = float(np.sum(top20_bids[:, 1]))
    ask20 = float(np.sum(top20_asks[:, 1]))

    bid50 = float(np.sum(top50_bids[:, 1]))
    ask50 = float(np.sum(top50_asks[:, 1]))

    total20 = bid20 + ask20
    total50 = bid50 + ask50

    obi20 = (
        (bid20 - ask20) / total20
        if total20 > 0
        else 0.0
    )

    obi50 = (
        (bid50 - ask50) / total50
        if total50 > 0
        else 0.0
    )

    best_bid = float(bids[0, 0])
    best_ask = float(asks[0, 0])

    spread = best_ask - best_bid

    mid_price = (best_bid + best_ask) / 2.0

    spread_pct = (
        spread / mid_price
        if mid_price > 0
        else 0.0
    )

    return {
        "top20_bid_sum": bid20,
        "top20_ask_sum": ask20,
        "top50_bid_sum": bid50,
        "top50_ask_sum": ask50,
        "obi_top20": obi20,
        "obi_top50": obi50,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_pct": spread_pct,
        "mid_price": mid_price,
    }


# ============================================================
# MAIN COLLECTION
# ============================================================

def collect_one_symbol(symbol):
    price = fetch_binance_price(symbol)

    bids, asks = fetch_binance_order_book(
        symbol,
        DEPTH_LIMIT,
    )

    if price is None or bids is None or asks is None:
        return None

    features = calculate_orderbook_features(
        bids,
        asks,
    )

    if not features:
        return None

    timestamp = datetime.datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    row = {
        "timestamp": timestamp,
        "symbol": symbol,

        # IMPORTANT:
        # This is live Binance order-book data.
        "data_source": "BINANCE",

        "timeframe": "LIVE",

        "current_price": price,

        "top20_bid_sum": features["top20_bid_sum"],
        "top20_ask_sum": features["top20_ask_sum"],

        "top50_bid_sum": features["top50_bid_sum"],
        "top50_ask_sum": features["top50_ask_sum"],

        "obi_top20": features["obi_top20"],
        "obi_top50": features["obi_top50"],

        "best_bid": features["best_bid"],
        "best_ask": features["best_ask"],

        "spread": features["spread"],
        "spread_pct": features["spread_pct"],

        "mid_price": features["mid_price"],
    }

    return row


# ============================================================
# CSV WRITER
# ============================================================

def save_row(row, file_path=MARKET_DATA_FILE):
    if row is None:
        return

    df = pd.DataFrame([row])

    file_exists = os.path.isfile(file_path)

    df.to_csv(
        file_path,
        mode="a",
        header=not file_exists,
        index=False,
    )


# ============================================================
# CONTINUOUS COLLECTOR
# ============================================================

def log_auto_data(
    file_path=MARKET_DATA_FILE,
    interval=COLLECTION_INTERVAL,
):

    print("=" * 70)
    print("BINANCE LIVE ORDER BOOK COLLECTOR")
    print("=" * 70)
    print(f"Symbols       : {', '.join(COINS_LIST)}")
    print(f"Depth         : {DEPTH_LIMIT}")
    print(f"Interval      : {interval} seconds")
    print(f"Output        : {file_path}")
    print("Execution     : NOT HERE")
    print("Execution API : MEXC will be handled separately")
    print("=" * 70)

    count = 0

    while True:

        cycle_start = time.time()

        for symbol in COINS_LIST:

            row = collect_one_symbol(symbol)

            if row is not None:

                save_row(row, file_path)

                count += 1

                print(
                    f"[{count}] "
                    f"{symbol} | "
                    f"Price={row['current_price']:.4f} | "
                    f"OBI20={row['obi_top20']:.4f} | "
                    f"OBI50={row['obi_top50']:.4f} | "
                    f"Spread={row['spread']:.6f}"
                )

            time.sleep(0.25)

        elapsed = time.time() - cycle_start

        remaining = max(
            0,
            interval - elapsed,
        )

        print(
            f"🔄 Cycle completed | "
            f"sleep={remaining:.2f}s"
        )

        time.sleep(remaining)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    try:
        log_auto_data()

    except KeyboardInterrupt:
        print("\n🛑 Collector stopped by user.")
