import datetime as dt
import os
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import requests

from src.feature_pipeline import (
    calculate_orderbook_values,
    normalized_ofi,
    calculate_research_features,
)

COINS_LIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XMRUSDT", "XRPUSDT", "TAOUSDT"]
BINANCE_BASE_URL = "https://api.binance.com"
DEPTH_LIMIT = 50
COLLECTION_INTERVAL = 5
MARKET_DATA_FILE = "market_data_log.csv"

price_history = defaultdict(lambda: deque(maxlen=200))
volume_history = defaultdict(lambda: deque(maxlen=200))
previous_depth = {}


def binance_get(endpoint, params):
    r = requests.get(BINANCE_BASE_URL + endpoint, params=params, timeout=5)
    r.raise_for_status()
    return r.json()


def fetch_order_book(symbol):
    data = binance_get("/api/v3/depth", {"symbol": symbol, "limit": DEPTH_LIMIT})
    return np.asarray(data["bids"], dtype=float), np.asarray(data["asks"], dtype=float)


def fetch_price(symbol):
    data = binance_get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])


def fetch_taker_flow(symbol, limit=100):
    data = binance_get("/api/v3/aggTrades", {"symbol": symbol, "limit": limit})
    buy = sum(float(x["q"]) for x in data if not x["m"])
    sell = sum(float(x["q"]) for x in data if x["m"])
    return (buy - sell) / (buy + sell + 1e-8)


def collect_one_symbol(symbol):
    try:
        price = fetch_price(symbol)
        bids, asks = fetch_order_book(symbol)
        ob = calculate_orderbook_values(bids, asks)

        price_history[symbol].append(price)
        # Binance depth has no candle volume; use recent traded quantity as a volume proxy
        taker = fetch_taker_flow(symbol)
        volume_proxy = abs(taker) + 1.0
        volume_history[symbol].append(volume_proxy)

        prev = previous_depth.get(symbol)
        ofi = normalized_ofi(
            prev[0] if prev else None,
            prev[1] if prev else None,
            ob["top20_bid_sum"],
            ob["top20_ask_sum"],
        )
        previous_depth[symbol] = (
            ob["top20_bid_sum"],
            ob["top20_ask_sum"],
        )

        research = calculate_research_features(
            list(price_history[symbol]),
            list(volume_history[symbol]),
            ob["top20_bid_sum"],
            ob["top20_ask_sum"],
            float(bids[0, 1]),
            float(asks[0, 1]),
            ofi,
            taker,
        )

        row = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": symbol,
            "data_source": "BINANCE",
            "current_price": price,
            **{k: ob[k] for k in (
                "top20_bid_sum", "top20_ask_sum", "top50_bid_sum",
                "top50_ask_sum", "obi_top20", "obi_top50",
                "best_bid", "best_ask", "spread", "spread_pct"
            )},
            **research,
        }
        return row
    except Exception as exc:
        print(f"Collector error [{symbol}]: {exc}")
        return None


def save_row(row):
    if row is None:
        return
    pd.DataFrame([row]).to_csv(
        MARKET_DATA_FILE,
        mode="a",
        header=not os.path.exists(MARKET_DATA_FILE),
        index=False,
    )


def log_auto_data():
    print("BINANCE LIVE DATA COLLECTOR | MEXC execution is separate")
    while True:
        started = time.time()
        for symbol in COINS_LIST:
            row = collect_one_symbol(symbol)
            save_row(row)
            if row:
                print(
                    f"{symbol} price={row['current_price']:.4f} "
                    f"OBI20={row['obi_top20']:.4f} OFI={row['OFI']:.4f} "
                    f"TAKER={row['TAKER_FLOW']:.4f}"
                )
            time.sleep(0.25)
        time.sleep(max(0, COLLECTION_INTERVAL - (time.time() - started)))


if __name__ == "__main__":
    try:
        log_auto_data()
    except KeyboardInterrupt:
        print("Collector stopped.")
