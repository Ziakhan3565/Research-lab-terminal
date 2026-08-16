import time
import datetime
import os
import numpy as np
import pandas as pd
import requests

COINS_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "LTCUSDT", "BCHUSDT", "APTUSDT", "TRXUSDT",
    "SHIBUSDT", "UNIUSDT", "ATOMUSDT", "SUIUSDT", "INJUSDT", "ICPUSDT"
]

def fetch_order_book_and_price(symbol, depth_limit=20):
    try:
        # Fetch Price
        price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        price_res = requests.get(price_url, timeout=5).json()
        current_price = float(price_res['price'])

        # Fetch Depth
        depth_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth_limit}"
        depth_res = requests.get(depth_url, timeout=5).json()
        bids = np.array(depth_res['bids'], dtype=float)
        asks = np.array(depth_res['asks'], dtype=float)
        
        return current_price, bids, asks
    except Exception as e:
        return None, None, None

def log_auto_data(file_path="market_data_log.csv"):
    print("🚀 Auto Data Collector Started... (Press Ctrl+C to stop)")
    count = 0
    
    while True:
        for symbol in COINS_LIST:
            price, bids, asks = fetch_order_book_and_price(symbol)
            
            if price is not None and bids is not None and asks is not None:
                top20_bid_sum = float(np.sum(bids[:20, 1]))
                top20_ask_sum = float(np.sum(asks[:20, 1]))
                
                total_vol = top20_bid_sum + top20_ask_sum
                obi_top20 = (top20_bid_sum - top20_ask_sum) / total_vol if total_vol > 0 else 0.0
                
                spread = asks[0, 0] - bids[0, 0]
                
                data_point = {
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'timeframe': '15m',
                    'top20_bid_sum': round(top20_bid_sum, 4),
                    'top20_ask_sum': round(top20_ask_sum, 4),
                    'obi_top20': round(obi_top20, 6),
                    'spread': round(spread, 6),
                    'current_price': price
                }
                
                df_new = pd.DataFrame([data_point])
                
                if not os.path.isfile(file_path):
                    df_new.to_csv(file_path, index=False)
                else:
                    df_new.to_csv(file_path, mode='a', header=False, index=False)
                
                count += 1
                print(f"✅ [{count}] Saved data for {symbol} | Price: ${price} | OBI: {obi_top20:.4f}")
            
            # Rate limit handling (0.5 sec pause per coin)
            time.sleep(0.5)
            
        print("🔄 Finished 1 full cycle of 21 coins. Pausing 10 seconds before next cycle...\n")
        time.sleep(10)

if __name__ == "__main__":
    log_auto_data()