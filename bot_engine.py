import os
import time
import json
import pandas as pd
import ccxt

# Exchange Setup
mexc = ccxt.mexc({
    'apiKey': os.getenv('MEXC_API_KEY', 'YOUR_MEXC_API_KEY'),
    'secret': os.getenv('MEXC_SECRET_KEY', 'YOUR_MEXC_SECRET_KEY'),
    'enableRateLimit': True,
})

def set_leverage(symbol, leverage):
    """Leverage aur Auto-margin mode set karna"""
    try:
        # MEXC Futures ke liye leverage set karna
        mexc.set_leverage(leverage, symbol)
        # Margin mode: 'cross' ya 'isolated'
        mexc.set_margin_mode('isolated', symbol) 
        print(f"Leverage set to {leverage}x for {symbol}")
    except Exception as e:
        print(f"Leverage error: {e}")

def run_bot():
    processed_signals = set()
    print("🤖 Bot Engine Ready. Check your dashboard to Start/Stop.")

    while True:
        # 1. Config Load Karein
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                cfg = json.load(f)
        else:
            cfg = {"is_running": False, "leverage": 5, "tp_pct": 1.0, "sl_pct": 0.5}

        # 2. Agar Bot OFF hai, toh wait karein
        if not cfg.get("is_running", False):
            time.sleep(5)
            continue

        # 3. Signals Monitor Karein
        if os.path.exists("signal_history.csv"):
            df = pd.read_csv("signal_history.csv")
            if not df.empty:
                latest = df.iloc[0]
                sig_id = f"{latest['timestamp']}_{latest['symbol']}"
                
                # Agar naya signal hai aur select kiye gaye coins mein se hai
                if sig_id not in processed_signals and latest['symbol'] in cfg.get("selected_coins", []):
                    # Leverage Set Karein
                    set_leverage(latest['symbol'], cfg.get("leverage", 5))
                    
                    # Trade Place Karein (Logic yahan pehle ki tarah rahega)
                    print(f"🚀 Executing Trade on {latest['symbol']} with {cfg['leverage']}x leverage!")
                    processed_signals.add(sig_id)

        time.sleep(5)

run_bot()
