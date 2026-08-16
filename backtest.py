import pandas as pd
import numpy as np
import joblib

def run_filtered_trend_backtest():
    print("🧪 Running Trend-Filtered Microstructure Backtest...\n")
    
    try:
        df = pd.read_csv("market_data_log.csv")
        model = joblib.load("xgboost_obi_model.pkl")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # Feature Engineering
    df['bid_ask_ratio'] = df['top20_bid_sum'] / (df['top20_ask_sum'] + 1e-5)
    df['total_depth'] = df['top20_bid_sum'] + df['top20_ask_sum']
    df['sma_20'] = df.groupby('symbol')['current_price'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['trend_signal'] = df['current_price'] - df['sma_20']

    features = ['top20_bid_sum', 'top20_ask_sum', 'obi_top20', 'spread', 'bid_ask_ratio', 'total_depth', 'trend_signal']
    
    df = df.dropna(subset=features + ['current_price', 'symbol']).copy()
    
    if 'timestamp' in df.columns:
        df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

    df['ml_signal'] = model.predict(df[features])
    
    try:
        probs = model.predict_proba(df[features])
        df['confidence'] = np.max(probs, axis=1)
    except:
        df['confidence'] = 1.0

    # Strong Filter: Only High Confidence + Strong Trend Match
    df_filtered = df[
        (df['confidence'] >= 0.55) & 
        (df['obi_top20'].abs() >= 0.20)
    ].copy()

    initial_balance = 100.0
    current_balance = initial_balance
    trade_amount = 10.0      
    fee_rate = 0.0004        
    
    # 1:1.5 Risk-Reward Ratio
    tp_pct = 0.0060          # 0.60% Take Profit
    sl_pct = 0.0040          # 0.40% Stop Loss

    wins = 0
    losses = 0
    total_profit_loss = 0.0

    for idx, row in df_filtered.iterrows():
        signal = row['ml_signal']
        entry = row['current_price']
        trend = row['trend_signal']

        # Rule: Filter Out Against-Trend Signals
        if signal == 1 and trend < 0: # Long only in Uptrend
            continue
        if signal == 0 and trend > 0: # Short only in Downtrend
            continue

        future_prices = df[
            (df['symbol'] == row['symbol']) & 
            (df.index > idx) & 
            (df.index <= idx + 6)
        ]['current_price']

        if len(future_prices) == 0:
            continue

        max_p = future_prices.max()
        min_p = future_prices.min()

        pnl_pct = 0.0

        if signal == 1:  # BUY
            if (max_p - entry) / entry >= tp_pct:
                pnl_pct = tp_pct - (2 * fee_rate)
                wins += 1
            elif (entry - min_p) / entry >= sl_pct:
                pnl_pct = -sl_pct - (2 * fee_rate)
                losses += 1
            else:
                continue

        elif signal == 0:  # SELL
            if (entry - min_p) / entry >= tp_pct:
                pnl_pct = tp_pct - (2 * fee_rate)
                wins += 1
            elif (max_p - entry) / entry >= sl_pct:
                pnl_pct = -sl_pct - (2 * fee_rate)
                losses += 1
            else:
                continue

        trade_pnl = trade_amount * pnl_pct
        current_balance += trade_pnl
        total_profit_loss += trade_pnl

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_return_pct = ((current_balance - initial_balance) / initial_balance) * 100

    print("=" * 40)
    print("📊 TREND-FILTERED P&L REPORT")
    print("=" * 40)
    print(f" Starting Capital     : ${initial_balance:.2f}")
    print(f" Total Executed Trades: {total_trades}")
    print(f" ✅ TP Hit (Wins)      : {wins}")
    print(f" ❌ SL Hit (Losses)    : {losses}")
    print(f" 🎯 Win Rate           : {win_rate:.2f}%")
    print("-" * 40)
    print(f" 💰 Total Net Profit   : ${total_profit_loss:.2f}")
    print(f" 📈 Account Growth %   : {total_return_pct:.2f}%")
    print(f" 🏦 Final Balance      : ${current_balance:.2f}")
    print("=" * 40)

if __name__ == "__main__":
    run_filtered_trend_backtest()