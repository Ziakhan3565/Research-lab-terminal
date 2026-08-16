import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_trend_aligned_model():
    print("🚀 Training Trend-Aligned Microstructure Model...")
    
    try:
        df = pd.read_csv("market_data_log.csv")
    except FileNotFoundError:
        print("❌ 'market_data_log.csv' nahi mili!")
        return

    # 1. Feature Engineering
    df['bid_ask_ratio'] = df['top20_bid_sum'] / (df['top20_ask_sum'] + 1e-5)
    df['total_depth'] = df['top20_bid_sum'] + df['top20_ask_sum']
    
    # Trend Indicator (Simple Moving Average Diff)
    df['sma_20'] = df.groupby('symbol')['current_price'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['trend_signal'] = df['current_price'] - df['sma_20']
    
    features = ['top20_bid_sum', 'top20_ask_sum', 'obi_top20', 'spread', 'bid_ask_ratio', 'total_depth', 'trend_signal']
    df = df.dropna(subset=features + ['current_price', 'symbol']).copy()

    # Target: Aglay 5 steps mein price move
    df['future_price'] = df.groupby('symbol')['current_price'].shift(-5)
    df = df.dropna(subset=['future_price']).copy()
    
    df['target'] = (df['future_price'] > df['current_price']).astype(int)

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {acc * 100:.2f}%")

    joblib.dump(model, "xgboost_obi_model.pkl")
    print("💾 Model saved successfully as 'xgboost_obi_model.pkl'!")

if __name__ == "__main__":
    train_trend_aligned_model()