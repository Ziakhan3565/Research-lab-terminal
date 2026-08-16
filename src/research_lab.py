import numpy as np
import pandas as pd

class TenPaperResearchLab:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol

    def calculate_all_signals(self, df, bids, asks, current_inventory=0):
        results = {}
        
        # 1. OFI (Order Flow Imbalance) - Cont et al. (2014)
        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        # Normalized OFI
        results['OFI'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        # 2. TSMOM (Time-Series Momentum) - Moskowitz et al. (2012)
        # Using 5-candle ROC for faster bearish/bullish detection
        returns_h = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
        realized_vol = df['Close'].pct_change().std() + 1e-8
        results['TSMOM'] = np.clip((returns_h / realized_vol) * 2.0, -1, 1)

        # 3. MICRO (Micro-Price Imbalance) - Stoikov (2018)
        best_bid, best_ask = bids[0, 0], asks[0, 0]
        q_b, q_a = bids[0, 1], asks[0, 1]
        micro_price = (q_b * best_bid + q_a * best_ask) / (q_b + q_a + 1e-8)
        mid_price = (best_bid + best_ask) / 2
        # Sensitivity boosted by 5x
        results['MICRO'] = np.clip((micro_price - mid_price) / (mid_price * 0.0002), -1, 1)

        # 4. AVST (Avellaneda & Stoikov MM Model) - (2008)
        gamma = 0.1
        reservation_price = mid_price - current_inventory * gamma * (realized_vol ** 2)
        results['AVST'] = 1.0 if reservation_price > mid_price else (-1.0 if reservation_price < mid_price else 0.0)

        # 5. INVAR (Inventory Variance Adjustment) - Guéant et al. (2012)
        inventory_penalty = -current_inventory * 0.2 * (realized_vol ** 2)
        results['INVAR'] = np.clip(1.0 + inventory_penalty, -1, 1)

        # 6. VPIN (Volume-Synchronized Toxicity) - Easley et al. (2012)
        buy_vol = df['Volume'].iloc[-5:].mean() * (1.2 if returns_h > 0 else 0.3)
        sell_vol = df['Volume'].iloc[-5:].mean() * (1.2 if returns_h <= 0 else 0.3)
        vpin = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)
        results['VPIN'] = np.clip(vpin * 2.5, -1, 1)

        # 7. QUEUE (L1 Queue Imbalance) - Huang et al. (2015)
        results['QUEUE'] = np.clip((q_b - q_a) / (q_b + q_a + 1e-8) * 1.5, -1, 1)

        # 8. VRATIO (Variance Ratio Test) - Lo & MacKinlay (1988)
        var_1 = df['Close'].pct_change().var() + 1e-8
        var_5 = (df['Close'].pct_change(5)).var() / 5.0 + 1e-8
        v_ratio = var_5 / var_1
        results['VRATIO'] = 1.0 if (v_ratio > 1.0 and returns_h > 0) else (-1.0 if (v_ratio > 1.0 and returns_h < 0) else 0.0)

        # 9. BURST (Volatility Burst Detection) - Christensen et al. (2014)
        vol_short = df['Close'].pct_change().iloc[-3:].std()
        vol_long = df['Close'].pct_change().iloc[-20:].std() + 1e-8
        burst_ratio = vol_short / vol_long
        results['BURST'] = 1.0 if (burst_ratio > 1.2 and returns_h > 0) else (-1.0 if (burst_ratio > 1.2 and returns_h < 0) else 0.0)

        # 10. FUND (Implied Fundamental Value) - Cartea et al. (2014)
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        results['FUND'] = np.clip(obi * 1.5, -1, 1)

        # Weighted Ensemble Model
        weights = {
            'OFI': 0.15, 'TSMOM': 0.15, 'MICRO': 0.12, 'QUEUE': 0.10,
            'AVST': 0.08, 'INVAR': 0.08, 'VPIN': 0.08, 'VRATIO': 0.08,
            'BURST': 0.08, 'FUND': 0.08
        }
        
        final_score = sum(results[paper] * weights[paper] for paper in results)
        
        return results, final_score