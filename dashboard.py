import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TenPaperResearchLab:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol
        self.scaler = StandardScaler()
        
        # Initial feature weights for all 12 notebook formulas
        self.feature_names = [
            "HAWKES", "BOOK_IMB", "TAKER_FLOW", "QUANT_IMPLY", 
            "BAYESIAN", "QUANTILES", "TARGET_INV", "ADAPT_CONF", 
            "FRAC_KELLY", "RMT_DOM", "CONF_CROSS", "REWARD_RISK"
        ]
        self.dynamic_weights = {k: 1.0 / len(self.feature_names) for k in self.feature_names}

    def extract_features(self, df, bids, asks):
        results = {}
        if len(bids) == 0 or len(asks) == 0 or df.empty or len(df) < 15:
            return {k: 0.0 for k in self.feature_names}

        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        mid_price = (bids[0, 0] + asks[0, 0]) / 2
        returns = df["Close"].pct_change().dropna()
        realized_vol = returns.std() + 1e-8
        returns_h = (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / (df["Close"].iloc[-5] + 1e-8)
        delta_p = df["Close"].iloc[-1] - df["Close"].iloc[-2]

        # 1. Hawkes Intensity Process
        vol_changes = df["Volume"].pct_change().dropna().values
        hawkes_intensity = (np.mean(vol_changes[-3:]) / (np.mean(vol_changes[-15:]) + 1e-8)) if len(vol_changes) >= 15 else 1.0
        results["HAWKES"] = np.clip((hawkes_intensity - 1.0) * np.sign(returns_h), -1, 1)

        # 2. Book Imbalance
        results["BOOK_IMB"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        # 3. Taker Flow
        taker_buy = df["Volume"].iloc[-1] * (1.0 if delta_p > 0 else 0.3)
        taker_sell = df["Volume"].iloc[-1] * (1.0 if delta_p <= 0 else 0.3)
        results["TAKER_FLOW"] = (taker_buy - taker_sell) / (taker_buy + taker_sell + 1e-8)

        # 4. Quantities Imply
        depth_skew = (bids[0, 1] - asks[0, 1]) / (bids[0, 1] + asks[0, 1] + 1e-8)
        results["QUANT_IMPLY"] = np.clip(depth_skew * 1.5, -1, 1)

        # 5. Bayesian Probability (P = 74.5% baseline dynamic update)
        prior = 0.745
        likelihood = 1.0 if results["BOOK_IMB"] > 0 else 0.25
        posterior = (likelihood * prior) / ((likelihood * prior) + ((1 - likelihood) * (1 - prior)) + 1e-8)
        results["BAYESIAN"] = np.clip((posterior - 0.5) * 2.0, -1, 1)

        # 6. Quantiles Imply
        q90 = returns.quantile(0.90) if len(returns) > 5 else 0.01
        q10 = returns.quantile(0.10) if len(returns) > 5 else -0.01
        results["QUANTILES"] = np.clip((returns_h - q10) / (q90 - q10 + 1e-8) * 2.0 - 1.0, -1, 1)

        # 7. Target versus 0.060% Invalidation Threshold
        target_diff = delta_p / (df["Close"].iloc[-1] + 1e-8)
        results["TARGET_INV"] = 1.0 if target_diff >= 0.0006 else (-1.0 if target_diff <= -0.0006 else 0.0)

        # 8. Adaptive Conformal Band Crosses Zero
        ma_fast = df["Close"].rolling(3).mean().iloc[-1]
        ma_slow = df["Close"].rolling(10).mean().iloc[-1]
        results["ADAPT_CONF"] = np.clip((ma_fast - ma_slow) / (realized_vol * mid_price + 1e-8), -1, 1)

        # 9. Fractional Kelly Risk
        win_prob = 0.55 + (0.15 * np.sign(results["BOOK_IMB"]))
        kelly_fraction = win_prob - ((1 - win_prob) / 1.5)
        results["FRAC_KELLY"] = np.clip(kelly_fraction * 2.0 * np.sign(returns_h), -1, 1)

        # 10. RMT Market Dominance
        rmt_dom = (abs(returns_h) / (realized_vol * np.sqrt(5) + 1e-8)) / 3.0
        results["RMT_DOM"] = np.clip(rmt_dom * np.sign(returns_h), -1, 1)

        # 11. Conformal Interval Crosses Zero
        conformal_spread = realized_vol * 1.96
        upper_b = mid_price * (1 + conformal_spread)
        lower_b = mid_price * (1 - conformal_spread)
        results["CONF_CROSS"] = 1.0 if mid_price > (upper_b + lower_b) / 2 else (-1.0 if mid_price < (upper_b + lower_b) / 2 else 0.0)

        # 12. Quantiles Reward / Risk Below 1.2 Filter
        rr_ratio = abs(q90) / (abs(q10) + 1e-8)
        results["REWARD_RISK"] = 1.0 if rr_ratio >= 1.2 else (-1.0 if rr_ratio < 0.8 else 0.0)

        return results

    def calculate_all_signals(self, df, bids, asks, current_inventory=0, performance_history=None):
        results = self.extract_features(df, bids, asks)
        feature_vector = np.array([results[k] for k in self.feature_names]).reshape(1, -1)
        
        # Compute final score via Weighted Linear Ensemble directly (Clean & Fast)
        weight_vector = np.array(list(self.dynamic_weights.values()))
        final_score = float(np.dot(feature_vector[0], weight_vector))

        return results, final_score, self.dynamic_weights


class PowerTradingRiskEngine:
    def __init__(self):
        pass

    def calculate_risk_metrics(self, liquidation_volumes, displayed_vol, cancelled_vol, time_exists, obs_window, open_interest, leverage, volatility):
        total_ltz = np.sum(liquidation_volumes) if len(liquidation_volumes) > 0 else 0.0
        max_ltz = np.max(liquidation_volumes) if len(liquidation_volumes) > 0 else 0.0
        ltz_score = (max_ltz / (total_ltz + 1e-8)) * 100

        spoof_ratio = cancelled_vol / (displayed_vol + 1e-8)
        persistence = min(max(time_exists / (obs_window + 1e-8), 0), 1)
        spoof_score = spoof_ratio * (1 - persistence)

        squeeze_risk = total_ltz * open_interest * leverage * volatility
        market_risk = ltz_score + spoof_score + squeeze_risk

        return {
            "LTZ_Score": ltz_score,
            "Spoof_Score": spoof_score,
            "Squeeze_Risk": squeeze_risk,
            "Market_Risk": market_risk
        }
