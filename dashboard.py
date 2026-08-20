import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


class TenPaperResearchLab:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol

        # ML scaler
        self.scaler = StandardScaler()

        # Online Machine Learning Classifier
        # partial_fit() ke zariye live feedback se continuously learn karega
        self.ml_model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )

        self.is_model_trained = False

        # Initial feature fallback weights for all 12 notebook formulas
        self.feature_names = [
            "HAWKES",
            "BOOK_IMB",
            "TAKER_FLOW",
            "QUANT_IMPLY",
            "BAYESIAN",
            "QUANTILES",
            "TARGET_INV",
            "ADAPT_CONF",
            "FRAC_KELLY",
            "RMT_DOM",
            "CONF_CROSS",
            "REWARD_RISK"
        ]

        self.dynamic_weights = {
            k: 1.0 / len(self.feature_names)
            for k in self.feature_names
        }

    # ============================================================
    # FEATURE EXTRACTION
    # ============================================================

    def extract_features(self, df, bids, asks):

        results = {
            k: 0.0 for k in self.feature_names
        }

        # --------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------

        if df is None or df.empty:
            return results

        if len(df) < 15:
            return results

        if bids is None or asks is None:
            return results

        if len(bids) == 0 or len(asks) == 0:
            return results

        try:
            bids = np.asarray(bids, dtype=float)
            asks = np.asarray(asks, dtype=float)

            if bids.ndim != 2 or asks.ndim != 2:
                return results

            if bids.shape[1] < 2 or asks.shape[1] < 2:
                return results

            close = pd.to_numeric(
                df["Close"],
                errors="coerce"
            )

            volume = pd.to_numeric(
                df["Volume"],
                errors="coerce"
            )

            valid = (
                close.notna()
                & volume.notna()
                & np.isfinite(close)
                & np.isfinite(volume)
            )

            close = close[valid]
            volume = volume[valid]

            if len(close) < 15:
                return results

            # ----------------------------------------------------
            # Order Book
            # ----------------------------------------------------

            bid_prices = bids[:, 0]
            bid_sizes = bids[:, 1]

            ask_prices = asks[:, 0]
            ask_sizes = asks[:, 1]

            bid_sizes = np.clip(
                np.nan_to_num(
                    bid_sizes,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                ),
                0,
                None
            )

            ask_sizes = np.clip(
                np.nan_to_num(
                    ask_sizes,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                ),
                0,
                None
            )

            bid_vol = float(np.sum(bid_sizes))
            ask_vol = float(np.sum(ask_sizes))

            best_bid = float(bid_prices[0])
            best_ask = float(ask_prices[0])

            mid_price = (best_bid + best_ask) / 2.0

            if not np.isfinite(mid_price) or mid_price <= 0:
                return results

            # ----------------------------------------------------
            # Returns / Volatility
            # ----------------------------------------------------

            returns = close.pct_change().replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if len(returns) < 10:
                return results

            realized_vol = float(
                returns.std(ddof=1)
            )

            if not np.isfinite(realized_vol):
                realized_vol = 0.0

            realized_vol = max(realized_vol, 1e-8)

            current_price = float(close.iloc[-1])
            previous_price = float(close.iloc[-2])
            five_price = float(close.iloc[-5])

            returns_h = (
                (current_price - five_price)
                / (five_price + 1e-8)
            )

            delta_p = current_price - previous_price

            # ====================================================
            # 1. HAWKES INTENSITY
            # ====================================================

            vol_changes = (
                volume.pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )

            vol_changes = vol_changes[
                np.isfinite(vol_changes)
            ]

            if len(vol_changes) >= 15:

                recent = np.mean(
                    np.abs(vol_changes[-3:])
                )

                baseline = np.mean(
                    np.abs(vol_changes[-15:])
                )

                hawkes_intensity = (
                    recent / (baseline + 1e-8)
                )

            else:
                hawkes_intensity = 1.0

            results["HAWKES"] = np.clip(
                (hawkes_intensity - 1.0)
                * np.sign(returns_h),
                -1.0,
                1.0
            )

            # ====================================================
            # 2. BOOK IMBALANCE
            # ====================================================

            results["BOOK_IMB"] = np.clip(
                (bid_vol - ask_vol)
                / (bid_vol + ask_vol + 1e-8),
                -1.0,
                1.0
            )

            # ====================================================
            # 3. TAKER FLOW
            # ====================================================

            last_volume = max(
                float(volume.iloc[-1]),
                0.0
            )

            if delta_p > 0:

                taker_buy = last_volume
                taker_sell = last_volume * 0.3

            elif delta_p < 0:

                taker_buy = last_volume * 0.3
                taker_sell = last_volume

            else:

                taker_buy = last_volume * 0.5
                taker_sell = last_volume * 0.5

            results["TAKER_FLOW"] = np.clip(
                (taker_buy - taker_sell)
                / (taker_buy + taker_sell + 1e-8),
                -1.0,
                1.0
            )

            # ====================================================
            # 4. QUANTITIES IMPLY
            # ====================================================

            best_bid_size = float(bid_sizes[0])
            best_ask_size = float(ask_sizes[0])

            depth_skew = (
                (best_bid_size - best_ask_size)
                / (
                    best_bid_size
                    + best_ask_size
                    + 1e-8
                )
            )

            results["QUANT_IMPLY"] = np.clip(
                depth_skew * 1.5,
                -1.0,
                1.0
            )

            # ====================================================
            # 5. BAYESIAN PROBABILITY
            # ====================================================

            prior = 0.745

            if results["BOOK_IMB"] > 0:
                likelihood = 1.0
            elif results["BOOK_IMB"] < 0:
                likelihood = 0.25
            else:
                likelihood = 0.5

            numerator = likelihood * prior

            denominator = (
                numerator
                + (
                    (1.0 - likelihood)
                    * (1.0 - prior)
                )
                + 1e-8
            )

            posterior = numerator / denominator

            results["BAYESIAN"] = np.clip(
                (posterior - 0.5) * 2.0,
                -1.0,
                1.0
            )

            # ====================================================
            # 6. QUANTILES
            # ====================================================

            if len(returns) > 5:

                q90 = float(
                    returns.quantile(0.90)
                )

                q10 = float(
                    returns.quantile(0.10)
                )

            else:

                q90 = 0.01
                q10 = -0.01

            quantile_range = (
                q90 - q10
            )

            if abs(quantile_range) < 1e-8:

                results["QUANTILES"] = 0.0

            else:

                results["QUANTILES"] = np.clip(
                    (
                        (returns_h - q10)
                        / quantile_range
                        * 2.0
                    ) - 1.0,
                    -1.0,
                    1.0
                )

            # ====================================================
            # 7. TARGET / INVALIDATION
            # ====================================================

            target_diff = (
                delta_p
                / (current_price + 1e-8)
            )

            invalidation_threshold = 0.0006

            if target_diff >= invalidation_threshold:

                results["TARGET_INV"] = 1.0

            elif target_diff <= -invalidation_threshold:

                results["TARGET_INV"] = -1.0

            else:

                results["TARGET_INV"] = 0.0

            # ====================================================
            # 8. ADAPTIVE CONFORMAL
            # ====================================================

            ma_fast = float(
                close.rolling(3).mean().iloc[-1]
            )

            ma_slow = float(
                close.rolling(10).mean().iloc[-1]
            )

            denominator = (
                realized_vol
                * mid_price
                + 1e-8
            )

            results["ADAPT_CONF"] = np.clip(
                (ma_fast - ma_slow)
                / denominator,
                -1.0,
                1.0
            )

            # ====================================================
            # 9. FRACTIONAL KELLY
            # ====================================================

            book_direction = np.sign(
                results["BOOK_IMB"]
            )

            win_prob = (
                0.55
                + (
                    0.15
                    * book_direction
                )
            )

            win_prob = np.clip(
                win_prob,
                0.01,
                0.99
            )

            reward_ratio = 1.5

            kelly_fraction = (
                win_prob
                - (
                    (1.0 - win_prob)
                    / reward_ratio
                )
            )

            results["FRAC_KELLY"] = np.clip(
                kelly_fraction
                * 2.0
                * np.sign(returns_h),
                -1.0,
                1.0
            )

            # ====================================================
            # 10. RMT MARKET DOMINANCE
            # ====================================================

            rmt_denominator = (
                realized_vol
                * np.sqrt(5.0)
                + 1e-8
            )

            rmt_dom = (
                abs(returns_h)
                / rmt_denominator
            ) / 3.0

            results["RMT_DOM"] = np.clip(
                rmt_dom
                * np.sign(returns_h),
                -1.0,
                1.0
            )

            # ====================================================
            # 11. CONFORMAL INTERVAL CROSS
            # ====================================================

            # Previous code mein midpoint exactly mid_price tha,
            # isliye result hamesha 0 aa raha tha.
            #
            # Ab previous-close based prediction center use hota hai.

            conformal_spread = (
                realized_vol * 1.96
            )

            predicted_center = (
                previous_price
                * (
                    1.0
                    + returns_h
                )
            )

            upper_b = (
                predicted_center
                * (1.0 + conformal_spread)
            )

            lower_b = (
                predicted_center
                * (1.0 - conformal_spread)
            )

            if current_price > upper_b:

                results["CONF_CROSS"] = 1.0

            elif current_price < lower_b:

                results["CONF_CROSS"] = -1.0

            else:

                # Band ke andar ho to neutral
                results["CONF_CROSS"] = 0.0

            # ====================================================
            # 12. REWARD / RISK
            # ====================================================

            rr_ratio = (
                abs(q90)
                / (abs(q10) + 1e-8)
            )

            if rr_ratio >= 1.2:

                results["REWARD_RISK"] = 1.0

            elif rr_ratio < 0.8:

                results["REWARD_RISK"] = -1.0

            else:

                results["REWARD_RISK"] = 0.0

            # ----------------------------------------------------
            # Final numerical cleanup
            # ----------------------------------------------------

            for key in self.feature_names:

                value = results[key]

                if not np.isfinite(value):
                    results[key] = 0.0

                results[key] = float(
                    np.clip(
                        results[key],
                        -1.0,
                        1.0
                    )
                )

            return results

        except Exception:
            return results

    # ============================================================
    # MACHINE LEARNING TRAINING
    # ============================================================

    def _train_online(self, X_train, y_train):

        try:

            X_train = np.asarray(
                X_train,
                dtype=float
            )

            y_train = np.asarray(
                y_train,
                dtype=int
            )

            if len(X_train) < 2:
                return False

            if len(np.unique(y_train)) < 2:
                return False

            # Fit scaler ONLY on actual training data
            self.scaler.fit(X_train)

            X_scaled = self.scaler.transform(
                X_train
            )

            self.ml_model.partial_fit(
                X_scaled,
                y_train,
                classes=np.array([0, 1])
            )

            self.is_model_trained = True

            return True

        except Exception:
            return False

    # ============================================================
    # MAIN SIGNAL CALCULATION
    # ============================================================

    def calculate_all_signals(
        self,
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=None
    ):

        results = self.extract_features(
            df,
            bids,
            asks
        )

        feature_vector = np.array(
            [
                results[k]
                for k in self.feature_names
            ],
            dtype=float
        ).reshape(1, -1)

        feature_vector = np.nan_to_num(
            feature_vector,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        # ========================================================
        # ONLINE ML TRAINING
        # ========================================================

        if (
            performance_history
            and len(performance_history) >= 5
        ):

            X_train = []
            y_train = []

            # Last 30 actual records
            recent_history = (
                performance_history[-30:]
            )

            for hist in recent_history:

                if not isinstance(hist, dict):
                    continue

                # ------------------------------------------------
                # IMPORTANT:
                # Random features use nahi karte.
                #
                # Agar historical record mein "features"
                # available hain to wahi actual features use honge.
                # ------------------------------------------------

                hist_features = hist.get(
                    "features"
                )

                if (
                    isinstance(
                        hist_features,
                        dict
                    )
                ):

                    row = []

                    for name in self.feature_names:

                        value = hist_features.get(
                            name,
                            0.0
                        )

                        try:
                            value = float(value)
                        except Exception:
                            value = 0.0

                        row.append(
                            np.clip(
                                value,
                                -1.0,
                                1.0
                            )
                        )

                    outcome = str(
                        hist.get(
                            "outcome",
                            ""
                        )
                    ).upper()

                    if outcome == "WIN":

                        X_train.append(row)
                        y_train.append(1)

                    elif outcome == "LOSS":

                        X_train.append(row)
                        y_train.append(0)

            if (
                len(X_train) >= 5
                and len(set(y_train)) > 1
            ):

                self._train_online(
                    X_train,
                    y_train
                )

        # ========================================================
        # FINAL SCORE
        # ========================================================

        if self.is_model_trained:

            try:

                scaled_features = (
                    self.scaler.transform(
                        feature_vector
                    )
                )

                ml_prob = float(
                    self.ml_model.predict_proba(
                        scaled_features
                    )[0][1]
                )

                final_score = (
                    ml_prob - 0.5
                ) * 2.0

            except Exception:

                weight_vector = np.array(
                    [
                        self.dynamic_weights[k]
                        for k in self.feature_names
                    ]
                )

                final_score = float(
                    np.dot(
                        feature_vector[0],
                        weight_vector
                    )
                )

        else:

            weight_vector = np.array(
                [
                    self.dynamic_weights[k]
                    for k in self.feature_names
                ]
            )

            final_score = float(
                np.dot(
                    feature_vector[0],
                    weight_vector
                )
            )

        final_score = float(
            np.clip(
                final_score,
                -1.0,
                1.0
            )
        )

        return (
            results,
            final_score,
            self.dynamic_weights
        )


# ================================================================
# POWER TRADING RISK ENGINE
# ================================================================

class PowerTradingRiskEngine:

    def __init__(self):
        pass

    def calculate_risk_metrics(
        self,
        liquidation_volumes,
        displayed_vol,
        cancelled_vol,
        time_exists,
        obs_window,
        open_interest,
        leverage,
        volatility
    ):

        try:

            liquidation_volumes = np.asarray(
                liquidation_volumes,
                dtype=float
            )

            liquidation_volumes = np.nan_to_num(
                liquidation_volumes,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            liquidation_volumes = np.clip(
                liquidation_volumes,
                0.0,
                None
            )

            # ====================================================
            # LIQUIDATION TARGET ZONE
            # ====================================================

            if len(liquidation_volumes) > 0:

                total_ltz = float(
                    np.sum(
                        liquidation_volumes
                    )
                )

                max_ltz = float(
                    np.max(
                        liquidation_volumes
                    )
                )

            else:

                total_ltz = 0.0
                max_ltz = 0.0

            ltz_score = (
                max_ltz
                / (total_ltz + 1e-8)
            ) * 100.0

            ltz_score = float(
                np.clip(
                    ltz_score,
                    0.0,
                    100.0
                )
            )

            # ====================================================
            # SPOOFING RISK
            # ====================================================

            displayed_vol = max(
                float(displayed_vol),
                0.0
            )

            cancelled_vol = max(
                float(cancelled_vol),
                0.0
            )

            time_exists = max(
                float(time_exists),
                0.0
            )

            obs_window = max(
                float(obs_window),
                1e-8
            )

            spoof_ratio = (
                cancelled_vol
                / (displayed_vol + 1e-8)
            )

            persistence = np.clip(
                time_exists / obs_window,
                0.0,
                1.0
            )

            spoof_score = (
                spoof_ratio
                * (1.0 - persistence)
            )

            # Keep numerical stability
            spoof_score = float(
                np.clip(
                    spoof_score,
                    0.0,
                    100.0
                )
            )

            # ====================================================
            # SQUEEZE RISK
            # ====================================================

            open_interest = max(
                float(open_interest),
                0.0
            )

            leverage = max(
                float(leverage),
                0.0
            )

            volatility = max(
                float(volatility),
                0.0
            )

            raw_squeeze_risk = (
                total_ltz
                * open_interest
                * leverage
                * volatility
            )

            # Numerical safety
            if not np.isfinite(
                raw_squeeze_risk
            ):

                raw_squeeze_risk = 0.0

            # ----------------------------------------------------
            # Normalized squeeze score
            # ----------------------------------------------------
            #
            # Raw value ko preserve bhi karte hain aur normalized
            # score bhi dete hain.
            #

            squeeze_score = (
                100.0
                * (
                    1.0
                    - np.exp(
                        -min(
                            raw_squeeze_risk,
                            50.0
                        )
                    )
                )
            )

            squeeze_score = float(
                np.clip(
                    squeeze_score,
                    0.0,
                    100.0
                )
            )

            # ====================================================
            # MARKET RISK
            # ====================================================

            market_risk = (
                ltz_score
                + spoof_score
                + squeeze_score
            )

            # Normalize combined score
            market_risk = np.clip(
                market_risk / 3.0,
                0.0,
                100.0
            )

            return {

                "LTZ_Score": float(
                    ltz_score
                ),

                "Spoof_Score": float(
                    spoof_score
                ),

                "Squeeze_Risk": float(
                    squeeze_score
                ),

                "Raw_Squeeze_Risk": float(
                    raw_squeeze_risk
                ),

                "Market_Risk": float(
                    market_risk
                )
            }

        except Exception:

            return {

                "LTZ_Score": 0.0,
                "Spoof_Score": 0.0,
                "Squeeze_Risk": 0.0,
                "Raw_Squeeze_Risk": 0.0,
                "Market_Risk": 0.0
            }
