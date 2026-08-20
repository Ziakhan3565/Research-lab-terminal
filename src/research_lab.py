import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# TRI LINE + 12 PAPER RESEARCH LAB
# ============================================================

class TenPaperResearchLab:

    def __init__(self, target_vol=0.15):

        self.target_vol = target_vol
        self.scaler = StandardScaler()

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
            "REWARD_RISK",

            "TRI_M_BODY",
            "TRI_M_UPPER",
            "TRI_M_LOWER",

            "TRI_W_BODY",
            "TRI_W_UPPER",
            "TRI_W_LOWER",

            "TRI_D_BODY",
            "TRI_D_UPPER",
            "TRI_D_LOWER",

            "TRI_DIRECTION"
        ]

        self.dynamic_weights = {
            name: 1.0 / len(self.feature_names)
            for name in self.feature_names
        }

        self.ml_model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )

        self.is_model_trained = False

    # ========================================================
    # TRI FEATURES
    # ========================================================

    def calculate_tri_features(self, price, tri_data):

        results = {
            name: 0.0
            for name in self.feature_names
            if name.startswith("TRI_")
        }

        if not tri_data:
            return results

        def distance(level):

            if level is None:
                return 0.0

            try:
                value = (
                    (float(price) - float(level))
                    / (abs(float(price)) + 1e-8)
                    * 1000
                )

                return float(np.clip(value, -1, 1))

            except Exception:
                return 0.0

        m_body = tri_data.get("mBody50")
        m_upper = tri_data.get("mUpper50")
        m_lower = tri_data.get("mLower50")

        w_body = tri_data.get("wBody50")
        w_upper = tri_data.get("wUpper50")
        w_lower = tri_data.get("wLower50")

        d_body = tri_data.get("dBody50")
        d_upper = tri_data.get("dUpper50")
        d_lower = tri_data.get("dLower50")

        results["TRI_M_BODY"] = distance(m_body)
        results["TRI_M_UPPER"] = distance(m_upper)
        results["TRI_M_LOWER"] = distance(m_lower)

        results["TRI_W_BODY"] = distance(w_body)
        results["TRI_W_UPPER"] = distance(w_upper)
        results["TRI_W_LOWER"] = distance(w_lower)

        results["TRI_D_BODY"] = distance(d_body)
        results["TRI_D_UPPER"] = distance(d_upper)
        results["TRI_D_LOWER"] = distance(d_lower)

        monthly = 0
        weekly = 0
        daily = 0

        if m_body is not None:
            monthly = 1 if price > m_body else -1

        if w_body is not None:
            weekly = 1 if price > w_body else -1

        if d_body is not None:
            daily = 1 if price > d_body else -1

        tri_score = (
            monthly * 0.40 +
            weekly * 0.35 +
            daily * 0.25
        )

        results["TRI_DIRECTION"] = float(
            np.clip(tri_score, -1, 1)
        )

        return results

    # ========================================================
    # 12 PAPER FEATURES
    # ========================================================

    def extract_features(
        self,
        df,
        bids,
        asks,
        tri_data=None
    ):

        results = {
            name: 0.0
            for name in self.feature_names
        }

        if df is None or df.empty:
            return results

        if len(df) < 15:
            return results

        if bids is None or asks is None:
            return results

        if len(bids) == 0 or len(asks) == 0:
            return results

        try:

            df = df.copy()

            df["Close"] = pd.to_numeric(
                df["Close"],
                errors="coerce"
            )

            df["Volume"] = pd.to_numeric(
                df["Volume"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["Close", "Volume"]
            )

            if len(df) < 15:
                return results

            bids = np.asarray(
                bids,
                dtype=float
            )

            asks = np.asarray(
                asks,
                dtype=float
            )

            bid_vol = np.sum(bids[:, 1])
            ask_vol = np.sum(asks[:, 1])

            mid_price = (
                bids[0, 0] +
                asks[0, 0]
            ) / 2

            returns = (
                df["Close"]
                .pct_change()
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )

            realized_vol = (
                returns.std() + 1e-8
            )

            returns_h = (
                df["Close"].iloc[-1]
                -
                df["Close"].iloc[-5]
            ) / (
                abs(df["Close"].iloc[-5])
                + 1e-8
            )

            delta_p = (
                df["Close"].iloc[-1]
                -
                df["Close"].iloc[-2]
            )

            # ------------------------------------------------
            # 1 HAWKES
            # ------------------------------------------------

            vol_changes = (
                df["Volume"]
                .pct_change()
                .replace(
                    [np.inf, -np.inf],
                    0
                )
                .fillna(0)
                .values
            )

            if len(vol_changes) >= 15:

                recent = np.mean(
                    vol_changes[-3:]
                )

                baseline = np.mean(
                    vol_changes[-15:]
                )

                hawkes = recent / (
                    baseline + 1e-8
                )

            else:

                hawkes = 1.0

            results["HAWKES"] = float(
                np.clip(
                    (hawkes - 1.0)
                    * np.sign(returns_h),
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 2 BOOK IMBALANCE
            # ------------------------------------------------

            results["BOOK_IMB"] = float(
                np.clip(
                    (bid_vol - ask_vol)
                    /
                    (
                        bid_vol +
                        ask_vol +
                        1e-8
                    ),
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 3 TAKER FLOW
            # ------------------------------------------------

            current_volume = float(
                df["Volume"].iloc[-1]
            )

            if delta_p > 0:

                taker_buy = current_volume
                taker_sell = current_volume * 0.3

            else:

                taker_buy = current_volume * 0.3
                taker_sell = current_volume

            results["TAKER_FLOW"] = float(
                np.clip(
                    (
                        taker_buy -
                        taker_sell
                    )
                    /
                    (
                        taker_buy +
                        taker_sell +
                        1e-8
                    ),
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 4 QUANTITY IMBALANCE
            # ------------------------------------------------

            depth_skew = (
                bids[0, 1] -
                asks[0, 1]
            ) / (
                bids[0, 1] +
                asks[0, 1] +
                1e-8
            )

            results["QUANT_IMPLY"] = float(
                np.clip(
                    depth_skew * 1.5,
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 5 BAYESIAN
            # ------------------------------------------------

            prior = 0.50

            likelihood = (
                0.75
                if results["BOOK_IMB"] > 0
                else 0.25
            )

            posterior = (
                likelihood * prior
            ) / (
                likelihood * prior
                +
                (1 - likelihood)
                * (1 - prior)
                +
                1e-8
            )

            results["BAYESIAN"] = float(
                np.clip(
                    (posterior - 0.5) * 2,
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 6 QUANTILES
            # ------------------------------------------------

            q90 = (
                returns.quantile(0.90)
                if len(returns) > 5
                else 0.01
            )

            q10 = (
                returns.quantile(0.10)
                if len(returns) > 5
                else -0.01
            )

            results["QUANTILES"] = float(
                np.clip(
                    (
                        (returns_h - q10)
                        /
                        (q90 - q10 + 1e-8)
                        * 2
                        - 1
                    ),
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 7 TARGET INVALIDATION
            # ------------------------------------------------

            target_diff = (
                delta_p
                /
                (
                    abs(
                        df["Close"].iloc[-1]
                    )
                    + 1e-8
                )
            )

            if target_diff >= 0.0006:
                results["TARGET_INV"] = 1.0

            elif target_diff <= -0.0006:
                results["TARGET_INV"] = -1.0

            else:
                results["TARGET_INV"] = 0.0

            # ------------------------------------------------
            # 8 ADAPTIVE CONFIDENCE
            # ------------------------------------------------

            ma_fast = (
                df["Close"]
                .rolling(3)
                .mean()
                .iloc[-1]
            )

            ma_slow = (
                df["Close"]
                .rolling(10)
                .mean()
                .iloc[-1]
            )

            results["ADAPT_CONF"] = float(
                np.clip(
                    (
                        ma_fast -
                        ma_slow
                    )
                    /
                    (
                        realized_vol *
                        mid_price +
                        1e-8
                    ),
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 9 FRACTIONAL KELLY
            # ------------------------------------------------

            win_prob = (
                0.55 +
                0.15 *
                np.sign(
                    results["BOOK_IMB"]
                )
            )

            kelly = (
                win_prob -
                (
                    (1 - win_prob)
                    / 1.5
                )
            )

            results["FRAC_KELLY"] = float(
                np.clip(
                    kelly *
                    2 *
                    np.sign(returns_h),
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 10 RMT
            # ------------------------------------------------

            rmt = (
                abs(returns_h)
                /
                (
                    realized_vol *
                    np.sqrt(5) +
                    1e-8
                )
            ) / 3

            results["RMT_DOM"] = float(
                np.clip(
                    rmt *
                    np.sign(returns_h),
                    -1,
                    1
                )
            )

            # ------------------------------------------------
            # 11 CONFORMAL CROSS
            # ------------------------------------------------

            conformal_spread = (
                realized_vol * 1.96
            )

            upper_b = (
                mid_price *
                (1 + conformal_spread)
            )

            lower_b = (
                mid_price *
                (1 - conformal_spread)
            )

            center = (
                upper_b +
                lower_b
            ) / 2

            if mid_price > center:
                results["CONF_CROSS"] = 1.0

            elif mid_price < center:
                results["CONF_CROSS"] = -1.0

            else:
                results["CONF_CROSS"] = 0.0

            # ------------------------------------------------
            # 12 REWARD RISK
            # ------------------------------------------------

            rr_ratio = (
                abs(q90)
                /
                (abs(q10) + 1e-8)
            )

            if rr_ratio >= 1.2:
                results["REWARD_RISK"] = 1.0

            elif rr_ratio < 0.8:
                results["REWARD_RISK"] = -1.0

            else:
                results["REWARD_RISK"] = 0.0

            # ------------------------------------------------
            # TRI
            # ------------------------------------------------

            tri_features = (
                self.calculate_tri_features(
                    df["Close"].iloc[-1],
                    tri_data
                )
            )

            results.update(
                tri_features
            )

        except Exception:

            return results

        return results

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    def calculate_all_signals(
        self,
        df,
        bids,
        asks,
        tri_data=None
    ):

        features = self.extract_features(
            df,
            bids,
            asks,
            tri_data
        )

        vector = np.array(
            [
                features[name]
                for name in self.feature_names
            ],
            dtype=float
        )

        weights = np.array(
            [
                self.dynamic_weights[name]
                for name in self.feature_names
            ],
            dtype=float
        )

        ensemble_score = np.dot(
            vector,
            weights
        )

        tri_direction = features[
            "TRI_DIRECTION"
        ]

        final_score = (
            ensemble_score * 0.75
            +
            tri_direction * 0.25
        )

        final_score = float(
            np.clip(
                final_score,
                -1,
                1
            )
        )

        if final_score >= 0.35:

            signal = "LONG"

        elif final_score <= -0.35:

            signal = "SHORT"

        else:

            signal = "WAIT"

        confidence = abs(
            final_score
        ) * 100

        return {
            "signal": signal,
            "score": final_score,
            "confidence": confidence,
            "features": features,
            "weights": self.dynamic_weights
        }


# ============================================================
# POWER RISK ENGINE
# ============================================================

class PowerTradingRiskEngine:

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

        liquidation_volumes = np.asarray(
            liquidation_volumes,
            dtype=float
        )

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
            /
            (total_ltz + 1e-8)
        ) * 100

        spoof_ratio = (
            cancelled_vol
            /
            (displayed_vol + 1e-8)
        )

        persistence = np.clip(
            time_exists
            /
            (obs_window + 1e-8),
            0,
            1
        )

        spoof_score = (
            spoof_ratio *
            (1 - persistence)
        ) * 100

        squeeze_risk = (
            total_ltz *
            open_interest *
            leverage *
            volatility
        )

        squeeze_normalized = (
            np.tanh(
                squeeze_risk / 1000000
            ) * 100
        )

        market_risk = (
            ltz_score
            +
            spoof_score
            +
            squeeze_normalized
        )

        market_risk = float(
            np.clip(
                market_risk,
                0,
                100
            )
        )

        if market_risk >= 75:

            risk_level = "EXTREME"

        elif market_risk >= 50:

            risk_level = "HIGH"

        elif market_risk >= 25:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        return {
            "LTZ_Score": float(
                np.clip(
                    ltz_score,
                    0,
                    100
                )
            ),
            "Spoof_Score": float(
                np.clip(
                    spoof_score,
                    0,
                    100
                )
            ),
            "Squeeze_Risk": float(
                squeeze_normalized
            ),
            "Market_Risk": market_risk,
            "Risk_Level": risk_level
        }


# ============================================================
# MASTER RESEARCH LAB ENGINE
# ============================================================

class IntegratedTradingEngine:

    def __init__(self):

        self.research = (
            TenPaperResearchLab()
        )

        self.risk = (
            PowerTradingRiskEngine()
        )

    def analyze(
        self,
        df,
        bids,
        asks,
        tri_data,
        liquidation_volumes=None,
        displayed_vol=0,
        cancelled_vol=0,
        time_exists=0,
        obs_window=60,
        open_interest=0,
        leverage=1,
        volatility=0
    ):

        if liquidation_volumes is None:
            liquidation_volumes = []

        signal_data = (
            self.research.calculate_all_signals(
                df=df,
                bids=bids,
                asks=asks,
                tri_data=tri_data
            )
        )

        risk_data = (
            self.risk.calculate_risk_metrics(
                liquidation_volumes=liquidation_volumes,
                displayed_vol=displayed_vol,
                cancelled_vol=cancelled_vol,
                time_exists=time_exists,
                obs_window=obs_window,
                open_interest=open_interest,
                leverage=leverage,
                volatility=volatility
            )
        )

        final_signal = signal_data["signal"]

        if risk_data["Risk_Level"] == "EXTREME":

            final_signal = "WAIT"

        elif (
            risk_data["Risk_Level"] == "HIGH"
            and
            signal_data["confidence"] < 75
        ):

            final_signal = "WAIT"

        return {
            "SIGNAL": final_signal,
            "RAW_SIGNAL": signal_data["signal"],
            "SCORE": signal_data["score"],
            "CONFIDENCE": signal_data["confidence"],
            "RISK": risk_data,
            "FEATURES": signal_data["features"]
        }
