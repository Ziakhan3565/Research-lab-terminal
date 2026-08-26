import numpy as np
import pandas as pd


# ============================================================
# RESEARCH ENGINE
# ============================================================

class ResearchEngine:

    FEATURE_NAMES = [
        "BOOK_IMB",
        "OFI",
        "TAKER_FLOW",
        "QUANT_IMPLY",
        "ADAPT_CONF",
        "BAYESIAN",
        "FOURIER_TREND",
        "EMA_TREND",
        "VWAP_DISTANCE",
        "VOLATILITY",
    ]

    def __init__(self):

        # Existing Research Lab weights
        self.weights = {
            "BOOK_IMB": 0.20,
            "OFI": 0.20,
            "TAKER_FLOW": 0.10,
            "QUANT_IMPLY": 0.08,
            "ADAPT_CONF": 0.10,
            "BAYESIAN": 0.05,
            "FOURIER_TREND": 0.08,
            "EMA_TREND": 0.07,
            "VWAP_DISTANCE": 0.05,
            "VOLATILITY": 0.07,
        }

        total = sum(self.weights.values())

        self.weights = {
            k: v / total
            for k, v in self.weights.items()
        }


    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def clip(value, low=-1.0, high=1.0):
        return float(
            np.clip(
                value,
                low,
                high,
            )
        )


    # ========================================================
    # OBI
    # ========================================================

    def calculate_obi(
        self,
        bid_volume,
        ask_volume,
    ):

        total = bid_volume + ask_volume

        if total <= 0:
            return 0.0

        return self.clip(
            (bid_volume - ask_volume) / total
        )


    # ========================================================
    # OFI
    # ========================================================

    def calculate_ofi(
        self,
        previous_bid,
        previous_ask,
        current_bid,
        current_ask,
    ):

        if (
            previous_bid is None
            or previous_ask is None
        ):
            return 0.0

        bid_change = current_bid - previous_bid
        ask_change = current_ask - previous_ask

        raw_ofi = bid_change - ask_change

        scale = (
            abs(current_bid)
            + abs(current_ask)
            + 1e-8
        )

        normalized = raw_ofi / scale

        return self.clip(normalized)


    # ========================================================
    # TAKER FLOW
    # ========================================================

    def calculate_taker_flow(
        self,
        buy_volume,
        sell_volume,
    ):

        total = buy_volume + sell_volume

        if total <= 0:
            return 0.0

        return self.clip(
            (buy_volume - sell_volume) / total
        )


    # ========================================================
    # QUANT IMPLY
    # ========================================================

    def calculate_quant_imply(
        self,
        best_bid_size,
        best_ask_size,
    ):

        total = (
            best_bid_size
            + best_ask_size
            + 1e-8
        )

        depth_skew = (
            best_bid_size
            - best_ask_size
        ) / total

        return self.clip(
            depth_skew * 1.5
        )


    # ========================================================
    # ADAPTIVE TREND
    # ========================================================

    def calculate_adaptive_conf(
        self,
        prices,
    ):

        if len(prices) < 10:
            return 0.0

        series = pd.Series(prices)

        fast = series.rolling(
            3
        ).mean().iloc[-1]

        slow = series.rolling(
            10
        ).mean().iloc[-1]

        returns = series.pct_change().dropna()

        volatility = (
            returns.rolling(
                15,
                min_periods=3,
            )
            .std()
            .iloc[-1]
        )

        if not np.isfinite(volatility):
            volatility = returns.std()

        price = float(series.iloc[-1])

        denominator = (
            volatility * price
            + 1e-8
        )

        value = (
            (fast - slow)
            / denominator
        )

        return self.clip(value)


    # ========================================================
    # BAYESIAN
    # ========================================================

    def calculate_bayesian(
        self,
        book_imbalance,
    ):

        # Existing prior retained as starting heuristic.
        # It should later be replaced by empirically estimated
        # probability from historical training data.

        prior = 0.745

        likelihood = (
            1.0
            if book_imbalance > 0
            else 0.25
        )

        numerator = likelihood * prior

        denominator = (
            numerator
            + (
                (1 - likelihood)
                * (1 - prior)
            )
            + 1e-8
        )

        posterior = numerator / denominator

        return self.clip(
            (posterior - 0.5) * 2.0
        )


    # ========================================================
    # ROLLING FOURIER
    # ========================================================

    def calculate_fourier_trend(
        self,
        prices,
        window=32,
    ):

        prices = np.asarray(
            prices,
            dtype=float,
        )

        if len(prices) < 15:
            return 0.0

        window_prices = prices[-window:]

        if len(window_prices) < 15:
            return 0.0

        centered = (
            window_prices
            - np.mean(window_prices)
        )

        fft_values = np.fft.fft(
            centered
        )

        n = len(fft_values)

        keep = max(
            1,
            int(n * 0.15),
        )

        filtered = np.zeros_like(
            fft_values
        )

        filtered[:keep] = (
            fft_values[:keep]
        )

        filtered[-keep:] = (
            fft_values[-keep:]
        )

        reconstructed = np.real(
            np.fft.ifft(filtered)
        )

        trend = (
            reconstructed[-1]
            - reconstructed[-2]
        )

        returns = (
            pd.Series(window_prices)
            .pct_change()
            .dropna()
        )

        volatility = (
            returns.std()
            + 1e-8
        )

        price = (
            window_prices[-1]
        )

        normalized = (
            trend
            / (volatility * price + 1e-8)
        )

        return self.clip(
            normalized
        )


    # ========================================================
    # EMA
    # ========================================================

    def calculate_ema_trend(
        self,
        prices,
    ):

        if len(prices) < 50:
            return 0.0

        series = pd.Series(
            prices
        )

        ema20 = (
            series
            .ewm(
                span=20,
                adjust=False,
            )
            .mean()
            .iloc[-1]
        )

        ema50 = (
            series
            .ewm(
                span=50,
                adjust=False,
            )
            .mean()
            .iloc[-1]
        )

        price = float(
            series.iloc[-1]
        )

        scale = (
            abs(price)
            + 1e-8
        )

        value = (
            (ema20 - ema50)
            / scale
        ) * 100

        return self.clip(
            value
        )


    # ========================================================
    # VWAP
    # ========================================================

    def calculate_vwap_distance(
        self,
        prices,
        volumes,
    ):

        if len(prices) == 0:
            return 0.0

        prices = np.asarray(
            prices,
            dtype=float,
        )

        volumes = np.asarray(
            volumes,
            dtype=float,
        )

        if len(volumes) != len(prices):
            volumes = np.ones_like(
                prices
            )

        volume_sum = (
            np.sum(volumes)
        )

        if volume_sum <= 0:
            return 0.0

        vwap = (
            np.sum(
                prices * volumes
            )
            / volume_sum
        )

        current_price = prices[-1]

        distance = (
            current_price - vwap
        ) / (
            abs(vwap) + 1e-8
        )

        return self.clip(
            distance * 100
        )


    # ========================================================
    # VOLATILITY
    # ========================================================

    def calculate_volatility(
        self,
        prices,
    ):

        if len(prices) < 5:
            return 0.0

        series = pd.Series(
            prices
        )

        returns = (
            series
            .pct_change()
            .dropna()
        )

        if len(returns) == 0:
            return 0.0

        vol = returns.std()

        # Positive volatility score is used only as a
        # market-activity feature, not as direction.
        return self.clip(
            vol * 100
        )


    # ========================================================
    # COMPLETE RESEARCH CALCULATION
    # ========================================================

    def calculate_all(
        self,
        prices,
        volumes,
        bid20,
        ask20,
        best_bid_size,
        best_ask_size,
        previous_bid20=None,
        previous_ask20=None,
        buy_volume=0.0,
        sell_volume=0.0,
    ):

        prices = np.asarray(
            prices,
            dtype=float,
        )

        volumes = np.asarray(
            volumes,
            dtype=float,
        )

        obi = self.calculate_obi(
            bid20,
            ask20,
        )

        ofi = self.calculate_ofi(
            previous_bid20,
            previous_ask20,
            bid20,
            ask20,
        )

        taker_flow = self.calculate_taker_flow(
            buy_volume,
            sell_volume,
        )

        quant_imply = self.calculate_quant_imply(
            best_bid_size,
            best_ask_size,
        )

        adaptive_conf = self.calculate_adaptive_conf(
            prices
        )

        bayesian = self.calculate_bayesian(
            obi
        )

        fourier = self.calculate_fourier_trend(
            prices
        )

        ema_trend = self.calculate_ema_trend(
            prices
        )

        vwap_distance = self.calculate_vwap_distance(
            prices,
            volumes,
        )

        volatility = self.calculate_volatility(
            prices
        )

        result = {
            "BOOK_IMB": obi,
            "OFI": ofi,
            "TAKER_FLOW": taker_flow,
            "QUANT_IMPLY": quant_imply,
            "ADAPT_CONF": adaptive_conf,
            "BAYESIAN": bayesian,
            "FOURIER_TREND": fourier,
            "EMA_TREND": ema_trend,
            "VWAP_DISTANCE": vwap_distance,
            "VOLATILITY": volatility,
        }

        weighted_score = 0.0

        for name, weight in self.weights.items():
            weighted_score += (
                result[name] * weight
            )

        result["RESEARCH_SCORE"] = float(
            np.clip(
                weighted_score,
                -1,
                1,
            )
        )

        return result
