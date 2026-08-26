import time


class SignalEngine:
    SCALPING_VALIDITY_SECONDS = 1800
    INTRADAY_VALIDITY_SECONDS = 28800

    # Research Lab is deliberately the largest directional component.
    STRONG_THRESHOLD = 0.65
    NORMAL_THRESHOLD = 0.40
    MIN_ML_CONFIDENCE = 0.55

    def validity_seconds(self, mode):
        return (
            self.INTRADAY_VALIDITY_SECONDS
            if str(mode).upper() == "INTRADAY"
            else self.SCALPING_VALIDITY_SECONDS
        )

    @staticmethod
    def ml_direction_score(long_probability, short_probability):
        return float(long_probability) - float(short_probability)

    def calculate_final_score(
        self, research_score, ml_score, ema_score,
        vwap_score, obi_score, ofi_score
    ):
        # OBI/OFI cannot independently create a trade.
        score = (
            float(research_score) * 0.45
            + float(ml_score) * 0.25
            + float(ema_score) * 0.10
            + float(vwap_score) * 0.10
            + float(obi_score) * 0.05
            + float(ofi_score) * 0.05
        )
        return max(-1.0, min(1.0, score))

    @staticmethod
    def ema_confirmation(price, ema20, ema50):
        if price > ema20 > ema50:
            return 1.0
        if price < ema20 < ema50:
            return -1.0
        if price > ema20:
            return 0.35
        if price < ema20:
            return -0.35
        return 0.0

    @staticmethod
    def vwap_confirmation(price, vwap):
        if vwap <= 0:
            return 0.0
        distance = (price - vwap) / vwap
        if distance > 0.001:
            return 1.0
        if distance < -0.001:
            return -1.0
        return 0.0

    def generate(
        self,
        research_score,
        long_probability,
        short_probability,
        price,
        ema20,
        ema50,
        vwap,
        obi,
        ofi,
        mode="SCALPING",
        current_time=None,
        ml_available=True,
    ):
        now = time.time() if current_time is None else float(current_time)
        ml_score = self.ml_direction_score(long_probability, short_probability)
        ema_score = self.ema_confirmation(price, ema20, ema50)
        vwap_score = self.vwap_confirmation(price, vwap)
        obi_score = max(-1.0, min(1.0, float(obi)))
        ofi_score = max(-1.0, min(1.0, float(ofi)))

        final_score = self.calculate_final_score(
            research_score, ml_score, ema_score,
            vwap_score, obi_score, ofi_score
        )

        bullish = sum(
            x > 0 for x in
            [obi, ofi, research_score, ml_score, ema_score, vwap_score]
        )
        bearish = sum(
            x < 0 for x in
            [obi, ofi, research_score, ml_score, ema_score, vwap_score]
        )

        signal = "NO TRADE"

        if ml_available:
            if (
                final_score >= self.STRONG_THRESHOLD
                and long_probability >= 0.70
                and bullish >= 4
            ):
                signal = "STRONG LONG"
            elif (
                final_score >= self.NORMAL_THRESHOLD
                and long_probability >= self.MIN_ML_CONFIDENCE
                and bullish >= 3
            ):
                signal = "LONG"
            elif (
                final_score <= -self.STRONG_THRESHOLD
                and short_probability >= 0.70
                and bearish >= 4
            ):
                signal = "STRONG SHORT"
            elif (
                final_score <= -self.NORMAL_THRESHOLD
                and short_probability >= self.MIN_ML_CONFIDENCE
                and bearish >= 3
            ):
                signal = "SHORT"

        validity = self.validity_seconds(mode)

        return {
            "signal": signal,
            "score": round(final_score, 6),
            "long_probability": round(float(long_probability), 6),
            "short_probability": round(float(short_probability), 6),
            "research_score": round(float(research_score), 6),
            "ema_score": round(float(ema_score), 6),
            "vwap_score": round(float(vwap_score), 6),
            "obi_score": round(float(obi_score), 6),
            "ofi_score": round(float(ofi_score), 6),
            "bullish_confirmations": int(bullish),
            "bearish_confirmations": int(bearish),
            "mode": str(mode).upper(),
            "generated_at": now,
            "expires_at": now + validity,
            "validity_seconds": validity,
            "ml_available": bool(ml_available),
        }

    @staticmethod
    def is_signal_valid(signal_payload, current_time=None):
        if not signal_payload or signal_payload.get("signal") == "NO TRADE":
            return False
        now = time.time() if current_time is None else float(current_time)
        return now <= float(signal_payload.get("expires_at", 0))
