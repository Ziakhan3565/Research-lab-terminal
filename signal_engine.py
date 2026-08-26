import time


class SignalEngine:

    SCALPING_VALIDITY_SECONDS = 1800
    INTRADAY_VALIDITY_SECONDS = 28800

    STRONG_THRESHOLD = 0.75
    NORMAL_THRESHOLD = 0.50

    MIN_ML_CONFIDENCE = 0.60

    def __init__(self):
        self.last_signal = None

    # ========================================================
    # SIGNAL VALIDITY
    # ========================================================

    def validity_seconds(self, mode):

        mode = str(
            mode
        ).upper()

        if mode == "INTRADAY":
            return self.INTRADAY_VALIDITY_SECONDS

        return self.SCALPING_VALIDITY_SECONDS

    # ========================================================
    # ML SCORE
    # ========================================================

    def calculate_ml_direction_score(
        self,
        long_probability,
        short_probability,
    ):

        return (
            float(long_probability)
            - float(short_probability)
        )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    def calculate_final_score(
        self,
        research_score,
        ml_score,
        ema_score,
        vwap_score,
        obi_score,
        ofi_score,
    ):

        # Main decision is MULTI-FEATURE.
        #
        # OBI/OFI are important but cannot independently
        # create a signal.

        final_score = (
            research_score * 0.35
            + ml_score * 0.30
            + ema_score * 0.10
            + vwap_score * 0.10
            + obi_score * 0.075
            + ofi_score * 0.075
        )

        return max(
            -1.0,
            min(
                1.0,
                final_score,
            ),
        )

    # ========================================================
    # EMA CONFIRMATION
    # ========================================================

    def ema_confirmation(
        self,
        price,
        ema20,
        ema50,
    ):

        if price > ema20 > ema50:
            return 1.0

        if price < ema20 < ema50:
            return -1.0

        if price > ema20:
            return 0.35

        if price < ema20:
            return -0.35

        return 0.0

    # ========================================================
    # VWAP CONFIRMATION
    # ========================================================

    def vwap_confirmation(
        self,
        price,
        vwap,
    ):

        if vwap <= 0:
            return 0.0

        distance = (
            price - vwap
        ) / vwap

        if distance > 0.001:
            return 1.0

        if distance < -0.001:
            return -1.0

        return 0.0

    # ========================================================
    # SIGNAL GENERATION
    # ========================================================

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
    ):

        if current_time is None:
            current_time = time.time()

        ml_score = self.calculate_ml_direction_score(
            long_probability,
            short_probability,
        )

        ema_score = self.ema_confirmation(
            price,
            ema20,
            ema50,
        )

        vwap_score = self.vwap_confirmation(
            price,
            vwap,
        )

        # Normalize OBI / OFI into direction scores.
        obi_score = max(
            -1.0,
            min(
                1.0,
                float(obi),
            ),
        )

        # OFI should already be normalized.
        ofi_score = max(
            -1.0,
            min(
                1.0,
                float(ofi),
            ),
        )

        final_score = self.calculate_final_score(
            research_score=research_score,
            ml_score=ml_score,
            ema_score=ema_score,
            vwap_score=vwap_score,
            obi_score=obi_score,
            ofi_score=ofi_score,
        )

        # ====================================================
        # CONFIRMATION
        # ====================================================

        bullish_confirmations = 0
        bearish_confirmations = 0

        if obi > 0:
            bullish_confirmations += 1
        elif obi < 0:
            bearish_confirmations += 1

        if ofi > 0:
            bullish_confirmations += 1
        elif ofi < 0:
            bearish_confirmations += 1

        if research_score > 0:
            bullish_confirmations += 1
        elif research_score < 0:
            bearish_confirmations += 1

        if ml_score > 0:
            bullish_confirmations += 1
        elif ml_score < 0:
            bearish_confirmations += 1

        if ema_score > 0:
            bullish_confirmations += 1
        elif ema_score < 0:
            bearish_confirmations += 1

        if vwap_score > 0:
            bullish_confirmations += 1
        elif vwap_score < 0:
            bearish_confirmations += 1

        # ====================================================
        # FINAL SIGNAL
        # ====================================================

        signal = "NO TRADE"

        # Strong signals need broad confirmation.
        if (
            final_score >= self.STRONG_THRESHOLD
            and long_probability >= 0.75
            and bullish_confirmations >= 4
        ):
            signal = "STRONG LONG"

        elif (
            final_score >= self.NORMAL_THRESHOLD
            and long_probability >= self.MIN_ML_CONFIDENCE
            and bullish_confirmations >= 3
        ):
            signal = "LONG"

        elif (
            final_score <= -self.STRONG_THRESHOLD
            and short_probability >= 0.75
            and bearish_confirmations >= 4
        ):
            signal = "STRONG SHORT"

        elif (
            final_score <= -self.NORMAL_THRESHOLD
            and short_probability >= self.MIN_ML_CONFIDENCE
            and bearish_confirmations >= 3
        ):
            signal = "SHORT"

        # ====================================================
        # EXPIRY
        # ====================================================

        validity = self.validity_seconds(
            mode
        )

        expires_at = (
            current_time
            + validity
        )

        payload = {
            "signal": signal,
            "score": round(
                float(final_score),
                6,
            ),

            "long_probability": round(
                float(long_probability),
                6,
            ),

            "short_probability": round(
                float(short_probability),
                6,
            ),

            "research_score": round(
                float(research_score),
                6,
            ),

            "ema_score": round(
                float(ema_score),
                6,
            ),

            "vwap_score": round(
                float(vwap_score),
                6,
            ),

            "obi_score": round(
                float(obi_score),
                6,
            ),

            "ofi_score": round(
                float(ofi_score),
                6,
            ),

            "bullish_confirmations": bullish_confirmations,
            "bearish_confirmations": bearish_confirmations,

            "mode": str(
                mode
            ).upper(),

            "generated_at": current_time,
            "expires_at": expires_at,

            "validity_seconds": validity,
        }

        self.last_signal = payload

        return payload

    # ========================================================
    # VALIDITY CHECK
    # ========================================================

    def is_signal_valid(
        self,
        signal_payload,
        current_time=None,
    ):

        if signal_payload is None:
            return False

        if current_time is None:
            current_time = time.time()

        if signal_payload.get("signal") == "NO TRADE":
            return False

        expires_at = signal_payload.get(
            "expires_at",
            0,
        )

        return (
            current_time <= expires_at
        )
