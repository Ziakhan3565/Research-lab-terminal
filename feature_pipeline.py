import numpy as np

FEATURES = [
    "top20_bid_sum", "top20_ask_sum", "top50_bid_sum", "top50_ask_sum",
    "obi_top20", "obi_top50", "spread", "spread_pct",
    "BOOK_IMB", "OFI", "TAKER_FLOW", "QUANT_IMPLY", "ADAPT_CONF",
    "BAYESIAN", "FOURIER_TREND", "EMA_TREND", "VWAP_DISTANCE", "VOLATILITY",
]

RESEARCH_FEATURES = [
    "BOOK_IMB", "OFI", "TAKER_FLOW", "QUANT_IMPLY", "ADAPT_CONF",
    "BAYESIAN", "FOURIER_TREND", "EMA_TREND", "VWAP_DISTANCE", "VOLATILITY",
]


def finite(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def clip(value, low=-1.0, high=1.0):
    return float(np.clip(finite(value), low, high))


def calculate_orderbook_values(bids, asks):
    bid20 = float(np.asarray(bids)[:20, 1].sum())
    ask20 = float(np.asarray(asks)[:20, 1].sum())
    bid50 = float(np.asarray(bids)[:50, 1].sum())
    ask50 = float(np.asarray(asks)[:50, 1].sum())

    total20 = bid20 + ask20
    total50 = bid50 + ask50

    obi20 = (bid20 - ask20) / (total20 + 1e-8)
    obi50 = (bid50 - ask50) / (total50 + 1e-8)

    best_bid = float(bids[0, 0])
    best_ask = float(asks[0, 0])
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2.0

    return {
        "top20_bid_sum": bid20,
        "top20_ask_sum": ask20,
        "top50_bid_sum": bid50,
        "top50_ask_sum": ask50,
        "obi_top20": clip(obi20),
        "obi_top50": clip(obi50),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_pct": spread / (mid + 1e-8),
        "mid_price": mid,
    }


def normalized_ofi(previous_bid20, previous_ask20, bid20, ask20):
    if previous_bid20 is None or previous_ask20 is None:
        return 0.0
    raw = (bid20 - previous_bid20) - (ask20 - previous_ask20)
    scale = abs(bid20) + abs(ask20) + 1e-8
    return clip(raw / scale)


def calculate_research_features(
    prices,
    volumes,
    bid20,
    ask20,
    best_bid_size,
    best_ask_size,
    ofi,
    taker_flow,
):
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if len(prices) == 0:
        return {name: 0.0 for name in RESEARCH_FEATURES}

    # OBI / book imbalance
    book_imb = clip((bid20 - ask20) / (bid20 + ask20 + 1e-8))

    # Quant implied depth skew
    depth_skew = (best_bid_size - best_ask_size) / (
        best_bid_size + best_ask_size + 1e-8
    )
    quant_imply = clip(depth_skew * 1.5)

    # Adaptive confidence
    if len(prices) >= 10:
        p = prices[-200:]
        fast = np.mean(p[-3:])
        slow = np.mean(p[-10:])
        returns = np.diff(p) / (p[:-1] + 1e-8)
        vol = np.std(returns[-15:]) if len(returns) else 0.0
        adapt = clip((fast - slow) / (vol * p[-1] + 1e-8))
    else:
        adapt = 0.0

    # Bayesian heuristic retained from the original research lab.
    prior = 0.745
    likelihood = 1.0 if book_imb > 0 else 0.25
    numerator = likelihood * prior
    denominator = numerator + ((1 - likelihood) * (1 - prior)) + 1e-8
    posterior = numerator / denominator
    bayesian = clip((posterior - 0.5) * 2.0)

    # Rolling Fourier trend.
    fourier = 0.0
    segment = prices[-32:]
    if len(segment) >= 15:
        centered = segment - np.mean(segment)
        fft_values = np.fft.fft(centered)
        n = len(fft_values)
        keep = max(1, int(n * 0.15))
        filtered = np.zeros_like(fft_values)
        filtered[:keep] = fft_values[:keep]
        filtered[-keep:] = fft_values[-keep:]
        curve = np.real(np.fft.ifft(filtered))
        trend = curve[-1] - curve[-2]
        r = np.diff(segment) / (segment[:-1] + 1e-8)
        volatility = np.std(r) + 1e-8
        fourier = clip(trend / (volatility * segment[-1] + 1e-8))

    # EMA trend
    p = prices[-200:]
    ema20 = p[0]
    ema50 = p[0]
    a20 = 2.0 / 21.0
    a50 = 2.0 / 51.0
    for value in p[1:]:
        ema20 = a20 * value + (1 - a20) * ema20
        ema50 = a50 * value + (1 - a50) * ema50
    ema_trend = clip(((ema20 - ema50) / (abs(p[-1]) + 1e-8)) * 100.0)

    # VWAP
    v = volumes[-len(p):] if len(volumes) else np.ones(len(p))
    if len(v) != len(p) or np.sum(v) <= 0:
        v = np.ones(len(p))
    vwap = np.sum(p * v) / (np.sum(v) + 1e-8)
    vwap_distance = clip(((p[-1] - vwap) / (abs(vwap) + 1e-8)) * 100.0)

    returns = np.diff(p) / (p[:-1] + 1e-8)
    volatility = clip(np.std(returns) * 100.0, 0.0, 1.0)

    return {
        "BOOK_IMB": book_imb,
        "OFI": clip(ofi),
        "TAKER_FLOW": clip(taker_flow),
        "QUANT_IMPLY": quant_imply,
        "ADAPT_CONF": adapt,
        "BAYESIAN": bayesian,
        "FOURIER_TREND": fourier,
        "EMA_TREND": ema_trend,
        "VWAP_DISTANCE": vwap_distance,
        "VOLATILITY": volatility,
    }


def make_feature_vector(orderbook, research):
    return np.asarray(
        [
            finite(orderbook.get("top20_bid_sum")),
            finite(orderbook.get("top20_ask_sum")),
            finite(orderbook.get("top50_bid_sum")),
            finite(orderbook.get("top50_ask_sum")),
            finite(orderbook.get("obi_top20")),
            finite(orderbook.get("obi_top50")),
            finite(orderbook.get("spread")),
            finite(orderbook.get("spread_pct")),
            finite(research.get("BOOK_IMB")),
            finite(research.get("OFI")),
            finite(research.get("TAKER_FLOW")),
            finite(research.get("QUANT_IMPLY")),
            finite(research.get("ADAPT_CONF")),
            finite(research.get("BAYESIAN")),
            finite(research.get("FOURIER_TREND")),
            finite(research.get("EMA_TREND")),
            finite(research.get("VWAP_DISTANCE")),
            finite(research.get("VOLATILITY")),
        ],
        dtype=float,
    ).reshape(1, -1)
