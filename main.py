#!/usr/bin/env python3
"""Quantile forecaster: reads <id>.in JSON cases, writes <id>.out forecasts.

Standard library only, deterministic, no network.
"""

import json
import math
import os
import sys

Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
C = [0.003, 0.0006, 0.0002, 0.0002, 0.001]
K = [1.0, 0.5, 0.0, 0.1, 0.5]
LAM = 0.3
HORIZON = 240
MAD_SCALE = 0.67449
DIFF_LAG = 60
LOOKBACK = 1440
REVERSION_MINUTES = 110.0


def _as_float(value):
    """Coerce a value to a finite float, or None if it is unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _mad_volatility(prices):
    """MAD of |p[t+60]-p[t]| over the last 1440 minutes, scaled to per-minute sd."""
    clean = [_as_float(p) for p in prices]
    n = len(clean)
    if n == 0:
        return 0.0

    first = max(0, n - LOOKBACK - DIFF_LAG)
    abs_diffs = []
    for t in range(first, n - DIFF_LAG):
        a = clean[t]
        b = clean[t + DIFF_LAG]
        if a is None or b is None:
            continue
        d = b - a
        if not math.isfinite(d):
            continue
        abs_diffs.append(abs(d))

    if not abs_diffs:
        return 0.0

    abs_diffs.sort()
    m = len(abs_diffs)
    if m % 2 == 1:
        mad = abs_diffs[m // 2]
    else:
        mad = 0.5 * (abs_diffs[m // 2 - 1] + abs_diffs[m // 2])

    s = mad / MAD_SCALE / math.sqrt(DIFF_LAG)
    if not math.isfinite(s):
        return 0.0
    return s


def _last_price(prices):
    for p in reversed(prices):
        f = _as_float(p)
        if f is not None:
            return f
    return 0.0


def forecast_market(market, total_last):
    """Return 240 rows of 5 quantiles for one market."""
    prices = market.get("prices") or []
    if not isinstance(prices, list):
        prices = []

    last = _last_price(prices)
    s = _mad_volatility(prices)

    rows = []
    for h in range(1, HORIZON + 1):
        if total_last > 0.0:
            ratio = (total_last + (1.0 - total_last) * (1.0 - math.exp(-h / REVERSION_MINUTES))) / total_last
            ratio = _clamp(ratio, 0.6, 1.6)
            m = last * ratio
        else:
            m = last

        sqrt_h = math.sqrt(h)
        row = []
        for i in range(5):
            spread = LAM * C[i] + (1.0 - LAM) * K[i] * s
            v = m + Z[i] * spread * sqrt_h
            v = _clamp(v, 0.0, 1.0)
            v = round(v, 6)
            row.append(v)
        rows.append(row)

    return rows


def process_case(case):
    markets = case.get("markets") or []
    if not isinstance(markets, list):
        markets = []

    prepared = []
    total_last = 0.0
    for market in markets:
        if not isinstance(market, dict):
            market = {}
        prices = market.get("prices") or []
        if not isinstance(prices, list):
            prices = []
        last = _last_price(prices)
        prepared.append((market.get("id"), prices, last))
        total_last += last

    forecasts = []
    for mid, prices, last in prepared:
        rows = forecast_market({"id": mid, "prices": prices}, total_last)
        forecasts.append({"id": mid, "q": rows})

    return {"forecasts": forecasts}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError as exc:
        print(f"error: cannot list input dir {in_dir}: {exc}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)

    for name in names:
        path = os.path.join(in_dir, name)
        stem = name[:-3]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            return 1

        if not isinstance(case, dict):
            case = {}

        result = process_case(case)

        out_path = os.path.join(out_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"), allow_nan=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
