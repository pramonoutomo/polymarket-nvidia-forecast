"""Forecast the Yes price of each market of Polymarket's "Next CEO of NVIDIA?" event,
minute by minute, four hours past each checkpoint, as five quantiles.

Self-contained: Python 3 standard library only, no network, deterministic.  Reads every
<case>.in in the input directory and writes <case>.out next to the output directory.

Model
-----
* Median: the event's markets are mutually exclusive, so the sum of their Yes prices
  tends back toward 1.  Each market's median is scaled by the reverted sum path
  S(h) = S + (1 - S) * (1 - exp(-h / TAU)).
* Spread: a blend of a fixed width and a per-market width from the market's own recent
  volatility (MAD of 60-minute changes over the last 24 hours), applied as a
  random-walk width sqrt(h) around the median, asymmetric across quantile levels.
"""

import json
import math
import os

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
# Fixed width per sqrt(minute), in price units, per quantile level.
C = [0.003, 0.0006, 0.0002, 0.0002, 0.001]
# Multipliers of the per-market volatility, per quantile level.
K = [1.0, 0.5, 0.0, 0.1, 0.5]
LAM = 0.3
TAU = 120.0
HORIZON = 240


def mad(values):
    if not values:
        return 0.0
    values.sort()
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def vol(prices):
    """Price units per sqrt(minute), from the MAD of 60-minute changes over the last 24h."""
    n = len(prices)
    first = max(0, n - 1 - 1440)
    diffs = []
    t = first
    while t + 60 < n:
        a = prices[t]
        b = prices[t + 60]
        if a is not None and b is not None:
            diffs.append(abs(b - a))
        t += 2
    return (mad(diffs) / 0.67449) / math.sqrt(60.0)


def price6(x):
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return round(x, 6) + 0.0


def forecast_case(case):
    markets = case["markets"]
    horizon = int(case.get("horizon_minutes", HORIZON))
    total = 0.0
    for m in markets:
        p = m["prices"]
        total += p[-1] if p else 0.0

    rows = []
    for m in markets:
        prices = m["prices"]
        x = prices[-1] if prices else 0.0
        s = vol(prices)
        out = []
        for h in range(1, horizon + 1):
            if total > 0.0:
                ratio = (total + (1.0 - total) * (1.0 - math.exp(-h / TAU))) / total
                ratio = min(1.6, max(0.6, ratio))
                med = x * ratio
            else:
                med = x
            row = []
            for i in range(5):
                w = (LAM * C[i] + (1.0 - LAM) * K[i] * s) * math.sqrt(h)
                row.append(price6(med + Z[i] * w))
            out.append(row)
        rows.append({"id": m["id"], "q": out})
    return {"forecasts": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        case_id = name[:-3]
        with open(os.path.join(out_dir, case_id + ".out"), "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
