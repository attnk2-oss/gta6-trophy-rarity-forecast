#!/usr/bin/env python3
"""Baseline forecast for GTA VI trophy earn rates at the requested horizon."""

import json
import math
import os
from pathlib import Path


INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
NORMAL_Z = (-1.645, -0.674, 0.0, 0.674, 1.645)


def clamp_rate(value):
    return min(99.99, max(0.01, float(value)))


def logit_rate(value):
    rate = clamp_rate(value)
    return math.log(rate / (100.0 - rate))


def inverse_logit(value):
    if value >= 0:
        exp_value = math.exp(-value)
        return 100.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return 100.0 * exp_value / (1.0 + exp_value)


def round4(value):
    return math.floor(value * 10000.0 + 0.5) / 10000.0


def forecast(case):
    if tuple(case["levels"]) != LEVELS:
        raise ValueError("case uses unsupported quantile levels")
    history = case["history"]
    if not history:
        raise ValueError("case has no historical reads")

    last_rates = history[-1]["rates"]
    spread = 0.5 * math.sqrt(max(0.0, case["horizon_minutes"] / 1440.0))
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        raw_rate = last_rates[index]
        center = logit_rate(0.01 if raw_rate is None else raw_rate)
        quantiles = [round4(inverse_logit(center + z * spread)) for z in NORMAL_Z]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(input_path.read_text(encoding="utf-8"))
        output = forecast(case)
        output_path = OUTPUT_DIR / (input_path.stem + ".out")
        output_path.write_text(
            json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
