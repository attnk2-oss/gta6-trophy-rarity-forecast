#!/usr/bin/env python3
"""Launch-aware GTA VI trophy earn-rate forecaster."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
NORMAL_Z = (-1.644853626951, -0.674489750196, 0.0, 0.674489750196, 1.644853626951)
STORY_IDS = (3, 4, 5, 6, 7)
LAUNCH_MULTIPLIER = {"chapter": 1.25, "mission": 1.25, "side": 1.1,
                     "collectible": 1.1, "platinum": 1.1}
RATE_CAP = {"chapter": 90.0, "mission": 75.0, "side": 55.0,
            "collectible": 45.0, "platinum": 15.0}


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    m = len(values) // 2
    return values[m] if len(values) % 2 else (values[m - 1] + values[m]) / 2.0


def pava(values, increasing=True):
    sign = 1.0 if increasing else -1.0
    blocks = []
    for value in values:
        blocks.append([sign * float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            count = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / count, count])
    result = []
    for value, count in blocks:
        result.extend([sign * value] * count)
    return result


def parse_epochs(history):
    first = {}
    for row in history:
        key = (int(row["players_tracked"]), tuple(row["rates"]))
        stamp = parse_time(row.get("at") or row["slot"])
        if key not in first or stamp < first[key][0]:
            first[key] = (stamp, row)
    return sorted(first.values(), key=lambda pair: pair[0])


def earners(rate, players):
    return int(math.floor(float(rate) * players / 100.0 + 0.5))


def trophy_class(trophy):
    tid = int(trophy["id"])
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if tid == 1 or trophy.get("tier") == "platinum":
        return "platinum"
    if tid in (3, 4, 5, 6, 7) or "chapter" in text or "final mission" in text:
        return "chapter"
    if any(word in text for word in ("collectible", "collect", "card set", "stunt", "100% completion")):
        return "collectible"
    if "mission" in text or "complete " in text:
        return "mission"
    return "side"


def robust_power(xs, ys, fallback=0.5, slope_cap=1.6):
    # Use only the most recent four distinct epochs. Pairwise slopes over a
    # launch-wide window let early zeros dominate long-horizon extrapolation.
    start = max(0, len(xs) - 4)
    xx, yy = xs[start:], ys[start:]
    slopes = [(yy[j] - yy[i]) / (xx[j] - xx[i])
              for i in range(len(xx)) for j in range(i + 1, len(xx))
              if xx[j] != xx[i]]
    slope = median(slopes) if slopes else fallback
    slope = min(slope_cap, max(0.0, slope))
    intercept = median([y - slope * x for x, y in zip(xx, yy)])
    return slope, intercept


def saturating_quadratic(xs, ys, x_target, fallback=0.5):
    """Recent log-count quadratic with non-positive curvature."""
    start = max(0, len(xs) - 4)
    xx, yy = xs[start:], ys[start:]
    if len(xx) < 3 or len(set(xx)) < 3:
        slope, intercept = robust_power(xx, yy, fallback=fallback, slope_cap=1.25)
        prediction = intercept + slope * x_target if xx else 0.0
        return max(yy[-1], prediction) if xx else 0.0, slope, 0.0

    n = float(len(xx))
    sums = [sum(x ** p for x in xx) for p in range(5)]
    sy = sum(yy)
    sxy = sum(x * y for x, y in zip(xx, yy))
    sx2y = sum(x * x * y for x, y in zip(xx, yy))
    matrix = [
        [n, sums[1], sums[2], sy],
        [sums[1], sums[2], sums[3], sxy],
        [sums[2], sums[3], sums[4], sx2y],
    ]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(matrix[r][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            slope, intercept = robust_power(xx, yy, fallback=fallback, slope_cap=1.25)
            return max(yy[-1], intercept + slope * x_target), slope, 0.0
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        matrix[col] = [v / divisor for v in matrix[col]]
        for r in range(3):
            if r == col:
                continue
            factor = matrix[r][col]
            matrix[r] = [a - factor * b for a, b in zip(matrix[r], matrix[col])]
    a, b, c = (matrix[i][3] for i in range(3))
    c = min(0.0, c)
    if c == 0.0:
        b, a = robust_power(xx, yy, fallback=fallback, slope_cap=1.25)
    else:
        # Refit the intercept and non-negative linear term with fixed curvature.
        residual = [y - c * x * x for x, y in zip(xx, yy)]
        b, a = robust_power(xx, residual, fallback=fallback, slope_cap=1.25)
    prediction = max(yy[-1], a + b * x_target + c * x_target * x_target)
    return prediction, b, c


def project_story(prediction, trophies):
    by_id = {int(t["id"]): i for i, t in enumerate(trophies)}
    indices = [by_id[i] for i in STORY_IDS if i in by_id]
    if len(indices) < 2:
        return
    fitted = pava([prediction[i] for i in indices], increasing=False)
    lower = [prediction[i] for i in indices]
    for j in range(len(lower) - 2, -1, -1):
        lower[j] = max(lower[j], lower[j + 1])
    for i, value, floor in zip(indices, fitted, lower):
        prediction[i] = max(value, floor)


def launch_forecast(epoch, trophies, horizon_hours):
    stamp, row = epoch[0]
    players = int(row["players_tracked"])
    # The explicit 90-minute prior is a six-fold tracked-population refresh.
    player_factor = 6.0 ** min(1.0, max(0.0, horizon_hours) / 1.5)
    target_players = players * player_factor
    predicted = []
    for i, trophy in enumerate(trophies):
        cls = trophy_class(trophy)
        observed = earners(row["rates"][i], players)
        projected = observed * LAUNCH_MULTIPLIER[cls]
        predicted.append(max(observed, projected))
    project_story(predicted, trophies)
    rates = [min(RATE_CAP[trophy_class(t)], 100.0 * e / target_players)
             for e, t in zip(predicted, trophies)]
    return rates


def trend_models(epochs, trophies, target_time):
    players = [int(row["players_tracked"]) for _, row in epochs]
    ages = [(stamp - RELEASE) / 3600.0 for stamp, _ in epochs]
    xs = [math.log(max(0.05, age + 1.0)) for age in ages]
    target_age = (target_time - RELEASE) / 3600.0
    x_target = math.log(max(0.05, target_age + 1.0))
    p_slope, p_intercept = robust_power(xs, [math.log(max(1, p)) for p in players],
                                        fallback=0.6, slope_cap=1.25)
    p_power = max(players[-1], math.exp(p_intercept + p_slope * x_target))
    p_sat_log, _, _ = saturating_quadratic(
        xs, [math.log(max(1, p)) for p in players], x_target, fallback=0.6)
    p_sat = max(players[-1], math.exp(p_sat_log))

    matrix = [[earners(row["rates"][i], players[e]) for e, (_, row) in enumerate(epochs)]
              for i in range(len(trophies))]
    baseline_epoch = next((e for e, p in enumerate(players) if p == 638), 0)
    result_power, result_sat = [], []
    for i, trophy in enumerate(trophies):
        history_e = [max(matrix[i][:e + 1]) for e in range(len(epochs))]
        base = matrix[i][baseline_epoch]
        ys = [math.log(max(1, value - base)) for value in history_e]
        cls = trophy_class(trophy)
        slope_cap = {"chapter": 1.3, "mission": 1.05, "side": 0.8,
                     "collectible": 0.75, "platinum": 0.65}[cls]
        slope, intercept = robust_power(xs, ys, fallback=0.35, slope_cap=slope_cap)
        last = float(max(history_e))
        projected_power = max(last, base + math.exp(min(700.0, intercept + slope * x_target)))
        sat_log, _, _ = saturating_quadratic(xs, ys, x_target, fallback=0.35)
        projected_sat = max(last, base + math.exp(min(700.0, sat_log)))
        result_power.append(projected_power)
        result_sat.append(projected_sat)
    project_story(result_power, trophies)
    project_story(result_sat, trophies)
    power_rates = [min(RATE_CAP[trophy_class(t)], 100.0 * e / p_power)
                   for e, t in zip(result_power, trophies)]
    sat_rates = [min(RATE_CAP[trophy_class(t)], 100.0 * e / p_sat)
                 for e, t in zip(result_sat, trophies)]
    center = [math.sqrt(max(0.0001, a) * max(0.0001, b)) for a, b in zip(power_rates, sat_rates)]
    return center, power_rates, sat_rates


def logit(rate):
    p = min(99.99, max(0.01, rate)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 100.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return 100.0 * z / (1.0 + z)


def forecast(case):
    if tuple(case["levels"]) != LEVELS:
        raise ValueError("unsupported quantile levels")
    epochs = parse_epochs(case["history"])
    target_time = parse_time(case["target"])
    horizon_hours = max(0.0, (target_time - epochs[-1][0]) / 3600.0)
    launch_mode = len(epochs) == 1 and horizon_hours <= 6.0
    if launch_mode:
        center = launch_forecast(epochs, case["trophies"], horizon_hours)
        power_rates, sat_rates = center, center
        for i, trophy in enumerate(case["trophies"]):
            if int(trophy["id"]) == 2:
                center[i] = power_rates[i] = sat_rates[i] = 99.99
            elif int(trophy["id"]) == 3:
                observed = earners(epochs[-1][1]["rates"][i], int(epochs[-1][1]["players_tracked"]))
                center[i] = power_rates[i] = sat_rates[i] = min(90.0, 100.0 * observed * 2.4 /
                    (int(epochs[-1][1]["players_tracked"]) * 6.0))
    else:
        center, power_rates, sat_rates = trend_models(epochs, case["trophies"], target_time)

    residuals = [[] for _ in case["trophies"]]
    for k in range(2, len(epochs)):
        prefix = epochs[:k]
        predicted, _, _ = trend_models(prefix, case["trophies"], epochs[k][0])
        row = epochs[k][1]
        p = int(row["players_tracked"])
        for i, rate in enumerate(row["rates"]):
            actual = 100.0 * earners(rate, p) / p
            residuals[i].append(logit(actual) - logit(predicted[i]))

    gaps = [(epochs[i][0] - epochs[i - 1][0]) / 3600.0 for i in range(1, len(epochs))]
    step = median([g for g in gaps if g > 0]) if gaps else 1.0
    scale = math.sqrt(max(1.0, horizon_hours / max(step, 1.0 / 60.0)))
    all_errors = [e for series in residuals for e in series]
    outputs = []
    for i, trophy in enumerate(case["trophies"]):
        rate = center[i]
        if rate < 0.02:
            quantiles = [0.01] * 5
        else:
            sample = residuals[i] or all_errors
            if len(sample) >= 2:
                mid = median(sample)
                observed_sigma = 1.4826 * median([abs(x - mid) for x in sample])
            elif sample:
                observed_sigma = abs(sample[0])
            else:
                observed_sigma = 0.30 if launch_mode else 0.40
            model_diff = abs(logit(power_rates[i]) - logit(sat_rates[i]))
            sigma = 0.30 if launch_mode else max(0.35, observed_sigma * scale) + model_diff
            sigma = min(3.0, sigma)
            c = logit(rate)
            quantiles = [round(inv_logit(c + z * sigma), 4) for z in NORMAL_Z]
            quantiles = [min(99.99, max(0.01, q)) for q in quantiles]
        outputs.append({"id": int(trophy["id"]), "q": quantiles})
    return {"trophies": outputs}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(input_path.read_text(encoding="utf-8"))
        output = forecast(case)
        (OUTPUT_DIR / (input_path.stem + ".out")).write_text(
            json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
