#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates from distinct historical snapshots."""

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


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def pava(values, increasing=True):
    """Unweighted pool-adjacent-violators projection."""
    blocks = []
    sign = 1.0 if increasing else -1.0
    for value in values:
        blocks.append([sign * float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            count = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / count, count])
    out = []
    for value, count in blocks:
        out.extend([sign * value] * count)
    return out


def robust_line(xs, ys, fallback=0.0):
    """Theil-Sen slope with a median intercept, robust to a bad snapshot."""
    pairs = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    slope = median(pairs) if pairs else fallback
    slope = min(4.0, max(0.0, slope))
    intercept = median([y - slope * x for x, y in zip(xs, ys)])
    return slope, intercept


def round_earners(rate, players):
    return int(math.floor(float(rate) * players / 100.0 + 0.5))


def distinct_epochs(history):
    # Repeated reads with an unchanged tracked-player count and rate vector are
    # one epoch; retain the timestamp of the first read in that epoch.
    first = {}
    for row in history:
        players = int(row["players_tracked"])
        rates = tuple(row["rates"])
        key = (players, rates)
        stamp = parse_time(row.get("at") or row["slot"])
        if key not in first or stamp < first[key][0]:
            first[key] = (stamp, row)
    return sorted(first.values(), key=lambda item: item[0])


def trophy_class(trophy):
    tid = int(trophy["id"])
    name = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if tid == 1 or trophy.get("tier") == "platinum":
        return "platinum"
    if "chapter" in name or "final mission" in name or "final" in name and "mission" in name:
        return "chapter"
    if any(word in name for word in ("collectible", "collect", "card set", "stunt", "100% completion")):
        return "collectible"
    if "mission" in name or "complete " in name:
        return "mission"
    return "side"


def project_story(predictions, trophies):
    id_to_index = {int(t["id"]): i for i, t in enumerate(trophies)}
    inds = [id_to_index[i] for i in STORY_IDS if i in id_to_index]
    if len(inds) < 2:
        return
    fitted = pava([predictions[i] for i in inds], increasing=False)
    # Preserve each trophy's observed earners while keeping the chain ordered.
    lower = [predictions[i] for i in inds]
    for pos in range(len(lower) - 2, -1, -1):
        lower[pos] = max(lower[pos], lower[pos + 1])
    fitted = [max(value, floor) for value, floor in zip(fitted, lower)]
    for i, value in zip(inds, fitted):
        predictions[i] = value


def fit_predict(rows, trophies, target_time):
    """Fit the specified robust log-log trends and return predicted rates."""
    if not rows:
        raise ValueError("history has no distinct epochs")
    players = [int(row["players_tracked"]) for _, row in rows]
    rates = [row["rates"] for _, row in rows]
    times = [stamp for stamp, _ in rows]
    # Keep t positive around launch while preserving the release-relative clock.
    ages = [(stamp - RELEASE) / 3600.0 for stamp in times]
    target_age = (target_time - RELEASE) / 3600.0
    xs = [math.log(max(0.05, age + 1.0)) for age in ages]
    x_target = math.log(max(0.05, target_age + 1.0))

    player_logs = [math.log(max(1, p)) for p in players]
    player_logs = pava(player_logs, increasing=True)
    player_slope, player_intercept = robust_line(xs, player_logs, fallback=0.8)
    target_players = max(players[-1], math.exp(player_intercept + player_slope * x_target))

    earns = [[round_earners(rates[e][i], players[e]) for e in range(len(rows))]
             for i in range(len(trophies))]
    # B is the earliest tracked-638 snapshot; use the first epoch if absent.
    baseline_epoch = next((e for e, p in enumerate(players) if p == 638), 0)
    baseline = [earns[i][baseline_epoch] for i in range(len(trophies))]
    last_seen = [max(values) for values in earns]

    classes = [trophy_class(t) for t in trophies]
    raw_slopes = []
    raw_intercepts = []
    for i in range(len(trophies)):
        ys = [math.log(max(1, earns[i][e] - baseline[i])) for e in range(len(rows))]
        slope, intercept = robust_line(xs, ys, fallback=0.8)
        raw_slopes.append(slope)
        raw_intercepts.append(intercept)
    group_slopes = {}
    for group in set(classes):
        group_slopes[group] = median([s for s, c in zip(raw_slopes, classes) if c == group])

    predicted_earners = []
    for i in range(len(trophies)):
        # Partial pooling keeps sparse trophy histories near their trophy class.
        n = max(0, len(rows) - 1)
        weight = n / (n + 4.0) if n else 0.0
        slope = weight * raw_slopes[i] + (1.0 - weight) * group_slopes[classes[i]]
        intercept = median([
            math.log(max(1, earns[i][e] - baseline[i])) - slope * xs[e]
            for e in range(len(rows))
        ])
        delta = math.exp(min(700.0, intercept + slope * x_target))
        predicted_earners.append(max(float(last_seen[i]), baseline[i] + delta))
    project_story(predicted_earners, trophies)
    return [min(99.99, max(0.0, 100.0 * e / target_players)) for e in predicted_earners]


def logit_percent(rate):
    rate = min(99.99, max(0.01, rate)) / 100.0
    return math.log(rate / (1.0 - rate))


def inv_logit_percent(value):
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 100.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return 100.0 * z / (1.0 + z)


def forecast(case):
    if tuple(case["levels"]) != LEVELS:
        raise ValueError("unsupported quantile levels")
    epochs = distinct_epochs(case["history"])
    target_time = parse_time(case["target"])
    center_rates = fit_predict(epochs, case["trophies"], target_time)

    # Rolling-origin one-epoch-ahead errors estimate uncertainty without using
    # any answer file. Every fold refits only on the prefix available then.
    residuals = [[] for _ in case["trophies"]]
    for k in range(2, len(epochs)):
        predicted = fit_predict(epochs[:k], case["trophies"], epochs[k][0])
        stamp, row = epochs[k]
        for i, rate in enumerate(row["rates"]):
            actual = 100.0 * round_earners(rate, int(row["players_tracked"])) / int(row["players_tracked"])
            residuals[i].append(logit_percent(actual) - logit_percent(predicted[i]))

    gaps = [(epochs[i][0] - epochs[i - 1][0]) / 3600.0 for i in range(1, len(epochs))]
    step = median([g for g in gaps if g > 0]) if gaps else 1.0
    horizon = max(0.0, (target_time - epochs[-1][0]) / 3600.0)
    scale = math.sqrt(max(1.0, horizon / max(step, 1.0 / 60.0)))
    group_residuals = {}
    groups = [trophy_class(t) for t in case["trophies"]]
    for i, group in enumerate(groups):
        group_residuals.setdefault(group, []).extend(residuals[i])
    all_residuals = [x for values in residuals for x in values]

    outputs = []
    for i, trophy in enumerate(case["trophies"]):
        center = center_rates[i]
        if center < 0.01:
            quantiles = [0.01] * 5
        else:
            sample = residuals[i] or group_residuals.get(groups[i], []) or all_residuals
            if len(sample) >= 2:
                bias = median(sample)
                sigma = 1.4826 * median([abs(x - bias) for x in sample])
            elif sample:
                bias = sample[0]
                sigma = max(0.15, abs(bias))
            else:
                bias, sigma = 0.0, 0.25
            sigma = min(3.0, max(0.12, sigma * scale))
            c_logit = logit_percent(center) + bias
            quantiles = [round(inv_logit_percent(c_logit + z * sigma), 4) for z in NORMAL_Z]
            quantiles = [min(99.99, max(0.01, q)) for q in quantiles]
        outputs.append({"id": int(trophy["id"]), "q": quantiles})
    return {"trophies": outputs}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(input_path.read_text(encoding="utf-8"))
        output = forecast(case)
        output_path = OUTPUT_DIR / (input_path.stem + ".out")
        output_path.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
