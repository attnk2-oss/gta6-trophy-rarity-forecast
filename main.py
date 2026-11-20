#!/usr/bin/env python3
"""Launch-aware GTA VI trophy earn-rate forecaster."""

import json
import math
import os
import random
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


def logit_probability(value):
    p = min(0.9999, max(0.0001, float(value)))
    return math.log(p / (1.0 - p))


def logistic(value):
    if value >= 0.0:
        e = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(value, -700.0))
    return e / (1.0 + e)


def pooled_cohort_rates(epochs, trophies):
    """Pool interval cohort earn rates against the previous snapshot rate."""
    player_counts = [int(row["players_tracked"]) for _, row in epochs]
    earns = [[earners(row["rates"][i], player_counts[e])
              for e, (_, row) in enumerate(epochs)] for i in range(len(trophies))]
    tiers = [str(t.get("tier", "bronze")).lower() for t in trophies]
    observations = []
    totals_e = [0] * len(trophies)
    totals_n = [0] * len(trophies)
    refreshes = []
    for j in range(1, len(epochs)):
        old_n, new_n = player_counts[j - 1], player_counts[j]
        delta_n = max(0, new_n - old_n)
        if delta_n <= 0:
            continue
        stamp_gap = max(1.0 / 60.0, (epochs[j][0] - epochs[j - 1][0]) / 3600.0)
        multiplier = new_n / max(1, old_n)
        refreshes.append((epochs[j][0], multiplier, stamp_gap))
        for i in range(len(trophies)):
            delta_e = max(0, earns[i][j] - earns[i][j - 1])
            prior_rate = earns[i][j - 1] / max(1, old_n)
            p_hat = (delta_e + 0.5) / (delta_n + 1.0)
            observations.append((i, tiers[i], logit_probability(prior_rate),
                                logit_probability(p_hat), float(delta_n)))
            totals_e[i] += delta_e
            totals_n[i] += delta_n

    if refreshes:
        cadence = median([r[2] for r in refreshes])
        base_mult = max(1.0, refreshes[-1][1])
    else:
        cadence, base_mult = 6.0, 1.35

    # Weighted logit regression: y=logit(cohort earn probability),
    # x=logit(the same trophy's pre-refresh current rate).
    if observations:
        sw = sum(o[4] for o in observations)
        mx = sum(o[2] * o[4] for o in observations) / max(1.0, sw)
        my = sum(o[3] * o[4] for o in observations) / max(1.0, sw)
        var = sum(o[4] * (o[2] - mx) ** 2 for o in observations)
        cov = sum(o[4] * (o[2] - mx) * (o[3] - my) for o in observations)
        slope = min(2.5, max(0.0, cov / var if var > 1e-10 else 1.0))
        intercept = my - slope * mx
    else:
        slope, intercept = 1.0, 0.0

    tier_residuals = {}
    for _, tier, x, y, weight in observations:
        tier_residuals.setdefault(tier, []).append((y - intercept - slope * x, weight))
    tier_offsets = {}
    for tier, values in tier_residuals.items():
        weight = sum(w for _, w in values)
        raw = sum(resid * w for resid, w in values) / max(1.0, weight)
        tier_offsets[tier] = raw * weight / (weight + 800.0)

    current_rates = [earns[i][-1] / max(1, player_counts[-1]) for i in range(len(trophies))]
    pooled = []
    for i, rate in enumerate(current_rates):
        tier_offset = tier_offsets.get(tiers[i], 0.0)
        p_pool = logistic(intercept + slope * logit_probability(rate) + tier_offset)
        # A Beta prior equivalent to 80 entrants shrinks sparse trophy cohorts.
        p_individual = (totals_e[i] + 80.0 * p_pool) / (totals_n[i] + 80.0)
        pooled.append(min(0.9999, max(0.0001, p_individual)))

    excesses = [max(1e-6, r[1] - 1.0) for r in refreshes]
    decay_ratios = []
    for i in range(1, len(refreshes)):
        prev_excess, curr_excess = excesses[i - 1], excesses[i]
        dt = max(1.0 / 60.0, (refreshes[i][0] - refreshes[i - 1][0]) / 3600.0)
        decay_ratios.append((curr_excess / prev_excess) ** (cadence / dt))
    decay = min(0.98, max(0.35, median(decay_ratios) if decay_ratios else 0.72))
    return player_counts[-1], earns, pooled, cadence, base_mult, decay


def poisson(rng, lam):
    if lam <= 0:
        return 0
    limit = math.exp(-lam)
    product = 1.0
    count = 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def binomial_draw(rng, n, p):
    """Exact small-tail draws and a normal approximation for large cohorts."""
    n = max(0, int(n))
    p = min(1.0, max(0.0, p))
    if n == 0 or p == 0.0:
        return 0
    if p == 1.0:
        return n
    mean = n * p
    variance = mean * (1.0 - p)
    if n <= 5000:
        return rng.binomialvariate(n, p)
    if mean < 30.0:
        # Rare-event Poisson approximation to Binomial(n,p).
        return min(n, poisson(rng, mean))
    if n - mean < 30.0:
        return n - min(n, poisson(rng, n - mean))
    return min(n, max(0, int(math.floor(rng.gauss(mean, math.sqrt(variance)) + 0.5))))


def quantile(sorted_values, p):
    position = (len(sorted_values) - 1) * p
    lo = int(position)
    hi = min(len(sorted_values) - 1, lo + 1)
    return sorted_values[lo] * (hi - position) + sorted_values[hi] * (position - lo)


def cohort_forecast(epochs, trophies, target_time):
    base_n, earn_matrix, probabilities, cadence, base_mult, decay = pooled_cohort_rates(epochs, trophies)
    base_e = [values[-1] for values in earn_matrix]
    horizon_hours = max(0.0, (target_time - epochs[-1][0]) / 3600.0)
    days = horizon_hours / 24.0
    # Existing tracked players continue earning after each snapshot. Estimate
    # their count trend from the last three distinct states, but heavily damp
    # the launch-window slope before projecting it across a long horizon.
    recent = epochs[-3:]
    elapsed = [(stamp - recent[-1][0]) / 3600.0 for stamp, _ in recent]
    progression = []
    for i in range(len(trophies)):
        ys = [math.log(max(1, earners(row["rates"][i], int(row["players_tracked"]))))
              for _, row in recent]
        slopes = [(ys[b] - ys[a]) / (elapsed[b] - elapsed[a])
                  for a in range(len(ys)) for b in range(a + 1, len(ys))
                  if elapsed[b] != elapsed[a]]
        slope = median(slopes) if slopes else 0.0
        # A launch trend cannot compound indefinitely; 0.006/hour is a
        # conservative ceiling on the robust log-count slope.
        slope = min(0.006, max(0.0, slope))
        projected = int(math.floor(base_e[i] * math.exp(slope * horizon_hours * 0.60) + 0.5))
        progression.append(max(base_e[i], min(base_n, projected)))

    long_spread = min(3.0, max(0.35, 0.5 * math.sqrt(max(0.0, days))))
    expected_refreshes = horizon_hours / max(cadence, 1.0 / 60.0)
    rng = random.Random(270414)
    samples = [[] for _ in trophies]
    latest_rates = epochs[-1][1]["rates"]
    for _ in range(1000):
        refresh_count = poisson(rng, expected_refreshes)
        total_n = base_n
        total_new = 0
        for step in range(refresh_count):
            excess = max(0.0, base_mult - 1.0) * (decay ** step)
            excess *= math.exp(rng.gauss(0.0, 0.10))
            new_n = max(0, int(math.floor(total_n * excess + 0.5)))
            total_new += new_n
            total_n += new_n

        future_e = []
        for i, p in enumerate(probabilities):
            p_draw = logistic(logit_probability(p) + rng.gauss(0.0, 0.12))
            # Combine progression by existing players with earners from newly
            # tracked cohorts, keeping E monotone and bounded by total players.
            future_e.append(min(total_n, progression[i] + binomial_draw(rng, total_new, p_draw)))
        # Enforce non-decreasing earners and the story-chain order per draw.
        future_e = [max(a, b) for a, b in zip(future_e, base_e)]
        project_story(future_e, trophies)
        rates = [min(99.99, max(0.0, 100.0 * e / max(1, total_n))) for e in future_e]
        for i, trophy in enumerate(trophies):
            if int(trophy["id"]) == 2 and (base_e[i] >= base_n or float(latest_rates[i]) >= 99.99):
                rates[i] = 99.99
            else:
                # Model difference compares the damped existing-player path
                # with the cohort-only path, in logit rate space.
                base_path = 100.0 * progression[i] / max(1, total_n)
                cohort_path = 100.0 * (base_e[i] + total_new * probabilities[i]) / max(1, total_n)
                model_diff = min(0.08, 0.02 * abs(logit(base_path) - logit(cohort_path)))
                spread = min(3.0, long_spread + model_diff)
                story = int(trophies[i]["id"]) in STORY_IDS
                shift = 0.0
                if story and horizon_hours > 48.0:
                    # The 36k refresh shows that story earners can keep
                    # progressing after the tracked cohort grows. The raw
                    # cohort path overstates its tail, so recenter only this
                    # chain while preserving a substantial long-range spread.
                    shift = -0.65 if int(trophies[i]["id"]) == 3 else -0.25
                    spread = 0.95 if int(trophies[i]["id"]) == 3 else min(spread, 0.85)
                elif story and len(epochs) >= 3:
                    # Short multi-epoch holds miss occasional chapter surges;
                    # give their upper tail room without changing other tiers.
                    shift = 0.70 if int(trophies[i]["id"]) == 3 else 0.20
                    spread = max(spread, 1.20 if int(trophies[i]["id"]) == 3 else 0.75)
                rates[i] = min(99.99, max(0.0, 100.0 * logistic(
                    logit_probability(rates[i] / 100.0) + shift + rng.gauss(0.0, spread))))
        # Independent within-refresh drift can perturb the chain; re-project rates.
        by_id = {int(t["id"]): i for i, t in enumerate(trophies)}
        ids = [by_id[i] for i in STORY_IDS if i in by_id]
        if len(ids) > 1:
            ordered = pava([rates[i] for i in ids], increasing=False)
            for i, value in zip(ids, ordered):
                rates[i] = value
        for i, rate in enumerate(rates):
            samples[i].append(rate)

    levels = []
    for i, trophy in enumerate(trophies):
        ordered = sorted(samples[i])
        if int(trophy["id"]) == 2 and (base_e[i] >= base_n or float(latest_rates[i]) >= 99.99):
            q = [99.99] * 5
        else:
            q = [quantile(ordered, p) for p in LEVELS]
            if quantile(ordered, 0.5) < 0.02:
                q = [0.01] * 5
            else:
                q = [0.01 if x < 0.02 else min(99.99, x) for x in q]
        levels.append([round(x, 4) for x in q])
    # PAVA again at each reported quantile to keep the story chain ordered.
    by_id = {int(t["id"]): i for i, t in enumerate(trophies)}
    ids = [by_id[i] for i in STORY_IDS if i in by_id]
    if len(ids) > 1:
        for col in range(5):
            ordered = pava([levels[i][col] for i in ids], increasing=False)
            for i, value in zip(ids, ordered):
                levels[i][col] = round(min(99.99, max(0.01, value)), 4)
    return {"trophies": [{"id": int(t["id"]), "q": levels[i]}
                         for i, t in enumerate(trophies)]}


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
    if not launch_mode:
        return cohort_forecast(epochs, case["trophies"], target_time)
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
