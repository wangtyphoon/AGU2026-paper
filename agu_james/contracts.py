"""Explicit paper settings shared by data, training, validation and analysis."""
from datetime import datetime, timedelta
import math
import numpy as np

VALIDATION_YEARS = (1980, 1990, 2000, 2010, 2020)
INIT_HOURS = (0, 6, 12, 18)
LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
BOUNDARY_WIDTH = 9
EFFECTIVE_LENGTH_KM = 5376.0

def split_for_year(year):
    year = int(year)
    if year == 2024:
        return "test"
    if year in VALIDATION_YEARS:
        return "val"
    if 1979 <= year <= 2023:
        return "train"
    raise ValueError(f"Year outside paper population: {year}")

def valid_initializations(times, rollout_steps, hours=INIT_HOURS):
    """Require previous analysis and every future frame in the same year split."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    population = set(times)
    result = []
    for t in sorted(population):
        if t.hour not in hours:
            continue
        try:
            split = split_for_year(t.year)
            frames = [t + timedelta(hours=6 * k) for k in range(-1, rollout_steps + 1)]
            valid = all(f in population and split_for_year(f.year) == split for f in frames)
        except ValueError:
            valid = False
        if valid:
            result.append(t)
    return result

def boundary_mask(height=192, width=192, boundary_width=BOUNDARY_WIDTH):
    if not 0 <= boundary_width < min(height, width) / 2:
        raise ValueError("Boundary must leave a non-empty interior")
    y, x = np.indices((height, width))
    return np.minimum.reduce((y, x, height - 1 - y, width - 1 - x)) < boundary_width

def spatial_weights(latitudes, width=None, boundary_width=BOUNDARY_WIDTH):
    latitudes = np.asarray(latitudes, dtype=np.float64)
    if latitudes.ndim == 1:
        if width is None:
            raise ValueError("width required for 1-D latitude")
        latitudes = np.repeat(latitudes[:, None], width, axis=1)
    weights = np.cos(np.deg2rad(latitudes))
    weights[boundary_mask(*weights.shape, boundary_width)] = 0
    if not np.isfinite(weights).all() or weights.mean() <= 0:
        raise ValueError("Invalid spatial weights")
    return weights / weights.mean()

def accumulation_steps(effective_batch, micro_batch, world_size=1):
    divisor = micro_batch * world_size
    if min(effective_batch, micro_batch, world_size) < 1 or effective_batch % divisor:
        raise ValueError("effective_batch must be divisible by micro_batch * world_size")
    return effective_batch // divisor

def schedule(stage, rollout=1):
    stage = str(stage).lower()
    settings = {
        "1": {1: (16, 150, None)}, "2": {1: (16, 30, None)},
        "3": dict(zip((2, 4, 8, 12, 20, 28),
                      ((16, 4, None), (16, 2, None), (16, 1, None), (16, 1, None), (8, None, 1000), (8, None, 1000)))),
        "4": dict(zip((1, 2, 4, 8, 12, 20, 28),
                      ((16, 8, None), (16, 4, None), (16, 2, None), (16, 1, None), (16, 1, None), (8, None, 1000), (8, None, 1000)))),
    }
    try:
        batch, epochs, updates = settings[stage][rollout]
    except KeyError:
        raise ValueError(f"Unsupported stage/rollout: {stage}/{rollout}") from None
    lr = {"1": 1e-3, "2": 1e-4}.get(stage, 1e-5 if rollout <= 12 else 1e-6)
    return dict(stage=stage, rollout_steps=rollout, effective_batch=batch,
                epochs=epochs, max_optimizer_updates=updates, peak_lr=lr)

def learning_rate(update_index, total_updates, warmup_updates, peak_lr, floor_ratio=0.01):
    """The index counts completed optimizer updates; first update is index zero."""
    if not 0 <= update_index < total_updates or not 1 <= warmup_updates <= total_updates:
        raise ValueError("Invalid optimizer update/warm-up budget")
    if update_index < warmup_updates:
        return peak_lr * (update_index + 1) / warmup_updates
    progress = (update_index - warmup_updates) / max(1, total_updates - warmup_updates - 1)
    return peak_lr * (floor_ratio + (1 - floor_ratio) * (1 + math.cos(math.pi * progress)) / 2)

def cutoff_from_energy(k, forecast, target, threshold=0.9):
    k, forecast, target = map(lambda x: np.asarray(x, dtype=np.float64), (k, forecast, target))
    if k.ndim != 1 or forecast.shape != k.shape or target.shape != k.shape:
        raise ValueError("Expected aligned one-dimensional k/forecast/target")
    valid = (k > 0) & np.isfinite(forecast) & np.isfinite(target) & (target > 0)
    candidates = k[valid & (forecast <= threshold * target)]
    return float(candidates.min()) if candidates.size else float("nan")

def short_wavelength_mask(k):
    k = np.asarray(k)
    return (k > 0) & (k > EFFECTIVE_LENGTH_KM / 1000.0)
