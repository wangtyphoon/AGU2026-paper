# Derived from gnn/ver_autoregressive/output/spectral_threshold_summary.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import math
import os
from typing import Sequence
import numpy as np
import pandas as pd
try:
    from scipy import stats
except ImportError:
    stats = None
EPS = 1e-12
SQRT2 = math.sqrt(2.0)

def welch_t_from_summary(m1: np.ndarray, s1: np.ndarray, n1: int, m2: np.ndarray, s2: np.ndarray, n2: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Welch t-test using summary statistics."""
    s1_2, s2_2 = (s1 ** 2, s2 ** 2)
    se = np.sqrt(s1_2 / n1 + s2_2 / n2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_vals = np.where(se > 0, (m1 - m2) / se, 0.0)
    num = (s1_2 / n1 + s2_2 / n2) ** 2
    den = s1_2 ** 2 / (n1 ** 2 * (n1 - 1)) + s2_2 ** 2 / (n2 ** 2 * (n2 - 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        df = np.where(den > 0, num / den, np.inf)
    if stats is not None:
        p_vals = 2 * stats.t.sf(np.abs(t_vals), df)
    else:
        abs_t = np.abs(t_vals)
        flat = [math.erfc(val / SQRT2) for val in abs_t.flat]
        p_vals = np.asarray(flat, dtype=abs_t.dtype).reshape(abs_t.shape)
    return (t_vals, df, p_vals)

def compute_power_ratio(mean_fore: np.ndarray, std_fore: np.ndarray, mean_targ: np.ndarray, std_targ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute energy ratio Forecast/Target (inputs are energy spectra)."""
    energy_fore = np.clip(mean_fore, EPS, None)
    energy_targ = np.clip(mean_targ, EPS, None)
    ratio = np.clip(energy_fore / np.clip(energy_targ, EPS, None), EPS, None)
    rel_fore = np.clip(np.clip(std_fore, 0.0, None) / np.clip(energy_fore, EPS, None), 0.0, None)
    rel_targ = np.clip(np.clip(std_targ, 0.0, None) / np.clip(energy_targ, EPS, None), 0.0, None)
    s_ratio = ratio * np.sqrt(rel_fore ** 2 + rel_targ ** 2)
    return (ratio, s_ratio)

def find_first_k(k_vals: np.ndarray, mask: np.ndarray) -> float:
    """Return smallest k where mask is True; NaN if never satisfied."""
    matched = k_vals[mask]
    return float(matched.min()) if matched.size else float('nan')

def locate_feature_csv(candidates: Sequence[str]) -> str:
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Feature CSV not found. Tried: {', '.join(candidates)}")

def resolve_channel_indices(features: list[str], keywords: Sequence[str] | None) -> list[int]:
    if not keywords:
        return list(range(len(features)))
    indices: set[int] = set()
    for key in keywords:
        indices.update((i for i, name in enumerate(features) if key in name))
    return sorted(indices)

def main(npz_path: str='per_channel_energy.npz', feature_csv_candidates: Sequence[str]=('../next_feature_table_weight.csv', 'next_feature_table_weight.csv', 'next_feature_weight.csv'), output_csv: str='spectral_threshold_summary.csv', keywords: Sequence[str] | None=None, n_fore: int=16000, n_targ: int=16000, sigma_thresholds: Sequence[int]=(1, 2, 3), significance_level: float=0.01, amplitude_drop: float=0.1, energy_drop: float=0.1) -> None:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'NPZ not found: {npz_path}')
    data = np.load(npz_path)
    required_base = ['forecast_mean', 'forecast_std', 'target_mean', 'target_std']
    suffix = ''
    if all((key in data for key in required_base)):
        suffix = ''
    else:
        suffixes: list[str] = []
        for key in data.keys():
            if key.startswith('forecast_mean') and key != 'forecast_mean':
                cand = key[len('forecast_mean'):]
                if all((base + cand in data for base in required_base)):
                    suffixes.append(cand)
        if not suffixes:
            raise KeyError(f'Missing keys {required_base} (or suffixed variants) in {npz_path}')
        if len(suffixes) > 1:
            raise ValueError(f'Multiple suffixes found {suffixes} in {npz_path}; please keep a single step in the NPZ or adjust the script.')
        suffix = suffixes[0]
    k_vals = np.asarray(data['k']).ravel()
    mean_fore = np.asarray(data[f'forecast_mean{suffix}'])
    std_fore = np.asarray(data[f'forecast_std{suffix}'])
    mean_targ = np.asarray(data[f'target_mean{suffix}'])
    std_targ = np.asarray(data[f'target_std{suffix}'])
    if not mean_fore.shape == mean_targ.shape == std_fore.shape == std_targ.shape:
        raise ValueError('Forecast/target mean/std shapes do not align')
    num_channels, num_k = mean_fore.shape
    if k_vals.shape[0] < num_k:
        num_k = k_vals.shape[0]
        mean_fore = mean_fore[:, :num_k]
        std_fore = std_fore[:, :num_k]
        mean_targ = mean_targ[:, :num_k]
        std_targ = std_targ[:, :num_k]
    elif k_vals.shape[0] > num_k:
        k_vals = k_vals[:num_k]
    feature_csv = locate_feature_csv(feature_csv_candidates)
    feature_series = pd.read_csv(feature_csv)['feature'].astype(str)
    features = feature_series.tolist()
    if len(features) < num_channels:
        features.extend([f'ch_{i}' for i in range(len(features), num_channels)])
    elif len(features) > num_channels:
        features = features[:num_channels]
    mask_positive_k = k_vals > 0
    k_positive = k_vals[mask_positive_k]
    channel_indices = resolve_channel_indices(features, keywords)
    rows: list[dict[str, float | int | str]] = []
    for c in channel_indices:
        feature_name = features[c]
        mean_fore_c = mean_fore[c]
        std_fore_c = std_fore[c]
        mean_targ_c = mean_targ[c]
        std_targ_c = std_targ[c]
        mean_fore_pos = mean_fore_c[mask_positive_k]
        std_fore_pos = std_fore_c[mask_positive_k]
        mean_targ_pos = mean_targ_c[mask_positive_k]
        std_targ_pos = std_targ_c[mask_positive_k]
        diff_pos = mean_fore_pos - mean_targ_pos
        target_std_pos = np.maximum(std_targ_pos, EPS)
        row: dict[str, float | int | str] = {'channel': c, 'feature': feature_name}
        for n in sigma_thresholds:
            exceed_mask = np.abs(diff_pos) > n * target_std_pos
            row[f'k_ge_{n}sigma'] = find_first_k(k_positive, exceed_mask)
        _, _, p_vals = welch_t_from_summary(mean_targ_pos, std_targ_pos, n_targ, mean_fore_pos, std_fore_pos, n_fore)
        row['k_sig_p01'] = find_first_k(k_positive, p_vals < significance_level)
        energy_ratio, _ = compute_power_ratio(mean_fore_pos, std_fore_pos, mean_targ_pos, std_targ_pos)
        amp_ratio = np.sqrt(np.clip(energy_ratio, EPS, None))
        amp_threshold = 1.0 - amplitude_drop
        row['k_amp_decay_10pct'] = find_first_k(k_positive, amp_ratio <= amp_threshold)
        energy_threshold = 1.0 - energy_drop
        row['k_energy_decay_10pct'] = find_first_k(k_positive, energy_ratio <= energy_threshold)
        rows.append(row)
    column_order = ['channel', 'feature', *[f'k_ge_{n}sigma' for n in sigma_thresholds], 'k_sig_p01', 'k_amp_decay_10pct', 'k_energy_decay_10pct']
    df = pd.DataFrame(rows, columns=column_order)
    df.to_csv(output_csv, index=False)
    print(f'Wrote {output_csv} with {len(df)} rows')
