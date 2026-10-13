# Derived from gnn/ver_autoregressive/output/forecast_activity_by_step.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def load_feature_names(csv_path):
    df = pd.read_csv(csv_path)
    if 'feature' not in df.columns:
        raise KeyError(f"{csv_path} missing 'feature' column")
    return df['feature'].astype(str).tolist()

def create_spatial_weights(height=192, width=192, boundary=8):
    lats_deg = np.linspace(0, 47.75, height)
    lat_w = np.cos(np.deg2rad(lats_deg))
    w2d = np.tile(lat_w[:, None], (1, width))
    w2d = np.flipud(w2d).copy()
    y_idx, x_idx = np.indices((height, width))
    dist_top = y_idx
    dist_bottom = height - 1 - y_idx
    dist_left = x_idx
    dist_right = width - 1 - x_idx
    min_dist = np.minimum.reduce([dist_top, dist_bottom, dist_left, dist_right])
    w2d[min_dist <= boundary] = 0.0
    mean_w = np.mean(w2d)
    if mean_w != 0:
        w2d = w2d / mean_w
    return w2d.astype(np.float64)

def parse_timestamp_from_filename(path, time_format='%Y%m%d%H'):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return datetime.strptime(name, time_format)
    except ValueError:
        pass
    digits = re.findall('(\\d{10})', name)
    for cand in digits:
        try:
            return datetime.strptime(cand, '%Y%m%d%H')
        except ValueError:
            continue
    return None

def parse_step_from_filename(path):
    name = os.path.basename(path)
    match = re.search('_step(\\d+)', name)
    if match:
        return int(match.group(1))
    match = re.search('step(\\d+)', name)
    return int(match.group(1)) if match else None

def load_pred_label_from_npz(npz_path, target_folder=None):
    data = np.load(npz_path)
    print(f'Loading npz: {npz_path}')
    keys = set(getattr(data, 'files', []))
    pred = None
    label = None
    for k_pred in ('forecast', 'pred', 'y_pred'):
        if k_pred in keys:
            pred = data[k_pred]
            break
    for k_label in ('target', 'label', 'y_true'):
        if k_label in keys:
            label = data[k_label]
            break
    if label is None and target_folder is not None:
        step_idx = parse_step_from_filename(npz_path)
        ts = parse_timestamp_from_filename(npz_path)
        candidate_name = os.path.basename(npz_path)
        if step_idx is not None and step_idx > 1 and (ts is not None):
            shifted_ts = ts + timedelta(hours=6 * (step_idx - 1))
            ts_str = shifted_ts.strftime('%Y%m%d%H')
            candidate_name = re.sub('\\d{10}', ts_str, candidate_name, count=1)
            candidate_name = re.sub('_step\\d+', '_step01', candidate_name, count=1)
        target_path = os.path.join(target_folder, candidate_name)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f'Missing target npz: {target_path}')
        tgt = np.load(target_path)
        tgt_keys = set(getattr(tgt, 'files', []))
        for k_label in ('target', 'label', 'y_true'):
            if k_label in tgt_keys:
                label = tgt[k_label]
                break
        if label is None:
            raise KeyError(f'{target_path} missing target/label/y_true')
    if pred is None or label is None:
        raise KeyError(f'{npz_path} missing forecast/pred or target/label')
    hw = None

    def reshape(arr, arr_name):
        nonlocal hw
        if arr.ndim == 4:
            batch, height, width, channels = arr.shape
            hw_here = (height, width)
            if hw is not None and hw != hw_here:
                raise ValueError(f'{npz_path} {arr_name} shape mismatch: {hw} vs {hw_here}')
            hw = hw_here
            return arr.reshape(batch * height * width, channels)
        if arr.ndim == 2:
            return arr
        raise ValueError(f'{npz_path} {arr_name} unsupported shape: {arr.shape}')
    pred = reshape(pred, 'pred')
    label = reshape(label, 'label')
    if pred.shape != label.shape:
        raise ValueError(f'{npz_path} pred/label mismatch: {pred.shape} vs {label.shape}')
    return (pred, label, hw)

def build_time_file_map(clim_dir):
    mapping = {}
    for path in sorted(glob.glob(os.path.join(clim_dir, 'time_*.npz'))):
        data = np.load(path)
        if 'time' not in data:
            continue
        ts = pd.to_datetime(data['time']).to_pydatetime()
        if isinstance(ts, np.ndarray):
            ts = ts.item()
        key = (ts.timetuple().tm_yday, ts.hour)
        mapping[key] = path
    return mapping

def load_clim_npz_for_time(clim_dir, ts, time_file_map=None):
    dayofyear = ts.timetuple().tm_yday
    hour = ts.hour
    path = os.path.join(clim_dir, f'day{dayofyear:03d}_hour{hour:02d}.npz')
    if os.path.exists(path):
        return (np.load(path), time_file_map)
    if time_file_map is None:
        time_file_map = build_time_file_map(clim_dir)
    path = time_file_map.get((dayofyear, hour))
    if path and os.path.exists(path):
        return (np.load(path), time_file_map)
    return (None, time_file_map)

def compute_activity_weighted(values, climatology, weights, epsilon=1e-12):
    f_anom = values - climatology
    weight_sum = np.sum(weights)
    if weight_sum <= epsilon:
        return np.nan
    f_mean = np.sum(weights * f_anom) / weight_sum
    f_prime = f_anom - f_mean
    f_var = np.sum(weights * f_prime ** 2)
    return np.sqrt(f_var / weight_sum)

def main(forecast_folder='.', target_folder=None, clim_folder='climate_meteology', feature_csv='next_feature_table_weight.csv', save_csv='per_channel_forecast_activity_summary.csv', per_file_csv='per_file_forecast_activity.csv', step_hours=6, boundary=8):
    feature_names = load_feature_names(feature_csv)
    npz_files = sorted(glob.glob(os.path.join(forecast_folder, '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'No npz files under {forecast_folder}')
    per_file_rows = []
    per_channel_target_activity = []
    per_step_target_activity = {}
    warnings = []
    time_file_map = None
    weights_cache = {}
    for npz_path in npz_files:
        _, label, _ = load_pred_label_from_npz(npz_path, target_folder=target_folder)
        step_idx = parse_step_from_filename(npz_path)
        ts = parse_timestamp_from_filename(npz_path)
        if ts is None:
            warnings.append(f'Skip {os.path.basename(npz_path)}: missing timestamp')
            continue
        if step_idx is not None:
            ts = ts + timedelta(hours=step_hours * step_idx)
        clim_data, time_file_map = load_clim_npz_for_time(clim_folder, ts, time_file_map)
        if clim_data is None:
            warnings.append(f'Skip {os.path.basename(npz_path)}: missing climatology file')
            continue
        bn, channels = label.shape
        target_activity_arr = np.full(channels, np.nan, dtype=np.float64)
        weights_flat = None
        for ch_idx in range(channels):
            feature = feature_names[ch_idx] if ch_idx < len(feature_names) else f'channel_{ch_idx:02d}'
            if feature not in clim_data:
                continue
            clim_arr = np.asarray(clim_data[feature]).squeeze()
            if clim_arr.ndim != 2:
                warnings.append(f'Skip {feature} in {os.path.basename(npz_path)}: invalid clim shape {clim_arr.shape}')
                continue
            clim_flat = clim_arr.reshape(-1)
            if weights_flat is None:
                height, width = clim_arr.shape
                if (height, width) not in weights_cache:
                    weights_cache[height, width] = create_spatial_weights(height=height, width=width, boundary=boundary).reshape(-1)
                weights_flat = weights_cache[height, width]
            node = clim_flat.size
            if node == 0 or bn % node != 0:
                warnings.append(f'Skip {feature} in {os.path.basename(npz_path)}: shape mismatch pred={bn}, clim={clim_arr.shape}')
                continue
            repeat = bn // node
            clim_rep = np.tile(clim_flat, repeat)
            weight_rep = np.tile(weights_flat, repeat)
            target_activity = compute_activity_weighted(label[:, ch_idx], clim_rep, weight_rep)
            target_activity_arr[ch_idx] = target_activity
        per_file_rows.append({'file': os.path.basename(npz_path), 'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'), 'step': step_idx if step_idx is not None else 0, 'target_activity_mean': float(np.nanmean(target_activity_arr)) if np.isfinite(target_activity_arr).any() else np.nan})
        per_channel_target_activity.append(target_activity_arr)
        if step_idx is not None:
            per_step_target_activity.setdefault(step_idx, []).append(target_activity_arr)
    if not per_channel_target_activity:
        raise RuntimeError('No valid files processed. Check climatology and timestamps.')
    per_channel_target_activity_arr = np.stack(per_channel_target_activity, axis=0)
    target_activity_mean = np.nanmean(per_channel_target_activity_arr, axis=0)
    channel_names = [feature_names[idx] if idx < len(feature_names) else f'channel_{idx:02d}' for idx in range(len(target_activity_mean))]
    summary_dict = {'feature': channel_names, 'target_activity_overall': target_activity_mean}
    for step_idx in sorted(per_step_target_activity.keys()):
        step_target_activity_arr = np.stack(per_step_target_activity[step_idx], axis=0)
        summary_dict[f'target_activity_step{step_idx:02d}'] = np.nanmean(step_target_activity_arr, axis=0)
    summary_df = pd.DataFrame(summary_dict)
    valid_mask = summary_df[['target_activity_overall']].notna().any(axis=1)
    summary_df = summary_df[valid_mask]
    summary_df.to_csv(save_csv, index=False)
    per_file_df = pd.DataFrame(per_file_rows).sort_values(['timestamp', 'file'])
    per_file_df.to_csv(per_file_csv, index=False)
    print('Saved per-channel summary to:', save_csv)
    print('Saved per-file summary to:', per_file_csv)
    if warnings:
        print('Warnings:')
        for msg in warnings:
            print(' -', msg)
