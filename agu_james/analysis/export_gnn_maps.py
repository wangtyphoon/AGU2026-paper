# Derived from vis_multi_steps/vis_muliti_step_ours_save.py
# See provenance/sources.json for original hash and modifications.
from __future__ import annotations
from agu_james import runtime
import re
from datetime import datetime, timedelta
from pathlib import Path
import torch
import numpy as np

def create_spatial_weights(H=192, W=192, boundary=8, device=torch.device('cpu')):
    """
    緯度餘弦加權 + 邊界距離≤boundary 的格點置 0。
    回傳 shape = (H, W) 的 torch.Tensor
    """
    lats_deg = np.linspace(0, 47.75, H)
    lat_w = np.cos(np.deg2rad(lats_deg))
    w2d = np.tile(lat_w[:, None], (1, W))
    w2d = np.flipud(w2d).copy()
    y_idx, x_idx = np.indices((H, W))
    dist_top = y_idx
    dist_bottom = H - 1 - y_idx
    dist_left = x_idx
    dist_right = W - 1 - x_idx
    min_dist = np.minimum.reduce([dist_top, dist_bottom, dist_left, dist_right])
    w2d[min_dist <= boundary] = 0.0
    w2d = w2d / np.mean(w2d)
    return torch.tensor(w2d, dtype=torch.float32, device=device)
TIME_TAG = '2024092712'
INPUT_DIR = runtime.setting('agu_james.analysis.export_gnn_maps.INPUT_DIR', lambda: Path(runtime.path('agu_james.analysis.export_gnn_maps.UnetGNNgate7_fix_attention_autoregressive_step28_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')))
TARGET_DIR = runtime.setting('agu_james.analysis.export_gnn_maps.TARGET_DIR', lambda: Path(runtime.path('agu_james.analysis.export_gnn_maps.UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')))
OUTPUT_DIR = runtime.setting('agu_james.analysis.export_gnn_maps.OUTPUT_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.export_gnn_maps')
STEP_PATTERN = f'*{TIME_TAG}*_step*.npz'
STEPS_TO_SAVE = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28]
CHANNELS = [53, 59, 74, 76, 85, 86, 87, 88, 115]
WIND_SPEED_CHANNELS = (86, 87)

def parse_timestamp_from_filename(path: Path, time_format: str='%Y%m%d%H'):
    name = path.stem
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

def parse_step_from_filename(path: Path):
    m = re.search('_step(\\d+)', path.name)
    if m:
        return int(m.group(1))
    m = re.search('step(\\d+)', path.name)
    return int(m.group(1)) if m else None

def load_forecast_and_target(path: Path):
    with np.load(path) as data:
        if 'forecast' not in data:
            raise KeyError(f"{path.name} is missing the 'forecast' array.")
        forecast = data['forecast']
        target = data.get('target')
    if target is None:
        step_idx = parse_step_from_filename(path)
        ts = parse_timestamp_from_filename(path)
        candidate_name = path.name
        if step_idx is not None and step_idx > 1 and (ts is not None):
            shifted_ts = ts + timedelta(hours=6 * (step_idx - 1))
            ts_str = shifted_ts.strftime('%Y%m%d%H')
            candidate_name = re.sub('\\d{10}', ts_str, candidate_name, count=1)
            candidate_name = re.sub('_step\\d+', '_step01', candidate_name, count=1)
        target_path = TARGET_DIR / candidate_name
        if not target_path.exists():
            raise FileNotFoundError(f'Missing target file: {target_path}')
        with np.load(target_path) as tgt:
            if 'target' not in tgt:
                raise KeyError(f"{target_path.name} is missing the 'target' array.")
            target = tgt['target']
    return (forecast, target)

def collect_step_files():
    step_files = sorted(INPUT_DIR.glob(STEP_PATTERN))
    if not step_files:
        raise FileNotFoundError(f'No {STEP_PATTERN} files found in {INPUT_DIR}')
    if STEPS_TO_SAVE:
        requested = {int(step) for step in STEPS_TO_SAVE}
        filtered = []
        missing = set(requested)
        for path in step_files:
            step_num = parse_step_from_filename(path)
            if step_num in requested:
                filtered.append(path)
                missing.discard(step_num)
        step_files = sorted(filtered, key=lambda p: (parse_step_from_filename(p) or 0, p.name))
        if missing:
            missing_text = ', '.join((str(step) for step in sorted(missing)))
            print(f'[Info] Missing step files for steps: {missing_text}')
        if not step_files:
            raise FileNotFoundError(f'No step files found for STEPS_TO_SAVE={sorted(requested)} in {INPUT_DIR}')
    return step_files

def get_layout(array: np.ndarray):
    if array.ndim == 1:
        if array.size != FLAT_SIZE:
            raise ValueError(f'Expected flattened array of size {FLAT_SIZE}, got {array.size}')
        return (1, 'flat')
    if array.ndim == 2:
        if array.shape[0] == FLAT_SIZE:
            return (array.shape[1], 'columns')
        if array.shape[1] == FLAT_SIZE:
            return (array.shape[0], 'rows')
    raise ValueError(f'Array shape {array.shape} does not match grid {GRID_SHAPE}.')

def normalize_index(index: int, count: int) -> int:
    if count <= 0:
        raise ValueError('Sample count must be positive.')
    if index < 0:
        index += count
    if not 0 <= index < count:
        raise IndexError(f'Sample index {index} out of range for {count} samples.')
    return index

def flatten_sample(array: np.ndarray, index: int, layout: str) -> np.ndarray:
    if layout == 'flat':
        return array
    if layout == 'columns':
        return array[:, index]
    if layout == 'rows':
        return array[index, :]
    raise ValueError(f'Unknown layout {layout!r}')

def reshape_grid(flat: np.ndarray) -> np.ndarray:
    if flat.size != FLAT_SIZE:
        raise ValueError(f'Cannot reshape array of size {flat.size} to {GRID_SHAPE}')
    return flat.reshape(GRID_SHAPE)

def load_channel_grid(array: np.ndarray, channel_index: int):
    sample_count, layout = get_layout(array)
    normalized_index = normalize_index(channel_index, sample_count)
    flat = flatten_sample(array, normalized_index, layout)
    return reshape_grid(flat)

def prepare_step_data(step_files, channel_index):
    prepared = []
    for step_id, path in enumerate(step_files, start=1):
        step_num = parse_step_from_filename(path) or step_id
        forecast_raw, target_raw = load_forecast_and_target(path)
        forecast = load_channel_grid(forecast_raw, channel_index)
        target = load_channel_grid(target_raw, channel_index)
        diff = forecast - target
        squared_error = diff ** 2
        weighted = squared_error * spatial_weights
        denom = spatial_weights.sum().clip(min=1e-12)
        mse = weighted.sum() / denom
        rmse = float(np.sqrt(mse))
        bias_mean = float((diff * spatial_weights).sum() / denom)
        diff_debiased = diff - bias_mean
        squared_error_debiased = diff_debiased ** 2
        mse_debiased = (squared_error_debiased * spatial_weights).sum() / denom
        rmse_debiased = float(np.sqrt(mse_debiased))
        prepared.append({'step_number': step_num, 'path': path, 'forecast': forecast, 'target': target, 'bias': diff, 'mse': squared_error, 'rmse': rmse, 'bias_mean': bias_mean, 'bias_debiased': diff_debiased, 'mse_debiased': squared_error_debiased, 'rmse_debiased': rmse_debiased})
    return prepared

def prepare_wind_speed_data(step_files, u_channel, v_channel):
    prepared = []
    for step_id, path in enumerate(step_files, start=1):
        step_num = parse_step_from_filename(path) or step_id
        forecast_raw, target_raw = load_forecast_and_target(path)
        forecast_u = load_channel_grid(forecast_raw, u_channel)
        forecast_v = load_channel_grid(forecast_raw, v_channel)
        target_u = load_channel_grid(target_raw, u_channel)
        target_v = load_channel_grid(target_raw, v_channel)
        forecast = np.sqrt(forecast_u ** 2 + forecast_v ** 2)
        target = np.sqrt(target_u ** 2 + target_v ** 2)
        diff = forecast - target
        squared_error = diff ** 2
        weighted = squared_error * spatial_weights
        denom = spatial_weights.sum().clip(min=1e-12)
        mse = weighted.sum() / denom
        rmse = float(np.sqrt(mse))
        bias_mean = float((diff * spatial_weights).sum() / denom)
        diff_debiased = diff - bias_mean
        squared_error_debiased = diff_debiased ** 2
        mse_debiased = (squared_error_debiased * spatial_weights).sum() / denom
        rmse_debiased = float(np.sqrt(mse_debiased))
        prepared.append({'step_number': step_num, 'path': path, 'forecast': forecast, 'target': target, 'bias': diff, 'mse': squared_error, 'rmse': rmse, 'bias_mean': bias_mean, 'bias_debiased': diff_debiased, 'mse_debiased': squared_error_debiased, 'rmse_debiased': rmse_debiased})
    return prepared

def save_steps(prepared, channel_label):
    if not prepared:
        print(f'[Info] No steps to save for {channel_label}.')
        return
    output_base = runtime.setting('agu_james.analysis.export_gnn_maps.output_base', lambda: OUTPUT_DIR / TIME_TAG)
    output_base.mkdir(parents=True, exist_ok=True)
    for step_info in prepared:
        step_dir = output_base / f"step{step_info['step_number']:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        out_path = step_dir / f'{channel_label}.npz'
        np.savez_compressed(out_path, forecast=step_info['forecast'], target=step_info['target'], bias=step_info['bias'], mse=step_info['mse'], rmse=np.array(step_info['rmse']), bias_mean=np.array(step_info['bias_mean']), bias_debiased=step_info['bias_debiased'], mse_debiased=step_info['mse_debiased'], rmse_debiased=np.array(step_info['rmse_debiased']))
    print(f'Saved {channel_label} data to {output_base}')

def main():
    step_files = collect_step_files()
    for channel_index in CHANNELS:
        prepared = prepare_step_data(step_files, channel_index)
        save_steps(prepared, channel_label=f'channel_{channel_index:03d}')
    if WIND_SPEED_CHANNELS:
        prepared = prepare_wind_speed_data(step_files, WIND_SPEED_CHANNELS[0], WIND_SPEED_CHANNELS[1])
        save_steps(prepared, channel_label=f'wind_speed_{WIND_SPEED_CHANNELS[0]:03d}_{WIND_SPEED_CHANNELS[1]:03d}')

def initialize():
    global FLAT_SIZE, GRID_SHAPE, lat, lon, spatial_weights
    lat = np.load(runtime.path('agu_james.analysis.export_gnn_maps.lat.npy'))
    lon = np.load(runtime.path('agu_james.analysis.export_gnn_maps.lon.npy'))
    GRID_SHAPE = lat.shape
    FLAT_SIZE = GRID_SHAPE[0] * GRID_SHAPE[1]
    if create_spatial_weights is None:
        spatial_weights = np.ones(GRID_SHAPE, dtype=np.float32)
    else:
        spatial_weights = create_spatial_weights(H=GRID_SHAPE[0], W=GRID_SHAPE[1], boundary=8)
        spatial_weights = np.array(spatial_weights, dtype=np.float32)
