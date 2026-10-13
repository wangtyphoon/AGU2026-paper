# Derived from gnn/ver_autoregressive/output/tc_bst_msl_intensity.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import re
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
BASE_DIR = runtime.setting('agu_james.analysis.gnn.tc_base.BASE_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.tc_base')
CSV_PATH = runtime.setting('agu_james.analysis.gnn.tc_base.CSV_PATH', lambda: BASE_DIR / 'cyclone_data_2024.csv')
NPZ_DIR = runtime.setting('agu_james.analysis.gnn.tc_base.NPZ_DIR', lambda: BASE_DIR / 'UnetGNNgate7_fix_attention_step12_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
TARGET_DIR = runtime.setting('agu_james.analysis.gnn.tc_base.TARGET_DIR', lambda: BASE_DIR / 'UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
FEATURE_TABLE = runtime.setting('agu_james.analysis.gnn.tc_base.FEATURE_TABLE', lambda: BASE_DIR / 'next_feature_table_weight.csv')
LAT_NPY = runtime.setting('agu_james.analysis.gnn.tc_base.LAT_NPY', lambda: BASE_DIR / 'lat.npy')
LON_NPY = runtime.setting('agu_james.analysis.gnn.tc_base.LON_NPY', lambda: BASE_DIR / 'lon.npy')
OUTPUT_CSV = runtime.setting('agu_james.analysis.gnn.tc_base.OUTPUT_CSV', lambda: BASE_DIR / 'tc_bst_msl_intensity35_step12.csv')
STEP_HOURS = 6
WINDOW_RADIUS_LAT = 12
WINDOW_RADIUS_LON = 16
CENTROID_RADIUS = 6
INTENSITY_RADIUS = 6
RECURSIVE_SEARCH = False
BORDER_MARGIN_GRID = 20
MIN_VMAX = 35.0
INERTIA_WEIGHT = 1
EXTRAP_MAX_DEG_PER_6H = 5.5
SEED_DISTANCE_RATIO = 1

def parse_timestamp_from_filename(path: str | Path) -> datetime | None:
    name = os.path.splitext(os.path.basename(str(path)))[0]
    try:
        return datetime.strptime(name, '%Y%m%d%H')
    except ValueError:
        pass
    digits = re.findall('(\\d{10})', name)
    for cand in digits:
        try:
            return datetime.strptime(cand, '%Y%m%d%H')
        except ValueError:
            continue
    return None

def parse_step_from_filename(path: str | Path) -> int | None:
    name = os.path.basename(str(path))
    m = re.search('_step(\\d+)', name)
    if m:
        return int(m.group(1))
    m = re.search('step(\\d+)', name)
    return int(m.group(1)) if m else None

def get_channel_map(feature_csv: Path, varnames: dict[str, list[str]]) -> dict[str, int]:
    df = pd.read_csv(feature_csv)
    features = df['feature'].astype(str).tolist()
    result: dict[str, int] = {}
    for name, keys in varnames.items():
        found = None
        for i, f in enumerate(features):
            if any((k.lower() in f.lower() for k in keys)):
                found = i
                break
        if found is None:
            raise ValueError(f'{name} not found in feature table (keywords={keys})')
        result[name] = found
    return result

def normalize_to_CHW(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[1] == H and arr.shape[2] == W:
            return arr
        if arr.shape[0] == H and arr.shape[1] == W:
            return np.transpose(arr, (2, 0, 1))
        raise ValueError(f'Unsupported 3D shape: {arr.shape}, expected H={H}, W={W}')
    if arr.ndim == 2:
        n0, n1 = arr.shape
        if n0 == H * W:
            C = n1
            return np.transpose(arr.reshape(H, W, C), (2, 0, 1))
        if n1 == H * W:
            C = n0
            return arr.reshape(C, H, W)
        raise ValueError(f'Unsupported 2D shape: {arr.shape}, expected H*W={H * W}')
    raise ValueError(f'Only 2D/3D arrays supported, got {arr.ndim}D')

def to_1d_lat_lon(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lat.ndim == 2:
        lat1d = lat[:, 0]
    elif lat.ndim == 1:
        lat1d = lat
    else:
        raise ValueError(f'Unsupported lat shape: {lat.shape}')
    if lon.ndim == 2:
        lon1d = lon[0, :]
    elif lon.ndim == 1:
        lon1d = lon
    else:
        raise ValueError(f'Unsupported lon shape: {lon.shape}')
    return (lat1d.astype(float), lon1d.astype(float))

def align_lon_system(lon_axis_1d: np.ndarray, lon_value: float) -> float:
    lo_min, lo_max = (np.nanmin(lon_axis_1d), np.nanmax(lon_axis_1d))
    v = float(lon_value)
    if lo_min >= 0 and v < 0:
        v = v + 360.0
    if lo_max <= 180 and v > 180:
        v = v - 360.0
    return v

def normalize_lon(lon_axis_1d: np.ndarray, lon_value: float) -> float:
    v = align_lon_system(lon_axis_1d, lon_value)
    lo_min, lo_max = (np.nanmin(lon_axis_1d), np.nanmax(lon_axis_1d))
    if lo_min >= 0 and lo_max > 180:
        v = v % 360.0
    elif lo_max <= 180:
        v = (v + 180.0) % 360.0 - 180.0
    return float(v)

def lon_delta(lon_from: float, lon_to: float) -> float:
    delta = lon_to - lon_from
    if delta > 180.0:
        delta -= 360.0
    elif delta < -180.0:
        delta += 360.0
    return float(delta)

def extrapolate_position(prev_lat: float, prev_lon: float, curr_lat: float, curr_lon: float, weight: float, lon_axis_1d: np.ndarray) -> tuple[float, float, float]:
    prev_lon = normalize_lon(lon_axis_1d, prev_lon)
    curr_lon = normalize_lon(lon_axis_1d, curr_lon)
    dlat = curr_lat - prev_lat
    dlon = lon_delta(prev_lon, curr_lon)
    extrap_lat = curr_lat + weight * dlat
    extrap_lon = curr_lon + weight * dlon
    extrap_lon = normalize_lon(lon_axis_1d, extrap_lon)
    disp = float(np.hypot(weight * dlat, weight * dlon))
    return (extrap_lat, extrap_lon, disp)

def grid_step(axis: np.ndarray) -> float:
    diffs = np.diff(axis.astype(float))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(np.abs(diffs)))

def seed_distance_ratio(seed_lat: float, seed_lon: float, pred_lat: float, pred_lon: float, lat_step: float, lon_step: float, radius_lat: int, radius_lon: int, lon_axis_1d: np.ndarray) -> float:
    if lat_step <= 0 or lon_step <= 0:
        return 0.0
    seed_lon = normalize_lon(lon_axis_1d, seed_lon)
    pred_lon = normalize_lon(lon_axis_1d, pred_lon)
    dlat = pred_lat - seed_lat
    dlon = lon_delta(seed_lon, pred_lon)
    di = dlat / lat_step
    dj = dlon / lon_step
    if radius_lat <= 0 and radius_lon <= 0:
        return 0.0
    if radius_lat <= 0:
        return float(abs(dj) / max(radius_lon, 1e-06))
    if radius_lon <= 0:
        return float(abs(di) / max(radius_lat, 1e-06))
    return float(np.sqrt((di / radius_lat) ** 2 + (dj / radius_lon) ** 2))

def select_seed_center(seed_time: datetime, forecast_history: list[tuple[datetime, float, float]], bst_map: dict[datetime, tuple[float, float]], lat1d: np.ndarray, lon1d: np.ndarray) -> tuple[float, float]:
    vmax = EXTRAP_MAX_DEG_PER_6H * (STEP_HOURS / 6.0)
    if len(forecast_history) >= 2:
        prev2 = forecast_history[-2]
        prev1 = forecast_history[-1]
        seed_lat, seed_lon, disp = extrapolate_position(prev2[1], prev2[2], prev1[1], prev1[2], INERTIA_WEIGHT, lon1d)
        if disp > vmax:
            return (prev1[1], normalize_lon(lon1d, prev1[2]))
        return (seed_lat, seed_lon)
    bst_times = sorted((t for t in bst_map if t <= seed_time))
    if len(bst_times) >= 2:
        t0, t1 = (bst_times[-2], bst_times[-1])
        prev2_lat, prev2_lon = bst_map[t0]
        prev1_lat, prev1_lon = bst_map[t1]
        seed_lat, seed_lon, disp = extrapolate_position(prev2_lat, prev2_lon, prev1_lat, prev1_lon, INERTIA_WEIGHT, lon1d)
        if disp > vmax:
            return (prev1_lat, normalize_lon(lon1d, prev1_lon))
        return (seed_lat, seed_lon)
    curr = bst_map.get(seed_time)
    if curr is None and bst_times:
        curr = bst_map[bst_times[-1]]
    if curr is None and forecast_history:
        prev1 = forecast_history[-1]
        return (prev1[1], normalize_lon(lon1d, prev1[2]))
    if curr is None:
        return (float(lat1d[len(lat1d) // 2]), float(lon1d[len(lon1d) // 2]))
    return (curr[0], normalize_lon(lon1d, curr[1]))

def filter_df_by_domain_and_vmax(df: pd.DataFrame, lat1d: np.ndarray, lon1d: np.ndarray, border_margin_grid: int, min_vmax: float) -> pd.DataFrame:
    lat_min, lat_max = (float(np.nanmin(lat1d)), float(np.nanmax(lat1d)))
    lon_min, lon_max = (float(np.nanmin(lon1d)), float(np.nanmax(lon1d)))
    lat_margin = border_margin_grid * grid_step(lat1d)
    lon_margin = border_margin_grid * grid_step(lon1d)
    if lat_margin <= 0 or lon_margin <= 0:
        return df[df['VMAX'] >= min_vmax].copy()
    lat_ok = (df['LAT'] >= lat_min + lat_margin) & (df['LAT'] <= lat_max - lat_margin)
    lon_aligned = df['LON'].apply(lambda v: align_lon_system(lon1d, v))
    lon_ok = (lon_aligned >= lon_min + lon_margin) & (lon_aligned <= lon_max - lon_margin)
    vmax_ok = df['VMAX'] >= min_vmax
    return df[lat_ok & lon_ok & vmax_ok].copy()

def safe_window_center(lat1d: np.ndarray, lon1d: np.ndarray, lat0: float, lon0: float) -> tuple[int, int]:
    i = int(np.argmin(np.abs(lat1d - lat0)))
    lon0_adj = align_lon_system(lon1d, lon0)
    j = int(np.argmin(np.abs(lon1d - lon0_adj)))
    i = max(0, min(lat1d.size - 1, i))
    j = max(0, min(lon1d.size - 1, j))
    return (i, j)

def make_window(i: int, j: int, radius: int, H: int, W: int) -> tuple[int, int, int, int]:
    i0, i1 = (max(0, i - radius), min(H, i + radius + 1))
    j0, j1 = (max(0, j - radius), min(W, j + radius + 1))
    if i0 >= i1:
        i0, i1 = (i, i + 1)
    if j0 >= j1:
        j0, j1 = (j, j + 1)
    return (i0, i1, j0, j1)

def make_window_bounds(i: int, j: int, radius_lat: int, radius_lon: int, H: int, W: int) -> tuple[int, int, int, int]:
    i0, i1 = (max(0, i - radius_lat), min(H, i + radius_lat + 1))
    j0, j1 = (max(0, j - radius_lon), min(W, j + radius_lon + 1))
    if i0 >= i1:
        i0, i1 = (i, i + 1)
    if j0 >= j1:
        j0, j1 = (j, j + 1)
    return (i0, i1, j0, j1)

def ellipse_mask(i0: int, i1: int, j0: int, j1: int, center_i: int, center_j: int, radius_lat: int, radius_lon: int) -> np.ndarray:
    di = (np.arange(i0, i1) - center_i)[:, None]
    dj = (np.arange(j0, j1) - center_j)[None, :]
    if radius_lat <= 0 and radius_lon <= 0:
        return (di == 0) & (dj == 0)
    if radius_lat <= 0:
        return (di == 0) & (np.abs(dj) <= radius_lon)
    if radius_lon <= 0:
        return (np.abs(di) <= radius_lat) & (dj == 0)
    return (di / radius_lat) ** 2 + (dj / radius_lon) ** 2 <= 1.0

def weighted_centroid_latlon(field: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_i: int, center_j: int, radius: int, mode: str) -> tuple[float, float]:
    H, W = field.shape
    i0, i1, j0, j1 = make_window(center_i, center_j, radius, H, W)
    sub = field[i0:i1, j0:j1]
    if np.all(np.isnan(sub)):
        return (float(lat1d[center_i]), float(lon1d[center_j]))
    if mode == 'min':
        weights = np.nanmax(sub) - sub
    else:
        weights = sub - np.nanmin(sub)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if not np.any(weights > 0):
        return (float(lat1d[center_i]), float(lon1d[center_j]))
    lat_grid = lat1d[i0:i1][:, None]
    lon_grid = lon1d[j0:j1][None, :]
    wsum = float(np.sum(weights))
    lat_c = float(np.sum(weights * lat_grid) / wsum)
    lon_c = float(np.sum(weights * lon_grid) / wsum)
    return (lat_c, lon_c)

def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u * u + v * v)

def local_min(val: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> float:
    sub = val[i0:i1, j0:j1]
    if sub.size == 0:
        return float(val[i0, j0])
    k = int(np.nanargmin(sub))
    di, dj = np.unravel_index(k, sub.shape)
    return float(sub[di, dj])

def local_max(val: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> float:
    sub = val[i0:i1, j0:j1]
    if sub.size == 0:
        return float(val[i0, j0])
    k = int(np.nanargmax(sub))
    di, dj = np.unravel_index(k, sub.shape)
    return float(sub[di, dj])

def load_forecast_and_target(npz_path: Path, target_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        keys = set(getattr(data, 'files', []))
        forecast = None
        target = None
        for k_pred in ('forecast', 'pred', 'y_pred'):
            if k_pred in keys:
                forecast = data[k_pred]
                break
        for k_label in ('target', 'label', 'y_true'):
            if k_label in keys:
                target = data[k_label]
                break
    if target is None and target_dir is not None:
        step_idx = parse_step_from_filename(npz_path) or 1
        ts = parse_timestamp_from_filename(npz_path)
        candidate_name = npz_path.name
        if step_idx > 1 and ts is not None:
            shifted_ts = ts + timedelta(hours=STEP_HOURS * (step_idx - 1))
            ts_str = shifted_ts.strftime('%Y%m%d%H')
            candidate_name = re.sub('\\d{10}', ts_str, candidate_name, count=1)
            candidate_name = re.sub('_step\\d+', '_step01', candidate_name, count=1)
        target_path = target_dir / candidate_name
        if not target_path.exists():
            raise FileNotFoundError(f'Target file not found: {target_path}')
        with np.load(target_path) as tgt:
            tgt_keys = set(getattr(tgt, 'files', []))
            for k_label in ('target', 'label', 'y_true'):
                if k_label in tgt_keys:
                    target = tgt[k_label]
                    break
        if target is None:
            raise KeyError(f'{target_path} has no target field (target/label/y_true)')
    if forecast is None:
        raise KeyError(f'{npz_path} has no forecast field (forecast/pred/y_pred)')
    if target is None:
        raise KeyError(f'{npz_path} has no target field and TARGET_DIR is not set')
    return (forecast, target)

def collect_npz_files(npz_dir: Path) -> list[Path]:
    files = sorted(npz_dir.glob('*.npz'))
    if files or not RECURSIVE_SEARCH:
        return files
    return sorted(npz_dir.rglob('*.npz'))

def group_step_files_by_init(files: list[Path]) -> dict[datetime, list[Path]]:
    groups: dict[datetime, list[Path]] = {}
    for path in files:
        init_time = parse_timestamp_from_filename(path)
        if init_time is None:
            continue
        groups.setdefault(init_time, []).append(path)
    for paths in groups.values():
        paths.sort(key=lambda p: (parse_step_from_filename(p) or 0, p.name))
    return groups

def find_msl_weighted_center(msl: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, seed_i: int, seed_j: int, search_radius_lat: int, search_radius_lon: int, centroid_radius: int) -> tuple[float, float]:
    H, W = msl.shape
    i0, i1, j0, j1 = make_window_bounds(seed_i, seed_j, search_radius_lat, search_radius_lon, H, W)
    window = msl[i0:i1, j0:j1]
    mask = ellipse_mask(i0, i1, j0, j1, seed_i, seed_j, search_radius_lat, search_radius_lon)
    window = np.where(mask, window, np.nan)
    if np.all(np.isnan(window)):
        i, j = np.unravel_index(np.nanargmin(msl), msl.shape)
    else:
        local_idx = np.nanargmin(window)
        di, dj = np.unravel_index(local_idx, window.shape)
        i, j = (i0 + di, j0 + dj)
    return weighted_centroid_latlon(msl, lat1d, lon1d, i, j, centroid_radius, mode='min')

def compute_intensity_metrics(field: np.ndarray, ch_map: dict[str, int], lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, radius: int) -> tuple[float, float, float]:
    H, W = (lat1d.size, lon1d.size)
    i, j = safe_window_center(lat1d, lon1d, center_lat, center_lon)
    i0, i1, j0, j1 = make_window(i, j, radius, H, W)
    msl = field[ch_map['msl'], :, :]
    u10 = field[ch_map['u10'], :, :]
    v10 = field[ch_map['v10'], :, :]
    u850 = field[ch_map['u850'], :, :]
    v850 = field[ch_map['v850'], :, :]
    msl_min_hpa = local_min(msl, i0, i1, j0, j1) / 100.0
    s10_max = local_max(wind_speed(u10, v10), i0, i1, j0, j1)
    s850_max = local_max(wind_speed(u850, v850), i0, i1, j0, j1)
    return (msl_min_hpa, s10_max, s850_max)

def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y%m%d%H')
    lat_raw = np.load(LAT_NPY)
    lon_raw = np.load(LON_NPY)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    lat_step = grid_step(lat1d)
    lon_step = grid_step(lon1d)
    df = filter_df_by_domain_and_vmax(df, lat1d, lon1d, BORDER_MARGIN_GRID, MIN_VMAX)
    wanted = {'msl': ['msl'], 'u10': ['u10'], 'v10': ['v10'], 'u850': ['850u', 'u850'], 'v850': ['850v', 'v850']}
    ch_map = get_channel_map(FEATURE_TABLE, wanted)
    npz_files = collect_npz_files(NPZ_DIR)
    if not npz_files:
        raise FileNotFoundError(f'No npz files found in {NPZ_DIR}')
    init_groups = group_step_files_by_init(npz_files)
    if not init_groups:
        raise FileNotFoundError(f'No valid init time found in {NPZ_DIR}')
    bst_by_cyclone: dict[int, dict[datetime, tuple[float, float]]] = {}
    for row in df.itertuples(index=False):
        bst_time = pd.Timestamp(row.DATE_TIME).to_pydatetime()
        cyclone_num = int(row.CYCLONE_NUM)
        bst_by_cyclone.setdefault(cyclone_num, {})[bst_time] = (float(row.LAT), float(row.LON))
    records: list[dict] = []
    for cyclone_num, bst_map in bst_by_cyclone.items():
        for init_time, step_files in sorted(init_groups.items()):
            start_center = bst_map.get(init_time)
            if start_center is None:
                print(f'[WARN] No BST start center at {init_time:%Y%m%d%H} (CYCLONE_NUM={cyclone_num})')
                continue
            forecast_history: list[tuple[datetime, float, float]] = []
            for npz_path in step_files:
                step_idx = parse_step_from_filename(npz_path) or 1
                valid_time = init_time + timedelta(hours=STEP_HOURS * step_idx)
                seed_time = init_time + timedelta(hours=STEP_HOURS * (step_idx - 1))
                print(f'Processing {npz_path.name} for valid time {valid_time:%Y%m%d%H} (CYCLONE_NUM={cyclone_num})')
                try:
                    forecast_raw, target_raw = load_forecast_and_target(npz_path, TARGET_DIR)
                except (FileNotFoundError, KeyError) as exc:
                    print(f'[WARN] {exc}')
                    continue
                H, W = (lat1d.size, lon1d.size)
                forecast = normalize_to_CHW(forecast_raw, H, W)
                target = normalize_to_CHW(target_raw, H, W)
                seed_lat, seed_lon = select_seed_center(seed_time, forecast_history, bst_map, lat1d, lon1d)
                seed_i, seed_j = safe_window_center(lat1d, lon1d, seed_lat, seed_lon)
                msl = forecast[ch_map['msl'], :, :]
                pred_lat, pred_lon = find_msl_weighted_center(msl, lat1d, lon1d, seed_i, seed_j, WINDOW_RADIUS_LAT, WINDOW_RADIUS_LON, CENTROID_RADIUS)
                pred_lon = normalize_lon(lon1d, pred_lon)
                dist_ratio = seed_distance_ratio(seed_lat, seed_lon, pred_lat, pred_lon, lat_step, lon_step, WINDOW_RADIUS_LAT, WINDOW_RADIUS_LON, lon1d)
                seed_distance_flag = dist_ratio > SEED_DISTANCE_RATIO
                forecast_history.append((valid_time, pred_lat, pred_lon))
                bst_center = bst_map.get(valid_time)
                if bst_center is None:
                    print(f'[WARN] No BST center at {valid_time:%Y%m%d%H} (CYCLONE_NUM={cyclone_num}); skipping record.')
                    continue
                bst_lat, bst_lon = bst_center
                f_msl, f_s10, f_s850 = compute_intensity_metrics(forecast, ch_map, lat1d, lon1d, pred_lat, pred_lon, INTENSITY_RADIUS)
                t_msl, t_s10, t_s850 = compute_intensity_metrics(target, ch_map, lat1d, lon1d, bst_lat, bst_lon, INTENSITY_RADIUS)
                records.append({'time': valid_time.strftime('%Y%m%d%H'), 'forecast_init_time': init_time.strftime('%Y%m%d%H'), 'step': step_idx, 'CYCLONE_NUM': cyclone_num, 'bst_lat': bst_lat, 'bst_lon': bst_lon, 'forecast_center_lat': pred_lat, 'forecast_center_lon': pred_lon, 'seed_distance_flag': int(seed_distance_flag), 'forecast_msl_min_hPa': f_msl, 'forecast_s10_max': f_s10, 'forecast_s850_max': f_s850, 'target_msl_min_hPa': t_msl, 'target_s10_max': t_s10, 'target_s850_max': t_s850, 'source_file': npz_path.name})
    if not records:
        print('No records to write; check time alignment and paths.')
        return
    out_df = pd.DataFrame(records)
    out_df.sort_values(['time', 'CYCLONE_NUM', 'step'], inplace=True)
    out_df.to_csv(OUTPUT_CSV, index=False, float_format='%.3f')
    print(f'Wrote {OUTPUT_CSV} (rows={len(out_df)})')
