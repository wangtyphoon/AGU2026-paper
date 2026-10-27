# Derived from gnn/ver_autoregressive/output/tc_axisymmetric_metrics_to_csv.py
from __future__ import annotations
from agu_james import runtime
from pathlib import Path
import numpy as np
import pandas as pd
from agu_james.analysis.gnn.tc_base import get_channel_map, load_forecast_and_target, normalize_lon, normalize_to_CHW, to_1d_lat_lon, wind_speed

def haversine_km(lat1, lon1, lat2, lon2):
    r_earth_km = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r_earth_km * c

def compute_axisymmetric_profile(speed2d: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, bin_km: float, max_radius_km: float | None=None) -> tuple[np.ndarray, np.ndarray]:
    lat2d = lat1d[:, None]
    lon2d = lon1d[None, :]
    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon)
    max_r = float(np.nanmax(dist_km))
    if max_radius_km is not None:
        max_r = min(max_r, float(max_radius_km))
    if not np.isfinite(max_r) or not np.isfinite(bin_km) or bin_km <= 0.0 or (max_r <= 0.0):
        return (np.array([], dtype=float), np.array([], dtype=float))
    bins = np.arange(0.0, max_r + bin_km, bin_km)
    centers = (bins[:-1] + bins[1:]) * 0.5
    means = np.empty_like(centers)
    for i in range(len(centers)):
        mask = (dist_km >= bins[i]) & (dist_km < bins[i + 1])
        if np.any(mask):
            means[i] = float(np.nanmean(speed2d[mask]))
        else:
            means[i] = np.nan
    return (centers, means)

def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    num = np.convolve(np.where(valid, values, 0.0), kernel, mode='same')
    den = np.convolve(valid.astype(float), kernel, mode='same')
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

def build_npz_index(npz_root: Path) -> dict[str, Path]:
    files = sorted(npz_root.rglob('*.npz'))
    index: dict[str, Path] = {}
    for path in files:
        if path.name not in index:
            index[path.name] = path
    return index

def axisymmetric_metrics(speed2d: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, bin_km: float, max_radius_km: float | None, smooth_window_bins: int, r8_threshold: float) -> tuple[float, float, float]:
    radii_km, mean_speed = compute_axisymmetric_profile(speed2d, lat1d, lon1d, center_lat, center_lon, bin_km, max_radius_km)
    mean_speed = smooth_1d(mean_speed, smooth_window_bins)
    valid_mask = np.isfinite(mean_speed)
    if np.any(valid_mask):
        vmax = float(np.nanmax(mean_speed))
        r_max = float(radii_km[np.nanargmax(mean_speed)])
        r8_candidates = radii_km[valid_mask & (mean_speed >= r8_threshold)]
        r8 = float(r8_candidates.max()) if r8_candidates.size else float('nan')
    else:
        vmax = float('nan')
        r_max = float('nan')
        r8 = float('nan')
    return (vmax, r_max, r8)

def main() -> None:
    base_dir = runtime.setting('agu_james.analysis.gnn.tc_wind.base_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.tc_wind')
    csv_path = runtime.setting('agu_james.analysis.gnn.tc_wind.csv_path', lambda: base_dir / 'tc_aurora_intensity35_step12_spectrum.csv')
    output_csv = runtime.setting('agu_james.analysis.gnn.tc_wind.output_csv', lambda: csv_path)
    npz_root = runtime.setting('agu_james.analysis.gnn.tc_wind.npz_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_autoregressive_step12_spectrum_new_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    target_root = runtime.setting('agu_james.analysis.gnn.tc_wind.target_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    lat_path = runtime.setting('agu_james.analysis.gnn.tc_wind.lat_path', lambda: base_dir / 'lat.npy')
    lon_path = runtime.setting('agu_james.analysis.gnn.tc_wind.lon_path', lambda: base_dir / 'lon.npy')
    feature_table = runtime.setting('agu_james.analysis.gnn.tc_wind.feature_table', lambda: base_dir / 'next_feature_table_weight.csv')
    bin_km = 25.0
    max_radius_km = 2000.0
    smooth_window_bins = 5
    r8_threshold = 8.0
    df = pd.read_csv(csv_path)
    lat_raw = np.load(lat_path)
    lon_raw = np.load(lon_path)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    ch_map = get_channel_map(feature_table, {'u10': ['u10'], 'v10': ['v10']})
    npz_index = build_npz_index(npz_root)
    forecast_vmax = []
    forecast_rmax = []
    forecast_r8 = []
    target_vmax = []
    target_rmax = []
    target_r8 = []
    for idx, row in df.iterrows():
        source_file = str(row['source_file'])
        npz_path = npz_index.get(source_file)
        if npz_path is None:
            print(f'[WARN] Missing npz for {source_file} (row {idx})')
            forecast_vmax.append(np.nan)
            forecast_rmax.append(np.nan)
            forecast_r8.append(np.nan)
            target_vmax.append(np.nan)
            target_rmax.append(np.nan)
            target_r8.append(np.nan)
            continue
        try:
            forecast_raw, target_raw = load_forecast_and_target(npz_path, target_root)
        except Exception as exc:
            print(f'[WARN] Failed to load {npz_path.name} (row {idx}): {exc}')
            forecast_vmax.append(np.nan)
            forecast_rmax.append(np.nan)
            forecast_r8.append(np.nan)
            target_vmax.append(np.nan)
            target_rmax.append(np.nan)
            target_r8.append(np.nan)
            continue
        H, W = (lat1d.size, lon1d.size)
        forecast = normalize_to_CHW(forecast_raw, H, W)
        target = normalize_to_CHW(target_raw, H, W)
        u10_f = forecast[ch_map['u10'], :, :]
        v10_f = forecast[ch_map['v10'], :, :]
        u10_t = target[ch_map['u10'], :, :]
        v10_t = target[ch_map['v10'], :, :]
        s10_f = wind_speed(u10_f, v10_f)
        s10_t = wind_speed(u10_t, v10_t)
        f_lat = float(row['forecast_center_lat'])
        f_lon = normalize_lon(lon1d, float(row['forecast_center_lon']))
        t_lat = float(row['bst_lat'])
        t_lon = normalize_lon(lon1d, float(row['bst_lon']))
        vmax_f, rmax_f, r8_f = axisymmetric_metrics(s10_f, lat1d, lon1d, f_lat, f_lon, bin_km, max_radius_km, smooth_window_bins, r8_threshold)
        vmax_t, rmax_t, r8_t = axisymmetric_metrics(s10_t, lat1d, lon1d, t_lat, t_lon, bin_km, max_radius_km, smooth_window_bins, r8_threshold)
        forecast_vmax.append(vmax_f)
        forecast_rmax.append(rmax_f)
        forecast_r8.append(r8_f)
        target_vmax.append(vmax_t)
        target_rmax.append(rmax_t)
        target_r8.append(r8_t)
        if (idx + 1) % 200 == 0:
            print(f'Processed {idx + 1}/{len(df)} rows')
    df = df.copy()
    df['forecast_vmax'] = forecast_vmax
    df['forecast_rmax_km'] = forecast_rmax
    df['forecast_r8_km'] = forecast_r8
    df['target_vmax'] = target_vmax
    df['target_rmax_km'] = target_rmax
    df['target_r8_km'] = target_r8
    df.to_csv(output_csv, index=False)
    print(f'Saved updated CSV to {output_csv}')
