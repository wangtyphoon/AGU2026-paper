# Derived from gnn/ver_autoregressive/output/tc_ike_calculation_to_csv.py
from __future__ import annotations
from agu_james import runtime
from pathlib import Path
from glob import escape as glob_escape
import numpy as np
import pandas as pd
from agu_james.analysis.gnn.tc_base import get_channel_map, load_forecast_and_target, normalize_lon, normalize_to_CHW, to_1d_lat_lon
R_EARTH_M = 6371000.0
R_D = 287.05

def find_npz(npz_root: Path, filename: str) -> Path:
    matches = sorted(npz_root.rglob(glob_escape(filename)))
    if not matches:
        raise FileNotFoundError(f'Cannot find {filename} under {npz_root}')
    if len(matches) > 1:
        print(f'[WARN] Multiple matches for {filename}; using {matches[0]}')
    return matches[0]

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

def compute_cell_area_m2(lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    lat_rad = np.radians(lat1d)
    lon_rad = np.radians(lon1d)
    lat_edges = np.empty(lat_rad.size + 1)
    lon_edges = np.empty(lon_rad.size + 1)
    lat_edges[1:-1] = 0.5 * (lat_rad[:-1] + lat_rad[1:])
    lon_edges[1:-1] = 0.5 * (lon_rad[:-1] + lon_rad[1:])
    lat_edges[0] = lat_rad[0] - 0.5 * (lat_rad[1] - lat_rad[0])
    lat_edges[-1] = lat_rad[-1] + 0.5 * (lat_rad[-1] - lat_rad[-2])
    lon_edges[0] = lon_rad[0] - 0.5 * (lon_rad[1] - lon_rad[0])
    lon_edges[-1] = lon_rad[-1] + 0.5 * (lon_rad[-1] - lon_rad[-2])
    dlon = np.diff(lon_edges)
    lat_band = np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])
    area = R_EARTH_M ** 2 * lat_band[:, None] * dlon[None, :]
    return np.abs(area)

def compute_ike_700(data_chw: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, radius_km: float, ch_map: dict[str, int], thickness_m: float) -> float:
    u = data_chw[ch_map['700u'], :, :]
    v = data_chw[ch_map['700v'], :, :]
    t = data_chw[ch_map['700t'], :, :]
    rho = 700.0 * 100.0 / (R_D * t)
    energy_density = 0.5 * rho * (u ** 2 + v ** 2) * thickness_m
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon)
    mask = dist_km <= radius_km
    area_m2 = compute_cell_area_m2(lat1d, lon1d)
    return float(np.nansum(energy_density[mask] * area_m2[mask]))

def compute_ike_surface(data_chw: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, radius_km: float, ch_map: dict[str, int], thickness_m: float) -> float:
    u10 = data_chw[ch_map['u10'], :, :]
    v10 = data_chw[ch_map['v10'], :, :]
    t2m = data_chw[ch_map['t2m'], :, :]
    msl = data_chw[ch_map['msl'], :, :]
    rho = msl / (R_D * t2m)
    energy_density = 0.5 * rho * (u10 ** 2 + v10 ** 2) * thickness_m
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon)
    mask = dist_km <= radius_km
    area_m2 = compute_cell_area_m2(lat1d, lon1d)
    return float(np.nansum(energy_density[mask] * area_m2[mask]))

def main() -> None:
    base_dir = runtime.setting('agu_james.analysis.gnn.tc_ike.base_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.tc_ike')
    csv_name = 'tc_aurora_intensity35_step28_spectrum.csv'
    csv_path = runtime.setting('agu_james.analysis.gnn.tc_ike.csv_path', lambda: (base_dir / csv_name).resolve())
    output_csv = runtime.setting('agu_james.analysis.gnn.tc_ike.output_csv', lambda: csv_path)
    npz_root = runtime.setting('agu_james.analysis.gnn.tc_ike.npz_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_autoregressive_step28_spectrum_new_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    target_root = runtime.setting('agu_james.analysis.gnn.tc_ike.target_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    lat_path = runtime.setting('agu_james.analysis.gnn.tc_ike.lat_path', lambda: (base_dir / 'lat.npy').resolve())
    lon_path = runtime.setting('agu_james.analysis.gnn.tc_ike.lon_path', lambda: (base_dir / 'lon.npy').resolve())
    feature_table = runtime.setting('agu_james.analysis.gnn.tc_ike.feature_table', lambda: (base_dir / 'next_feature_table_weight.csv').resolve())
    radius_km = 750.0
    thickness_m = 1000.0
    df = pd.read_csv(csv_path)
    lat_raw = np.load(lat_path)
    lon_raw = np.load(lon_path)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    ch_map = get_channel_map(feature_table, {'700u': ['700u'], '700v': ['700v'], '700t': ['700t'], 'u10': ['u10'], 'v10': ['v10'], 't2m': ['t2m'], 'msl': ['msl']})
    forecast_ike700 = []
    target_ike700 = []
    forecast_ike_surface = []
    target_ike_surface = []
    for idx, row in df.iterrows():
        source_file = str(row['source_file'])
        try:
            npz_path = find_npz(npz_root, source_file)
        except FileNotFoundError as exc:
            print(f'[WARN] {exc} (row {idx})')
            forecast_ike700.append(np.nan)
            target_ike700.append(np.nan)
            forecast_ike_surface.append(np.nan)
            target_ike_surface.append(np.nan)
            continue
        try:
            forecast_raw, target_raw = load_forecast_and_target(npz_path, target_root)
        except Exception as exc:
            print(f'[WARN] Failed to load {npz_path.name} (row {idx}): {exc}')
            forecast_ike700.append(np.nan)
            target_ike700.append(np.nan)
            forecast_ike_surface.append(np.nan)
            target_ike_surface.append(np.nan)
            continue
        H, W = (lat1d.size, lon1d.size)
        forecast = normalize_to_CHW(forecast_raw, H, W)
        target = normalize_to_CHW(target_raw, H, W)
        f_lat = float(row['forecast_center_lat'])
        f_lon = normalize_lon(lon1d, float(row['forecast_center_lon']))
        t_lat = float(row['bst_lat'])
        t_lon = normalize_lon(lon1d, float(row['bst_lon']))
        forecast_ike700.append(compute_ike_700(forecast, lat1d, lon1d, f_lat, f_lon, radius_km, ch_map, thickness_m))
        target_ike700.append(compute_ike_700(target, lat1d, lon1d, t_lat, t_lon, radius_km, ch_map, thickness_m))
        forecast_ike_surface.append(compute_ike_surface(forecast, lat1d, lon1d, f_lat, f_lon, radius_km, ch_map, thickness_m))
        target_ike_surface.append(compute_ike_surface(target, lat1d, lon1d, t_lat, t_lon, radius_km, ch_map, thickness_m))
        if (idx + 1) % 200 == 0:
            print(f'Processed {idx + 1}/{len(df)} rows')
    df = df.copy()
    df['forecast_ike700'] = forecast_ike700
    df['target_ike700'] = target_ike700
    df['forecast_ike_surface'] = forecast_ike_surface
    df['target_ike_surface'] = target_ike_surface
    df.to_csv(output_csv, index=False)
    print(f'Saved updated CSV to {output_csv}')
