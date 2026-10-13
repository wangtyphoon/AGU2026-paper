# Derived from gnn/ver_autoregressive/output/tc_gwb_700z_to_csv.py
# See provenance/sources.json for original hash and modifications.
from __future__ import annotations
from agu_james import runtime
from pathlib import Path
from glob import escape as glob_escape
import numpy as np
import pandas as pd
from agu_james.analysis.gnn.tc_base import get_channel_map, load_forecast_and_target, normalize_to_CHW, to_1d_lat_lon
'\nGradient Wind Balance (GWB) height difference:\nΔZ = Z(R) − Z(0) = (1 / g) * ∫_0^R (Vθ^2 / r + f * Vθ) dr\n'


G = 9.80665

def integrate_gwb(r_m, vtheta, f: float, radius_m: float) -> tuple[float, str]:
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(vtheta, dtype=float)
    if r.ndim != 1 or v.shape != r.shape or not len(r):
        raise ValueError('Expected matching nonempty one-dimensional profiles')
    if not np.isfinite(r).all() or np.any(r <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError('Ring centers must be finite, positive and increasing')
    if not np.isfinite(f) or not np.isfinite(radius_m) or radius_m <= 0:
        return float('nan'), 'invalid_center_or_radius'
    upper = int(np.searchsorted(r, radius_m, side='left'))
    if upper == len(r):
        return float('nan'), 'missing_endpoint_support'
    needed = v[:upper + 1]
    if not np.isfinite(needed[0]):
        return float('nan'), 'missing_first_ring'
    if not np.isfinite(needed).all():
        return float('nan'), 'missing_required_ring'
    with np.errstate(over='ignore', invalid='ignore'):
        q = needed ** 2 / r[:upper + 1] + f * needed
    if not np.isfinite(q).all():
        return float('nan'), 'nonfinite_integrand'
    support_r = np.r_[0., r[:upper + 1]]
    support_q = np.r_[0., q]
    inner = r < radius_m
    nodes = np.r_[0., r[inner], radius_m]
    values = np.interp(nodes, support_r, support_q)
    # The trapezoid sum also supports NumPy versions predating np.trapezoid.
    integral = np.sum((values[1:] + values[:-1]) * np.diff(nodes) * 0.5)
    return float(integral / G), 'ok'


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

def axisymmetric_mean(field2d: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, bin_km: float, max_radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    lat2d = lat1d[:, None]
    lon2d = lon1d[None, :]
    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon)
    bins = np.arange(0.0, max_radius_km + bin_km, bin_km)
    centers = (bins[:-1] + bins[1:]) * 0.5
    means = np.full_like(centers, np.nan, dtype=float)
    for i in range(len(centers)):
        mask = (dist_km >= bins[i]) & (dist_km < bins[i + 1])
        if np.any(mask):
            means[i] = float(np.nanmean(field2d[mask]))
    return (centers, means)

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

def tangential_wind(u: np.ndarray, v: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    km_per_deg = 111.32
    lat0_rad = np.deg2rad(center_lat)
    dy = (lat2d - center_lat) * km_per_deg
    dx = (lon2d - center_lon) * km_per_deg * np.cos(lat0_rad)
    theta = np.arctan2(dy, dx)
    return -u * np.sin(theta) + v * np.cos(theta)

def nearest_grid_value(field2d: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float) -> float:
    if not np.isfinite(center_lat) or not np.isfinite(center_lon):
        return float('nan')
    if not np.any(np.isfinite(lat1d)) or not np.any(np.isfinite(lon1d)):
        return float('nan')
    i = int(np.nanargmin(np.abs(lat1d - center_lat)))
    j = int(np.nanargmin(np.abs(lon1d - center_lon)))
    return float(field2d[i, j])

def compute_delta_z(u700: np.ndarray, v700: np.ndarray, z700: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, bin_km: float, radii_km: list[float]) -> tuple[list[float], list[float]]:
    g = 9.80665
    z700_hgt = z700 / g
    center_lon = normalize_lon(lon1d, center_lon)
    lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing='ij')
    z0 = nearest_grid_value(z700_hgt, lat1d, lon1d, center_lat, center_lon)
    max_radius_km = float(max(radii_km) + bin_km)
    r_centers, z_mean = axisymmetric_mean(z700_hgt, lat1d, lon1d, center_lat, center_lon, bin_km, max_radius_km)
    valid_z = np.isfinite(z_mean)
    v_theta = tangential_wind(u700, v700, lat2d, lon2d, center_lat, center_lon)
    r_bins, vtheta_mean = axisymmetric_mean(v_theta, lat1d, lon1d, center_lat, center_lon, bin_km, max_radius_km)
    r_bins_m = r_bins * 1000.0
    omega = 7.2921159e-05
    f = 2.0 * omega * np.sin(np.radians(center_lat))
    delta_z_sim_list: list[float] = []
    delta_z_gwb_list: list[float] = []
    for radius_km in radii_km:
        if not np.isfinite(z0) or not np.any(valid_z):
            delta_z_sim_list.append(float('nan'))
        else:
            r_valid = r_centers[valid_z]
            z_valid = z_mean[valid_z]
            z_r = float(np.interp(radius_km, r_valid, z_valid))
            delta_z_sim_list.append(z_r - z0)
        delta_z_gwb, _ = integrate_gwb(r_bins_m, vtheta_mean, f, radius_km * 1000.0)
        delta_z_gwb_list.append(float(delta_z_gwb))
    return (delta_z_sim_list, delta_z_gwb_list)

def main() -> None:
    base_dir = runtime.setting('agu_james.analysis.gnn.tc_gwb.base_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.tc_gwb')
    csv_name = 'tc_aurora_intensity35_step28_spectrum.csv'
    csv_path = runtime.setting('agu_james.analysis.gnn.tc_gwb.csv_path', lambda: (base_dir / csv_name).resolve())
    output_csv = runtime.setting('agu_james.analysis.gnn.tc_gwb.output_csv', lambda: csv_path)
    npz_root = runtime.setting('agu_james.analysis.gnn.tc_gwb.npz_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_autoregressive_step28_spectrum_new_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    target_root = runtime.setting('agu_james.analysis.gnn.tc_gwb.target_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    lat_path = runtime.setting('agu_james.analysis.gnn.tc_gwb.lat_path', lambda: (base_dir / 'lat.npy').resolve())
    lon_path = runtime.setting('agu_james.analysis.gnn.tc_gwb.lon_path', lambda: (base_dir / 'lon.npy').resolve())
    feature_table = runtime.setting('agu_james.analysis.gnn.tc_gwb.feature_table', lambda: (base_dir / 'next_feature_table_weight.csv').resolve())
    bin_km = 25.0
    radii_km = [100.0, 500.0]
    df = pd.read_csv(csv_path)
    lat_raw = np.load(lat_path)
    lon_raw = np.load(lon_path)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    ch_map = get_channel_map(feature_table, {'u700': ['700u'], 'v700': ['700v'], 'z700': ['700z']})
    delta_z_sim_100 = []
    delta_z_sim_500 = []
    delta_z_gwb_100 = []
    delta_z_gwb_500 = []
    target_delta_z_sim_100 = []
    target_delta_z_sim_500 = []
    target_delta_z_gwb_100 = []
    target_delta_z_gwb_500 = []
    for idx, row in df.iterrows():
        source_file = str(row['source_file'])
        try:
            npz_path = find_npz(npz_root, source_file)
        except FileNotFoundError as exc:
            print(f'[WARN] {exc} (row {idx})')
            delta_z_sim_100.append(np.nan)
            delta_z_sim_500.append(np.nan)
            delta_z_gwb_100.append(np.nan)
            delta_z_gwb_500.append(np.nan)
            target_delta_z_sim_100.append(np.nan)
            target_delta_z_sim_500.append(np.nan)
            target_delta_z_gwb_100.append(np.nan)
            target_delta_z_gwb_500.append(np.nan)
            continue
        try:
            forecast_raw, target_raw = load_forecast_and_target(npz_path, target_root)
        except Exception as exc:
            print(f'[WARN] Failed to load {npz_path.name} (row {idx}): {exc}')
            delta_z_sim_100.append(np.nan)
            delta_z_sim_500.append(np.nan)
            delta_z_gwb_100.append(np.nan)
            delta_z_gwb_500.append(np.nan)
            target_delta_z_sim_100.append(np.nan)
            target_delta_z_sim_500.append(np.nan)
            target_delta_z_gwb_100.append(np.nan)
            target_delta_z_gwb_500.append(np.nan)
            continue
        H, W = (lat1d.size, lon1d.size)
        forecast = normalize_to_CHW(forecast_raw, H, W)
        target = normalize_to_CHW(target_raw, H, W)
        u700_f = forecast[ch_map['u700'], :, :]
        v700_f = forecast[ch_map['v700'], :, :]
        z700_f = forecast[ch_map['z700'], :, :]
        u700_t = target[ch_map['u700'], :, :]
        v700_t = target[ch_map['v700'], :, :]
        z700_t = target[ch_map['z700'], :, :]
        forecast_center_lat = float(row['forecast_center_lat'])
        forecast_center_lon = float(row['forecast_center_lon'])
        target_center_lat = float(row['bst_lat'])
        target_center_lon = float(row['bst_lon'])
        delta_z_sim_list, delta_z_gwb_list = compute_delta_z(u700_f, v700_f, z700_f, lat1d, lon1d, forecast_center_lat, forecast_center_lon, bin_km, radii_km)
        target_delta_z_sim_list, target_delta_z_gwb_list = compute_delta_z(u700_t, v700_t, z700_t, lat1d, lon1d, target_center_lat, target_center_lon, bin_km, radii_km)
        delta_z_sim_100.append(delta_z_sim_list[0])
        delta_z_sim_500.append(delta_z_sim_list[1])
        delta_z_gwb_100.append(delta_z_gwb_list[0])
        delta_z_gwb_500.append(delta_z_gwb_list[1])
        target_delta_z_sim_100.append(target_delta_z_sim_list[0])
        target_delta_z_sim_500.append(target_delta_z_sim_list[1])
        target_delta_z_gwb_100.append(target_delta_z_gwb_list[0])
        target_delta_z_gwb_500.append(target_delta_z_gwb_list[1])
        if (idx + 1) % 200 == 0:
            print(f'Processed {idx + 1}/{len(df)} rows')
    df = df.copy()
    df['delta_z_sim_100km'] = delta_z_sim_100
    df['delta_z_sim_500km'] = delta_z_sim_500
    df['delta_z_gwb_100km'] = delta_z_gwb_100
    df['delta_z_gwb_500km'] = delta_z_gwb_500
    df['target_delta_z_sim_100km'] = target_delta_z_sim_100
    df['target_delta_z_sim_500km'] = target_delta_z_sim_500
    df['target_delta_z_gwb_100km'] = target_delta_z_gwb_100
    df['target_delta_z_gwb_500km'] = target_delta_z_gwb_500
    df.to_csv(output_csv, index=False)
    print(f'Saved updated CSV to {output_csv}')
