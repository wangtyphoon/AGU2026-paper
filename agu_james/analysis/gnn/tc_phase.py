# Derived from gnn/ver_autoregressive/output/tc_phase_core_to_csv.py
# See provenance/sources.json for original hash and modifications.
from __future__ import annotations
from agu_james import runtime
from pathlib import Path
from glob import escape as glob_escape
import numpy as np
import pandas as pd
from agu_james.analysis.gnn.tc_base import get_channel_map, load_forecast_and_target, lon_delta, normalize_lon, normalize_to_CHW, to_1d_lat_lon

def find_npz(npz_root: Path, filename: str) -> Path:
    matches = sorted(npz_root.rglob(glob_escape(filename)))
    if not matches:
        raise FileNotFoundError(f'Cannot find {filename} under {npz_root}')
    if len(matches) > 1:
        print(f'[WARN] Multiple matches for {filename}; using {matches[0]}')
    return matches[0]

def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
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

def adjust_lon_grid(lon1d: np.ndarray, center_lon: float) -> tuple[np.ndarray, float]:
    lon_min, lon_max = (float(np.nanmin(lon1d)), float(np.nanmax(lon1d)))
    center_lon_norm = normalize_lon(lon1d, center_lon)
    lon_grid = lon1d.copy()
    if lon_min >= 0 and center_lon_norm < 0:
        lon_grid = lon_grid + 360.0
        center_lon_norm += 360.0
    if lon_max <= 180 and center_lon_norm > 180:
        lon_grid = lon_grid - 360.0
        center_lon_norm -= 360.0
    return (lon_grid, center_lon_norm)

def movement_vector_km(prev_lat: float, prev_lon: float, next_lat: float, next_lon: float, ref_lat: float, lon1d: np.ndarray) -> tuple[float, float]:
    prev_lon = normalize_lon(lon1d, prev_lon)
    next_lon = normalize_lon(lon1d, next_lon)
    dlat = next_lat - prev_lat
    dlon = lon_delta(prev_lon, next_lon)
    km_per_deg = 111.32
    dy = dlat * km_per_deg
    dx = dlon * km_per_deg * np.cos(np.radians(ref_lat))
    return (float(dx), float(dy))

def compute_motion_vectors_by_group(df: pd.DataFrame, lat_col: str, lon_col: str, lon1d: np.ndarray, group_cols: list[str], sort_col: str) -> tuple[np.ndarray, np.ndarray]:
    dx_all = np.full(len(df), np.nan, dtype=float)
    dy_all = np.full(len(df), np.nan, dtype=float)
    for _, group in df.groupby(group_cols):
        group_sorted = group.sort_values(sort_col)
        idxs = group_sorted.index.to_numpy()
        if idxs.size < 2:
            continue
        lats = group_sorted[lat_col].to_numpy(dtype=float)
        lons = group_sorted[lon_col].to_numpy(dtype=float)
        lons = np.array([normalize_lon(lon1d, val) for val in lons], dtype=float)
        for i, row_idx in enumerate(idxs):
            if i == 0:
                prev_lat, prev_lon = (lats[i], lons[i])
                next_lat, next_lon = (lats[i + 1], lons[i + 1])
            elif i == len(idxs) - 1:
                prev_lat, prev_lon = (lats[i - 1], lons[i - 1])
                next_lat, next_lon = (lats[i], lons[i])
            else:
                prev_lat, prev_lon = (lats[i - 1], lons[i - 1])
                next_lat, next_lon = (lats[i + 1], lons[i + 1])
            if not (np.isfinite(prev_lat) and np.isfinite(prev_lon) and np.isfinite(next_lat) and np.isfinite(next_lon) and np.isfinite(lats[i])):
                continue
            dx_km, dy_km = movement_vector_km(float(prev_lat), float(prev_lon), float(next_lat), float(next_lon), float(lats[i]), lon1d)
            if np.hypot(dx_km, dy_km) == 0.0:
                continue
            dx_all[row_idx] = dx_km
            dy_all[row_idx] = dy_km
    return (dx_all, dy_all)

def left_right_thickness(thickness2d: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, motion_dx_km: float, motion_dy_km: float, radius_km: float) -> tuple[float, float]:
    if not np.isfinite(motion_dx_km) or not np.isfinite(motion_dy_km):
        return (float('nan'), float('nan'))
    if np.hypot(motion_dx_km, motion_dy_km) == 0.0:
        return (float('nan'), float('nan'))
    lon1d_adj, center_lon_adj = adjust_lon_grid(lon1d, center_lon)
    lat2d, lon2d = np.meshgrid(lat1d, lon1d_adj, indexing='ij')
    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon_adj)
    radius_mask = dist_km <= radius_km
    km_per_deg = 111.32
    dx = (lon2d - center_lon_adj) * km_per_deg * np.cos(np.radians(center_lat))
    dy = (lat2d - center_lat) * km_per_deg
    motion_norm = np.hypot(motion_dx_km, motion_dy_km)
    motion_unit_x = motion_dx_km / motion_norm
    motion_unit_y = motion_dy_km / motion_norm
    cross = motion_unit_x * dy - motion_unit_y * dx
    valid = np.isfinite(thickness2d) & radius_mask
    left_mask = valid & (cross > 0.0)
    right_mask = valid & (cross < 0.0)
    left_values = thickness2d[radius_mask & left_mask]
    right_values = thickness2d[radius_mask & right_mask]
    left_mean = float(np.nanmean(left_values)) if left_values.size else float('nan')
    right_mean = float(np.nanmean(right_values)) if right_values.size else float('nan')
    return (left_mean, right_mean)

def radius_mask(lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, radius_km: float) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing='ij')
    dist_km = haversine_km(lat2d, lon2d, center_lat, center_lon)
    return dist_km <= radius_km

def z_range_by_levels(field_chw: np.ndarray, level_map: dict[str, int], levels: list[int], mask: np.ndarray, g: float) -> tuple[np.ndarray, np.ndarray]:
    level_arr = np.array(levels, dtype=float)
    deltas = np.full(len(levels), np.nan, dtype=float)
    for i, lvl in enumerate(levels):
        idx = level_map[f'z{lvl}']
        values = field_chw[idx][mask] / g
        values = values[np.isfinite(values)]
        if values.size:
            deltas[i] = float(np.nanmax(values) - np.nanmin(values))
    return (level_arr, deltas)

def interpolate_delta_z(levels: np.ndarray, deltas: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
    mask = np.isfinite(levels) & np.isfinite(deltas)
    if np.count_nonzero(mask) < 2:
        return np.full_like(p_grid, np.nan, dtype=float)
    order = np.argsort(levels[mask])
    levels_sorted = levels[mask][order]
    deltas_sorted = deltas[mask][order]
    return np.interp(p_grid, levels_sorted, deltas_sorted, left=np.nan, right=np.nan)

def regression_slope_ln_p(p_grid: np.ndarray, deltas: np.ndarray) -> float:
    mask = np.isfinite(p_grid) & np.isfinite(deltas)
    if np.count_nonzero(mask) < 2:
        return float('nan')
    ln_p = np.log(p_grid[mask])
    slope, _ = np.polyfit(ln_p, deltas[mask], 1)
    return float(slope)


LOWER_PRESSURES = (925, 850, 700, 600)
UPPER_PRESSURES = (600, 500, 400, 300)

def native_delta_z(levels, deltas, requested):
    """Select measured height ranges only: never interpolate missing layers."""
    lookup = dict(zip(levels, deltas))
    return np.asarray([lookup.get(p, np.nan) for p in requested], dtype=float)

def compute_native_vt(levels, deltas):
    """Signed -V_T: equal-weight OLS of 500-km height range against ln(p/1 hPa)."""
    lower = native_delta_z(levels, deltas, LOWER_PRESSURES)
    upper = native_delta_z(levels, deltas, UPPER_PRESSURES)
    return {
        'vt_lower': regression_slope_ln_p(np.asarray(LOWER_PRESSURES), lower),
        'vt_upper': regression_slope_ln_p(np.asarray(UPPER_PRESSURES), upper),
        'lower_valid_levels': int(np.isfinite(lower).sum()),
        'upper_valid_levels': int(np.isfinite(upper).sum()),
    }

def write_thermal_metadata(output_path, counts):
    """Keep legacy CSV columns intact; save valid-layer counts and method beside it."""
    import json
    path = Path(output_path)
    pd.DataFrame(counts, columns=[
        'forecast_lower_valid_levels', 'forecast_upper_valid_levels',
        'target_lower_valid_levels', 'target_upper_valid_levels',
    ]).to_csv(path.with_suffix('.thermal_wind_counts.csv'), index_label='row')
    path.with_suffix('.thermal_wind_method.json').write_text(json.dumps({
        'method': 'signed-native-pressure-levels-v1', 'radius_km': 500,
        'lower_pressures_hpa': LOWER_PRESSURES, 'upper_pressures_hpa': UPPER_PRESSURES,
        'regression': 'equal-weight OLS of Delta Z_core against ln(p/1 hPa)',
        'interpolation': False, 'minimum_finite_layers': 2, 'units': 'm',
        'positive': 'warm core', 'negative': 'cold core',
    }, indent=2), encoding='utf-8')

def main() -> None:
    base_dir = runtime.setting('agu_james.analysis.gnn.tc_phase.base_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.tc_phase')
    csv_name = 'tc_aurora_intensity35_step12_spectrum.csv'
    csv_path = runtime.setting('agu_james.analysis.gnn.tc_phase.csv_path', lambda: (base_dir / csv_name).resolve())
    output_csv = runtime.setting('agu_james.analysis.gnn.tc_phase.output_csv', lambda: csv_path)
    npz_root = runtime.setting('agu_james.analysis.gnn.tc_phase.npz_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_autoregressive_step12_spectrum_new_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    target_root = runtime.setting('agu_james.analysis.gnn.tc_phase.target_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    lat_path = runtime.setting('agu_james.analysis.gnn.tc_phase.lat_path', lambda: (base_dir / 'lat.npy').resolve())
    lon_path = runtime.setting('agu_james.analysis.gnn.tc_phase.lon_path', lambda: (base_dir / 'lon.npy').resolve())
    feature_table = runtime.setting('agu_james.analysis.gnn.tc_phase.feature_table', lambda: (base_dir / 'next_feature_table_weight.csv').resolve())
    radius_km = 500.0
    group_cols = ['CYCLONE_NUM', 'forecast_init_time']
    sort_col = 'step'
    available_levels = [925, 850, 700, 600, 500, 400, 300]
    wanted = {f'z{lvl}': [f'{lvl}z'] for lvl in available_levels}
    df = pd.read_csv(csv_path)
    lat_raw = np.load(lat_path)
    lon_raw = np.load(lon_path)
    lat1d, lon1d = to_1d_lat_lon(lat_raw, lon_raw)
    ch_map = get_channel_map(feature_table, wanted)
    g = 9.80665
    f_dx_all, f_dy_all = compute_motion_vectors_by_group(df, 'forecast_center_lat', 'forecast_center_lon', lon1d, group_cols, sort_col)
    t_dx_all, t_dy_all = compute_motion_vectors_by_group(df, 'bst_lat', 'bst_lon', lon1d, group_cols, sort_col)
    forecast_b_param = []
    target_b_param = []
    forecast_vtu = []
    forecast_vtl = []
    target_vtu = []
    target_vtl = []
    thermal_counts = []
    for idx, row in df.iterrows():
        source_file = str(row['source_file'])
        try:
            npz_path = find_npz(npz_root, source_file)
        except FileNotFoundError as exc:
            print(f'[WARN] {exc} (row {idx})')
            forecast_b_param.append(np.nan)
            target_b_param.append(np.nan)
            forecast_vtu.append(np.nan)
            forecast_vtl.append(np.nan)
            target_vtu.append(np.nan)
            target_vtl.append(np.nan)
            thermal_counts.append((0, 0, 0, 0))
            continue
        try:
            forecast_raw, target_raw = load_forecast_and_target(npz_path, target_root)
        except Exception as exc:
            print(f'[WARN] Failed to load {npz_path.name} (row {idx}): {exc}')
            forecast_b_param.append(np.nan)
            target_b_param.append(np.nan)
            forecast_vtu.append(np.nan)
            forecast_vtl.append(np.nan)
            target_vtu.append(np.nan)
            target_vtl.append(np.nan)
            thermal_counts.append((0, 0, 0, 0))
            continue
        H, W = (lat1d.size, lon1d.size)
        forecast = normalize_to_CHW(forecast_raw, H, W)
        target = normalize_to_CHW(target_raw, H, W)
        z600_f = forecast[ch_map['z600'], :, :] / g
        z925_f = forecast[ch_map['z925'], :, :] / g
        z600_t = target[ch_map['z600'], :, :] / g
        z925_t = target[ch_map['z925'], :, :] / g
        thickness_f = z600_f - z925_f
        thickness_t = z600_t - z925_t
        f_lat = float(row['forecast_center_lat'])
        f_lon = normalize_lon(lon1d, float(row['forecast_center_lon']))
        t_lat = float(row['bst_lat'])
        t_lon = normalize_lon(lon1d, float(row['bst_lon']))
        f_left, f_right = left_right_thickness(thickness_f, lat1d, lon1d, f_lat, f_lon, f_dx_all[idx], f_dy_all[idx], radius_km)
        t_left, t_right = left_right_thickness(thickness_t, lat1d, lon1d, t_lat, t_lon, t_dx_all[idx], t_dy_all[idx], radius_km)
        f_b = f_right - f_left if np.isfinite(f_left) and np.isfinite(f_right) else np.nan
        t_b = t_right - t_left if np.isfinite(t_left) and np.isfinite(t_right) else np.nan
        mask_forecast = radius_mask(lat1d, lon1d, f_lat, f_lon, radius_km)
        mask_target = radius_mask(lat1d, lon1d, t_lat, t_lon, radius_km)
        lvl_f_all, delta_f_all = z_range_by_levels(forecast, ch_map, available_levels, mask_forecast, g)
        lvl_t_all, delta_t_all = z_range_by_levels(target, ch_map, available_levels, mask_target, g)
        vt_f = compute_native_vt(lvl_f_all, delta_f_all)
        vt_t = compute_native_vt(lvl_t_all, delta_t_all)
        forecast_vtl.append(vt_f['vt_lower'])
        forecast_vtu.append(vt_f['vt_upper'])
        target_vtl.append(vt_t['vt_lower'])
        target_vtu.append(vt_t['vt_upper'])
        thermal_counts.append((vt_f['lower_valid_levels'], vt_f['upper_valid_levels'],
                               vt_t['lower_valid_levels'], vt_t['upper_valid_levels']))
        forecast_b_param.append(f_b)
        target_b_param.append(t_b)
        if (idx + 1) % 200 == 0:
            print(f'Processed {idx + 1}/{len(df)} rows')
    df = df.copy()
    df['forecast_b_param'] = forecast_b_param
    df['forecast_vtu'] = forecast_vtu
    df['forecast_vtl'] = forecast_vtl
    df['target_b_param'] = target_b_param
    df['target_vtu'] = target_vtu
    df['target_vtl'] = target_vtl
    df.to_csv(output_csv, index=False)
    write_thermal_metadata(output_csv, thermal_counts)
    print(f'Saved updated CSV to {output_csv}')
