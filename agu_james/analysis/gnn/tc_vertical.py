# Derived from gnn/ver_autoregressive/output/tc_axisymmetric_wcc_to_npz.py
# See provenance/sources.json for original hash and modifications.
from __future__ import annotations
from agu_james import runtime
import re
from pathlib import Path
import numpy as np
import pandas as pd
from agu_james.analysis.gnn.tc_base import load_forecast_and_target, normalize_lon, normalize_to_CHW, to_1d_lat_lon

def haversine_deg(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return np.degrees(c)

def compute_axisymmetric_profile_deg(field2d: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray, center_lat: float, center_lon: float, radius_edges_deg: np.ndarray) -> np.ndarray:
    lat2d = lat1d[:, None]
    lon2d = lon1d[None, :]
    dist_deg = haversine_deg(lat2d, lon2d, center_lat, center_lon)
    prof = np.full(radius_edges_deg.size - 1, np.nan, dtype=np.float32)
    for i in range(radius_edges_deg.size - 1):
        r0 = radius_edges_deg[i]
        r1 = radius_edges_deg[i + 1]
        mask = (dist_deg >= r0) & (dist_deg < r1)
        if np.any(mask):
            prof[i] = np.float32(np.nanmean(field2d[mask]))
    return prof

def compute_horizontal_divergence(u: np.ndarray, v: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    a = 6371000.0
    phi = np.radians(lat1d.astype(np.float64))
    lam = np.radians(lon1d.astype(np.float64))
    cos_phi = np.cos(phi)
    cos_safe = np.where(np.abs(cos_phi) < 1e-06, np.nan, cos_phi)
    dudlam = np.gradient(u.astype(np.float64), lam, axis=1, edge_order=2)
    dvdphi = np.gradient(v.astype(np.float64), phi, axis=0, edge_order=2)
    term1 = dudlam / cos_safe[:, None]
    div = (term1 + dvdphi) / a
    return div.astype(np.float32)

def build_npz_index(npz_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(npz_root.rglob('*.npz')):
        index.setdefault(path.name, path)
    return index

def extract_pressure_levels(feature_table: Path) -> list[int]:
    feat = pd.read_csv(feature_table)['feature'].astype(str)
    levels = set()
    for name in feat:
        m = re.match('^(\\d+)[a-z]+$', name, flags=re.IGNORECASE)
        if m:
            levels.add(int(m.group(1)))
    if not levels:
        raise ValueError(f'No pressure levels found in {feature_table}')
    return sorted(levels)

def build_level_map(feature_table: Path, levels: list[int], suffix: str) -> dict[int, int | None]:
    features = pd.read_csv(feature_table)['feature'].astype(str).str.lower().tolist()
    idx_map = {name: i for i, name in enumerate(features)}
    out: dict[int, int | None] = {}
    for lv in levels:
        out[lv] = idx_map.get(f'{lv}{suffix}'.lower())
    return out

def sanitize_npz_name(name: str) -> str:
    return name.replace('[', '').replace(']', '').replace("'", '')

def main() -> None:
    base_dir = runtime.setting('agu_james.analysis.gnn.tc_vertical.base_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.tc_vertical')
    csv_path = runtime.setting('agu_james.analysis.gnn.tc_vertical.csv_path', lambda: base_dir / 'tc_aurora_intensity35_step28_spectrum.csv')
    npz_root = runtime.setting('agu_james.analysis.gnn.tc_vertical.npz_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_autoregressive_step28_spectrum_new_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    target_root = runtime.setting('agu_james.analysis.gnn.tc_vertical.target_root', lambda: base_dir / 'UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2')
    lat_path = runtime.setting('agu_james.analysis.gnn.tc_vertical.lat_path', lambda: base_dir / 'lat.npy')
    lon_path = runtime.setting('agu_james.analysis.gnn.tc_vertical.lon_path', lambda: base_dir / 'lon.npy')
    feature_table = runtime.setting('agu_james.analysis.gnn.tc_vertical.feature_table', lambda: base_dir / 'next_feature_table_weight.csv')
    output_dir = runtime.setting('agu_james.analysis.gnn.tc_vertical.output_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.tc_vertical')
    max_radius_deg = 10.0
    bin_deg = 0.25
    max_groups: int | None = None
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path).reset_index(drop=False).rename(columns={'index': 'row_index'})
    lat1d, lon1d = to_1d_lat_lon(np.load(lat_path), np.load(lon_path))
    H, W = (lat1d.size, lon1d.size)
    levels = extract_pressure_levels(feature_table)
    w_map = build_level_map(feature_table, levels, 'w')
    u_map = build_level_map(feature_table, levels, 'u')
    v_map = build_level_map(feature_table, levels, 'v')
    radius_edges = np.arange(0.0, max_radius_deg + bin_deg, bin_deg, dtype=np.float32)
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    npz_index = build_npz_index(npz_root)
    grouped = df.groupby('source_file', sort=False)
    total_groups = len(grouped)
    skipped_bst_nan = 0
    for gi, (source_file, sub_df) in enumerate(grouped, start=1):
        if max_groups is not None and gi > max_groups:
            break
        npz_path = npz_index.get(str(source_file))
        if npz_path is None:
            print(f'[WARN] Missing forecast npz for {source_file}, skipped')
            continue
        valid_mask = sub_df['bst_lat'].notna() & sub_df['bst_lon'].notna()
        dropped = int((~valid_mask).sum())
        if dropped > 0:
            skipped_bst_nan += dropped
            sub_df = sub_df.loc[valid_mask].copy()
        if sub_df.empty:
            continue
        try:
            forecast_raw, target_raw = load_forecast_and_target(npz_path, target_root)
        except Exception as exc:
            print(f'[WARN] Failed to load {source_file}: {exc}')
            continue
        forecast = normalize_to_CHW(forecast_raw, H, W)
        target = normalize_to_CHW(target_raw, H, W)
        n_case = len(sub_df)
        n_level = len(levels)
        n_bin = len(radius_centers)
        fw = np.full((n_case, n_level, n_bin), np.nan, dtype=np.float32)
        tw = np.full((n_case, n_level, n_bin), np.nan, dtype=np.float32)
        fdiv = np.full((n_case, n_level, n_bin), np.nan, dtype=np.float32)
        tdiv = np.full((n_case, n_level, n_bin), np.nan, dtype=np.float32)
        for ci, (_, row) in enumerate(sub_df.iterrows()):
            f_lat = float(row['forecast_center_lat'])
            f_lon = normalize_lon(lon1d, float(row['forecast_center_lon']))
            t_lat = float(row['bst_lat'])
            t_lon = normalize_lon(lon1d, float(row['bst_lon']))
            for li, lv in enumerate(levels):
                w_idx = w_map.get(lv)
                u_idx = u_map.get(lv)
                v_idx = v_map.get(lv)
                if w_idx is not None:
                    fw[ci, li, :] = compute_axisymmetric_profile_deg(forecast[w_idx], lat1d, lon1d, f_lat, f_lon, radius_edges)
                    tw[ci, li, :] = compute_axisymmetric_profile_deg(target[w_idx], lat1d, lon1d, t_lat, t_lon, radius_edges)
                if u_idx is not None and v_idx is not None:
                    f_div2d = compute_horizontal_divergence(forecast[u_idx], forecast[v_idx], lat1d, lon1d)
                    t_div2d = compute_horizontal_divergence(target[u_idx], target[v_idx], lat1d, lon1d)
                    fdiv[ci, li, :] = compute_axisymmetric_profile_deg(f_div2d, lat1d, lon1d, f_lat, f_lon, radius_edges)
                    tdiv[ci, li, :] = compute_axisymmetric_profile_deg(t_div2d, lat1d, lon1d, t_lat, t_lon, radius_edges)
        out_name = sanitize_npz_name(str(source_file))
        out_path = output_dir / out_name
        np.savez_compressed(out_path, source_file=str(source_file), row_index=sub_df['row_index'].to_numpy(np.int32), time=sub_df['time'].to_numpy(np.int64), cyclone_num=sub_df['CYCLONE_NUM'].to_numpy(np.int32), step=sub_df['step'].to_numpy(np.int16), forecast_center_lat=sub_df['forecast_center_lat'].to_numpy(np.float32), forecast_center_lon=sub_df['forecast_center_lon'].to_numpy(np.float32), bst_lat=sub_df['bst_lat'].to_numpy(np.float32), bst_lon=sub_df['bst_lon'].to_numpy(np.float32), pressure_levels_hpa=np.asarray(levels, dtype=np.int32), radius_edges_deg=radius_edges.astype(np.float32), radius_centers_deg=radius_centers.astype(np.float32), forecast_w_axisym=fw, target_w_axisym=tw, forecast_div_axisym=fdiv, target_div_axisym=tdiv, forecast_conv_axisym=(-fdiv).astype(np.float32), target_conv_axisym=(-tdiv).astype(np.float32))
        if gi % 200 == 0 or gi == total_groups:
            print(f'Saved {gi}/{total_groups}: {out_path.name}')
    print(f'Done. Output directory: {output_dir}')
    print(f'Skipped rows with missing bst center: {skipped_bst_nan}')
