# Derived from gnn/ver_autoregressive/output/download_clim.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import os
import re
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
FEATURE_CSV = runtime.setting('agu_james.analysis.gnn.climatology.FEATURE_CSV', lambda: 'next_feature_table_weight.csv')
ZARR_PATH = 'weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr'
OUTPUT_DIR = runtime.setting('agu_james.analysis.gnn.climatology.OUTPUT_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.climatology')
FEATURE_VAR_ALIASES = {'t': ['temperature', 't'], 'u': ['u_component_of_wind', 'u'], 'v': ['v_component_of_wind', 'v'], 'w': ['vertical_velocity', 'w'], 'q': ['specific_humidity', 'q'], 'z': ['geopotential', 'z', 'geopotential_height'], 'pv': ['potential_vorticity', 'pv'], 'cc': ['cloud_cover', 'fraction_of_cloud_cover', 'cc'], 't2m': ['2m_temperature', 't2m'], 'd2m': ['2m_dewpoint_temperature', 'd2m'], 'u10': ['10m_u_component_of_wind', 'u10'], 'v10': ['10m_v_component_of_wind', 'v10'], 'msl': ['mean_sea_level_pressure', 'msl'], 'sst': ['sea_surface_temperature', 'sst'], 'tp': ['total_precipitation', 'tp'], 'blh': ['boundary_layer_height', 'blh'], 'slhf': ['mean_surface_latent_heat_flux', 'slhf'], 'sshf': ['mean_surface_sensible_heat_flux', 'sshf'], 'ssr': ['mean_surface_net_short_wave_radiation_flux', 'ssr', 'surface_solar_radiation_downwards'], 'str': ['mean_surface_net_long_wave_radiation_flux', 'str', 'surface_thermal_radiation_downwards'], 'swh': ['significant_height_of_combined_wind_waves_and_swell', 'swh'], 'flsr': ['forecast_log_surface_roughness', 'flsr'], 'ewss': ['eastward_turbulent_surface_stress', 'ewss'], 'nsss': ['northward_turbulent_surface_stress', 'nsss'], 'lgws': ['large_scale_precipitation', 'lgws'], 'mgws': ['mean_wave_direction', 'mgws']}

def parse_feature(feature: str):
    match = re.match('^(\\d+)([a-z]+)$', feature)
    if match:
        return (int(match.group(1)), match.group(2))
    return (None, feature)

def resolve_var_name(var_code: str, data_vars):
    candidates = FEATURE_VAR_ALIASES.get(var_code, []) + [var_code]
    for name in candidates:
        if name in data_vars:
            return name
    return None

def pick_level_dim(dims):
    for name in ('level', 'pressure_level', 'plev', 'isobaricInhPa'):
        if name in dims:
            return name
    return None

def main():
    features = pd.read_csv(FEATURE_CSV)['feature'].astype(str).tolist()
    lat = np.load('lat.npy')
    lon = np.load('lon.npy')
    lat_min, lat_max = (np.min(lat), np.max(lat))
    lon_min, lon_max = (np.min(lon), np.max(lon))
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper(ZARR_PATH)
    ds_clim = xr.open_zarr(store, consolidated=True)
    data_vars = set(ds_clim.data_vars)
    missing = []
    output_dir = runtime.setting('agu_james.analysis.gnn.climatology.output_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.analysis.gnn.climatology')
    os.makedirs(output_dir, exist_ok=True)
    feature_specs = {}
    for feature in features:
        level, var_code = parse_feature(feature)
        var_name = resolve_var_name(var_code, data_vars)
        if not var_name:
            missing.append(feature)
            print(f'[MISS] {feature} -> no matching variable in ds_clim')
            continue
        da = ds_clim[var_name]
        level_dim = pick_level_dim(da.dims)
        if level is not None and level_dim:
            da = da.sel({level_dim: level})
        elif level is not None and (not level_dim):
            missing.append(feature)
            print(f'[MISS] {feature} -> no pressure level dimension in {var_name}')
            continue
        feature_specs[feature] = (var_name, da)
        print(f'[OK] {feature} -> {var_name}, dims={da.dims}, shape={da.shape}')
    if missing:
        print(f'Missing {len(missing)} features:', missing)
    if not feature_specs:
        print('No features available to save.')
        return
    time_dim = None
    if 'time' in ds_clim.dims:
        time_dim = 'time'
        time_vals = ds_clim['time'].values
        time_indices = list(enumerate(time_vals))
    else:
        hour_vals = ds_clim['hour'].values if 'hour' in ds_clim.dims else [0]
        day_vals = ds_clim['dayofyear'].values if 'dayofyear' in ds_clim.dims else [0]
        time_indices = [(day_idx, hour_idx, day_val, hour_val) for day_idx, day_val in enumerate(day_vals) for hour_idx, hour_val in enumerate(hour_vals)]
    print(f'Preparing to save data for {len(time_indices)} time steps.')
    if time_dim == 'time':
        for idx, time_val in time_indices:
            out_path = os.path.join(output_dir, f'time_{idx:04d}.npz')
            if os.path.exists(out_path):
                print(f'[SKIP] time_{idx:04d} already exists')
                continue
            payload = {'time': time_val, 'lat': lat, 'lon': lon}
            for feature, (_, da) in feature_specs.items():
                da_sel = da
                if 'time' in da_sel.dims:
                    da_sel = da_sel.isel(time=idx)
                da_sel = da_sel.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                payload[feature] = da_sel.load().values
            np.savez(out_path, **payload)
    elif time_indices:
        for day_idx, hour_idx, day_val, hour_val in time_indices:
            out_path = os.path.join(output_dir, f'day{int(day_val):03d}_hour{int(hour_val):02d}.npz')
            if os.path.exists(out_path):
                print(f'[SKIP] day{int(day_val):03d}_hour{int(hour_val):02d} already exists')
                continue
            payload = {'dayofyear': day_val, 'hour': hour_val, 'lat': lat, 'lon': lon}
            for feature, (_, da) in feature_specs.items():
                da_sel = da
                if 'dayofyear' in da_sel.dims:
                    da_sel = da_sel.isel(dayofyear=day_idx)
                if 'hour' in da_sel.dims:
                    da_sel = da_sel.isel(hour=hour_idx)
                da_sel = da_sel.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                payload[feature] = da_sel.load().values
            np.savez(out_path, **payload)
    else:
        payload = {'lat': lat, 'lon': lon}
        for feature, (_, da) in feature_specs.items():
            da_sel = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
            payload[feature] = da_sel.load().values
        out_path = os.path.join(output_dir, 'clim_all.npz')
        if os.path.exists(out_path):
            print('[SKIP] clim_all.npz already exists')
            return
        np.savez(out_path, **payload)
