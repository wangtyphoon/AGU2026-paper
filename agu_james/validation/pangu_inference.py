# Derived from func_inference.py
from agu_james import runtime
import os
import numpy as np
import onnxruntime as ort
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd

def _normalize_path(path: str) -> str:
    return None if path is None else os.path.normpath(os.fspath(path))

def _build_region_selector(nc_path: str, lat_range=(0.0, 47.75), lon_range=(102.25, 150.0)):
    """
    Mirror validation/subset.py logic to find the lat/lon slice for a target region.
    Returns (lat_slice, lon_slice, lat_values, lon_values).
    """
    nc_path = _normalize_path(nc_path)
    with Dataset(nc_path) as ds:
        lats = np.asarray(ds.variables['latitude'][:])
        lons = np.asarray(ds.variables['longitude'][:])
    lons = np.mod(lons, 360.0)
    lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
    if not lat_mask.any():
        raise ValueError(f'lat_range {lat_range} selects no points in {nc_path}')
    if not lon_mask.any():
        raise ValueError(f'lon_range {lon_range} selects no points in {nc_path}')
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    lat_sl = slice(lat_idx.min(), lat_idx.max() + 1)
    lon_sl = slice(lon_idx.min(), lon_idx.max() + 1)
    return ((lat_sl, lon_sl), lats[lat_sl], lons[lon_sl])

def run_pangu_inference(start_time: datetime, end_time: datetime, model_paths: dict, base_dir: str=None, use_gpu: bool=True, intra_threads: int=30, crop_region: bool=True, lat_range=(0.0, 47.75), lon_range=(102.25, 150.0), reference_nc: str='upper.nc', overwrite: bool=False, save_every_hours: int=6, direct_hours: tuple[int, ...]=(24, 48, 72), save_after_hours: int=72) -> None:
    """
    Run Pangu-Weather ONNX inference from start_time to end_time.

    每次往前推最大的 interval，直到 current_time >= end_time。若 crop_region=True，
    會依照 validation/subset.py 的預設範圍 (lat_range/lon_range) 只儲存子域資料，
    以降低磁碟用量；推理鏈仍在記憶體中使用全域格點資料。

    save_every_hours: 連鎖推理時的存檔間隔（例如 6 小時）；若為 0/None 則只在末端存檔。
    direct_hours: 多個「直推」小時數；若該小時數 <= 預報長度且在 model_paths 中存在，
                  則會直接用對應模型跑一次，並以直推輸出作為後續延伸的起點。
    """
    if base_dir is None:
        base_dir = runtime.path('pangu_root', runtime.artifact_root() / 'pangu')
    else:
        base_dir = _normalize_path(base_dir)
    forecasts_dir = os.path.join(base_dir, 'forecasts', start_time.strftime('%Y-%m-%d-%H-%M'))
    results_dir = os.path.join(base_dir, 'results', f"{start_time.strftime('%Y-%m-%d-%H-%M')}")
    if os.path.exists(results_dir):
        if not overwrite:
            print(f'[INFO] results_dir 已存在且 overwrite=False，跳過運算：{results_dir}')
            return
    os.makedirs(results_dir, exist_ok=True)
    region_selector = None
    lat_subset = None
    lon_subset = None
    nc_for_slice = reference_nc
    if crop_region:
        region_selector, lat_subset, lon_subset = _build_region_selector(nc_for_slice, lat_range=lat_range, lon_range=lon_range)
        np.save(os.path.join(results_dir, 'latitudes_subset.npy'), lat_subset)
        np.save(os.path.join(results_dir, 'longitudes_subset.npy'), lon_subset)
    total_hours = int((end_time - start_time).total_seconds() // 3600)
    if total_hours <= 0:
        print(f'[INFO] end_time 不大於 start_time，跳過：{start_time} -> {end_time}')
        return
    direct_hours_sorted = sorted((h for h in direct_hours if h is not None and h > 0 and (h <= total_hours)))
    init_upper = np.load(os.path.join(forecasts_dir, 'input_upper.npy')).astype(np.float32)
    init_surface = np.load(os.path.join(forecasts_dir, 'input_surface.npy')).astype(np.float32)
    session_cache: dict[str, ort.InferenceSession] = {}

    def _ensure_session(model_file: str) -> ort.InferenceSession:
        if model_file in session_cache:
            return session_cache[model_file]
        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False
        sess_opts.enable_mem_reuse = False
        sess_opts.intra_op_num_threads = intra_threads
        providers = ['CPUExecutionProvider']
        if use_gpu:
            cuda_opts = {'arena_extend_strategy': 'kSameAsRequested'}
            providers = [('CUDAExecutionProvider', cuda_opts)]
        ort_sess = ort.InferenceSession(model_file, sess_options=sess_opts, providers=providers)
        session_cache[model_file] = ort_sess
        return ort_sess

    def _run_chain_step(target_time: datetime, current_time: datetime, upper, surface):
        out_u = out_s = None
        while current_time < target_time:
            remaining_hours = (target_time - current_time).total_seconds() / 3600
            for interval in sorted(model_paths.keys(), reverse=True):
                if interval <= remaining_hours:
                    break
            model_file = model_paths[interval]
            print(f'[{current_time}] → +{interval}h  使用模型：{os.path.basename(model_file)}')
            ort_session = _ensure_session(model_file)
            out_u, out_s = ort_session.run(None, {'input': upper, 'input_surface': surface})
            upper = out_u.astype(np.float32)
            surface = out_s.astype(np.float32)
            current_time += timedelta(hours=interval)
        return (out_u, out_s, upper, surface, current_time)
    upper = init_upper.copy()
    surface = init_surface.copy()
    current_time = start_time
    saved_times: set[str] = set()
    segment_base_upper = init_upper.copy()
    segment_base_surface = init_surface.copy()
    last_direct_hour = 0

    def _save_output(ts: datetime, out_u, out_s):
        stamp = ts.strftime('%Y-%m-%d-%H-%M')
        if stamp in saved_times:
            return
        upper_to_save = out_u
        surface_to_save = out_s
        if crop_region and region_selector is not None:
            lat_sl, lon_sl = region_selector
            upper_to_save = out_u[:, :, lat_sl, lon_sl]
            surface_to_save = out_s[:, lat_sl, lon_sl]
        np.save(os.path.join(results_dir, f'output_upper_{stamp}.npy'), upper_to_save)
        np.save(os.path.join(results_dir, f'output_surface_{stamp}.npy'), surface_to_save)
        saved_times.add(stamp)

    def _iter_checkpoints(start_h: int, end_h: int):
        if not save_every_hours:
            return []
        h = start_h + save_every_hours
        checkpoints = []
        while h < end_h:
            if h >= save_after_hours:
                checkpoints.append(start_time + timedelta(hours=h))
            h += save_every_hours
        return checkpoints
    for direct_h in direct_hours_sorted:
        target_time = start_time + timedelta(hours=direct_h)
        for ck_time in _iter_checkpoints(last_direct_hour, direct_h):
            out_upper, out_surface, upper, surface, current_time = _run_chain_step(ck_time, current_time, upper, surface)
            if out_upper is not None and out_surface is not None:
                _save_output(ck_time, out_upper, out_surface)
        interval = direct_h - last_direct_hour
        if interval in model_paths:
            model_file = model_paths[interval]
            print(f'[DIRECT] {start_time} → +{interval}h  使用模型：{os.path.basename(model_file)}')
            ort_session = _ensure_session(model_file)
            direct_upper, direct_surface = ort_session.run(None, {'input': segment_base_upper, 'input_surface': segment_base_surface})
            _save_output(target_time, direct_upper, direct_surface)
            segment_base_upper = direct_upper.astype(np.float32)
            segment_base_surface = direct_surface.astype(np.float32)
            upper = segment_base_upper.copy()
            surface = segment_base_surface.copy()
            current_time = target_time
        else:
            out_upper, out_surface, upper, surface, current_time = _run_chain_step(target_time, current_time, upper, surface)
            if out_upper is not None and out_surface is not None:
                _save_output(target_time, out_upper, out_surface)
                segment_base_upper = upper.copy()
                segment_base_surface = surface.copy()
        last_direct_hour = direct_h
    if last_direct_hour < total_hours:
        for ck_time in _iter_checkpoints(last_direct_hour, total_hours):
            out_upper, out_surface, upper, surface, current_time = _run_chain_step(ck_time, current_time, upper, surface)
            if out_upper is not None and out_surface is not None:
                _save_output(ck_time, out_upper, out_surface)
        out_upper, out_surface, upper, surface, current_time = _run_chain_step(end_time, current_time, upper, surface)
        if out_upper is not None and out_surface is not None:
            _save_output(end_time, out_upper, out_surface)
    print('Inference complete.')

def consecutive_6h_steps(files, ascending=True):
    files_sorted = sorted(files, reverse=not ascending)
    times = [datetime.strptime(f[:10], '%Y%m%d%H') for f in files_sorted]
    step_list = [0]
    for i in range(1, len(times)):
        delta = times[i] - times[i - 1]
        target = timedelta(hours=6) if ascending else timedelta(hours=-6)
        if delta == target:
            step_list.append(step_list[-1] + 1)
        else:
            step_list.append(0)
    return step_list if ascending else step_list[::-1]

def filename_to_datetime(fn: str) -> datetime:
    return datetime.strptime(fn[:10], '%Y%m%d%H')
