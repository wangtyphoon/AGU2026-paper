# Derived from gnn/ver_autoregressive/output/rmse.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import os
import glob
import numpy as np
import torch
import pandas as pd
import re
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

def create_channel_weights(device):
    """讀取 next_feature_table_weight.csv，回傳權重張量與對應特徵名稱。

    權重長度固定為 116，CSV 未涵蓋的通道使用 1.0；若 CSV 內含 feature 欄位，
    會拿來標記通道名稱，缺漏時以 channel_XX 補齊。"""
    num_channels = 116
    channel_names = [f'channel_{idx:02d}' for idx in range(num_channels)]
    base = torch.ones(num_channels, device=device)
    df = pd.read_csv('next_feature_table_weight.csv')
    if 'weight' not in df.columns:
        raise KeyError("next_feature_table_weight.csv 缺少 'weight' 欄位")
    weights_tensor = torch.tensor(df['weight'].values, device=device, dtype=torch.float32)
    base[:len(weights_tensor)] = weights_tensor[:num_channels]
    if 'feature' in df.columns:
        feature_names = [str(name) for name in df['feature'].values[:num_channels]]
        channel_names[:len(feature_names)] = feature_names
    seen = {}
    unique_names = []
    for name in channel_names:
        key = name if name else 'channel'
        count = seen.get(key, 0)
        unique_name = key if count == 0 else f'{key}_{count}'
        unique_names.append(unique_name)
        seen[key] = count + 1
    return (base, unique_names)

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

def weighted_rmse_loss(pred, label, channel_weights, spatial_weights, epsilon=1e-08):
    """
    pred, label:       (batch*node, C)
    channel_weights:   (C,)
    spatial_weights:   (H, W)
    """
    Bn, C = pred.shape
    node = spatial_weights.numel()
    assert Bn % node == 0, 'batch*node 必須能被 node 整除'
    batch = Bn // node
    spatial_flat = spatial_weights.view(-1)
    spatial_repeat = spatial_flat.repeat(batch)
    w_c = channel_weights.unsqueeze(0)
    w_s = spatial_repeat.unsqueeze(1)
    total_w = w_s * w_c
    diff2 = (pred - label) ** 2
    weighted = diff2 * total_w
    mse = weighted.mean()
    rmse = torch.sqrt(mse + epsilon)
    return rmse

def per_channel_mse_spatial(pred, label, spatial_weights, use_mask=True, epsilon=1e-16):
    """
    計算每通道 **MSE**，僅依空間權重決定加權方式。
    - use_mask=True: 使用二值遮罩 (spatial_weights > 0 為 1，否則 0)
    - use_mask=False: 使用連續權重 (直接乘 spatial_weights)
    輸出: (C,)
    """
    Bn, C = pred.shape
    node = spatial_weights.numel()
    assert Bn % node == 0, 'batch*node 必須能被 node 整除'
    batch = Bn // node
    sw = spatial_weights
    if use_mask:
        sw = (spatial_weights > 0).to(spatial_weights.dtype)
    spatial_flat = sw.view(-1)
    spatial_repeat = spatial_flat.repeat(batch)
    diff2 = (pred - label) ** 2
    w = spatial_repeat.unsqueeze(1)
    weighted = diff2 * w
    denom = w.sum(dim=0).clamp_min(1e-12)
    mse_c = weighted.sum(dim=0) / denom
    return mse_c

def per_channel_bias_spatial(pred, label, spatial_weights, use_mask=True, epsilon=1e-16):
    """
    計算每通道偏差（BIAS），沿用與 per_channel_mse_spatial 相同的權重處理。
    """
    Bn, C = pred.shape
    node = spatial_weights.numel()
    assert Bn % node == 0, 'batch*node 必須可被 node 整除'
    batch = Bn // node
    sw = spatial_weights
    if use_mask:
        sw = (spatial_weights > 0).to(spatial_weights.dtype)
    spatial_flat = sw.view(-1)
    spatial_repeat = spatial_flat.repeat(batch)
    diff = pred - label
    w = spatial_repeat.unsqueeze(1)
    weighted = diff * w
    denom = w.sum(dim=0).clamp_min(1e-12)
    bias_c = weighted.sum(dim=0) / denom
    return bias_c

def load_pred_label_from_npz(npz_path, target_folder=None):
    """
    嘗試從 .npz 取得 (pred, label) numpy 陣列。
    - 先從同一個檔案尋找常見鍵名 (forecast/pred/y_pred、target/label/y_true)
    - 若 label 不在同檔且提供 target_folder，會以檔名（含時間與 step）在該資料夾尋找對應 target npz
    支援常見形狀：
    - (B, H, W, C) 會自動 reshape 為 (B*H*W, C)
    - (B*H*W, C) 保持不變
    """
    data = np.load(npz_path)
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
            raise FileNotFoundError(f'找不到對應的 target 檔案：{target_path}')
        tgt = np.load(target_path)
        tgt_keys = set(getattr(tgt, 'files', []))
        for k_label in ('target', 'label', 'y_true'):
            if k_label in tgt_keys:
                label = tgt[k_label]
                break
        if label is None:
            raise KeyError(f'{target_path} 找不到可辨識的 target 鍵名 (target/label/y_true)')
    if pred is None or label is None:
        raise KeyError(f'{npz_path} 找不到可辨識的預報或答案鍵名，請在 load_pred_label_from_npz() 中手動指定。')
    hw = None

    def reshape(arr, arr_name):
        nonlocal hw
        if arr.ndim == 4:
            B, H, W, C = arr.shape
            hw_here = (H, W)
            if hw is not None and hw != hw_here:
                raise ValueError(f'{npz_path} 的 {arr_name} 空間尺寸與預期不符: {hw} vs {hw_here}')
            hw = hw_here
            return arr.reshape(B * H * W, C)
        if arr.ndim == 2:
            return arr
        raise ValueError(f'{npz_path} 的 {arr_name} 形狀不支援: {arr.shape}')
    pred = reshape(pred, 'pred')
    label = reshape(label, 'label')
    if pred.shape != label.shape:
        raise ValueError(f'{npz_path} 的預報與答案形狀不一致: pred {pred.shape}, label {label.shape}')
    return (pred, label, hw)

def parse_timestamp_from_filename(path, time_format='%Y%m%d%H'):
    """自檔名解析時間，失敗則回傳 None"""
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
    """
    依照 val_autoregressive 輸出的檔名慣例，從檔名取出 step 編號。
    支援 ..._step01.npz 或 ...step01.npz，未找到則回傳 None。
    """
    name = os.path.basename(path)
    m = re.search('_step(\\d+)', name)
    if m:
        return int(m.group(1))
    m = re.search('step(\\d+)', name)
    return int(m.group(1)) if m else None

def process_single_file(file_path, channel_weights_np, channel_names, base_spatial_weights_np, H, W, boundary, device_str, use_mask_for_channel_rmse, temporal_breakdown, target_folder=None):
    """
    處理單一 .npz 檔案，回傳彙整所需的數值（方便平行化）。
    僅使用 numpy 作為跨程序傳遞的容器，避免 GPU 資源衝突。
    """
    device = torch.device(device_str)
    pred_np, label_np, hw = load_pred_label_from_npz(file_path, target_folder=target_folder)
    if hw is not None:
        H_here, W_here = hw
        if (H_here, W_here) == (H, W):
            sp_w_here = torch.tensor(base_spatial_weights_np, device=device)
        else:
            sp_w_here = create_spatial_weights(H=H_here, W=W_here, boundary=boundary, device=device)
    else:
        sp_w_here = torch.tensor(base_spatial_weights_np, device=device)
    ch_w = torch.tensor(channel_weights_np, device=device)
    pred_t = torch.tensor(pred_np, dtype=torch.float32, device=device)
    label_t = torch.tensor(label_np, dtype=torch.float32, device=device)
    overall_rmse = weighted_rmse_loss(pred_t, label_t, ch_w, sp_w_here).item()
    mse_c = per_channel_mse_spatial(pred_t, label_t, sp_w_here, use_mask=use_mask_for_channel_rmse)
    bias_c = per_channel_bias_spatial(pred_t, label_t, sp_w_here, use_mask=use_mask_for_channel_rmse)
    mse_c_cpu = mse_c.detach().cpu().numpy()
    bias_c_cpu = bias_c.detach().cpu().numpy()
    step_idx = parse_step_from_filename(file_path)
    per_time_records = []
    month_accum = {}
    hour_accum = {}
    warnings = []
    if temporal_breakdown:
        ts = parse_timestamp_from_filename(file_path)
        if ts is None:
            warnings.append(f'警告：無法自檔名解析時間資訊，已略過 {os.path.basename(file_path)} 的時間統計')
        else:
            mse_per_channel = np.array(mse_c_cpu, dtype=np.float64, copy=True)
            rmse_per_channel = np.sqrt(mse_per_channel)
            timestamp_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            file_name = os.path.basename(file_path)
            step_idx_for_time = step_idx if step_idx is not None else 0
            for ch_idx, rmse_val in enumerate(rmse_per_channel):
                feature_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'channel_{ch_idx:02d}'
                per_time_records.append({'file': file_name, 'timestamp': timestamp_str, 'step': step_idx_for_time, 'channel': ch_idx, 'feature': feature_name, 'rmse': float(rmse_val)})
            month_key = (ts.year, ts.month, step_idx_for_time)
            month_accum[month_key] = {'sum': mse_per_channel.copy(), 'count': 1}
            hour_key = (ts.hour, step_idx_for_time)
            hour_accum[hour_key] = {'sum': mse_per_channel.copy(), 'count': 1}
    return {'file': os.path.basename(file_path), 'overall_rmse': overall_rmse, 'mse_c': mse_c_cpu, 'bias_c': bias_c_cpu, 'step': step_idx, 'per_time_records': per_time_records, 'month_accum': month_accum, 'hour_accum': hour_accum, 'warnings': warnings}

def main(folder='.', target_folder=None, H=192, W=192, boundary=8, device=None, use_mask_for_channel_rmse=True, save_csv='per_channel_rmse_summary.csv', temporal_breakdown=False, per_time_csv='per_time_channel_rmse.csv', per_month_csv='per_month_channel_rmse.csv', per_hour_csv='per_hour_channel_rmse.csv', num_workers=0):
    device = torch.device(device) if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if num_workers and num_workers > 1 and (device.type == 'cuda'):
        print('警告：num_workers>1 且使用 CUDA，為避免 GPU 多進程衝突，已改為單執行緒。')
        num_workers = 0
    if target_folder is None:
        target_folder = runtime.setting('agu_james.analysis.gnn.rmse.target_folder', lambda: folder)
    ch_w, channel_names = create_channel_weights(device=device)
    sp_w = create_spatial_weights(H=H, W=W, boundary=boundary, device=device)
    channel_weights_np = ch_w.detach().cpu().numpy()
    base_spatial_weights_np = sp_w.detach().cpu().numpy()
    device_str = str(device)
    npz_files = sorted(glob.glob(os.path.join(folder, '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'資料夾 {folder} 下找不到任何 .npz 檔案')
    per_file_rows = []
    per_chan_mse_list = []
    per_chan_bias_list = []
    per_time_records = []
    month_accum = {}
    hour_accum = {}
    per_step_mse = {}
    per_step_bias = {}
    warnings = []

    def add_result(res):
        per_file_rows.append({'file': res['file'], 'overall_weighted_rmse': res['overall_rmse']})
        per_chan_mse_list.append(res['mse_c'])
        per_chan_bias_list.append(res['bias_c'])
        step_idx = res['step']
        if step_idx is not None:
            per_step_mse.setdefault(step_idx, []).append(res['mse_c'])
            per_step_bias.setdefault(step_idx, []).append(res['bias_c'])
        per_time_records.extend(res['per_time_records'])
        warnings.extend(res['warnings'])
        for key, data in res['month_accum'].items():
            if key not in month_accum:
                month_accum[key] = {'sum': data['sum'].copy(), 'count': data['count']}
            else:
                month_accum[key]['sum'] += data['sum']
                month_accum[key]['count'] += data['count']
        for key, data in res['hour_accum'].items():
            if key not in hour_accum:
                hour_accum[key] = {'sum': data['sum'].copy(), 'count': data['count']}
            else:
                hour_accum[key]['sum'] += data['sum']
                hour_accum[key]['count'] += data['count']
        print(f"{res['file']}: Overall weighted RMSE = {res['overall_rmse']}")
    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_file, f, channel_weights_np, channel_names, base_spatial_weights_np, H, W, boundary, device_str, use_mask_for_channel_rmse, temporal_breakdown, target_folder) for f in npz_files]
            for fut in as_completed(futures):
                res = fut.result()
                add_result(res)
    else:
        for f in npz_files:
            res = process_single_file(f, channel_weights_np, channel_names, base_spatial_weights_np, H, W, boundary, device_str, use_mask_for_channel_rmse, temporal_breakdown, target_folder)
            add_result(res)
    per_file_df = pd.DataFrame(per_file_rows).sort_values('file')
    per_chan_mse_arr = np.stack(per_chan_mse_list, axis=0)
    per_chan_bias_arr = np.stack(per_chan_bias_list, axis=0)
    per_chan_mse_mean = per_chan_mse_arr.mean(axis=0)
    per_chan_bias_mean = per_chan_bias_arr.mean(axis=0)
    per_chan_rmse_mean = np.sqrt(per_chan_mse_mean)
    channel_cols = channel_names[:len(per_chan_rmse_mean)]
    summary_dict = {'feature': channel_cols, 'rmse_overall': per_chan_rmse_mean[:len(channel_cols)], 'bias_overall': per_chan_bias_mean[:len(channel_cols)]}
    for step_idx in sorted(per_step_mse.keys()):
        mse_step_arr = np.stack(per_step_mse[step_idx], axis=0)
        bias_step_arr = np.stack(per_step_bias[step_idx], axis=0)
        mse_step_mean = mse_step_arr.mean(axis=0)
        bias_step_mean = bias_step_arr.mean(axis=0)
        summary_dict[f'rmse_step{step_idx:02d}'] = np.sqrt(mse_step_mean)[:len(channel_cols)]
        summary_dict[f'bias_step{step_idx:02d}'] = bias_step_mean[:len(channel_cols)]
    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv(save_csv, index=False)
    print('=== Per-file overall weighted RMSE ===')
    print(per_file_df.to_string(index=False))
    print('\n=== Mean per-channel metrics (overall + per-step; using ' + ('spatial mask)' if use_mask_for_channel_rmse else 'spatial weights)'))
    print(summary_df.head(10).to_string(index=False))
    print(f'\nSaved per-channel RMSE/bias to: {save_csv}')
    if temporal_breakdown:
        if per_time_records:
            per_time_df = pd.DataFrame(per_time_records)
            per_time_df.sort_values(['timestamp', 'step', 'channel'], inplace=True)
            per_time_df['feature_step'] = per_time_df.apply(lambda r: f"{r['feature']}_step{int(r['step']):02d}", axis=1)
            step_values = sorted(per_time_df['step'].unique())
            feature_step_cols = [f'{feat}_step{int(s):02d}' for s in step_values for feat in channel_cols]
            per_time_pivot = per_time_df.pivot_table(index=['file', 'timestamp'], columns='feature_step', values='rmse', aggfunc='mean')
            per_time_pivot = per_time_pivot.reindex(columns=feature_step_cols)
            per_time_pivot.sort_index(inplace=True)
            per_time_pivot.columns.name = None
            per_time_pivot = per_time_pivot.reset_index()
            per_time_pivot.to_csv(per_time_csv, index=False)
            month_rows = []
            for (year, month, step_val), data in sorted(month_accum.items()):
                mean_mse = data['sum'] / max(data['count'], 1)
                rmse_vals = np.sqrt(mean_mse)
                for ch_idx, rmse_val in enumerate(rmse_vals):
                    feature_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'channel_{ch_idx:02d}'
                    month_rows.append({'year': year, 'month': month, 'step': step_val, 'channel': ch_idx, 'feature': feature_name, 'rmse': float(rmse_val)})
            if month_rows:
                month_df = pd.DataFrame(month_rows)
                month_df['feature_step'] = month_df.apply(lambda r: f"{r['feature']}_step{int(r['step']):02d}", axis=1)
                month_pivot = month_df.pivot_table(index=['year', 'month'], columns='feature_step', values='rmse', aggfunc='mean')
                month_pivot = month_pivot.reindex(columns=feature_step_cols)
                month_pivot.sort_index(inplace=True)
                month_pivot.columns.name = None
                month_pivot = month_pivot.reset_index()
                month_pivot.to_csv(per_month_csv, index=False)
            hour_rows = []
            for (hour, step_val), data in sorted(hour_accum.items()):
                mean_mse = data['sum'] / max(data['count'], 1)
                rmse_vals = np.sqrt(mean_mse)
                for ch_idx, rmse_val in enumerate(rmse_vals):
                    feature_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'channel_{ch_idx:02d}'
                    hour_rows.append({'hour': hour, 'step': step_val, 'channel': ch_idx, 'feature': feature_name, 'rmse': float(rmse_val)})
            if hour_rows:
                hour_df = pd.DataFrame(hour_rows)
                hour_df['feature_step'] = hour_df.apply(lambda r: f"{r['feature']}_step{int(r['step']):02d}", axis=1)
                hour_pivot = hour_df.pivot_table(index='hour', columns='feature_step', values='rmse', aggfunc='mean')
                hour_pivot = hour_pivot.reindex(columns=feature_step_cols)
                hour_pivot.sort_index(inplace=True)
                hour_pivot.columns.name = None
                hour_pivot = hour_pivot.reset_index()
                hour_pivot.to_csv(per_hour_csv, index=False)
            print(f'\n已輸出時間分組 RMSE：{per_time_csv}, {per_month_csv}, {per_hour_csv}')
        else:
            print('\n警告：啟用時間分組，但未能自檔名解析任何時間資訊。')
    for msg in warnings:
        print(msg)
