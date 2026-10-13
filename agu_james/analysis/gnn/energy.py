# Derived from gnn/ver_autoregressive/output/spectrum_all_stat.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import os
import glob
import re
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

def radial_power_spectrum(field2d, area2d=None, detrend_mean=True, amplitude=True, density_norm=False):
    """
    面積加權幅度/功率徑向譜：
      1) 若提供 area2d，先做面積加權去均值：f' = (f - mean_w) * sqrt(A_norm)
      2) 2D FFT → 取 |F| (或 |F|^2)
      3) 以像素半徑做徑向平均
    回傳: k (0..kmax-1), spec
    """
    H, W = field2d.shape
    if area2d is not None:
        if area2d.shape != (H, W):
            raise ValueError(f'area2d 形狀需為 {(H, W)}，目前是 {area2d.shape}')
        A = area2d.astype(np.float64, copy=False)
        A = A / np.nanmean(A)
        if detrend_mean:
            mean_w = np.nansum(field2d * A) / np.nansum(A)
        else:
            mean_w = 0.0
        x = (field2d.astype(np.float64) - mean_w) * np.sqrt(A)
    else:
        x = field2d.astype(np.float64, copy=False)
        if detrend_mean:
            x = x - np.nanmean(x)
    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)
    if amplitude:
        P = np.abs(F)
        if density_norm:
            P = P / (H * W)
    else:
        P = np.abs(F) ** 2
        if density_norm:
            P = P / (H * W) ** 2
    cy, cx = (H // 2, W // 2)
    y, xg = np.indices((H, W))
    r = np.sqrt((xg - cx) ** 2 + (y - cy) ** 2).astype(int)
    kmax = min(cx, cy) + 1
    spec_sum = np.bincount(r.ravel(), weights=P.ravel(), minlength=kmax)
    spec_cnt = np.bincount(r.ravel(), minlength=kmax)
    spec = spec_sum[:kmax] / np.maximum(spec_cnt[:kmax], 1)
    k = np.arange(kmax)
    return (k, spec)

def parse_timestamp_from_filename(path, time_format='%Y%m%d%H'):
    """自檔名解析時間，失敗則回傳 None。"""
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

def load_pred_label_from_npz(npz_path, target_folder=None):
    data = np.load(npz_path)
    pred = data.get('forecast')
    label = data.get('target')
    if pred is None:
        raise KeyError(f'{npz_path} 缺少 forecast 鍵')
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
        label = tgt.get('target')
        if label is None:
            raise KeyError(f'{target_path} 找不到 target 鍵')
    if label is None:
        raise KeyError(f'{npz_path} 缺少 target 鍵，且未提供 target_folder')
    if pred.ndim == 4:
        B, H, W, C = pred.shape
        hw = (H, W)
        return (pred, label, hw)
    elif pred.ndim == 2:
        return (pred, label, None)
    else:
        raise ValueError(f'{npz_path} 的 pred 形狀不支援: {pred.shape}')

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

def _make_empty_acc(names):
    return {nm: {'sum': None, 'sumsq': None, 'cnt': None} for nm in names}

def _truncate_acc(acc, klen):
    for nm in acc:
        for key in ('sum', 'sumsq', 'cnt'):
            if acc[nm][key] is not None:
                acc[nm][key] = acc[nm][key][:, :klen]

def _init_acc(names, C, klen):
    out = {}
    for nm in names:
        out[nm] = {'sum': np.zeros((C, klen), dtype=np.float64), 'sumsq': np.zeros((C, klen), dtype=np.float64), 'cnt': np.zeros((C, klen), dtype=np.int64)}
    return out

def _merge_acc(k_ref, acc, res, names, C, target_klen=None):
    """對齊 k 軸後把單檔統計量併入累計器。"""
    k_new = res['k']
    if k_ref is None:
        k_ref = k_new.copy()
    klen = min(len(k_ref), len(k_new))
    if target_klen is not None:
        klen = min(klen, target_klen)
    if len(k_ref) != klen:
        k_ref = k_ref[:klen]
    if not acc or acc[names[0]]['sum'] is None:
        acc = _init_acc(names, C, klen)
    else:
        cur_len = acc[names[0]]['sum'].shape[1]
        if cur_len != klen:
            _truncate_acc(acc, klen)
    for nm in names:
        la = res['acc'][nm]
        for key in ('sum', 'sumsq', 'cnt'):
            acc[nm][key][:, :klen] += la[key][:, :klen]
    return (k_ref, acc, klen)

def _compute_mean_std(acc, names):
    stats = {}
    for nm in names:
        S = acc[nm]['sum']
        SS = acc[nm]['sumsq']
        N = acc[nm]['cnt'].astype(np.float64)
        mean = np.divide(S, np.maximum(N, 1.0), out=np.zeros_like(S), where=N > 0)
        var_num = SS - S * S / np.maximum(N, 1.0)
        var_den = np.maximum(N - 1.0, 1.0)
        var = np.divide(var_num, var_den, out=np.zeros_like(SS), where=N > 1)
        std = np.sqrt(np.clip(var, 0.0, None))
        stats[nm] = {'mean': mean, 'std': std, 'cnt': N}
    return stats

def _process_one_file(path, H0, W0, C, y0, y1, x0, x1, names, amplitude, detrend_mean, apply_lat_weight, lat_path, target_folder):
    """
    單檔 worker：讀檔→裁切→(可選)面積權重→逐樣本逐通道頻譜→回傳每個 source 的 (sum, sumsq, count)
    為避免大量 IPC，這裡只回傳統計量，shape = (C, klen)
    """
    pred, label, hw = load_pred_label_from_npz(path, target_folder=target_folder)
    if pred.ndim == 2:
        N = pred.shape[0]
        assert N % (H0 * W0) == 0, f'{path} 的 N 不能被 H0*W0={H0 * W0} 整除'
        B = N // (H0 * W0)
        pred = pred.reshape(B, H0, W0, C)
        label = label.reshape(B, H0, W0, C)
    pred_c = pred[:, y0:y1, x0:x1, :]
    label_c = label[:, y0:y1, x0:x1, :]
    if apply_lat_weight:
        lat2d_full = np.load(lat_path)
        lat2d = lat2d_full[y0:y1, x0:x1]
        A2d = np.cos(np.deg2rad(lat2d)).astype(np.float64)
    else:
        A2d = None
    k_ref = None
    acc = {}
    for name in names:
        acc[name] = {'sum': None, 'sumsq': None, 'cnt': None}
    B_here = pred_c.shape[0]
    for b in range(B_here):
        for c in range(C):
            arr_f = pred_c[b, :, :, c]
            arr_t = label_c[b, :, :, c]
            to_compute = {}
            if 'forecast' in names:
                to_compute['forecast'] = arr_f
            if 'target' in names:
                to_compute['target'] = arr_t
            if 'error' in names:
                to_compute['error'] = arr_f - arr_t
            for name, arr in to_compute.items():
                k, spec = radial_power_spectrum(arr, area2d=A2d, detrend_mean=detrend_mean, amplitude=amplitude, density_norm=False)
                if k_ref is None:
                    k_ref = k.copy()
                    klen = len(k_ref)
                    for nm in names:
                        acc[nm]['sum'] = np.zeros((C, klen), dtype=np.float64)
                        acc[nm]['sumsq'] = np.zeros((C, klen), dtype=np.float64)
                        acc[nm]['cnt'] = np.zeros((C, klen), dtype=np.int64)
                else:
                    klen = min(len(k_ref), len(k))
                    if klen < len(k_ref):
                        k_ref = k_ref[:klen]
                        for nm in names:
                            for key in ('sum', 'sumsq', 'cnt'):
                                acc[nm][key] = acc[nm][key][:, :klen]
                    spec = spec[:klen]
                acc[name]['sum'][c, :klen] += spec
                acc[name]['sumsq'][c, :klen] += spec * spec
                acc[name]['cnt'][c, :klen] += 1
    return {'k': k_ref, 'acc': acc, 'C': C}

def _cumfrac(spec_ck: np.ndarray, eps: float=1e-12) -> np.ndarray:
    """
    spec_ck: (C, K) 的頻譜平均或總和（非負）
    回傳同形狀 (C, K) 的累積能量占比 C(k)
    """
    csum = np.cumsum(np.clip(spec_ck, 0.0, None), axis=-1)
    total = csum[:, [-1]]
    return csum / (total + eps)

def main(folder='.', target_folder=None, H=192, W=192, boundary=8, device=None, source='all', amplitude=True, detrend_mean=True, apply_lat_weight=False, lat_path='lat.npy', save_npz='per_channel_spectra.npz', save_csv_per_channel=False, max_workers=None, save_npz_per_step_prefix=None):
    """
    產出：
      - save_npz：包含 k 及各通道的 mean/std（依 source）
      - 檔名含 _stepXX 會額外輸出該 step 的統計（鍵名加上 _stepXX 後綴）
      - 另外輸出：snr (若同時計到 target 與 error)、各 source 的累積能量占比 cumfrac
      - 可選：每通道 CSV
    並行：以「檔案」為粒度做多製程。
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    npz_files = sorted(glob.glob(os.path.join(folder, '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'資料夾 {folder} 下找不到 *.npz')
    if target_folder is None:
        target_folder = runtime.setting('agu_james.analysis.gnn.energy.target_folder', lambda: folder)
    first_pred, first_label, hw = load_pred_label_from_npz(npz_files[0], target_folder=target_folder)
    if first_pred.ndim == 4:
        _, H0, W0, C = first_pred.shape
    else:
        H0, W0 = (H, W)
        C = first_pred.shape[1]
    if boundary < 0:
        raise ValueError('boundary 需為非負整數')
    if H0 <= 2 * boundary or W0 <= 2 * boundary:
        raise ValueError(f'原始大小 {H0}x{W0} 無法以 boundary={boundary} 裁切（需 H0>2b 且 W0>2b）')
    y0, y1 = (boundary, H0 - boundary)
    x0, x1 = (boundary, W0 - boundary)
    Hc, Wc = (y1 - y0, x1 - x0)
    names = []
    if source in ('forecast', 'all'):
        names.append('forecast')
    if source in ('target', 'all'):
        names.append('target')
    if source in ('error', 'all'):
        names.append('error')
    if not names:
        raise ValueError(f'source 需為 forecast/target/error/all，目前為: {source}')
    k_ref = None
    global_acc = _make_empty_acc(names)
    per_step_acc = {}
    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = max(1, cpu - 1)
    worker = partial(_process_one_file, H0=H0, W0=W0, C=C, y0=y0, y1=y1, x0=x0, x1=x1, names=names, amplitude=amplitude, detrend_mean=detrend_mean, apply_lat_weight=apply_lat_weight, lat_path=lat_path, target_folder=target_folder)
    print(f'開始平行處理 {len(npz_files)} 個檔案，max_workers={max_workers} ...')
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, p): p for p in npz_files}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f'[ERROR] {path}: {e}')
                continue
            step_idx = parse_step_from_filename(path)
            k_ref, global_acc, klen = _merge_acc(k_ref, global_acc, res, names, C)
            for info in per_step_acc.values():
                if info['k'] is not None and len(info['k']) > klen:
                    info['k'] = info['k'][:klen]
                    _truncate_acc(info['acc'], klen)
            if step_idx is not None:
                entry = per_step_acc.setdefault(step_idx, {'k': None, 'acc': _make_empty_acc(names)})
                entry['k'], entry['acc'], _ = _merge_acc(entry['k'], entry['acc'], res, names, C, target_klen=klen)
            print(f'完成：{os.path.basename(path)}')
    out = {'k': k_ref}
    stats_overall = _compute_mean_std(global_acc, names)
    for nm in names:
        mean = stats_overall[nm]['mean']
        out[f'{nm}_mean'] = mean
        out[f'{nm}_std'] = stats_overall[nm]['std']
        out[f'{nm}_cumfrac'] = _cumfrac(mean)
    eps = 1e-12
    if 'target' in names and 'error' in names:
        signal = out['target_mean']
        noise = out['error_mean']
        snr = signal / (noise + eps)
        out['snr'] = snr
    if 'forecast' in names and 'target' in names:
        f_spec = np.clip(out['forecast_mean'], 0.0, None)
        t_spec = np.clip(out['target_mean'], 0.0, None)
        f_cum = np.cumsum(f_spec, axis=-1)
        t_cum = np.cumsum(t_spec, axis=-1)
        cum_energy_ratio = f_cum / (t_cum + eps)
        out['cum_energy_ratio'] = cum_energy_ratio
    step_outputs = {}
    for step_idx in sorted(per_step_acc.keys()):
        info = per_step_acc[step_idx]
        if info['k'] is None:
            continue
        if len(info['k']) != len(k_ref):
            step_klen = min(len(info['k']), len(k_ref))
            info['k'] = info['k'][:step_klen]
            _truncate_acc(info['acc'], step_klen)
        step_stats = _compute_mean_std(info['acc'], names)
        suffix = f'_step{step_idx:02d}'
        for nm in names:
            mean = step_stats[nm]['mean']
            out[f'{nm}_mean{suffix}'] = mean
            out[f'{nm}_std{suffix}'] = step_stats[nm]['std']
            out[f'{nm}_cumfrac{suffix}'] = _cumfrac(mean)
        if 'target' in names and 'error' in names:
            signal_s = step_stats['target']['mean']
            noise_s = step_stats['error']['mean']
            out[f'snr{suffix}'] = signal_s / (noise_s + eps)
        if 'forecast' in names and 'target' in names:
            f_spec_s = np.clip(step_stats['forecast']['mean'], 0.0, None)
            t_spec_s = np.clip(step_stats['target']['mean'], 0.0, None)
            f_cum = np.cumsum(f_spec_s, axis=-1)
            t_cum = np.cumsum(t_spec_s, axis=-1)
            out[f'cum_energy_ratio{suffix}'] = f_cum / (t_cum + eps)
        step_out = {'k': info['k']}
        for nm in names:
            step_out[f'{nm}_mean'] = step_stats[nm]['mean']
            step_out[f'{nm}_std'] = step_stats[nm]['std']
            step_out[f'{nm}_cumfrac'] = _cumfrac(step_stats[nm]['mean'])
        if 'target' in names and 'error' in names:
            step_out['snr'] = step_out['target_mean'] / (step_out['error_mean'] + eps)
        if 'forecast' in names and 'target' in names:
            f_spec_s = np.clip(step_out['forecast_mean'], 0.0, None)
            t_spec_s = np.clip(step_out['target_mean'], 0.0, None)
            f_cum = np.cumsum(f_spec_s, axis=-1)
            t_cum = np.cumsum(t_spec_s, axis=-1)
            step_out['cum_energy_ratio'] = f_cum / (t_cum + eps)
        step_outputs[step_idx] = step_out
    np.savez_compressed(save_npz, **out)
    print(f'已輸出：{save_npz}（包含 k、各通道 mean/std、cumfrac；source={source}；裁切 boundary={boundary}）')
    if 'snr' in out:
        print('已包含 SNR(k)（以 target 為訊號、error 為雜訊）。')
    if per_step_acc:
        steps = ', '.join([f'step{int(s):02d}' for s in sorted(per_step_acc.keys())])
        print(f'已計算並輸出分步頻譜，鍵名附加於: {steps}')
    if save_npz_per_step_prefix:
        base_prefix = save_npz_per_step_prefix
    elif save_npz:
        base_prefix = os.path.splitext(save_npz)[0] + '_step'
    else:
        base_prefix = None
    if base_prefix and step_outputs:
        os.makedirs(os.path.dirname(base_prefix), exist_ok=True) if os.path.dirname(base_prefix) else None
        for step_idx, payload in step_outputs.items():
            step_path = f'{base_prefix}{step_idx:02d}.npz'
            np.savez_compressed(step_path, **payload)
            print(f'已輸出獨立 step 統計：{step_path}')
    if save_csv_per_channel:
        k = out['k']
        for c in range(C):
            rows = {'k': k}
            for nm in names:
                rows[f'{nm}_mean'] = out[f'{nm}_mean'][c]
                rows[f'{nm}_std'] = out[f'{nm}_std'][c]
                rows[f'{nm}_cumfrac'] = out[f'{nm}_cumfrac'][c]
            if 'snr' in out:
                rows['snr'] = out['snr'][c]
            df = pd.DataFrame(rows)
            csv_path = runtime.setting('agu_james.analysis.gnn.energy.csv_path', lambda: f'spectrum_channel_{c:03d}.csv')
            df.to_csv(csv_path, index=False)
        print(f'已輸出每通道 CSV：spectrum_channel_###.csv（共 {C} 份）')
