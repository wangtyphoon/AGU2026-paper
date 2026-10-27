# Derived from gnn/ver_autoregressive/output/coherence_spectrum_all_stat.py
from agu_james import runtime
import os
import glob
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
from agu_james.analysis.gnn.energy import load_pred_label_from_npz, parse_step_from_filename

def _prepare_field(field2d, area2d=None, detrend_mean=True):
    """面積加權去均值，與 spectrum_all_stat 的 radial_power_spectrum 保持一致。"""
    x = field2d.astype(np.float64, copy=False)
    if area2d is not None:
        if area2d.shape != x.shape:
            raise ValueError(f'area2d 形狀需為 {x.shape}，目前是 {area2d.shape}')
        A = area2d.astype(np.float64, copy=False)
        A = A / np.nanmean(A)
        if detrend_mean:
            mean_w = np.nansum(x * A) / np.nansum(A)
        else:
            mean_w = 0.0
        x = (x - mean_w) * np.sqrt(A)
    elif detrend_mean:
        x = x - np.nanmean(x)
    return x

def radial_coherence_spectrum(field_a, field_b, area2d=None, detrend_mean=True, density_norm=False, eps=1e-12):
    """
    計算兩個 2D 場的徑向相干譜：
      coherence(k) = |S_ab(k)|^2 / (S_aa(k) * S_bb(k))
    其中 S_ab = FFT(a) * conj(FFT(b))，S_aa = |FFT(a)|^2，S_bb 同理。
    先在全頻率點計算，再在徑向上做平均。
    回傳：k (0..kmax-1), coherence(k)
    """
    if field_a.shape != field_b.shape:
        raise ValueError(f'兩個場的形狀需相同，收到 {field_a.shape} 與 {field_b.shape}')
    a = _prepare_field(field_a, area2d=area2d, detrend_mean=detrend_mean)
    b = _prepare_field(field_b, area2d=area2d, detrend_mean=detrend_mean)
    H, W = a.shape
    Fa = np.fft.fftshift(np.fft.fft2(a))
    Fb = np.fft.fftshift(np.fft.fft2(b))
    Paa = np.abs(Fa) ** 2
    Pbb = np.abs(Fb) ** 2
    Pab = Fa * np.conjugate(Fb)
    if density_norm:
        norm = (H * W) ** 2
        Paa = Paa / norm
        Pbb = Pbb / norm
        Pab = Pab / norm
    cy, cx = (H // 2, W // 2)
    y, xg = np.indices((H, W))
    r = np.sqrt((xg - cx) ** 2 + (y - cy) ** 2).astype(int)
    kmax = min(cx, cy) + 1
    rr = r.ravel()
    cnt = np.bincount(rr, minlength=kmax)
    sum_paa = np.bincount(rr, weights=Paa.ravel().real, minlength=kmax)
    sum_pbb = np.bincount(rr, weights=Pbb.ravel().real, minlength=kmax)
    sum_pab_real = np.bincount(rr, weights=Pab.real.ravel(), minlength=kmax)
    sum_pab_imag = np.bincount(rr, weights=Pab.imag.ravel(), minlength=kmax)
    cnt_safe = np.maximum(cnt[:kmax], 1)
    mean_paa = sum_paa[:kmax] / cnt_safe
    mean_pbb = sum_pbb[:kmax] / cnt_safe
    mean_pab = (sum_pab_real[:kmax] + 1j * sum_pab_imag[:kmax]) / cnt_safe
    coh = np.abs(mean_pab) ** 2 / (mean_paa * mean_pbb + eps)
    k = np.arange(kmax)
    return (k, coh)

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
    """對齊 k 軸後，把單檔統計量併入累計器。"""
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

def _process_one_file(path, H0, W0, C, y0, y1, x0, x1, names, detrend_mean, apply_lat_weight, lat_path, density_norm, target_folder):
    """
    單檔 worker：讀檔→裁切→(可選)面積權重→逐樣本逐通道相干譜→回傳統計量。
    回傳的 acc 只包含 coherence，一律 shape=(C, klen)。
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
    acc = _make_empty_acc(names)
    B_here = pred_c.shape[0]
    for b in range(B_here):
        for c in range(C):
            arr_f = pred_c[b, :, :, c]
            arr_t = label_c[b, :, :, c]
            k, coh = radial_coherence_spectrum(arr_f, arr_t, area2d=A2d, detrend_mean=detrend_mean, density_norm=density_norm)
            if k_ref is None:
                k_ref = k.copy()
                klen = len(k_ref)
                acc['coherence']['sum'] = np.zeros((C, klen), dtype=np.float64)
                acc['coherence']['sumsq'] = np.zeros((C, klen), dtype=np.float64)
                acc['coherence']['cnt'] = np.zeros((C, klen), dtype=np.int64)
            else:
                klen = min(len(k_ref), len(k))
                if klen < len(k_ref):
                    k_ref = k_ref[:klen]
                    _truncate_acc(acc, klen)
                coh = coh[:klen]
            acc['coherence']['sum'][c, :klen] += coh
            acc['coherence']['sumsq'][c, :klen] += coh * coh
            acc['coherence']['cnt'][c, :klen] += 1
    return {'k': k_ref, 'acc': acc, 'C': C}

def main(folder='.', target_folder=None, H=192, W=192, boundary=8, detrend_mean=True, apply_lat_weight=False, lat_path='lat.npy', density_norm=False, save_npz='coherence_spectrum.npz', save_csv_per_channel=False, max_workers=None, save_npz_per_step_prefix=None):
    """
    以徑向平均的方式計算 forecast 與 target 之間的相干譜，輸出：
      - k
      - coherence_mean (C, K)
      - coherence_std  (C, K)
    若檔名含 _stepXX，會額外輸出 per-step 的統計（鍵名加上 _stepXX）。
    """
    npz_files = sorted(glob.glob(os.path.join(folder, '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'資料夾 {folder} 下找不到 *.npz')
    if target_folder is None:
        target_folder = runtime.setting('agu_james.analysis.gnn.coherence.target_folder', lambda: folder)
    first_pred, _, hw = load_pred_label_from_npz(npz_files[0], target_folder=target_folder)
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
    names = ['coherence']
    k_ref = None
    global_acc = _make_empty_acc(names)
    per_step_acc = {}
    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = max(1, cpu - 1)
    worker = partial(_process_one_file, H0=H0, W0=W0, C=C, y0=y0, y1=y1, x0=x0, x1=x1, names=names, detrend_mean=detrend_mean, apply_lat_weight=apply_lat_weight, lat_path=lat_path, density_norm=density_norm, target_folder=target_folder)
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
    out['coherence_mean'] = stats_overall['coherence']['mean']
    out['coherence_std'] = stats_overall['coherence']['std']
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
        out[f'coherence_mean{suffix}'] = step_stats['coherence']['mean']
        out[f'coherence_std{suffix}'] = step_stats['coherence']['std']
        step_out = {'k': info['k'], 'coherence_mean': step_stats['coherence']['mean'], 'coherence_std': step_stats['coherence']['std']}
        step_outputs[step_idx] = step_out
    np.savez_compressed(save_npz, **out)
    print(f'已輸出：{save_npz}（包含 k、coherence_mean/coherence_std；裁切 boundary={boundary}）')
    if per_step_acc:
        steps = ', '.join([f'step{int(s):02d}' for s in sorted(per_step_acc.keys())])
        print(f'已計算並輸出分步相干譜，鍵名附加於: {steps}')
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
            print(f'已輸出獨立 step 相干譜：{step_path}')
    if save_csv_per_channel:
        k = out['k']
        C = out['coherence_mean'].shape[0]
        for c in range(C):
            rows = {'k': k, 'coherence_mean': out['coherence_mean'][c], 'coherence_std': out['coherence_std'][c]}
            df = pd.DataFrame(rows)
            csv_path = runtime.setting('agu_james.analysis.gnn.coherence.csv_path', lambda: f'coherence_channel_{c:03d}.csv')
            df.to_csv(csv_path, index=False)
        print(f'已輸出每通道 CSV：coherence_channel_###.csv（共 {C} 份）')
