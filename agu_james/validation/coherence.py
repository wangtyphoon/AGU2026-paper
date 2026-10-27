# Derived from gnn/ver_autoregressive/train_script/val_autoregressive_with_spectrum.py
from agu_james import runtime
import os
from typing import Dict, Optional, Sequence, Tuple
from agu_james.logging_utils import logger
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from agu_james.training.loss import create_spatial_weights
from agu_james.training.dataset import process_data_step, move_to_device

def _build_radial_index(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    建立徑向 bin index（以像素半徑 floor(r) 分箱）：
      - r_flat: (Npix_valid,) int64
      - valid_flat: (H*W,) bool，標記哪些像素被納入 k=0..kmax-1
      - count_k: (kmax,) float32，每個 k 的像素數
    kmax = min(H//2, W//2) + 1
    """
    cy, cx = (H // 2, W // 2)
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    r = torch.sqrt((xx - cx).to(torch.float32) ** 2 + (yy - cy).to(torch.float32) ** 2).to(torch.int64)
    kmax = min(cx, cy) + 1
    valid = r < kmax
    r_flat = r.reshape(-1)[valid.reshape(-1)]
    count_k = torch.bincount(r_flat, minlength=kmax).to(torch.float32)
    k = torch.arange(kmax, device=device, dtype=torch.int64)
    return (r_flat, valid.reshape(-1), count_k)

def _radial_spectrum_torch(x_bchw: torch.Tensor, r_flat: torch.Tensor, valid_flat: torch.Tensor, count_k: torch.Tensor, *, amplitude: bool=True, detrend_mean: bool=True, area2d: Optional[torch.Tensor]=None, density_norm: bool=False) -> torch.Tensor:
    """
    GPU/CPU 皆可用的批次徑向幅度/功率譜，等價於 spectrum_all_stat.py 的 radial_power_spectrum()：
      - x_bchw: (B, C, H, W)
      - 回傳 spec: (B, C, K)
    註：
      * 若提供 area2d (H, W)，會做面積加權去均值並乘 sqrt(A_norm)。
      * 徑向平均以像素半徑分箱 (floor(r))。
    """
    B, C, H, W = x_bchw.shape
    device = x_bchw.device
    dtype = x_bchw.dtype
    x = x_bchw.to(torch.float64)
    if area2d is not None:
        if area2d.shape != (H, W):
            raise ValueError(f'area2d 形狀需為 {(H, W)}，目前是 {tuple(area2d.shape)}')
        A = area2d.to(torch.float64)
        A = A / (A.mean() + 1e-12)
        if detrend_mean:
            mean_w = (x * A).sum(dim=(-2, -1), keepdim=True) / (A.sum() + 1e-12)
        else:
            mean_w = 0.0
        x = (x - mean_w) * torch.sqrt(A)
    elif detrend_mean:
        x = x - x.mean(dim=(-2, -1), keepdim=True)
    F = torch.fft.fft2(x, dim=(-2, -1))
    F = torch.fft.fftshift(F, dim=(-2, -1))
    if amplitude:
        P = torch.abs(F)
        if density_norm:
            P = P / (H * W)
    else:
        P = torch.abs(F) ** 2
        if density_norm:
            P = P / (H * W) ** 2
    BC = B * C
    P_flat = P.reshape(BC, H * W)
    P_flat = P_flat[:, valid_flat]
    index = r_flat.unsqueeze(0).expand(BC, -1)
    K = int(count_k.numel())
    sum_per = torch.zeros((BC, K), device=device, dtype=torch.float64)
    sum_per.scatter_add_(1, index, P_flat)
    spec = sum_per / (count_k.unsqueeze(0) + 1e-12)
    spec = spec.reshape(B, C, K)
    return spec.to(dtype)

def _radial_coherence_torch(a, b, r_flat, valid_flat, count_k, detrend_mean=False, area2d=None, eps=1e-12):
    """Ring-average complex cross spectra before computing per-sample Q."""

    def prepare(x):
        x = x.to(torch.float64)
        if area2d is not None:
            weights = area2d.to(torch.float64)
            weights = weights / weights.mean()
            if detrend_mean:
                x = x - (x * weights).sum((-2, -1), keepdim=True) / weights.sum()
            x = x * torch.sqrt(weights)
        elif detrend_mean:
            x = x - x.mean((-2, -1), keepdim=True)
        return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    fa, fb = (prepare(a), prepare(b))
    B, C, H, W = a.shape
    K = count_k.numel()
    index = r_flat.unsqueeze(0).expand(B * C, -1)

    def radial(values):
        values = values.reshape(B * C, H * W)[:, valid_flat]
        result = torch.zeros((B * C, K), dtype=torch.float64, device=a.device)
        result.scatter_add_(1, index, values)
        return (result / count_k.to(torch.float64).clamp_min(1)).reshape(B, C, K)
    cross = fa * fb.conj()
    aa, bb = (radial(fa.abs().square()), radial(fb.abs().square()))
    real, imag = (radial(cross.real), radial(cross.imag))
    return (real.square() + imag.square()) / (aa * bb + eps)

def _acc_init(C: int, K: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {'sum': torch.zeros((C, K), device=device, dtype=torch.float64), 'sumsq': torch.zeros((C, K), device=device, dtype=torch.float64), 'cnt': torch.zeros((C, K), device=device, dtype=torch.int64)}

def _nanquantile_finite(x_bck: torch.Tensor, q: float) -> torch.Tensor:
    """
    沿 batch 維度計算分位數（忽略 NaN/Inf），回傳 (C, K)。
    """
    x = x_bck.to(torch.float32)
    if x.numel() == 0:
        return torch.zeros((x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
    if hasattr(torch, 'nanquantile'):
        x_q = x.clone()
        x_q[~torch.isfinite(x_q)] = float('nan')
        return torch.nanquantile(x_q, q, dim=0)
    finite = torch.isfinite(x)
    if not finite.any():
        return torch.zeros((x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
    inf = torch.tensor(float('inf'), device=x.device, dtype=x.dtype)
    x_fill = torch.where(finite, x, inf)
    x_sort, _ = torch.sort(x_fill, dim=0)
    n = finite.sum(dim=0)
    n_clamped = torch.clamp(n, min=1)
    idx = torch.floor(q * (n_clamped - 1).to(torch.float32)).to(torch.long)
    q_val = torch.gather(x_sort, 0, idx.unsqueeze(0)).squeeze(0)
    q_val = torch.where(n > 0, q_val, torch.zeros_like(q_val))
    return q_val

def _acc_update(acc: Dict[str, torch.Tensor], spec_bck: torch.Tensor, trim_ratio: float=0.01) -> None:
    """
    累計（以「每個樣本」為一筆觀測），並：
      - 過濾 NaN/Inf
      - 去除每個 (C,K) 上下 1% 的極端值（依 batch 維度）
      - spec_bck: (B, C, K)
    """
    if spec_bck.numel() == 0:
        return
    B = spec_bck.shape[0]
    spec = spec_bck
    finite = torch.isfinite(spec)
    mask = finite
    if trim_ratio > 0.0 and B > 1:
        n = finite.sum(dim=0)
        trim_ok = n.to(torch.float32) * trim_ratio >= 1.0
        if trim_ok.any():
            lo = _nanquantile_finite(spec, trim_ratio).to(spec.dtype)
            hi = _nanquantile_finite(spec, 1.0 - trim_ratio).to(spec.dtype)
            neg_inf = torch.full_like(lo, float('-inf'))
            pos_inf = torch.full_like(hi, float('inf'))
            lo = torch.where(trim_ok, lo, neg_inf)
            hi = torch.where(trim_ok, hi, pos_inf)
            mask = mask & (spec >= lo.unsqueeze(0)) & (spec <= hi.unsqueeze(0))
    spec64 = spec.to(torch.float64)
    spec_filtered = torch.where(mask, spec64, torch.zeros_like(spec64))
    acc['sum'] += spec_filtered.sum(dim=0)
    acc['sumsq'] += (spec_filtered ** 2).sum(dim=0)
    acc['cnt'] += mask.sum(dim=0).to(torch.int64)

def _acc_mean_std(acc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    由 sum/sumsq/cnt 得 mean/std（ddof=1）
    回傳：mean(C,K), std(C,K), cnt(C,K float64)
    """
    S = acc['sum']
    SS = acc['sumsq']
    N = acc['cnt'].to(torch.float64)
    mean = torch.where(N > 0, S / torch.clamp(N, min=1.0), torch.zeros_like(S))
    var_num = SS - S * S / torch.clamp(N, min=1.0)
    var_den = torch.clamp(N - 1.0, min=1.0)
    var = torch.where(N > 1, var_num / var_den, torch.zeros_like(SS))
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return (mean, std, N)

def _cumfrac(spec_ck: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    csum = torch.cumsum(torch.clamp(spec_ck, min=0.0), dim=-1)
    total = csum[:, [-1]]
    return csum / (total + eps)

@torch.no_grad()
def validate_autoregressive(model, val_loader, device, epoch, save_folder, rollout_steps: int, H: int=192, W: int=192, boundary: int=9, save_step_npz: bool=True, compute_spectrum: bool=False, spectrum_boundary: int=9, spectrum_metadata: Optional[dict]=None, spectrum_source: str='all', spectrum_amplitude: bool=False, spectrum_detrend_mean: bool=False, spectrum_apply_lat_weight: bool=False, spectrum_lat_path: str='lat.npy', spectrum_save_npz: Optional[str]=None, spectrum_per_step: bool=True, spectrum_batch_stride: int=1, spectrum_channel_indices: Optional[Sequence[int]]=None):
    """
    validate_autoregressive：
      - 保留你原本的 autoregressive rollout + 邊界覆寫流程
      - 另外可選擇在 validation 時即時計算徑向頻譜（不需要先把每步 npz 存到磁碟）
    回傳：
      - 若 compute_spectrum=True：回傳 dict（可直接 np.savez_compressed）
      - 否則回傳 None
    """
    feature_table = runtime.setting('agu_james.validation.coherence.feature_table', lambda: pd.read_csv(runtime.legacy_path('next_feature_table_weight.csv')))
    mean_np = feature_table['mean'].values
    std_np = feature_table['std'].values
    scale_df = pd.read_csv(runtime.legacy_path('target_mean_std.csv'))
    scale_std = torch.tensor(scale_df['std'].values, dtype=torch.float32, device=device)
    os.makedirs(save_folder, exist_ok=True)
    model.eval()
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch} [Val-AR {rollout_steps} steps]')
    spatial_w = create_spatial_weights(H=H, W=W, boundary=boundary, device=device)
    base_mask_flat = spatial_w.view(-1) == 0
    num_boundary_points = base_mask_flat.sum().item()
    logger.info(f'Boundary mask created with {num_boundary_points} boundary points out of {H * W} total points.')
    spectrum_out = None
    skipped_samples, skipped_steps, skipped_reasons = ([], [], [])
    if compute_spectrum:
        C_full = 116
        if spectrum_channel_indices is None:
            ch_idx = torch.arange(C_full, device=device, dtype=torch.long)
        else:
            ch_idx = torch.as_tensor(list(spectrum_channel_indices), device=device, dtype=torch.long)
        C_sel = int(ch_idx.numel())
        mean_t = torch.tensor(mean_np[:C_full], dtype=torch.float64, device=device)[ch_idx]
        std_t = torch.tensor(std_np[:C_full], dtype=torch.float64, device=device)[ch_idx]
        if spectrum_boundary < 0:
            raise ValueError('spectrum_boundary must be nonnegative')
        y0, y1 = (spectrum_boundary, H - spectrum_boundary)
        x0, x1 = (spectrum_boundary, W - spectrum_boundary)
        Hc, Wc = (y1 - y0, x1 - x0)
        if Hc <= 0 or Wc <= 0:
            raise ValueError(f'boundary={boundary} 導致裁切後大小為 {Hc}x{Wc}，請調整 boundary 或 H/W')
        area2d = None
        if spectrum_apply_lat_weight:
            lat2d_full = np.load(spectrum_lat_path)
            if lat2d_full.shape != (H, W):
                raise ValueError(f'lat.npy 形狀需為 {(H, W)}，目前是 {lat2d_full.shape}')
            lat2d = torch.tensor(lat2d_full[y0:y1, x0:x1], device=device, dtype=torch.float32)
            area2d = torch.cos(torch.deg2rad(lat2d)).to(torch.float32)
        r_flat, valid_flat, count_k = _build_radial_index(Hc, Wc, device=device)
        K = int(count_k.numel())
        names = []
        if spectrum_source in ('forecast', 'all'):
            names.append('forecast')
        if spectrum_source in ('target', 'all'):
            names.append('target')
        if spectrum_source in ('error', 'all'):
            names.append('error')
        if not names:
            raise ValueError(f'spectrum_source 需為 forecast/target/error/all，目前為: {spectrum_source}')
        names.append('coherence')
        global_acc = {nm: _acc_init(C_sel, K, device=device) for nm in names}
        step_acc = {s: {nm: _acc_init(C_sel, K, device=device) for nm in names} for s in range(1, rollout_steps + 1)} if spectrum_per_step else None
    for batch_idx, (current_data, next_data, sample_name) in enumerate(progress_bar, start=1):
        current_data = move_to_device(current_data, device)
        next_data = [move_to_device(g, device) for g in next_data]
        B_idx = current_data['grid'].batch
        batch_size = int(B_idx.max().item() + 1)
        active = torch.ones(batch_size, device=device, dtype=torch.bool)

        def skip_samples(bad, reason):
            bad = bad & active
            for i in bad.nonzero(as_tuple=True)[0].tolist():
                if isinstance(sample_name, (list, tuple, np.ndarray)) and len(sample_name) == batch_size:
                    name = str(sample_name[i])
                elif torch.is_tensor(sample_name) and sample_name.numel() == batch_size:
                    name = str(sample_name[i].item())
                else:
                    name = f'{sample_name}_b{i:02d}' if batch_size > 1 else str(sample_name)
                skipped_samples.append(name)
                skipped_steps.append(step)
                skipped_reasons.append(reason)
                logger.warning('Skipping validation sample | sample=%s | step=%02d | reason=%s', name, step, reason)
            active[bad] = False

        def bad_field_samples(field):
            bad_nodes = ~torch.isfinite(field).all(dim=1)
            return torch.bincount(B_idx[bad_nodes], minlength=batch_size) > 0
        node = H * W
        assert current_data['grid'].x.shape[0] == batch_size * node, 'H,W 與資料不一致，請確認'
        boundary_mask_bn = base_mask_flat.repeat(batch_size)
        boundary_mask_bnC = boundary_mask_bn.unsqueeze(1).expand(-1, 116)
        for step in range(1, rollout_steps + 1):
            with torch.amp.autocast(device_type=device.type):
                outputs = model(current_data)
                pred_step = outputs * scale_std + current_data['grid'].x[:, :116]
            if not compute_spectrum and (not torch.isfinite(pred_step).all()):
                raise ValueError(f'Non-finite forecast | samples={sample_name} | step={step:02d}')
            truth_step = next_data[step - 1]['grid'].x[:, :116]
            if compute_spectrum:
                skip_samples(bad_field_samples(pred_step), 'Non-finite forecast')
                skip_samples(bad_field_samples(truth_step), 'Non-finite target')
                if not active.any():
                    break
            if compute_spectrum and (spectrum_batch_stride <= 1 or (batch_idx - 1) % spectrum_batch_stride == 0):
                pred_phy = pred_step[:, ch_idx].to(torch.float64) * std_t.to(torch.float64) + mean_t.to(torch.float64)
                truth_phy = truth_step[:, ch_idx].to(torch.float64) * std_t.to(torch.float64) + mean_t.to(torch.float64)
                expected = torch.arange(batch_size, device=device).repeat_interleave(node)
                if torch.equal(B_idx, expected):
                    pred_bhwc = pred_phy.view(batch_size, H, W, C_sel)
                    truth_bhwc = truth_phy.view(batch_size, H, W, C_sel)
                else:
                    pred_bhwc = torch.empty((batch_size, H, W, C_sel), device=device, dtype=pred_phy.dtype)
                    truth_bhwc = torch.empty((batch_size, H, W, C_sel), device=device, dtype=truth_phy.dtype)
                    for i in range(batch_size):
                        m = B_idx == i
                        pred_bhwc[i] = pred_phy[m].view(H, W, C_sel)
                        truth_bhwc[i] = truth_phy[m].view(H, W, C_sel)
                pred_crop = pred_bhwc[:, y0:y1, x0:x1, :].permute(0, 3, 1, 2).contiguous()
                truth_crop = truth_bhwc[:, y0:y1, x0:x1, :].permute(0, 3, 1, 2).contiguous()
                bad_crop = ~torch.isfinite(pred_crop).flatten(1).all(1) | ~torch.isfinite(truth_crop).flatten(1).all(1)
                skip_samples(bad_crop, 'Non-finite physical spectrum input')
                if not active.any():
                    break
                sample_indices = active.nonzero(as_tuple=True)[0]
                pred_crop = pred_crop[active]
                truth_crop = truth_crop[active]
                coherence = _radial_coherence_torch(pred_crop, truth_crop, r_flat, valid_flat, count_k, detrend_mean=spectrum_detrend_mean, area2d=area2d)
                spectra = {'coherence': coherence}
                to_compute = {}
                if 'forecast' in global_acc:
                    to_compute['forecast'] = pred_crop
                if 'target' in global_acc:
                    to_compute['target'] = truth_crop
                if 'error' in global_acc:
                    to_compute['error'] = pred_crop - truth_crop
                for nm, arr_bchw in to_compute.items():
                    spec_bck = _radial_spectrum_torch(arr_bchw, r_flat=r_flat, valid_flat=valid_flat, count_k=count_k, amplitude=spectrum_amplitude, detrend_mean=spectrum_detrend_mean, area2d=area2d, density_norm=False)
                    spectra[nm] = spec_bck
                valid_spectra = torch.ones(len(sample_indices), device=device, dtype=torch.bool)
                for values in spectra.values():
                    valid_spectra &= torch.isfinite(values).flatten(1).all(1)
                    valid_spectra &= torch.isfinite(values.to(torch.float64).square()).flatten(1).all(1)
                bad = torch.zeros_like(active)
                bad[sample_indices[~valid_spectra]] = True
                skip_samples(bad, 'Non-finite spectrum/coherence or squared spectrum')
                for nm, values in spectra.items():
                    _acc_update(global_acc[nm], values[valid_spectra], trim_ratio=0.0)
                    if spectrum_per_step and step_acc is not None:
                        _acc_update(step_acc[step][nm], values[valid_spectra], trim_ratio=0.0)
                if not active.any():
                    break
            if save_step_npz:
                f_np = pred_step.detach().float().cpu().numpy() * std_np[:116] + mean_np[:116]
                bidx_np = B_idx.cpu().numpy()
                for i in range(batch_size):
                    if not active[i]:
                        continue
                    mask_i = bidx_np == i
                    f_i = f_np[mask_i].astype(np.float32)
                    if isinstance(sample_name, (list, tuple, np.ndarray)) and len(sample_name) == batch_size:
                        sample_i = sample_name[i]
                    elif torch.is_tensor(sample_name) and sample_name.numel() == batch_size:
                        sample_i = sample_name[i].item()
                    else:
                        sample_i = f'{sample_name}_b{i:02d}' if batch_size > 1 else sample_name
                    out_path = os.path.join(save_folder, f'{sample_i}_step{step:02d}.npz')
                    if np.isnan(f_i).any():
                        nan_channels = np.where(np.isnan(f_i).any(axis=0))[0]
                        raise ValueError(f'NaN found in forecast | sample={sample_i} | step={step:02d} | channels={nan_channels.tolist()}')
                    np.savez(out_path, forecast=f_i)
            pred_step_patched = torch.where(boundary_mask_bnC, truth_step, pred_step)
            if compute_spectrum:
                pred_step_patched = torch.where(active[B_idx, None], pred_step_patched, torch.zeros_like(pred_step_patched))
            if step < rollout_steps:
                current_data = process_data_step(current_data=current_data, next_data=next_data, step=step, first116=pred_step_patched)
    logger.info('Validation (autoregressive) completed successfully.')
    if skipped_samples:
        logger.warning('Validation skipped %d samples; earlier valid steps were retained.', len(skipped_samples))
    if compute_spectrum:
        spectrum_out = {'k': torch.arange(int(count_k.numel()), device=device, dtype=torch.int64).cpu().numpy()}
        eps = 1e-12
        for nm, acc in global_acc.items():
            mean_ck, std_ck, cnt_ck = _acc_mean_std(acc)
            mean_ck = mean_ck.detach().cpu().numpy()
            std_ck = std_ck.detach().cpu().numpy()
            cnt_ck = cnt_ck.detach().cpu().numpy()
            spectrum_out[f'{nm}_mean'] = mean_ck
            spectrum_out[f'{nm}_std'] = std_ck
            spectrum_out[f'{nm}_cnt'] = cnt_ck
            if nm != 'coherence':
                spectrum_out[f'{nm}_cumfrac'] = _cumfrac(torch.tensor(mean_ck)).numpy()
        if 'forecast' in global_acc and 'target' in global_acc:
            f_spec = np.clip(spectrum_out['forecast_mean'], 0.0, None)
            t_spec = np.clip(spectrum_out['target_mean'], 0.0, None)
            f_cum = np.cumsum(f_spec, axis=-1)
            t_cum = np.cumsum(t_spec, axis=-1)
            spectrum_out['cum_energy_ratio'] = f_cum / (t_cum + eps)
        if spectrum_per_step and step_acc is not None:
            for step in range(1, rollout_steps + 1):
                suffix = f'_step{step:02d}'
                for nm in global_acc.keys():
                    mean_ck, std_ck, cnt_ck = _acc_mean_std(step_acc[step][nm])
                    mean_ck = mean_ck.detach().cpu().numpy()
                    std_ck = std_ck.detach().cpu().numpy()
                    cnt_ck = cnt_ck.detach().cpu().numpy()
                    spectrum_out[f'{nm}_mean{suffix}'] = mean_ck
                    spectrum_out[f'{nm}_std{suffix}'] = std_ck
                    spectrum_out[f'{nm}_cnt{suffix}'] = cnt_ck
                    if nm != 'coherence':
                        spectrum_out[f'{nm}_cumfrac{suffix}'] = _cumfrac(torch.tensor(mean_ck)).numpy()
                if 'forecast' in global_acc and 'target' in global_acc:
                    f_spec = np.clip(spectrum_out[f'forecast_mean{suffix}'], 0.0, None)
                    t_spec = np.clip(spectrum_out[f'target_mean{suffix}'], 0.0, None)
                    f_cum = np.cumsum(f_spec, axis=-1)
                    t_cum = np.cumsum(t_spec, axis=-1)
                    spectrum_out[f'cum_energy_ratio{suffix}'] = f_cum / (t_cum + eps)
        spectrum_out['feature_names'] = feature_table['feature'].to_numpy(dtype=str)[ch_idx.cpu().numpy()]
        settings = dict(spectrum_boundary=spectrum_boundary, rollout_boundary=boundary, rollout_steps=rollout_steps, spectrum_amplitude=spectrum_amplitude, spectrum_detrend_mean=spectrum_detrend_mean, spectrum_apply_lat_weight=spectrum_apply_lat_weight, coherence_definition='ring_mean_cross_magnitude_squared_then_sample_mean')
        settings.update(spectrum_metadata or {})
        for key, value in settings.items():
            spectrum_out[key] = np.asarray(value)
        spectrum_out['skipped_samples'] = np.asarray(skipped_samples, dtype=str)
        spectrum_out['skipped_steps'] = np.asarray(skipped_steps, dtype=np.int64)
        spectrum_out['skipped_reasons'] = np.asarray(skipped_reasons, dtype=str)
        spectrum_out['skipped_sample_count'] = np.asarray(len(skipped_samples))
        spectrum_out['skip_policy'] = np.asarray('retain_valid_steps_skip_from_first_invalid_step')
        if spectrum_save_npz:
            os.makedirs(os.path.dirname(spectrum_save_npz), exist_ok=True) if os.path.dirname(spectrum_save_npz) else None
            np.savez_compressed(spectrum_save_npz, **spectrum_out)
            logger.info(f'Spectrum stats saved to: {spectrum_save_npz}')
        return spectrum_out
    return None
