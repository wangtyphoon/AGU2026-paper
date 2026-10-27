# Derived from gnn/ver_spectrum_v2/train_script/spectrum_loss_v2_ddp.py
from agu_james import runtime
import pandas as pd
import torch
import torch.distributed as dist
RANDOM_EPS = 1e-12
try:
    from torch.distributed.nn.functional import all_reduce as dist_all_reduce
except Exception:
    dist_all_reduce = None

def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()

def _dist_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not _dist_is_ready():
        return tensor
    if dist_all_reduce is None:
        if tensor.requires_grad:
            raise RuntimeError('torch.distributed.nn.functional.all_reduce is unavailable; cannot compute distributed spectral bias with autograd.')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor
    return dist_all_reduce(tensor, op=dist.ReduceOp.SUM)

def _dist_batch_mean(tensor: torch.Tensor, dim: int=0) -> torch.Tensor:
    """Mean over batch dimension with global aggregation when DDP is enabled."""
    if not _dist_is_ready():
        return tensor.mean(dim=dim)
    batch = tensor.shape[dim]
    d_sum = tensor.sum(dim=dim)
    d_sum = _dist_all_reduce_sum(d_sum)
    total = torch.tensor(float(batch), device=tensor.device, dtype=d_sum.dtype)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    total = total.clamp_min(1.0)
    return d_sum / total

def load_channel_weights(csv_path='next_feature_table_weight_ver3.csv', device=None, dtype=torch.float32, normalize=False):
    """Load per-channel weights from CSV."""
    df = pd.read_csv(csv_path)
    weights = torch.tensor(df['weight'].values, dtype=dtype)
    if normalize:
        mean = weights.mean().clamp_min(1e-12)
        weights = weights / mean
    if device is not None:
        weights = weights.to(device)
    return weights

def load_cutoffs(csv_path: str, num_channels: int=116, default_cutoff: float=46.0, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Load per-channel cutoffs into a torch tensor.
    - 優先使用 CSV 裡的 'channel' 和 'k_energy_decay_10pct' 欄位
    - 缺失的 channel 用 default_cutoff 補齊
    - 無效值 (NaN 或 <=0) 會被替換成 default_cutoff
    - 最大值會被限制在 default_cutoff
    """
    df = pd.read_csv(csv_path)
    if 'channel' not in df.columns or 'k_energy_decay_10pct' not in df.columns:
        raise KeyError("CSV must contain 'channel' and 'k_energy_decay_10pct' columns")
    cutoffs = torch.full((num_channels,), float(default_cutoff), dtype=dtype)
    for _, row in df.iterrows():
        ch = row['channel']
        if pd.isna(ch):
            continue
        ch = int(ch)
        if 0 <= ch < num_channels:
            k_val = row['k_energy_decay_10pct']
            if not pd.isna(k_val):
                cutoffs[ch] = torch.tensor(float(k_val), dtype=dtype)
    mask_invalid = ~torch.isfinite(cutoffs) | (cutoffs <= 0)
    cutoffs[mask_invalid] = float(default_cutoff)
    cutoffs = torch.minimum(cutoffs, torch.tensor(default_cutoff, dtype=dtype))
    if device is not None:
        cutoffs = cutoffs.to(device)
    return cutoffs

def load_feature_stats(mean_std_csv='next_feature_table_weight.csv', target_std_csv='target_mean_std.csv', device=None, dtype=torch.float32):
    """Load mean/std (per-channel) and target std for scaling."""
    mean_std_df = pd.read_csv(mean_std_csv)
    mean = torch.tensor(mean_std_df['mean'].values, dtype=dtype)
    std = torch.tensor(mean_std_df['std'].values, dtype=dtype)
    scale_df = pd.read_csv(target_std_csv)
    target_std = torch.tensor(scale_df['std'].values, dtype=dtype)
    if device is not None:
        mean = mean.to(device)
        std = std.to(device)
        target_std = target_std.to(device)
    return (mean, std, target_std)

def create_latitude_weights(H: int, W: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Cos(lat) 權重（無邊界清零），並以平均值歸一化。"""
    lats_deg = torch.linspace(0, 47.75, H)
    lat_w = torch.cos(torch.deg2rad(lats_deg))
    w2d = lat_w[:, None].repeat(1, W)
    w2d = torch.flip(w2d, dims=(0,))
    w2d = w2d / w2d.mean().clamp_min(1e-12)
    if device is not None:
        w2d = w2d.to(device=device, dtype=dtype)
    return w2d

def _radial_bins(H, W, device):
    ky = torch.fft.fftfreq(H, d=1.0) * H
    kx = torch.fft.rfftfreq(W, d=1.0) * W
    kxg = kx.unsqueeze(0).expand(H, -1)
    kyg = ky.unsqueeze(1).expand(-1, kx.shape[0])
    radial_cont = torch.sqrt(kxg ** 2 + kyg ** 2).to(device)
    k_bins = torch.floor(radial_cont + 1e-08).long()
    Lmax = int(k_bins.max().item())
    return (k_bins, Lmax, radial_cont)

def _iso_power(Xf, H, W):
    return (Xf.real ** 2 + Xf.imag ** 2) / (H * W) ** 2

def _reshape_to_bchw(t, C, H, W):
    if t.ndim == 2:
        N = t.shape[0]
        if N % (H * W) != 0:
            raise ValueError(f'Input size {N} is not divisible by H*W={H * W}')
        B = N // (H * W)
        return t.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    if t.ndim == 4:
        return t
    raise ValueError(f'Unsupported tensor shape for spectral loss: {t.shape}')

def spectral_composite_loss_v2(current_data, preds, targets, *, H=192, W=192, boundary=0, alpha=1.0, order=12, cutoffs=None, l_tilde=None, l_max=None, spatial_w=None, channel_weights=None, weights=(1.0, 1.0, 0.1), scale=None, mean=None, std=None, target_std=None):
    """
    Trainable spectral composite loss (v2):
      1) filtered_mse (Gaussian spectral filter + spatial weights)
      2) spectrum_mse_norm (energy-normalized spectral MSE)
      3) spectral_bias_norm (bias energy / target energy)
    """
    'Return total loss and component metrics for the spectral composite loss.'
    device = preds.device
    forcast = current_data + preds * scale
    truth = current_data + targets * scale
    C = targets.shape[-1] if preds.ndim == 2 else preds.shape[1]
    X = _reshape_to_bchw(forcast, C, H, W)
    Y = _reshape_to_bchw(truth, C, H, W)
    B, C, H0, W0 = X.shape
    if (H0, W0) != (H, W):
        raise ValueError(f'Input spatial size ({H0}, {W0}) does not match expected ({H}, {W})')
    y0, y1 = (boundary, H - boundary)
    x0, x1 = (boundary, W - boundary)
    X = X[..., y0:y1, x0:x1]
    Y = Y[..., y0:y1, x0:x1]
    Hc, Wc = (y1 - y0, x1 - x0)
    spatial_w = spatial_w[y0:y1, x0:x1] if spatial_w is not None else None
    Xf = torch.fft.rfftn(X, dim=(-2, -1))
    Yf = torch.fft.rfftn(Y, dim=(-2, -1))
    Df = Xf - Yf
    k_bins, Lmax, radial_cont = _radial_bins(Hc, Wc, device)
    Wr = Wc // 2 + 1
    if channel_weights is None:
        channel_w = torch.ones((C,), device=device, dtype=X.dtype)
    else:
        channel_w = channel_weights.to(device=device, dtype=X.dtype)
    channel_w = channel_w.view(-1)
    if channel_w.shape[0] != C:
        raise ValueError(f'channel_weights length {channel_w.shape[0]} does not match channel count {C}')
    channel_w = torch.where(torch.isfinite(channel_w) & (channel_w > 0), channel_w, torch.ones_like(channel_w))
    channel_w_sum = channel_w.sum().clamp_min(1e-12)
    if cutoffs is None:
        cutoffs = torch.full((C,), float(50), device=device, dtype=torch.float32)
    else:
        cutoffs = cutoffs.to(device=device, dtype=torch.float32)
        cutoffs = torch.where((cutoffs > 0) & torch.isfinite(cutoffs), cutoffs, torch.full_like(cutoffs, float(max(Hc, Wc))))
    k_for_mask = radial_cont.view(1, 1, Hc, Wr)
    cutoff_expand = cutoffs.view(1, C, 1, 1)
    alpha_tensor = torch.as_tensor(alpha, device=device, dtype=X.dtype).clamp_min(1e-12)
    gaussian_sigma = cutoff_expand / torch.sqrt(alpha_tensor)
    Wmask = torch.exp(-0.5 * (k_for_mask / gaussian_sigma) ** 2)
    d_w = torch.fft.irfftn(Df * Wmask, s=(Hc, Wc), dim=(-2, -1)).real
    if spatial_w is None:
        w = torch.ones((Hc, Wc), device=device, dtype=X.dtype)
    else:
        w = spatial_w.to(device=X.device, dtype=X.dtype)
        if w.shape != (Hc, Wc):
            raise ValueError(f'spatial_w must be (Hc, Wc)=({Hc}, {Wc})')
    denom = w.sum().clamp_min(1e-12)
    filtered_mse_per_channel = (d_w ** 2 * w).sum(dim=(-2, -1)) / denom
    filtered_mse_per_channel = filtered_mse_per_channel.mean(dim=0)
    filtered_mse = (filtered_mse_per_channel * channel_w / scale ** 2).sum() / channel_w_sum
    Px = _iso_power(Xf, Hc, Wc)
    Py = _iso_power(Yf, Hc, Wc)
    Pd = _iso_power(Df, Hc, Wc)
    BC = B * C
    k_flat = k_bins.view(-1)
    Px_flat = Px.view(BC, -1)
    Py_flat = Py.view(BC, -1)
    Pd_flat = Pd.view(BC, -1)
    bins = Lmax + 1
    Sx = torch.zeros((BC, bins), device=device, dtype=Px.dtype)
    Sy = torch.zeros((BC, bins), device=device, dtype=Py.dtype)
    Sd = torch.zeros((BC, bins), device=device, dtype=Pd.dtype)
    Sx.scatter_add_(1, k_flat.unsqueeze(0).expand(BC, -1), Px_flat)
    Sy.scatter_add_(1, k_flat.unsqueeze(0).expand(BC, -1), Py_flat)
    Sd.scatter_add_(1, k_flat.unsqueeze(0).expand(BC, -1), Pd_flat)
    Sx = Sx.view(B, C, bins)
    Sy = Sy.view(B, C, bins)
    Sd = Sd.view(B, C, bins)
    if l_tilde is not None:
        L_use = min(l_tilde + 1, bins)
        Sx_use = Sx[..., :L_use]
        Sy_use = Sy[..., :L_use]
    else:
        Sx_use, Sy_use = (Sx, Sy)
    diff_spec = Sx_use - Sy_use
    num = (diff_spec ** 2).sum(dim=-1)
    den = (Sy_use ** 2).sum(dim=-1).clamp_min(RANDOM_EPS)
    spec_mse_bc = num / den
    spectrum_mse_per_channel = spec_mse_bc.mean(dim=0)
    spectrum_mse = (spectrum_mse_per_channel * channel_w).sum() / channel_w_sum
    d_mean = _dist_batch_mean(X - Y, dim=0)
    Dmean_f = torch.fft.rfftn(d_mean, dim=(-2, -1))
    Pmean = _iso_power(Dmean_f, Hc, Wc)
    BC_k = k_bins.view(-1)
    Pm_f = Pmean.view(C, -1)
    bins = Lmax + 1
    Sbias = torch.zeros((C, bins), device=device, dtype=Pmean.dtype)
    Sbias.scatter_add_(1, BC_k.unsqueeze(0).expand(C, -1), Pm_f)
    Sy_mean = _dist_batch_mean(Sy, dim=0) if _dist_is_ready() else Sy.mean(dim=0)
    if l_max is not None:
        Lb = min(l_max + 1, bins)
        bias_energy = Sbias[:, :Lb].sum(dim=-1)
        target_energy = Sy_mean[:, :Lb].sum(dim=-1).clamp_min(RANDOM_EPS)
    else:
        bias_energy = Sbias.sum(dim=-1)
        target_energy = Sy_mean.sum(dim=-1).clamp_min(RANDOM_EPS)
    spectral_bias_norm_per_channel = bias_energy / target_energy
    spectrum_bias = (spectral_bias_norm_per_channel * channel_w).sum() / channel_w_sum
    wf, ws, wb = weights
    total = wf * filtered_mse + ws * spectrum_mse + wb * spectrum_bias
    metrics = {'filtered_mse': wf * filtered_mse.detach(), 'spectrum_mse_norm': ws * spectrum_mse.detach(), 'spectral_bias_norm': wb * spectrum_bias.detach(), 'total': total.detach()}
    return (total, metrics)
spectral_composite_loss = spectral_composite_loss_v2
