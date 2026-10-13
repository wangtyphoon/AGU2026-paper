# Derived from gnn/ver_autoregressive/train_script/val_autoregressive.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import os
from agu_james.logging_utils import logger
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from agu_james.training.loss import create_spatial_weights
from agu_james.training.dataset import process_data_step, move_to_device

@torch.no_grad()
def validate_autoregressive(model, val_loader, device, epoch, save_folder, rollout_steps: int, H: int=192, W: int=192, boundary: int=9, save_step_npz: bool=True, save_every_n_steps: int=2):
    feature_table = runtime.setting('agu_james.validation.rollout.feature_table', lambda: pd.read_csv(runtime.legacy_path('next_feature_table_weight.csv')))
    mean = feature_table['mean'].values
    std = feature_table['std'].values
    scale_df = pd.read_csv(runtime.legacy_path('target_mean_std.csv'))
    scale_std = torch.tensor(scale_df['std'].values, dtype=torch.float32, device=device)
    if save_every_n_steps < 1:
        raise ValueError(f'save_every_n_steps must be >= 1, got {save_every_n_steps}')
    os.makedirs(save_folder, exist_ok=True)
    model.eval()
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch} [Val-AR {rollout_steps} steps]')
    spatial_w = create_spatial_weights(H=H, W=W, boundary=boundary, device=device)
    base_mask_flat = spatial_w.view(-1) == 0
    num_boundary_points = base_mask_flat.sum().item()
    logger.info(f'Boundary mask created with {num_boundary_points} boundary points out of {H * W} total points.')
    for batch_idx, (current_data, next_data, sample_name) in enumerate(progress_bar):
        current_data = move_to_device(current_data, device)
        next_data = [move_to_device(g, device) for g in next_data]
        B_idx = current_data['grid'].batch
        batch_size = int(B_idx.max().item() + 1)
        node = H * W
        assert current_data['grid'].x.shape[0] == batch_size * node, 'H,W 與資料不一致，請確認'
        if isinstance(sample_name, (list, tuple)):
            sample_names = list(sample_name)
        else:
            sample_names = [sample_name] * batch_size
        boundary_mask_bn = base_mask_flat.repeat(batch_size)
        boundary_mask_bnC = boundary_mask_bn.unsqueeze(1).expand(-1, 116)
        active_samples = [True] * batch_size
        bidx_np = B_idx.cpu().numpy()
        for step in range(1, rollout_steps + 1):
            if not any(active_samples):
                print(f'All samples in batch {batch_idx} became inactive at step {step - 1:02d}, stop rollout early.')
                break
            with torch.amp.autocast(device_type=device.type):
                outputs = model(current_data)
                pred_step = outputs * scale_std + current_data['grid'].x[:, :116]
            truth_step = next_data[step - 1]['grid'].x[:, :116]
            for i in range(batch_size):
                if not active_samples[i]:
                    continue
                mask_i = B_idx == i
                pred_i = pred_step[mask_i]
                nan_mask_i = torch.isnan(pred_i)
                if nan_mask_i.any():
                    nan_channels = torch.where(nan_mask_i.any(dim=0))[0].detach().cpu().tolist()
                    name_i = sample_names[i] if i < len(sample_names) else sample_names[0]
                    print(f'NaN found in forecast | sample={name_i} | step={step:02d} | channels={nan_channels}')
                    active_samples[i] = False
            if save_step_npz and step % save_every_n_steps == 0:
                f_np = pred_step.detach().float().cpu().numpy() * std + mean
                for i in range(batch_size):
                    if not active_samples[i]:
                        continue
                    mask_i = bidx_np == i
                    f_i = f_np[mask_i].astype(np.float32)
                    name_i = sample_names[i] if i < len(sample_names) else sample_names[0]
                    name_i = str([name_i])
                    out_path = os.path.join(save_folder, f'{name_i}_step{step:02d}.npz')
                    np.savez(out_path, forecast=f_i)
            pred_step_patched = torch.where(boundary_mask_bnC, truth_step, pred_step)
            if step < rollout_steps:
                active_samples_t = torch.tensor(active_samples, dtype=torch.bool, device=device)
                active_nodes = active_samples_t[B_idx].unsqueeze(1)
                first116_next = torch.where(active_nodes, pred_step_patched, truth_step)
                current_data = process_data_step(current_data=current_data, next_data=next_data, step=step, first116=first116_next)
    logger.info('Validation (autoregressive) completed successfully.')
