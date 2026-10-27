# Derived from gnn/ver_autoregressive/train_script/loss.py
from agu_james import runtime
import numpy as np
import torch
import pandas as pd

def create_channel_weights(device):
    """
    讀取 next_feature_table_weight.csv，產生長度 116 的通道權重向量
    （其餘未在 CSV 出現的通道預設為 1.0）。
    """
    base = torch.ones(116, device=device)
    df = pd.read_csv(runtime.legacy_path('next_feature_table_weight_ver3.csv'))
    base[:len(df)] = torch.tensor(df['weight'].values, device=device)
    return base

def create_spatial_weights(H=192, W=192, boundary=9, device=None):
    from agu_james.contracts import spatial_weights
    lat = np.linspace(47.75, 0.0, H)
    return torch.tensor(spatial_weights(lat, W, boundary), dtype=torch.float32, device=device)

def weighted_rmse_loss(pred, label, channel_weights, spatial_weights, epsilon=1e-08):
    """
    pred, label:    (batch*node, C)
    channel_weights: (C,)
    spatial_weights_2d: (H, W)
    """
    Bn, C = pred.shape
    node = spatial_weights.numel()
    assert Bn % node == 0, 'batch*node 必须能被 node 整除'
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

def weighted_mse_loss(pred, label, channel_weights, spatial_weights):
    """
    加權 MSE（不開根號）。
    pred, label:    (batch*node, C)
    channel_weights: (C,)
    spatial_weights: (H, W)
    """
    Bn, C = pred.shape
    node = spatial_weights.numel()
    assert Bn % node == 0, 'batch*node 必須可被 node 整除'
    batch = Bn // node
    spatial_flat = spatial_weights.view(-1)
    spatial_repeat = spatial_flat.repeat(batch)
    w_c = channel_weights.unsqueeze(0)
    w_s = spatial_repeat.unsqueeze(1)
    total_w = w_s * w_c
    diff2 = (pred - label) ** 2
    weighted = diff2 * total_w
    mse = weighted.mean()
    return mse
