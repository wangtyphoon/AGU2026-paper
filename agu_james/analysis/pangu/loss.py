# Derived from validation/loss.py
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
    df = pd.read_csv('next_feature_table_weight.csv')
    base[:len(df)] = torch.tensor(df['weight'].values, device=device)
    return base

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
    return w2d

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
