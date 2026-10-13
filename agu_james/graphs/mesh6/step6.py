# Derived from gnn/ver_unet7/pre_process/pre_norm6_unet.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import torch
import os

def standardize_features(tensor, name, save_stats=True):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    if save_stats:
        stats = {'mean': mean.cpu().numpy(), 'std': std.cpu().numpy()}
        torch.save(stats, f'mesh_grid_data_unet/norm_stats/{name}_norm_stats.pt')
        print(f'Saved normalization stats for {name}')
    normalized = (tensor - mean) / std
    return normalized

def main():
    global data, dec_type, edge_attr, edge_type_decoder, edge_type_encoder, edge_type_proc, enc_type, fine_key, fine_lvl, grid_features, input_path, level, mesh_features, mesh_key, output_path, pool_type, ups_type
    input_path = runtime.setting('agu_james.graphs.mesh6.step6.input_path', lambda: 'mesh_grid_data_unet/hetero_graph.pt')
    data = torch.load(input_path, weights_only=False)
    os.makedirs('mesh_grid_data_unet/norm_stats', exist_ok=True)
    for level in [3, 4, 5, 6]:
        mesh_key = f'mesh{level}'
        if mesh_key in data.node_types:
            print(f'標準化 {mesh_key} 節點特徵...')
            mesh_features = data[mesh_key].x
            data[mesh_key].x = standardize_features(mesh_features, f'{mesh_key}_node')
            print(f'{mesh_key} node features standardized: {data[mesh_key].x.shape}')
    if 'grid' in data.node_types:
        print('標準化 grid 節點特徵...')
        grid_features = data['grid'].x
        data['grid'].x = standardize_features(grid_features, 'grid_node')
        print(f"Grid node features standardized: {data['grid'].x.shape}")
    for level in [3, 4, 5, 6]:
        mesh_key = f'mesh{level}'
        edge_type_proc = (mesh_key, 'processor', mesh_key)
        if edge_type_proc in data.edge_types:
            print(f'標準化 {mesh_key} processor 邊特徵...')
            data[edge_type_proc].edge_attr = standardize_features(data[edge_type_proc].edge_attr, f'{mesh_key}_processor_edge')
        fine_lvl = level + 1
        fine_key = f'mesh{fine_lvl}'
        if fine_key not in data.node_types:
            continue
        pool_type = (fine_key, f'pool{fine_lvl}_to_{level}', mesh_key)
        if pool_type in data.edge_types:
            print(f'標準化 {fine_key}→{mesh_key} pool 邊特徵...')
            data[pool_type].edge_attr = standardize_features(data[pool_type].edge_attr, f'{fine_key}_pool{fine_lvl}_to_{level}_edge')
        ups_type = (mesh_key, f'upsample{level}_to_{fine_lvl}', fine_key)
        if ups_type in data.edge_types:
            print(f'標準化 {mesh_key}→{fine_key} upsample 邊特徵...')
            data[ups_type].edge_attr = standardize_features(data[ups_type].edge_attr, f'{mesh_key}_upsample{level}_to_{fine_lvl}_edge')
    edge_type_encoder = ('grid', 'encoder6', 'mesh6')
    edge_type_decoder = ('mesh6', 'decoder6', 'grid')
    if edge_type_encoder in data.edge_types:
        print('標準化 grid→mesh6 encoder 邊特徵...')
        edge_attr = data[edge_type_encoder].edge_attr
        data[edge_type_encoder].edge_attr = standardize_features(edge_attr, 'grid_encoder6_mesh6_edge')
    for level in [3, 4, 5, 6]:
        mesh_key = f'mesh{level}'
        enc_type = ('grid', f'encoder{level}', mesh_key)
        if enc_type in data.edge_types:
            print(f'標準化 grid→{mesh_key} encoder{level} 邊特徵...')
            data[enc_type].edge_attr = standardize_features(data[enc_type].edge_attr, f'grid_encoder{level}_{mesh_key}_edge')
        dec_type = (mesh_key, f'decoder{level}', 'grid')
        if dec_type in data.edge_types:
            print(f'標準化 {mesh_key}→grid decoder{level} 邊特徵...')
            data[dec_type].edge_attr = standardize_features(data[dec_type].edge_attr, f'{mesh_key}_decoder{level}_grid_edge')
    output_path = 'mesh_grid_data_unet/hetero_graph_normalized.pt'
    torch.save(data, output_path)
    print(f'已保存標準化後的 mesh_grid_data_unet/hetero_graph_normalized.pt')
