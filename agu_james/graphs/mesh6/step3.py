# Derived from gnn/ver_unet7/pre_process/node_feature2_unet.py
from agu_james import runtime
import torch
import numpy as np
import os
from torch_geometric.utils import degree

def main():
    global data, deg_encoder_source, deg_features, deg_source, deg_target, edge_index, edge_index_encoder, edge_type, edge_type_encoder, grid_features, lat, lat_angle, lat_cos, lat_lon_transformed, lat_max, lat_min, lat_scaled, lat_sin, level, lon, lon_angle, lon_cos, lon_max, lon_min, lon_scaled, lon_sin, mesh_features, mesh_key, new_grid_features, new_mesh_features, num_grid_nodes, num_nodes, output_dir
    data = torch.load('mesh_grid_data_unet/hetero_graph.pt', weights_only=False)
    if 'grid' in data.node_types:
        grid_features = data['grid'].x.clone()
        lat = grid_features[:, 1]
        lon = grid_features[:, 0]
        lat_min, lat_max = (torch.min(lat), torch.max(lat))
        lon_min, lon_max = (torch.min(lon), torch.max(lon))
        lat_scaled = (lat - lat_min) / (lat_max - lat_min)
        lon_scaled = (lon - lon_min) / (lon_max - lon_min)
        lat_angle = lat_scaled * 2 * torch.pi
        lon_angle = lon_scaled * 2 * torch.pi
        lat_sin = torch.sin(lat_angle)
        lat_cos = torch.cos(lat_angle)
        lon_sin = torch.sin(lon_angle)
        lon_cos = torch.cos(lon_angle)
        lat_lon_transformed = torch.stack([lat_sin, lat_cos, lon_sin, lon_cos], dim=1)
        new_grid_features = torch.cat([grid_features, lat_lon_transformed], dim=1)
        data['grid'].x = new_grid_features
        print('更新後的 grid 節點特徵 shape:', data['grid'].x.shape)
    for level in [3, 4, 5, 6]:
        mesh_key = f'mesh{level}'
        if mesh_key in data.node_types:
            mesh_features = data[mesh_key].x.clone()
            lat = mesh_features[:, 1]
            lon = mesh_features[:, 0]
            lat_min, lat_max = (torch.min(lat), torch.max(lat))
            lon_min, lon_max = (torch.min(lon), torch.max(lon))
            lat_scaled = (lat - lat_min) / (lat_max - lat_min)
            lon_scaled = (lon - lon_min) / (lon_max - lon_min)
            lat_angle = lat_scaled * 2 * torch.pi
            lon_angle = lon_scaled * 2 * torch.pi
            lat_sin = torch.sin(lat_angle)
            lat_cos = torch.cos(lat_angle)
            lon_sin = torch.sin(lon_angle)
            lon_cos = torch.cos(lon_angle)
            lat_lon_transformed = torch.stack([lat_sin, lat_cos, lon_sin, lon_cos], dim=1)
            new_mesh_features = torch.cat([mesh_features, lat_lon_transformed], dim=1)
            edge_type = (mesh_key, 'processor', mesh_key)
            if edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                num_nodes = data[mesh_key].x.size(0)
                deg_source = degree(edge_index[0], num_nodes=num_nodes)
                deg_target = degree(edge_index[1], num_nodes=num_nodes)
                deg_features = torch.stack([deg_source, deg_target], dim=1)
                new_mesh_features = torch.cat([new_mesh_features, deg_features], dim=1)
            data[mesh_key].x = new_mesh_features
            print(f'更新後的 {mesh_key} 節點特徵 shape:', data[mesh_key].x.shape)
    edge_type_encoder = ('grid', 'encoder6', 'mesh6')
    if edge_type_encoder in data.edge_types:
        edge_index_encoder = data[edge_type_encoder].edge_index
        num_grid_nodes = data['grid'].x.size(0)
        deg_encoder_source = degree(edge_index_encoder[0], num_nodes=num_grid_nodes)
        data['grid'].x = torch.cat([data['grid'].x, deg_encoder_source.unsqueeze(1)], dim=1)
        print('grid 節點加上 encoder 邊度數 shape:', data['grid'].x.shape)
    for level in [3, 4, 5, 6]:
        mesh_key = f'mesh{level}'
        edge_type = (mesh_key, f'decoder{level}', 'grid')
        if edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            num_nodes = data[mesh_key].x.size(0)
            deg_source = degree(edge_index[0], num_nodes=num_nodes)
            data[mesh_key].x = torch.cat([data[mesh_key].x, deg_source.unsqueeze(1)], dim=1)
            print(f'{mesh_key} 節點加上 decoder{level} 度數後 shape:', data[mesh_key].x.shape)
    output_dir = runtime.setting('agu_james.graphs.mesh6.step3.output_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.graphs.mesh6.step3')
    torch.save(data, os.path.join(output_dir, 'hetero_graph.pt'))
    print('已更新並保存 mesh_grid_data_unet/hetero_graph.pt')
