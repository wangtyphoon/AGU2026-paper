# Derived from gnn/ver_unet7/pre_process/edge_feature4_unet7.py
from agu_james import runtime
import torch
import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees, asin
import os

def latlon_to_xyz(lat, lon, radius=6371.0):
    φ = radians(lat)
    θ = radians(lon)
    x = radius * cos(φ) * cos(θ)
    y = radius * cos(φ) * sin(θ)
    z = radius * sin(φ)
    return (x, y, z)

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return distance

def calculate_edge_features(edge_index, src_nodes, dst_nodes, edge_type='encoder', include_xyz=True):
    num_edges = edge_index.size(1)
    base_dim = 3
    xyz_dim = 3 if include_xyz else 0
    dir_dim = 0
    edge_features = torch.zeros((num_edges, base_dim + xyz_dim + dir_dim), dtype=torch.float)
    for i in range(num_edges):
        s = edge_index[0, i].item()
        d = edge_index[1, i].item()
        lon1, lat1 = map(radians, [src_nodes[s, 0].item(), src_nodes[s, 1].item()])
        lon2, lat2 = map(radians, [dst_nodes[d, 0].item(), dst_nodes[d, 1].item()])
        distance = haversine_distance(degrees(lat1), degrees(lon1), degrees(lat2), degrees(lon2))
        delta_lat = degrees(lat2 - lat1)
        delta_lon = degrees(lon2 - lon1)
        edge_features[i, 0] = distance
        edge_features[i, 1] = delta_lat
        edge_features[i, 2] = delta_lon
        if include_xyz:
            sx, sy, sz = latlon_to_xyz(degrees(lat1), degrees(lon1))
            dx, dy, dz = latlon_to_xyz(degrees(lat2), degrees(lon2))
            edge_features[i, 3] = dx - sx
            edge_features[i, 4] = dy - sy
            edge_features[i, 5] = dz - sz
    return edge_features

def main():
    global coarse_idx, coarse_lvl, data, decoder_edge_features, dst_features, edge_index, edge_type, edge_type_decoder, edge_type_encoder, edge_type_proc, encoder_edge_features, fine_lvl, grid_features, input_path, level, mesh_features, mesh_key, output_path, pool_edge_features, pool_type, proc_edge_features, refinement_levels, src_features, ups_edge_features, ups_type
    input_path = runtime.setting('agu_james.graphs.mesh7.step2.input_path', lambda: 'mesh_grid_data_unet7/hetero_graph.pt')
    data = torch.load(input_path, weights_only=False)
    refinement_levels = [3, 4, 5, 6, 7]
    for level in refinement_levels:
        mesh_key = f'mesh{level}'
        edge_type_proc = (mesh_key, 'processor', mesh_key)
        if mesh_key in data.node_types and edge_type_proc in data.edge_types:
            edge_index = data[edge_type_proc].edge_index
            mesh_features = data[mesh_key].x
            print(f'正在計算 {mesh_key} processor 邊特徵...')
            proc_edge_features = calculate_edge_features(edge_index, mesh_features, mesh_features, edge_type='processor')
            data[edge_type_proc].edge_attr = proc_edge_features
            print(f'{mesh_key} processor 邊特徵形狀: {proc_edge_features.shape}')
        for coarse_idx, coarse_lvl in enumerate(refinement_levels[:-1]):
            fine_lvl = refinement_levels[coarse_idx + 1]
            pool_type = (f'mesh{fine_lvl}', f'pool{fine_lvl}_to_{coarse_lvl}', f'mesh{coarse_lvl}')
            if pool_type in data.edge_types:
                edge_index = data[pool_type].edge_index
                src_features = data[f'mesh{fine_lvl}'].x
                dst_features = data[f'mesh{coarse_lvl}'].x
                print(f'正在計算 mesh{fine_lvl}→mesh{coarse_lvl} pool 邊特徵...')
                pool_edge_features = calculate_edge_features(edge_index, src_features, dst_features, edge_type='pool')
                data[pool_type].edge_attr = pool_edge_features
                print(f'mesh{fine_lvl}→mesh{coarse_lvl} pool 邊特徵形狀: {pool_edge_features.shape}')
            ups_type = (f'mesh{coarse_lvl}', f'upsample{coarse_lvl}_to_{fine_lvl}', f'mesh{fine_lvl}')
            if ups_type in data.edge_types:
                edge_index = data[ups_type].edge_index
                src_features = data[f'mesh{coarse_lvl}'].x
                dst_features = data[f'mesh{fine_lvl}'].x
                print(f'正在計算 mesh{coarse_lvl}→mesh{fine_lvl} upsample 邊特徵...')
                ups_edge_features = calculate_edge_features(edge_index, src_features, dst_features, edge_type='upsample')
                data[ups_type].edge_attr = ups_edge_features
                print(f'mesh{coarse_lvl}→mesh{fine_lvl} upsample 邊特徵形狀: {ups_edge_features.shape}')
    edge_type_encoder = ('grid', 'encoder7', 'mesh7')
    edge_type_decoder = ('mesh7', 'decoder', 'grid')
    if edge_type_encoder in data.edge_types:
        edge_index = data[edge_type_encoder].edge_index
        grid_features = data['grid'].x
        mesh_features = data['mesh7'].x
        print('正在計算 grid→mesh7 encoder 邊特徵...')
        encoder_edge_features = calculate_edge_features(edge_index, grid_features, mesh_features, edge_type='encoder')
        data[edge_type_encoder].edge_attr = encoder_edge_features
        print(f'grid→mesh7 encoder 邊特徵形狀: {encoder_edge_features.shape}')
    for level in refinement_levels:
        mesh_key = f'mesh{level}'
        edge_type = (mesh_key, f'decoder{level}', 'grid')
        if edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            src_features = data[mesh_key].x
            dst_features = data['grid'].x
            print(f'正在計算 {mesh_key}→grid decoder{level} 邊特徵...')
            decoder_edge_features = calculate_edge_features(edge_index, src_features, dst_features, edge_type='decoder')
            data[edge_type].edge_attr = decoder_edge_features
            print(f'{mesh_key}→grid decoder{level} 邊特徵形狀: {decoder_edge_features.shape}')
    output_path = 'mesh_grid_data_unet7/hetero_graph.pt'
    torch.save(data, output_path)
    print('已保存添加邊特徵的 mesh_grid_data_unet7/hetero_graph.pt')
