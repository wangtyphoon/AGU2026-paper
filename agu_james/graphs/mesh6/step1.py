# Derived from gnn/ver_unet7/pre_process/create_hetero_mesh_poolk.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import numpy as np
import torch
from torch_geometric.data import HeteroData
import os
import matplotlib.pyplot as plt
from scipy.linalg import norm
import math
import concurrent.futures
from functools import partial
import itertools
import time

def create_icosahedron():
    """建立一個二十面體。"""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0], [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi], [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], dtype=np.float64)
    for i in range(len(vertices)):
        n = norm(vertices[i])
        if n > 1e-09:
            vertices[i] = vertices[i] / n
    faces = np.array([[0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [0, 11, 5], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [3, 9, 4], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return (vertices, faces)

def refine_icosahedron(vertices, faces, refinement_level):
    """依照指定的細化層級細分二十面體網格。"""
    edge_midpoints = {}
    vertices_list = vertices.tolist()
    for _ in range(refinement_level):
        new_faces = []
        edge_midpoints_level = {}
        num_current_vertices = len(vertices_list)
        norms = {i: norm(v) for i, v in enumerate(vertices_list)}
        for face in faces:
            v1, v2, v3 = face
            midpoints_indices = []

            def get_midpoint_idx(idx1, idx2):
                edge_key = tuple(sorted((idx1, idx2)))
                if edge_key in edge_midpoints_level:
                    return edge_midpoints_level[edge_key]
                if edge_key in edge_midpoints:
                    return edge_midpoints[edge_key]
                p1 = np.array(vertices_list[idx1])
                p2 = np.array(vertices_list[idx2])
                midpoint = (p1 + p2) / 2.0
                midpoint_norm = norm(midpoint)
                if midpoint_norm > 1e-09:
                    midpoint = midpoint / midpoint_norm
                vertices_list.append(midpoint.tolist())
                new_idx = len(vertices_list) - 1
                edge_midpoints_level[edge_key] = new_idx
                edge_midpoints[edge_key] = new_idx
                norms[new_idx] = norm(vertices_list[new_idx])
                return new_idx
            v12 = get_midpoint_idx(v1, v2)
            v23 = get_midpoint_idx(v2, v3)
            v31 = get_midpoint_idx(v3, v1)
            new_faces.append([v1, v12, v31])
            new_faces.append([v2, v23, v12])
            new_faces.append([v3, v31, v23])
            new_faces.append([v12, v23, v31])
        faces = np.array(new_faces)
    vertices = np.array(vertices_list)
    return (vertices, faces)

def spherical_coordinates(vertices):
    """將笛卡爾座標轉換為經緯度 (度)"""
    x, y, z = (vertices[:, 0], vertices[:, 1], vertices[:, 2])
    r = np.linalg.norm(vertices, axis=1)
    r[r < 1e-09] = 1e-09
    z_over_r = np.clip(z / r, -1.0, 1.0)
    longitude = np.degrees(np.arctan2(y, x))
    latitude = np.degrees(np.arcsin(z_over_r))
    return (longitude, latitude)

def extract_valid_vertices(vertices, faces, lon_min, lon_max, lat_min, lat_max, earth_radius):
    """
    調整頂點至地球半徑、轉換經緯度並過濾指定範圍內的頂點。
    回傳：
      valid_idx: 符合範圍的頂點在 vertices 中的索引
      lon, lat: 所有頂點的經緯度資訊（度）
      valid_vertices_scaled: 符合範圍的頂點縮放至地球半徑後的座標
    """
    vertices_scaled = vertices * earth_radius
    lon_all, lat_all = spherical_coordinates(vertices_scaled)
    valid_idx = np.where((lon_all >= lon_min) & (lon_all <= lon_max) & (lat_all >= lat_min) & (lat_all <= lat_max))[0]
    valid_vertices_scaled = vertices_scaled[valid_idx]
    return (valid_idx, lon_all, lat_all, valid_vertices_scaled)

def haversine_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    計算兩點間的球面距離（公里），輸入的經緯度以度表示。
    Uses NumPy for vectorized calculation if inputs are arrays.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c

def process_encoder_chunk(mesh_indices_chunk, mesh_lons, mesh_lats, mesh_levels, grid_lat, grid_lon, num_grid_nodes, level_distance_thresholds):
    """Processes a chunk of mesh nodes to find encoder edges."""
    local_encoder_edges = []
    for mesh_idx in mesh_indices_chunk:
        mesh_level = int(mesh_levels[mesh_idx])
        if mesh_level not in level_distance_thresholds:
            continue
        distance_threshold = level_distance_thresholds[mesh_level]
        mesh_lat_val = mesh_lats[mesh_idx]
        mesh_lon_val = mesh_lons[mesh_idx]
        distances = haversine_distance(mesh_lat_val, mesh_lon_val, grid_lat, grid_lon)
        nearby_grid_indices = np.where(distances < distance_threshold)[0]
        for grid_local_idx in nearby_grid_indices:
            local_encoder_edges.append([grid_local_idx, mesh_idx])
    return local_encoder_edges

def process_decoder_chunk(grid_indices_chunk, grid_lat, grid_lon, mesh_lons, mesh_lats, mesh_levels, mesh_indices_per_level, refinement_levels):
    """Processes a chunk of grid nodes to find decoder edges."""
    local_decoder_edges = []
    mesh_data_per_level = {}
    for level in refinement_levels:
        indices = mesh_indices_per_level.get(level, np.array([], dtype=int))
        if len(indices) > 0:
            mesh_data_per_level[level] = {'indices': indices, 'lats': mesh_lats[indices], 'lons': mesh_lons[indices]}
    for grid_local_idx in grid_indices_chunk:
        connected_mesh_indices = set()
        grid_node_lat = grid_lat[grid_local_idx]
        grid_node_lon = grid_lon[grid_local_idx]
        for level in refinement_levels:
            if level not in mesh_data_per_level:
                continue
            level_data = mesh_data_per_level[level]
            level_specific_mesh_indices = level_data['indices']
            level_mesh_lats = level_data['lats']
            level_mesh_lons = level_data['lons']
            if len(level_specific_mesh_indices) == 0:
                continue
            distances_for_level = haversine_distance(grid_node_lat, grid_node_lon, level_mesh_lats, level_mesh_lons)
            sorted_indices_local = np.argsort(distances_for_level)
            num_to_take = min(3, len(level_specific_mesh_indices))
            closest_mesh_indices_global = level_specific_mesh_indices[sorted_indices_local[:num_to_take]]
            for mesh_idx in closest_mesh_indices_global:
                connected_mesh_indices.add(mesh_idx)
        for mesh_idx in connected_mesh_indices:
            local_decoder_edges.append([mesh_idx, grid_local_idx])
    return local_decoder_edges

def main():
    global _, a, base_f, base_v, c_lats, c_lons, ci, close, coarse, coarse_arr, d, data, dec_edge_tensor, dec_edges_dict, dec_list, dist, dlat, dlon, e, earth_radius, edge_index, edges, enc_edge_tensor, enc_edges, end_time_mesh, et, f, f_lats, f_lons, faces, feat, fi, fine, fine_arr, finest, finest_nodes, g, grid_features, grid_lat, grid_lon, i, idx, idx_loc, j, key, lat_all, lat_grid, lat_max, lat_min, level, level_distance_thresholds, level_node_map, lon_all, lon_grid, lon_max, lon_min, m, mesh_coords, mesh_edges, mesh_idx, mesh_lats, mesh_lons, mesh_nodes, nearest, nearest_idxs, node_counters, num_grid_nodes, num_mesh_finest, output_dir, pe, pool_k, pooling_edges, proc, refinement_levels, size, start_time_encoder, start_time_mesh, u, upsample_edges, v, valid_idx, valid_set, verts, widen_range
    refinement_levels = [3, 4, 5, 6]
    earth_radius = 6371.0
    lon_min, lon_max = (102.25, 150)
    lat_min, lat_max = (0, 47.75)
    widen_range = 2.5
    lon_min, lon_max = (102.25 - widen_range, 150.0 + widen_range)
    lat_min, lat_max = (0 - widen_range, 47.75 + widen_range)
    print('建立全域 Mesh 節點和邊（分層儲存）...')
    start_time_mesh = time.time()
    mesh_nodes = {}
    mesh_edges = {}
    level_node_map = {}
    node_counters = {}
    for level in refinement_levels:
        print(f'  處理細化層級 {level}...')
        base_v, base_f = create_icosahedron()
        verts, faces = refine_icosahedron(base_v, base_f, level)
        valid_idx, lon_all, lat_all, _ = extract_valid_vertices(verts, faces, lon_min, lon_max, lat_min, lat_max, earth_radius)
        mesh_nodes[level] = []
        mesh_edges[level] = set()
        level_node_map[level] = {}
        node_counters[level] = 0
        for i in valid_idx:
            idx_loc = node_counters[level]
            level_node_map[level][i] = idx_loc
            mesh_nodes[level].append([lon_all[i], lat_all[i]])
            node_counters[level] += 1
        valid_set = set(valid_idx)
        for f in faces:
            if all((v in valid_set for v in f)):
                g = [level_node_map[level][v] for v in f]
                for u, v in [(g[0], g[1]), (g[1], g[2]), (g[2], g[0])]:
                    mesh_edges[level].add((u, v))
                    mesh_edges[level].add((v, u))
    for level in refinement_levels:
        print(f'Level {level} 節點數: {len(mesh_nodes[level])}, 邊數: {len(mesh_edges[level])}')
    end_time_mesh = time.time()
    print('\n載入 ERA5 grid 資料...')
    try:
        lat_grid = np.load(runtime.path('agu_james.graphs.mesh6.step1.lat.npy'))
        lon_grid = np.load(runtime.path('agu_james.graphs.mesh6.step1.lon.npy'))
    except FileNotFoundError:
        print('錯誤：找不到 ERA5 grid 檔案。請檢查路徑。')
        exit()
    size = 192
    if lat_grid.shape != (size, size) or lon_grid.shape != (size, size):
        print(f'警告: 載入的 grid 資料形狀為 {lat_grid.shape}, 與預期的 ({size},{size}) 不同。偏移量可能超出範圍。')
    grid_lat = lat_grid.flatten()
    grid_lon = lon_grid.flatten()
    num_grid_nodes = grid_lat.shape[0]
    grid_features = np.stack([grid_lon, grid_lat], axis=1).astype(np.float32)
    print(f'Grid 節點數: {num_grid_nodes}')
    start_time_encoder = time.time()
    level_distance_thresholds = {3: 606.0, 4: 303.0, 5: 151.0, 6: 75.0}
    finest = max(refinement_levels)
    print(f'\nEncoder/Decoder 僅連到 level={finest} 節點')
    finest_nodes = np.array(mesh_nodes[finest], dtype=np.float32)
    mesh_lons = finest_nodes[:, 0]
    mesh_lats = finest_nodes[:, 1]
    num_mesh_finest = finest_nodes.shape[0]
    enc_edges = []
    for mesh_idx in range(num_mesh_finest):
        dist = haversine_distance(mesh_lats[mesh_idx], mesh_lons[mesh_idx], grid_lat, grid_lon)
        close = np.where(dist < level_distance_thresholds[finest])[0]
        for g in close:
            enc_edges.append([g, mesh_idx])
    enc_edge_tensor = torch.tensor(enc_edges, dtype=torch.long).t().contiguous()
    print(f'Encoder 邊 (grid→mesh{finest}): {enc_edge_tensor.shape[1]}')
    dec_edges_dict = {}
    for level in refinement_levels:
        print(f'建立 Decoder 邊 (mesh{level} → grid)...')
        mesh_coords = np.array(mesh_nodes[level], dtype=np.float32)
        mesh_lons = mesh_coords[:, 0]
        mesh_lats = mesh_coords[:, 1]
        dec_list = []
        for g in range(num_grid_nodes):
            dist = haversine_distance(grid_lat[g], grid_lon[g], mesh_lats, mesh_lons)
            nearest_idxs = np.argsort(dist)[:3]
            for m in nearest_idxs:
                dec_list.append([m, g])
        dec_edge_tensor = torch.tensor(dec_list, dtype=torch.long).t().contiguous()
        dec_edges_dict[level] = dec_edge_tensor
        print(f'  mesh{level}→grid 邊數: {dec_edge_tensor.shape[1]}')
    print('\n建立 Pooling / Upsample 邊…')
    pool_k = 7
    earth_radius = 6371.0
    for idx, level in enumerate(refinement_levels[:-1]):
        coarse = level
        fine = refinement_levels[idx + 1]
        coarse_arr = np.array(mesh_nodes[coarse])[:, :2]
        fine_arr = np.array(mesh_nodes[fine])[:, :2]
        c_lats = np.radians(coarse_arr[:, 1])
        c_lons = np.radians(coarse_arr[:, 0])
        f_lats = np.radians(fine_arr[:, 1])
        f_lons = np.radians(fine_arr[:, 0])
        pooling_edges = []
        upsample_edges = []
        for ci in range(coarse_arr.shape[0]):
            dlat = f_lats - c_lats[ci]
            dlon = f_lons - c_lons[ci]
            a = np.sin(dlat / 2) ** 2 + np.cos(c_lats[ci]) * np.cos(f_lats) * np.sin(dlon / 2) ** 2
            d = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) * earth_radius
            nearest = np.argsort(d)[:pool_k]
            for fi in nearest:
                pooling_edges.append([fi, ci])
                upsample_edges.append([ci, fi])
        mesh_edges[f'pool{fine}_to_{coarse}'] = pooling_edges
        mesh_edges[f'upsample{coarse}_to_{fine}'] = upsample_edges
        print(f'mesh{coarse}→mesh{fine} pool 邊: {len(pooling_edges)}')
        print(f'mesh{fine}→mesh{coarse} upsample 邊: {len(upsample_edges)}')
    print('\n建立 HeteroData 物件…')
    data = HeteroData()
    data['grid'].x = torch.tensor(grid_features, dtype=torch.float)
    data['grid', f'encoder{finest}', f'mesh{finest}'].edge_index = enc_edge_tensor
    for level, edge_index in dec_edges_dict.items():
        data[f'mesh{level}', f'decoder{level}', 'grid'].edge_index = edge_index
    for level in refinement_levels:
        feat = np.array(mesh_nodes[level])[:, :2].astype(np.float32)
        data[f'mesh{level}'].x = torch.tensor(feat, dtype=torch.float)
        proc = list(mesh_edges[level])
        pe = torch.tensor(proc, dtype=torch.long).t().contiguous()
        data[f'mesh{level}', 'processor', f'mesh{level}'].edge_index = pe
    for key, edges in mesh_edges.items():
        if str(key).startswith('pool'):
            fine, coarse = key.replace('pool', '').split('_to_')
            i = int(fine)
            j = int(coarse)
            e = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data[f'mesh{i}', f'pool{i}_to_{j}', f'mesh{j}'].edge_index = e
        elif str(key).startswith('upsample'):
            coarse, fine = key.replace('upsample', '').split('_to_')
            i = int(coarse)
            j = int(fine)
            e = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data[f'mesh{i}', f'upsample{i}_to_{j}', f'mesh{j}'].edge_index = e
    print(data)
    for et in data.edge_types:
        print(f'{et}: {data[et].num_edges} edges')
    output_dir = runtime.setting('agu_james.graphs.mesh6.step1.output_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.graphs.mesh6.step1')
    os.makedirs(output_dir, exist_ok=True)
    print(f'\n保存資料至 {output_dir}...')
    try:
        torch.save(data, os.path.join(output_dir, 'hetero_graph.pt'))
        print(f'所有資料已成功保存至 {output_dir} 目錄')
    except Exception as e:
        print(f'保存資料時發生錯誤: {e}')
