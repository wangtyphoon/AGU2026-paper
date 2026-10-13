# Derived from pre_process/chapter2_vis.py
# See provenance/sources.json for original hash and modifications.
from __future__ import annotations
from agu_james import runtime
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
GRAPH_PATH = runtime.setting('agu_james.plotting.graph_helpers.GRAPH_PATH', lambda: Path('mesh_grid_data_unet7') / 'hetero_graph.pt')
OUTPUT_DIR = runtime.setting('agu_james.plotting.graph_helpers.OUTPUT_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.graph_helpers')
FIG_A = 'panel_a_latlon_vs_icosahedral_mesh'
FIG_B = 'panel_b_multiscale_mesh_levels_3_7'
FIG_D = 'panel_d_local_connection_stencils'
COLORS = {'ink': '#17212b', 'muted': '#5f6f7a', 'grid': '#9aa8ad', 'grid_dark': '#65757d', 'latlon_dash': '#3f7fb7', 'domain_box': '#d04a3a', 'mesh': '#197d91', 'mesh_light': '#61bfc3', 'encoder': '#b84a39', 'decoder': '#2f6fb0', 'pool': '#587245', 'upsample': '#d28a1e', 'panel': '#f7fbfb'}

def import_torch():
    try:
        import torch
        return torch
    except ModuleNotFoundError:
        fallback = '/usr/bin/python3'
        if Path(fallback).exists() and Path(sys.executable).resolve() != Path(fallback).resolve():
            os.execv(fallback, [fallback, *sys.argv])
        raise SystemExit('PyTorch is required to read mesh_grid_data_unet7/hetero_graph.pt. Install torch and torch-geometric, or run this script with a Python environment that has them.')

def tensor_to_numpy(value):
    return value.detach().cpu().numpy()

def load_graph():
    torch = import_torch()
    try:
        import torch_geometric
    except ModuleNotFoundError as exc:
        raise SystemExit('torch-geometric is required because hetero_graph.pt stores a PyG HeteroData object.') from exc
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f'Cannot find graph file: {GRAPH_PATH}')
    return torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)

def coords(data, node_type):
    return tensor_to_numpy(data[node_type].x[:, :2]).astype(float)

def edges(data, edge_type):
    return tensor_to_numpy(data[edge_type].edge_index).astype(int)

def edge_segments(src_xy, dst_xy, edge_index, max_edges=None):
    edge_index = edge_index.T
    if max_edges is not None and edge_index.shape[0] > max_edges:
        step = int(np.ceil(edge_index.shape[0] / max_edges))
        edge_index = edge_index[::step]
    return np.stack([src_xy[edge_index[:, 0]], dst_xy[edge_index[:, 1]]], axis=1)

def add_edges(ax, src_xy, dst_xy, edge_index, color, lw=0.35, alpha=0.35, max_edges=None, zorder=1):
    segs = edge_segments(src_xy, dst_xy, edge_index, max_edges=max_edges)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=zorder))

def set_domain_axes(ax, all_xy, pad=1.4):
    lon_min, lat_min = all_xy.min(axis=0)
    lon_max, lat_max = all_xy.max(axis=0)
    ax.set_xlim(lon_min - pad, lon_max + pad)
    ax.set_ylim(lat_min - pad, lat_max + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, color='#d7e2e4', linewidth=0.7, alpha=0.75)

def set_grid_node_axes(ax, grid_xy, pad=1.4):
    lon_min, lat_min = grid_xy.min(axis=0)
    lon_max, lat_max = grid_xy.max(axis=0)
    ax.set_xlim(lon_min - pad, lon_max + pad)
    ax.set_ylim(lat_min - pad, lat_max + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    lon_start = int(np.floor(lon_min / 5.0) * 5)
    lon_end = int(np.ceil(lon_max / 5.0) * 5)
    lat_start = int(np.floor(lat_min / 5.0) * 5)
    lat_end = int(np.ceil(lat_max / 5.0) * 5)
    ax.set_xticks(np.arange(lon_start, lon_end + 1, 5))
    ax.set_yticks(np.arange(lat_start, lat_end + 1, 5))
    ax.grid(True, color=COLORS['latlon_dash'], linestyle='--', linewidth=0.75, alpha=0.5)

def style_axis(ax, title, fontsize=13):
    ax.set_facecolor(COLORS['panel'])
    for spine in ax.spines.values():
        spine.set_color('#cbdadd')
        spine.set_linewidth(0.8)
    ax.set_title(title, loc='left', fontsize=fontsize, fontweight='bold', color=COLORS['ink'], pad=10)

def panel_title(label, title):
    return f'({label}) {title}'

def save_figure(fig, basename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f'{basename}.png'
    pdf = OUTPUT_DIR / f'{basename}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png}')
    print(f'Saved: {pdf}')

def infer_grid_shape(grid_xy):
    unique_lon = np.unique(np.round(grid_xy[:, 0], 8))
    unique_lat = np.unique(np.round(grid_xy[:, 1], 8))
    return (len(unique_lat), len(unique_lon))

def draw_latlon_grid_from_nodes(ax, grid_xy):
    ax.scatter(grid_xy[:, 0], grid_xy[:, 1], s=2.4, color='#4b4f54', alpha=0.34, linewidths=0, zorder=2)

def resolution_token(resolution):
    return f'{resolution:g}'.replace('.', 'p')

def axis_values_for_resolution(start, stop, resolution):
    values = np.arange(start, stop + 1e-09, resolution)
    if not np.isclose(values[-1], stop):
        values = np.append(values, stop)
    return values

def latlon_grid_for_resolution(grid_xy, resolution):
    lon_min, lat_min = grid_xy.min(axis=0)
    lon_max, lat_max = grid_xy.max(axis=0)
    lons = axis_values_for_resolution(lon_min, lon_max, resolution)
    lats = axis_values_for_resolution(lat_min, lat_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

def draw_latlon_grid_lines(ax, grid_xy, resolution):
    lon_min, lat_min = grid_xy.min(axis=0)
    lon_max, lat_max = grid_xy.max(axis=0)
    lons = np.unique(np.round(grid_xy[:, 0], 8))
    lats = np.unique(np.round(grid_xy[:, 1], 8))
    lw = 0.22 if resolution <= 0.5 else 0.45
    alpha = 0.28 if resolution <= 0.5 else 0.48
    vertical = [((lon, lat_min), (lon, lat_max)) for lon in lons]
    horizontal = [((lon_min, lat), (lon_max, lat)) for lat in lats]
    ax.add_collection(LineCollection(vertical + horizontal, colors=COLORS['grid'], linewidths=lw, alpha=alpha, zorder=1))
    ax.scatter(grid_xy[:, 0], grid_xy[:, 1], s=5.2 if resolution >= 2 else 2.6, color='#4b4f54', alpha=0.5 if resolution >= 1 else 0.32, linewidths=0, zorder=2)

def latitude_scaled_xy(xy, reference_xy):
    lon0 = 0.5 * (reference_xy[:, 0].min() + reference_xy[:, 0].max())
    return np.column_stack([(xy[:, 0] - lon0) * np.cos(np.deg2rad(xy[:, 1])), xy[:, 1]])

def set_latitude_scaled_axes(ax, scaled_xy, original_xy):
    lon_min, lat_min = original_xy.min(axis=0)
    lon_max, lat_max = original_xy.max(axis=0)
    pad_x = max((scaled_xy[:, 0].max() - scaled_xy[:, 0].min()) * 0.06, 0.5)
    pad_y = max((lat_max - lat_min) * 0.06, 0.5)
    ax.set_xlim(scaled_xy[:, 0].min() - pad_x, scaled_xy[:, 0].max() + pad_x)
    ax.set_ylim(lat_min - pad_y, lat_max + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude distance scale')
    ax.set_ylabel('Latitude')
    ax.grid(True, color='#d7e2e4', linewidth=0.7, alpha=0.75)

def draw_scaled_domain_boundary(ax, grid_xy, reference_xy):
    lon_min, lat_min = grid_xy.min(axis=0)
    lon_max, lat_max = grid_xy.max(axis=0)
    lons = np.linspace(lon_min, lon_max, 180)
    lats = np.linspace(lat_min, lat_max, 180)
    boundary = [np.column_stack([lons, np.full_like(lons, lat_min)]), np.column_stack([lons, np.full_like(lons, lat_max)]), np.column_stack([np.full_like(lats, lon_min), lats]), np.column_stack([np.full_like(lats, lon_max), lats])]
    for side in boundary:
        scaled_side = latitude_scaled_xy(side, reference_xy)
        ax.plot(scaled_side[:, 0], scaled_side[:, 1], color=COLORS['domain_box'], lw=1.5, linestyle='--', zorder=4)

def draw_latitude_scaled_grid(ax, grid_xy, resolution=None):
    lon0 = 0.5 * (grid_xy[:, 0].min() + grid_xy[:, 0].max())
    scaled_xy = latitude_scaled_xy(grid_xy, grid_xy)
    lon_min, lat_min = grid_xy.min(axis=0)
    lon_max, lat_max = grid_xy.max(axis=0)
    lons = np.unique(np.round(grid_xy[:, 0], 8))
    lats = np.unique(np.round(grid_xy[:, 1], 8))
    lw = 0.22 if resolution is not None and resolution <= 0.5 else 0.45
    alpha = 0.28 if resolution is not None and resolution <= 0.5 else 0.48
    for lon in lons:
        line_lats = np.linspace(lat_min, lat_max, 180)
        line = np.column_stack([np.full_like(line_lats, lon), line_lats])
        scaled_line = latitude_scaled_xy(line, grid_xy)
        ax.plot(scaled_line[:, 0], scaled_line[:, 1], color=COLORS['grid'], lw=lw, alpha=alpha, zorder=1)
    for lat in lats:
        line_lons = np.linspace(lon_min, lon_max, 180)
        line = np.column_stack([line_lons, np.full_like(line_lons, lat)])
        scaled_line = latitude_scaled_xy(line, grid_xy)
        ax.plot(scaled_line[:, 0], scaled_line[:, 1], color=COLORS['grid'], lw=lw, alpha=alpha, zorder=1)
    ax.scatter(scaled_xy[:, 0], scaled_xy[:, 1], s=5.2 if resolution is not None and resolution >= 2 else 2.6, color='#4b4f54', alpha=0.5 if resolution is not None and resolution >= 1 else 0.32, linewidths=0, zorder=2)
    sample_lats = np.linspace(lat_min, lat_max, 4)
    for lat in sample_lats:
        x0 = (lon_min - lon0) * np.cos(np.deg2rad(lat))
        x1 = (lon_max - lon0) * np.cos(np.deg2rad(lat))
        ax.plot([x0, x1], [lat, lat], color=COLORS['latlon_dash'], lw=0.75, alpha=0.42, zorder=1)
    set_latitude_scaled_axes(ax, scaled_xy, grid_xy)

def draw_latitude_scaled_mesh(ax, mesh_xy, mesh_edges, grid_xy):
    scaled_mesh_xy = latitude_scaled_xy(mesh_xy, grid_xy)
    add_edges(ax, scaled_mesh_xy, scaled_mesh_xy, mesh_edges, COLORS['mesh'], lw=0.28, alpha=0.34, max_edges=90000)
    ax.scatter(scaled_mesh_xy[:, 0], scaled_mesh_xy[:, 1], s=2.8, color=COLORS['mesh'], alpha=0.65, zorder=2)
    draw_scaled_domain_boundary(ax, grid_xy, grid_xy)
    set_latitude_scaled_axes(ax, scaled_mesh_xy, mesh_xy)

def draw_mesh_with_domain(ax, mesh_xy, mesh_edges, grid_xy, level):
    add_edges(ax, mesh_xy, mesh_xy, mesh_edges, COLORS['mesh'], lw=0.34, alpha=0.36, max_edges=90000)
    ax.scatter(mesh_xy[:, 0], mesh_xy[:, 1], s=5 if level <= 4 else 2.8, color=COLORS['mesh'], alpha=0.68, zorder=2)
    grid_lon_min, grid_lat_min = grid_xy.min(axis=0)
    grid_lon_max, grid_lat_max = grid_xy.max(axis=0)
    ax.add_patch(Rectangle((grid_lon_min, grid_lat_min), grid_lon_max - grid_lon_min, grid_lat_max - grid_lat_min, fill=False, edgecolor=COLORS['domain_box'], linewidth=1.8, linestyle='--', zorder=4))
    set_domain_axes(ax, mesh_xy)
    ax.text(grid_lon_max - 0.35, grid_lat_max - 0.55, 'grid domain', fontsize=9.5, color=COLORS['domain_box'], fontweight='bold', ha='right', va='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75), zorder=5)

def plot_panel_a(data):
    grid_xy = coords(data, 'grid')
    scale_specs = [(0.25, 7), (0.5, 6), (1.0, 5), (2.0, 4), (4.0, 3)]
    for resolution, mesh_level in scale_specs:
        resolution_grid_xy = latlon_grid_for_resolution(grid_xy, resolution)
        mesh_xy = coords(data, f'mesh{mesh_level}')
        mesh_edges = edges(data, (f'mesh{mesh_level}', 'processor', f'mesh{mesh_level}'))
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), constrained_layout=True)
        style_axis(axes[0], panel_title('a', f'{resolution:g} degree latitude-longitude grid'))
        draw_latlon_grid_lines(axes[0], resolution_grid_xy, resolution)
        set_grid_node_axes(axes[0], grid_xy)
        style_axis(axes[1], panel_title('b', 'Grid distance projection'))
        draw_latitude_scaled_grid(axes[1], resolution_grid_xy, resolution)
        style_axis(axes[2], panel_title('c', f'Icosahedral mesh level {mesh_level}'))
        draw_mesh_with_domain(axes[2], mesh_xy, mesh_edges, grid_xy, mesh_level)
        style_axis(axes[3], panel_title('d', 'Mesh distance projection'))
        draw_latitude_scaled_mesh(axes[3], mesh_xy, mesh_edges, grid_xy)
        save_figure(fig, f'panel_a_grid_mesh_scale_{resolution_token(resolution)}deg')

def plot_panel_b(data):
    levels = [3, 4, 5, 6, 7]
    fig, axes = plt.subplots(1, len(levels), figsize=(18, 5.2), sharex=True, sharey=True, constrained_layout=True)
    all_xy = np.vstack([coords(data, f'mesh{level}') for level in levels])
    colors = plt.cm.YlGnBu(np.linspace(0.42, 0.86, len(levels)))
    for label, ax, level, color in zip('abcde', axes, levels, colors):
        node_xy = coords(data, f'mesh{level}')
        proc_edges = edges(data, (f'mesh{level}', 'processor', f'mesh{level}'))
        title = panel_title(label, f'Mesh level {level}   {node_xy.shape[0]:,} nodes | {proc_edges.shape[1]:,} edges')
        style_axis(ax, title, fontsize=9.8)
        add_edges(ax, node_xy, node_xy, proc_edges, color, lw=0.34, alpha=0.38, max_edges=75000)
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=5 if level <= 4 else 2.3, color=color, alpha=0.78, zorder=2)
        set_domain_axes(ax, all_xy, pad=1.2)
    save_figure(fig, FIG_B)

def nearest_node(xy, lon, lat):
    return int(np.argmin((xy[:, 0] - lon) ** 2 + (xy[:, 1] - lat) ** 2))

def edge_targets_from_source(edge_index, source_idx):
    return edge_index[1, edge_index[0] == source_idx]

def edge_sources_to_target(edge_index, target_idx):
    return edge_index[0, edge_index[1] == target_idx]

def local_limits(ax, points, pad_ratio=0.35):
    lon_min, lat_min = points.min(axis=0)
    lon_max, lat_max = points.max(axis=0)
    span = max(lon_max - lon_min, lat_max - lat_min, 0.5)
    pad = span * pad_ratio
    ax.set_xlim(lon_min - pad, lon_max + pad)
    ax.set_ylim(lat_min - pad, lat_max + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, color='#d7e2e4', linewidth=0.65, alpha=0.75)

def plot_encoder_stencil(ax, data):
    grid_xy = coords(data, 'grid')
    mesh_xy = coords(data, 'mesh7')
    enc = edges(data, ('grid', 'encoder7', 'mesh7'))
    center_mesh = nearest_node(mesh_xy, 126.0, 24.0)
    grid_ids = edge_sources_to_target(enc, center_mesh)
    pts = np.vstack([mesh_xy[[center_mesh]], grid_xy[grid_ids]])
    style_axis(ax, panel_title('a', 'Fixed-radius grid-to-mesh encoding'))
    for gid in grid_ids:
        ax.plot([grid_xy[gid, 0], mesh_xy[center_mesh, 0]], [grid_xy[gid, 1], mesh_xy[center_mesh, 1]], color=COLORS['encoder'], lw=0.75, alpha=0.85)
    radius = np.max(np.linalg.norm(grid_xy[grid_ids] - mesh_xy[center_mesh], axis=1))
    ax.add_patch(Circle(mesh_xy[center_mesh], radius, fill=False, edgecolor=COLORS['encoder'], linewidth=1.5, linestyle='--'))
    circle_bounds = mesh_xy[center_mesh] + np.array([[-radius, -radius], [-radius, radius], [radius, -radius], [radius, radius]])
    ax.scatter(grid_xy[grid_ids, 0], grid_xy[grid_ids, 1], marker='s', s=36, color=COLORS['grid_dark'], zorder=3, label='grid nodes')
    ax.scatter(mesh_xy[center_mesh, 0], mesh_xy[center_mesh, 1], marker='^', s=110, color=COLORS['mesh'], zorder=4, label='mesh7 node')
    ax.legend(loc='upper right', fontsize=9)
    local_limits(ax, np.vstack([pts, circle_bounds]))

def plot_decoder_stencil(ax, data):
    grid_xy = coords(data, 'grid')
    mesh_xy = coords(data, 'mesh7')
    dec = edges(data, ('mesh7', 'decoder7', 'grid'))
    grid_id = nearest_node(grid_xy, 126.0, 24.0)
    mesh_ids = edge_sources_to_target(dec, grid_id)
    pts = np.vstack([grid_xy[[grid_id]], mesh_xy[mesh_ids]])
    style_axis(ax, panel_title('b', '3-nearest-neighbor mesh-to-grid decoding'))
    if len(mesh_ids) >= 3:
        tri = mesh_xy[mesh_ids[:3]]
        ax.fill(tri[:, 0], tri[:, 1], color='#dbeaf7', alpha=0.65, zorder=0)
    for mid in mesh_ids:
        ax.plot([mesh_xy[mid, 0], grid_xy[grid_id, 0]], [mesh_xy[mid, 1], grid_xy[grid_id, 1]], color=COLORS['decoder'], lw=1.1, alpha=0.9)
    ax.scatter(mesh_xy[mesh_ids, 0], mesh_xy[mesh_ids, 1], s=70, color=COLORS['mesh'], zorder=3, label='mesh7 nodes')
    ax.scatter(grid_xy[grid_id, 0], grid_xy[grid_id, 1], marker='s', s=90, color=COLORS['decoder'], zorder=4, label='grid node')
    ax.legend(loc='upper right', fontsize=9)
    local_limits(ax, pts)

def plot_crossscale_stencil(ax, data):
    coarse_xy = coords(data, 'mesh6')
    fine_xy = coords(data, 'mesh7')
    up = edges(data, ('mesh6', 'upsample6_to_7', 'mesh7'))
    pool = edges(data, ('mesh7', 'pool7_to_6', 'mesh6'))
    coarse_id = nearest_node(coarse_xy, 126.0, 24.0)
    fine_ids = edge_targets_from_source(up, coarse_id)
    pts = np.vstack([coarse_xy[[coarse_id]], fine_xy[fine_ids]])
    style_axis(ax, panel_title('c', '7-neighbor pooling / upsampling connection'))
    for fid in fine_ids:
        ax.plot([coarse_xy[coarse_id, 0], fine_xy[fid, 0]], [coarse_xy[coarse_id, 1], fine_xy[fid, 1]], color=COLORS['pool'], lw=1.0, alpha=0.85)
    ax.scatter(fine_xy[fine_ids, 0], fine_xy[fine_ids, 1], s=62, color=COLORS['mesh'], zorder=3, label='mesh7 fine nodes')
    ax.scatter(coarse_xy[coarse_id, 0], coarse_xy[coarse_id, 1], marker='D', s=105, color=COLORS['upsample'], edgecolor='white', linewidth=0.8, zorder=4, label='mesh6 coarse node')
    ax.legend(loc='upper right', fontsize=9)
    local_limits(ax, pts)

def plot_panel_d(data):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3), constrained_layout=True)
    plot_encoder_stencil(axes[0], data)
    plot_decoder_stencil(axes[1], data)
    plot_crossscale_stencil(axes[2], data)
    for ax in axes:
        ax.set_anchor('N')
    save_figure(fig, FIG_D)

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 160, 'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})
    data = load_graph()
    print(data)
    plot_panel_a(data)
    plot_panel_b(data)
    plot_panel_d(data)
