# Derived from pre_process/chapter2_vis_composite.py
# See provenance/sources.json for original hash and modifications.
"""Create a three-row composite of the chapter 2 graph visualizations."""
from __future__ import annotations
from agu_james import runtime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import agu_james.plotting.graph_helpers as source
ROOT = runtime.setting('agu_james.plotting.graph.ROOT', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.graph')
GRAPH_PATH = runtime.setting('agu_james.plotting.graph.GRAPH_PATH', lambda: ROOT / 'mesh_grid_data_unet7' / 'hetero_graph.pt')
OUTPUT_DIR = runtime.setting('agu_james.plotting.graph.OUTPUT_DIR', lambda: runtime.artifact_root() / 'jobs/agu_james.plotting.graph')
OUTPUT_NAME = 'chapter2_mesh7_multiscale_connections_composite'

def load_data():
    torch = source.import_torch()
    import torch_geometric
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f'Cannot find graph file: {GRAPH_PATH}')
    return torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)

def set_title(ax, label, title, fontsize=16.5):
    ax.set_title(source.panel_title(label, title), loc='left', fontsize=fontsize, fontweight='bold', color=source.COLORS['ink'], pad=10)

def draw_mesh7_row(axes, data):
    grid_xy = source.coords(data, 'grid')
    resolution = 0.25
    resolution_grid_xy = source.latlon_grid_for_resolution(grid_xy, resolution)
    mesh_xy = source.coords(data, 'mesh7')
    mesh_edges = source.edges(data, ('mesh7', 'processor', 'mesh7'))
    titles = ('0.25° latitude-longitude grid', 'Grid distance projection', 'Icosahedral mesh7', 'Mesh distance projection')
    for label, ax, title in zip('abcd', axes, titles):
        source.style_axis(ax, source.panel_title(label, title), fontsize=15.75)
    source.draw_latlon_grid_lines(axes[0], resolution_grid_xy, resolution)
    source.set_grid_node_axes(axes[0], grid_xy)
    source.draw_latitude_scaled_grid(axes[1], resolution_grid_xy, resolution)
    source.draw_mesh_with_domain(axes[2], mesh_xy, mesh_edges, grid_xy, 7)
    source.draw_latitude_scaled_mesh(axes[3], mesh_xy, mesh_edges, grid_xy)

def draw_multiscale_row(axes, data):
    levels = range(3, 8)
    all_xy = np.vstack([source.coords(data, f'mesh{level}') for level in levels])
    colors = plt.cm.YlGnBu(np.linspace(0.42, 0.86, len(axes)))
    for label, ax, level, color in zip('efghi', axes, levels, colors):
        node_xy = source.coords(data, f'mesh{level}')
        proc_edges = source.edges(data, (f'mesh{level}', 'processor', f'mesh{level}'))
        title = f'Mesh{level}\n{len(node_xy):,} nodes | {proc_edges.shape[1]:,} edges'
        source.style_axis(ax, source.panel_title(label, title), fontsize=15)
        source.add_edges(ax, node_xy, node_xy, proc_edges, color, lw=0.34, alpha=0.38, max_edges=75000)
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=5 if level <= 4 else 2.3, color=color, alpha=0.78, zorder=2)
        source.set_domain_axes(ax, all_xy, pad=1.2)

def draw_processor_stencil(ax, data):
    mesh_xy = source.coords(data, 'mesh7')
    proc_edges = source.edges(data, ('mesh7', 'processor', 'mesh7'))
    center_id = source.nearest_node(mesh_xy, 126.0, 24.0)
    targets = np.unique(source.edge_targets_from_source(proc_edges, center_id))
    neighbors = targets[targets != center_id]
    if len(neighbors) != 6 or center_id not in targets:
        raise ValueError('The selected mesh7 processor node must have six neighbors and a self-loop')
    center = mesh_xy[center_id]
    points = np.vstack([center, mesh_xy[neighbors]])
    source.style_axis(ax, source.panel_title('k', 'Mesh7 processor: 6 neighbors + self-loop'), fontsize=16.5)
    for neighbor_id in neighbors:
        ax.plot([center[0], mesh_xy[neighbor_id, 0]], [center[1], mesh_xy[neighbor_id, 1]], color=source.COLORS['pool'], lw=1.15, alpha=0.9, zorder=1)
    ax.scatter(mesh_xy[neighbors, 0], mesh_xy[neighbors, 1], s=64, color=source.COLORS['mesh'], zorder=3, label='neighbor nodes')
    ax.scatter(center[0], center[1], marker='D', s=105, color=source.COLORS['upsample'], edgecolor='white', linewidth=0.8, zorder=4, label='center node')
    loop_radius = np.min(np.linalg.norm(mesh_xy[neighbors] - center, axis=1)) * 0.2
    ax.add_patch(Circle(center, loop_radius, fill=False, edgecolor=source.COLORS['encoder'], linewidth=1.8, linestyle='--', zorder=5, label='self-loop'))
    ax.legend(loc='upper right', fontsize=9)
    source.local_limits(ax, points)

def draw_connections_row(axes, data):
    source.plot_encoder_stencil(axes[0], data)
    set_title(axes[0], 'j', 'Grid-to-mesh7 encoder', fontsize=16.5)
    draw_processor_stencil(axes[1], data)
    source.plot_decoder_stencil(axes[2], data)
    set_title(axes[2], 'l', 'Mesh7-to-grid decoder', fontsize=16.5)
    for ax in axes:
        ax.set_anchor('N')

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 160, 'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})
    data = load_data()
    fig = plt.figure(figsize=(24, 17), layout='constrained')
    rows = fig.subfigures(3, 1, height_ratios=(1.0, 1.0, 1.0), hspace=0.06)
    mesh7_axes = rows[0].subplots(1, 4)
    multiscale_axes = rows[1].subplots(1, 5, sharex=True, sharey=True)
    connection_axes = rows[2].subplots(1, 3)
    draw_mesh7_row(mesh7_axes, data)
    draw_multiscale_row(multiscale_axes, data)
    draw_connections_row(connection_axes, data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (('png', {'dpi': 300}), ('pdf', {})):
        output = OUTPUT_DIR / f'{OUTPUT_NAME}.{extension}'
        fig.savefig(output, bbox_inches='tight', **kwargs)
        print(f'Saved: {output}')
    plt.close(fig)
