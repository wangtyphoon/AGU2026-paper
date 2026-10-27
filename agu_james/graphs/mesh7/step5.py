# Derived from gnn/ver_unet7/pre_process/self_loop5_unet7.py
from agu_james import runtime
import torch
import os

def add_self_loops_to_edge_type(data, edge_type):
    src, edge_name, dst = edge_type
    num_src_nodes = data[src].x.size(0)
    num_dst_nodes = data[dst].x.size(0)
    if src == dst:
        self_loops = torch.arange(num_src_nodes, device=data[edge_type].edge_index.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        existing_edges = data[edge_type].edge_index
        new_edge_index = torch.cat([existing_edges, self_loops], dim=1)
        if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
            existing_attr = data[edge_type].edge_attr
            attr_dim = existing_attr.size(1)
            self_loop_attr = torch.zeros(num_src_nodes, attr_dim, device=existing_attr.device)
            new_edge_attr = torch.cat([existing_attr, self_loop_attr], dim=0)
            data[edge_type].edge_attr = new_edge_attr
        data[edge_type].edge_index = new_edge_index
    return data

def main():
    global data, edge_type_proc, input_path, level, mesh_key, output_path
    input_path = runtime.setting('agu_james.graphs.mesh7.step5.input_path', lambda: 'mesh_grid_data_unet7/hetero_graph.pt')
    data = torch.load(input_path, weights_only=False)
    for level in [3, 4, 5, 6, 7]:
        mesh_key = f'mesh{level}'
        edge_type_proc = (mesh_key, 'processor', mesh_key)
        if mesh_key in data.node_types and edge_type_proc in data.edge_types:
            print(f'Adding self-loops to {mesh_key} processor edges...')
            data = add_self_loops_to_edge_type(data, edge_type_proc)
    output_path = 'mesh_grid_data_unet7/hetero_graph.pt'
    torch.save(data, output_path)
    print('已完成 self-loop，並保存 mesh_grid_data_unet/hetero_graph.pt')
