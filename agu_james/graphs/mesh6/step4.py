# Derived from gnn/ver_unet7/pre_process/PE+SE3_unet.py
from agu_james import runtime
import torch
from torch_geometric.transforms import AddRandomWalkPE, AddLaplacianEigenvectorPE
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import os

def main():
    global _, data, device, edge_type, input_path, laplacian_pe_transform, level, mesh_data, mesh_edge_index, mesh_key, mesh_x, num_mesh_nodes, output_dir, random_walk_pe_transform
    input_path = runtime.setting('agu_james.graphs.mesh6.step4.input_path', lambda: 'mesh_grid_data_unet/hetero_graph.pt')
    data = torch.load(input_path, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    for level in [3, 4, 5, 6]:
        mesh_key = f'mesh{level}'
        edge_type = (mesh_key, 'processor', mesh_key)
        if mesh_key in data.node_types and edge_type in data.edge_types:
            mesh_x = data[mesh_key].x.to(device)
            mesh_edge_index = data[edge_type].edge_index.to(device)
            num_mesh_nodes = mesh_x.size(0)
            mesh_data = Data(x=mesh_x, edge_index=mesh_edge_index, num_nodes=num_mesh_nodes).to(device)
            mesh_data.edge_index, _ = add_self_loops(mesh_data.edge_index, num_nodes=mesh_data.num_nodes)
            laplacian_pe_transform = AddLaplacianEigenvectorPE(k=6, attr_name=None, is_undirected=True)
            mesh_data = laplacian_pe_transform(mesh_data)
            random_walk_pe_transform = AddRandomWalkPE(walk_length=6, attr_name=None)
            mesh_data = random_walk_pe_transform(mesh_data)
            data[mesh_key].x = mesh_data.x.cpu()
            print(f'{mesh_key} 節點特徵 shape (PE+SE3):', data[mesh_key].x.shape)
    output_dir = runtime.setting('agu_james.graphs.mesh6.step4.output_dir', lambda: runtime.artifact_root() / 'jobs/agu_james.graphs.mesh6.step4')
    torch.save(data, os.path.join(output_dir, 'hetero_graph.pt'))
    print('已更新並保存 mesh_grid_data_unet/hetero_graph.pt (PE+SE3)')
