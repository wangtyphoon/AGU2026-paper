# Derived from gnn/ver_autoregressive/model_unet.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from agu_james.training.models.shared.gatv2convNG import GATv2ConvN
from agu_james.training.models.shared.gat_edge_ffn_upG_fix import GATv2Conv_up
from torch.utils.checkpoint import checkpoint
from agu_james.training.models.shared.processor__blockG import ProcessorBlock

class HeteroGNN(nn.Module):

    def __init__(self, in_grid, in_mesh, hidden_dim, out_grid, heads=1, processor_layers=[1, 16, 8, 4, 2], use_transconv=True, use_checkpoint=True, ffn_ratio=1.0):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        levels = [7, 6, 5, 4, 3]
        self.edge_dim = 6
        self.encoders = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        self.processors = nn.ModuleDict()
        for i, lvl in enumerate(levels):
            if lvl == levels[0]:
                self.encoders[f'enc{lvl}'] = GATv2ConvN(in_channels=(in_grid, in_mesh), out_channels=hidden_dim, heads=heads, edge_dim=self.edge_dim, residual=True, add_self_loops=False, ffn_ratio=ffn_ratio)
            if i < len(levels) - 1:
                next_lvl = levels[i + 1]
                self.pools[f'pool{lvl}_{next_lvl}'] = GATv2ConvN(in_channels=(hidden_dim, in_mesh), out_channels=hidden_dim, heads=heads, edge_dim=self.edge_dim, residual=True, add_self_loops=False, ffn_ratio=ffn_ratio)
            self.processors[f'proc{lvl}'] = ProcessorBlock(hidden_dim=hidden_dim, heads=heads, edge_dim_in=self.edge_dim, num_layers=processor_layers[i], use_transconv=use_transconv, ffn_ratio=ffn_ratio)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.pool_norms = nn.ModuleDict({f'norm_pool{lvl}_{next_lvl}': nn.LayerNorm(hidden_dim) for lvl, next_lvl in zip(levels[:-1], levels[1:])})
        self.upsamples = nn.ModuleDict({f'up{lvl}_{next_lvl}': GATv2Conv_up(in_channels=(hidden_dim // heads, hidden_dim), out_channels=hidden_dim, upd_hidden_dim=hidden_dim, edge_dim=self.edge_dim, heads=heads, residual=True, add_self_loops=False, ffn_ratio=ffn_ratio) for lvl, next_lvl in zip(levels[::-1][:-1], levels[::-1][1:])})
        self.up_norms = nn.ModuleDict({f'norm_up{lvl}_{next_lvl}': nn.LayerNorm(hidden_dim) for lvl, next_lvl in zip(levels[::-1][:-1], levels[::-1][1:])})
        self.decoders = nn.ModuleDict()
        for i, lvl in enumerate(levels):
            grid_in_channels = in_grid if lvl == 3 else out_grid
            self.decoders[f'dec{lvl}'] = GATv2ConvN(in_channels=(hidden_dim // heads, grid_in_channels), out_channels=out_grid, edge_dim=self.edge_dim, heads=heads, residual=True, add_self_loops=False, ffn_ratio=ffn_ratio)

    def run_ckpt(self, module, *args):
        if self.use_checkpoint:
            return checkpoint(module, *args, use_reentrant=False)
        else:
            return module(*args)

    def forward(self, data: HeteroData):
        grid_x = data['grid'].x
        mesh_x = {lvl: data[f'mesh{lvl}'].x for lvl in [7, 6, 5, 4, 3]}
        mesh_enc = {}
        mesh_enc[7] = self.run_ckpt(self.encoders['enc7'], (grid_x, mesh_x[7]), data['grid', 'encoder7', 'mesh7'].edge_index, data['grid', 'encoder7', 'mesh7'].edge_attr)
        mesh_enc[7] = self.encoder_norm(mesh_enc[7])
        for lvl, next_lvl in zip([7, 6, 5, 4], [6, 5, 4, 3]):
            pool = self.pools[f'pool{lvl}_{next_lvl}']
            mesh_enc[next_lvl] = self.run_ckpt(pool, (mesh_enc[lvl], mesh_x[next_lvl]), data[f'mesh{lvl}', f'pool{lvl}_to_{next_lvl}', f'mesh{next_lvl}'].edge_index, data[f'mesh{lvl}', f'pool{lvl}_to_{next_lvl}', f'mesh{next_lvl}'].edge_attr)
            norm = self.pool_norms[f'norm_pool{lvl}_{next_lvl}']
            mesh_enc[next_lvl] = norm(mesh_enc[next_lvl])
        mesh_proc = {}
        for lvl in [7, 6, 5, 4, 3]:
            proc = self.processors[f'proc{lvl}']
            mesh_proc[lvl] = proc(mesh_enc[lvl], data[f'mesh{lvl}', 'processor', f'mesh{lvl}'].edge_index, data[f'mesh{lvl}', 'processor', f'mesh{lvl}'].edge_attr, getattr(data[f'mesh{lvl}'], 'batch', None))
        for lvl, next_lvl in zip([3, 4, 5, 6], [4, 5, 6, 7]):
            up = self.upsamples[f'up{lvl}_{next_lvl}']
            mesh_proc[next_lvl] = self.run_ckpt(up, (mesh_proc[lvl], mesh_proc[next_lvl]), data[f'mesh{lvl}', f'upsample{lvl}_to_{next_lvl}', f'mesh{next_lvl}'].edge_index, data[f'mesh{lvl}', f'upsample{lvl}_to_{next_lvl}', f'mesh{next_lvl}'].edge_attr)
            norm = self.up_norms[f'norm_up{lvl}_{next_lvl}']
            mesh_proc[next_lvl] = norm(mesh_proc[next_lvl])
        grid_pred = None
        for lvl in [3, 4, 5, 6, 7]:
            dec = self.decoders[f'dec{lvl}']
            input_grid = grid_x if grid_pred is None else grid_pred
            grid_pred = self.run_ckpt(dec, (mesh_proc[lvl], input_grid), data[f'mesh{lvl}', f'decoder{lvl}', 'grid'].edge_index, data[f'mesh{lvl}', f'decoder{lvl}', 'grid'].edge_attr)
        return grid_pred
import torch.optim as optim
