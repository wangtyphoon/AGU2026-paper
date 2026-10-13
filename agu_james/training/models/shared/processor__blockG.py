# Derived from gnn/ver_unet_scale/layer/processor__blockG.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import torch
import torch.nn as nn
from agu_james.training.models.shared.gat_edge_before_ffnG_fix import GATv2Conv
from agu_james.training.models.shared.SG_module import SGModule
from torch.utils.checkpoint import checkpoint

class ProcessorBlock(nn.Module):

    def __init__(self, hidden_dim: int, heads: int, edge_dim_in: int, num_layers: int, use_transconv: bool=True, use_checkpoint: bool=True, ffn_ratio: float=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.use_transconv = use_transconv
        self.use_checkpoint = use_checkpoint
        self.edge_proj = nn.Linear(edge_dim_in, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else hidden_dim // heads
            out_ch = hidden_dim // heads
            self.gat_layers.append(GATv2Conv(in_channels=in_ch, out_channels=out_ch, heads=heads, edge_dim=hidden_dim, upd_hidden_dim=hidden_dim, residual=True, add_self_loops=False, ffn_ratio=ffn_ratio))
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(out_ch))
        if use_transconv:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.trans = SGModule(in_channels=hidden_dim, hidden_channels=hidden_dim // heads, num_layers=1, num_heads=heads, dropout=0.1)
            self.concat_norm = nn.LayerNorm(hidden_dim // heads)

    def forward(self, mesh_enc, edge_index, edge_attr, batch=None):
        x = mesh_enc
        e = self.edge_proj(edge_attr)
        for i, gat in enumerate(self.gat_layers):
            if self.use_checkpoint:
                x, e = checkpoint(gat, x, edge_index, e, use_reentrant=False)
            else:
                x, e = gat(x, edge_index, e)
            if i < self.num_layers - 1:
                x = self.norms[i](x)
        if not self.use_transconv:
            return x
        with torch.amp.autocast('cuda', dtype=torch.float32):
            trans_out = self.trans(mesh_enc, batch)
        alpha = torch.sigmoid(self.alpha)
        out = alpha * x + (1.0 - alpha) * trans_out
        return self.concat_norm(out)
