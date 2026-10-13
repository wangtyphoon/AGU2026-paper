# Derived from gnn/ver_unet7/layer/gat_edge_ffn_dec.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import typing
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, NoneType, OptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, is_torch_sparse_tensor, remove_self_loops, softmax
from torch_geometric.utils.sparse import set_sparse_value
if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload

class GATv2Conv_dec(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int=1, concat: bool=True, negative_slope: float=0.2, dropout: float=0.0, add_self_loops: bool=True, edge_dim: Optional[int]=None, upd_hidden_dim: int=64, fill_value: Union[float, Tensor, str]='mean', bias: bool=True, share_weights: bool=False, residual: bool=True, **kwargs):
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights
        self.ffn_ratio = 2
        self.att = Parameter(torch.empty(1, heads, out_channels))
        if isinstance(self.in_channels, int):
            node_dim_l = node_dim_r = self.in_channels
        else:
            node_dim_l, node_dim_r = self.in_channels
        if edge_dim is not None and upd_hidden_dim is not None:
            self.edge_upd = nn.Sequential(nn.Linear(node_dim_l + node_dim_r + edge_dim, upd_hidden_dim * self.ffn_ratio), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(upd_hidden_dim * self.ffn_ratio, heads * out_channels))
            self.edge_norm = nn.LayerNorm(heads * out_channels)
        else:
            self.edge_upd = None
        if edge_dim != heads * out_channels:
            self.edge_res_lin = nn.Linear(edge_dim, heads * out_channels, bias=False)
        total_out = out_channels * (heads if concat else 1)
        self.node_upd = nn.Sequential(nn.Linear(total_out + node_dim_r, upd_hidden_dim * self.ffn_ratio), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(upd_hidden_dim * self.ffn_ratio, total_out))
        if node_dim_l != total_out:
            self.node_res_lin = nn.Linear(node_dim_r, total_out, bias=False)
        if bias:
            self.bias = Parameter(torch.empty(total_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.edge_upd is not None:
            for layer in self.edge_upd:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.node_upd:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    @overload
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor=None, return_attention_weights: NoneType=None) -> Tensor:
        ...

    @overload
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor=None, return_attention_weights: bool=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        ...

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor=None, return_attention_weights: Optional[bool]=None):
        H, C = (self.heads, self.out_channels)
        if isinstance(x, Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x
        if self.edge_upd is not None and edge_attr is not None:
            src, dst = edge_index
            orig_edge = edge_attr
            h_src = x_src[src]
            h_dst = x_dst[dst]
            e_in = torch.cat([h_src, h_dst, edge_attr], dim=-1)
            new_edge = self.edge_upd(e_in)
            if hasattr(self, 'edge_res_lin'):
                res_edge = self.edge_res_lin(orig_edge)
            else:
                res_edge = orig_edge
            edge_attr = self.edge_norm(new_edge + res_edge)
        if self.add_self_loops:
            num_nodes = x_src.size(0)
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes)
        alpha = self.edge_update(edge_attr=edge_attr, index=edge_index[0], ptr=None, dim_size=None)
        out = self.propagate(edge_index, alpha=alpha, edge_attr=edge_attr)
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        res = x_dst
        out = torch.cat([out, x_dst], dim=-1)
        out = self.node_upd(out)
        if hasattr(self, 'node_res_lin'):
            res = self.node_res_lin(res)
            out = out + res + (self.bias if self.bias is not None else 0)
        else:
            out = out + res + (self.bias if self.bias is not None else 0)
        if isinstance(return_attention_weights, bool):
            return (out, edge_attr, (edge_index, alpha))
        else:
            return out

    def edge_update(self, edge_attr: OptTensor, index: Tensor, ptr: OptTensor, dim_size: Optional[int]) -> Tensor:
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, alpha: Tensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is not None:
            msg = edge_attr.view(-1, self.heads, self.out_channels) * alpha.unsqueeze(-1)
            return msg

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})'
