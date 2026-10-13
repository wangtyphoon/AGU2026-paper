# Derived from gnn/ver_unet_scale/SG_module.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

class SGFormerAttention(torch.nn.Module):
    """The simple global attention mechanism from the
    `"SGFormer: Simplifying and Empowering Transformers for
    Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper.

    Args:
        channels (int): Size of each input sample.
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
    """

    def __init__(self, channels: int, head_channels: int=64, qkv_bias: bool=False) -> None:
        super().__init__()
        if head_channels is None:
            head_channels = channels
        self.head_channels = head_channels
        inner_channels = head_channels
        self.q = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)

    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\\mathbf{M} \\in {\\{ 0, 1 \\}}^{B \\times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        B, N, *_ = x.shape
        qs, ks, vs = (self.q(x), self.k(x), self.v(x))
        qs, ks, vs = map(lambda t: t.reshape(B, N, 1, self.head_channels), (qs, ks, vs))
        if mask is not None:
            mask = mask[:, :, None, None]
            vs.masked_fill_(~mask, 0.0)
        epsilon = 1e-06
        qs[qs == 0] = epsilon
        ks[ks == 0] = epsilon
        qs, ks = map(lambda t: t / torch.linalg.norm(t, ord=2, dim=-1, keepdim=True), (qs, ks))
        kvs = torch.einsum('blhm,blhd->bhmd', ks, vs)
        attention_num = torch.einsum('bnhm,bhmd->bnhd', qs, kvs)
        attention_num += N * vs
        all_ones = torch.ones([B, N]).to(ks.device)
        ks_sum = torch.einsum('blhm,bl->bhm', ks, all_ones)
        attention_normalizer = torch.einsum('bnhm,bhm->bnh', qs, ks_sum)
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer
        return attn_output.mean(dim=2)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(head_channels={self.head_channels})'

class SGModule(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.attns = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers):
            self.attns.append(SGFormerAttention(hidden_channels, head_channels=hidden_channels))
            self.bns.append(torch.nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for attn in self.attns:
            attn.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x: Tensor, batch: Tensor):
        batch, indices = batch.sort(stable=True)
        rev_perm = torch.empty_like(indices)
        rev_perm[indices] = torch.arange(len(indices), device=indices.device)
        x = x[indices]
        x, mask = to_dense_batch(x, batch)
        layer_ = []
        x = self.fcs[0](x)
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)
        for i, attn in enumerate(self.attns):
            x = attn(x, mask)
            x = (x + layer_[i]) / 2.0
            x = self.bns[i + 1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
        x_mask = x[mask]
        unsorted_x_mask = x_mask[rev_perm]
        return unsorted_x_mask
