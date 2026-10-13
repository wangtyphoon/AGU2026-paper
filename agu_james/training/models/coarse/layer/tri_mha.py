# Derived from gnn/ver_unetG/layer/tri_mha.py
# See provenance/sources.json for original hash and modifications.
from agu_james import runtime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import SparseTensor
from pathlib import Path
from torch_geometric.utils import to_dense_batch

class TriBlockDiagMHA(nn.Module):

    def __init__(self, d_model: int, num_heads: int, key_size: int=None, value_size: int=None, attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_size = key_size or d_model // num_heads
        self.value_size = value_size or d_model // num_heads
        self.scale = self.key_size ** (-0.5)
        self.wq = nn.Linear(d_model, num_heads * self.key_size, bias=False)
        self.wk = nn.Linear(d_model, num_heads * self.key_size, bias=False)
        self.wv = nn.Linear(d_model, num_heads * self.value_size, bias=False)
        self.wo = nn.Linear(num_heads * self.value_size, d_model, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _expand_mask(self, m: torch.Tensor, B: int, H: int):
        return m[None, :, None, :, :].expand(B, -1, H, -1, -1)

    @staticmethod
    def _tri_softmax(logits_d, logits_u, logits_l):
        m = torch.maximum(torch.maximum(logits_d.amax(dim=-1, keepdim=True), logits_u.amax(dim=-1, keepdim=True)), logits_l.amax(dim=-1, keepdim=True))
        ud = torch.exp(logits_d - m)
        uu = torch.exp(logits_u - m)
        ul = torch.exp(logits_l - m)
        denom = (ud.sum(dim=-1, keepdim=True) + uu.sum(dim=-1, keepdim=True) + ul.sum(dim=-1, keepdim=True)).clamp_min(1e-30)
        return (ud / denom, uu / denom, ul / denom)

    def forward(self, x: torch.Tensor, mask_info: dict):
        """
        x: (B, N, d_model)
        mask_info: {
          'block_size': int, 'num_blocks': int, 'pad_nodes': int,
          'diag_mask': (nb, bs, bs) bool, 'up_mask': (nb, bs, bs) bool, 'low_mask': (nb, bs, bs) bool
        }
        """
        B, N, D = x.shape
        device = x.device
        bs = int(mask_info['block_size'])
        nb = int(mask_info['num_blocks'])
        pad_nodes = int(mask_info['pad_nodes'])
        if pad_nodes > 0:
            x = F.pad(x, (0, 0, 0, pad_nodes))
        x_blocks = x.view(B, nb, bs, D)
        q = self.wq(x_blocks).view(B, nb, bs, self.num_heads, self.key_size)
        k = self.wk(x_blocks).view(B, nb, bs, self.num_heads, self.key_size)
        v = self.wv(x_blocks).view(B, nb, bs, self.num_heads, self.value_size)
        zero_k = torch.zeros_like(k[:, :1])
        zero_v = torch.zeros_like(v[:, :1])
        k_pad = torch.cat([zero_k, k, zero_k], dim=1)
        v_pad = torch.cat([zero_v, v, zero_v], dim=1)

        def qk(q, kblk):
            return torch.einsum('bnqhd,bnkhd->bnhqk', q, kblk) * self.scale
        logits_d = qk(q, k_pad[:, 1:-1, ...])
        logits_u = qk(q, k_pad[:, 2:, ...])
        logits_l = qk(q, k_pad[:, :-2, ...])
        diag_mask = self._expand_mask(mask_info['diag_mask'].to(device), B, self.num_heads)
        up_mask = self._expand_mask(mask_info['up_mask'].to(device), B, self.num_heads)
        low_mask = self._expand_mask(mask_info['low_mask'].to(device), B, self.num_heads)
        neg_inf = torch.finfo(x.dtype).min
        logits_d = torch.where(diag_mask, logits_d, torch.tensor(neg_inf, device=device, dtype=x.dtype))
        logits_u = torch.where(up_mask, logits_u, torch.tensor(neg_inf, device=device, dtype=x.dtype))
        logits_l = torch.where(low_mask, logits_l, torch.tensor(neg_inf, device=device, dtype=x.dtype))
        w_d, w_u, w_l = self._tri_softmax(logits_d, logits_u, logits_l)
        w_d = self.attn_drop(w_d)
        w_u = self.attn_drop(w_u)
        w_l = self.attn_drop(w_l)

        def av(w, vblk):
            return torch.einsum('bnhqk,bnkhd->bnqhd', w, vblk)
        out = av(w_d, v_pad[:, 1:-1, ...]) + av(w_u, v_pad[:, 2:, ...]) + av(w_l, v_pad[:, :-2, ...])
        out = out.reshape(B, nb, bs, self.num_heads * self.value_size).reshape(B, nb * bs, -1)[:, :N, :]
        out = self.wo(out)
        return self.proj_drop(out)

class SwiGLU(nn.Module):

    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w = nn.Linear(d_in, d_hidden)

    def forward(self, x):
        v, g = self.w(x).chunk(2, dim=-1)
        return v * F.silu(g)

class GraphCastStyleBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, ffw_hidden: int | None=None, attn_drop: float=0.0, proj_drop: float=0.0, d_global_cond: int | None=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mha = TriBlockDiagMHA(d_model, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        f = ffw_hidden or 2 * d_model
        self.ffn = nn.Sequential(SwiGLU(d_model, f), nn.Linear(f // 2, d_model))
        self.dropout = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask_info: dict, global_cond: torch.Tensor | None=None):
        xn = self.ln1(x)
        x = x + self.dropout(self.mha(xn, mask_info))
        yn = self.ln2(x)
        x = x + self.dropout(self.ffn(yn))
        return x

class GraphCastStyleTransformer(nn.Module):
    """
    在 __init__ 內自行讀取 graph 檔案以建立：
      - 依緯度排序的 perm_sort / perm_unsort
      - 以排序後 edge_index 推導的 tri-block 遮罩
    forward() 仍接受「原始節點順序」的 x，內部完成排序/還原。
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, graph_path: str | Path, node_type: str='mesh6', rel: tuple[str, str, str]=('mesh6', 'processor', 'mesh6'), k_hop: int=16, include_leq_k: bool=True, ffw_hidden: int | None=None, attn_drop: float=0.0, proj_drop: float=0.0, d_global_cond: int | None=None, precomputed_mask: dict | None=None, build_device: str='cpu', sort_key_col: int=1, do_sort: bool=True):
        super().__init__()
        graph_path = runtime.setting('agu_james.training.models.coarse.layer.tri_mha.graph_path', lambda: runtime.path('graph', runtime.artifact_root() / 'graphs/mesh6/mesh_grid_data_unet/hetero_graph_normalized.pt'))
        g = torch.load(graph_path, weights_only=False, map_location='cpu')
        node_store = g[node_type]
        rel_store = g[rel]
        node_x = node_store.x
        edge_index = rel_store.edge_index
        edge_attr = getattr(rel_store, 'edge_attr', None)
        num_nodes = node_x.size(0)
        self.layers = nn.ModuleList([GraphCastStyleBlock(d_model, num_heads, ffw_hidden, attn_drop, proj_drop, d_global_cond) for _ in range(num_layers)])
        if do_sort and node_x is not None:
            p, pinv = self._sort_nodes_by_lat_from_nodefeat(node_x, lat_col=sort_key_col)
            self.register_buffer('perm_sort', p, persistent=False)
            self.register_buffer('perm_unsort', pinv, persistent=False)
            edge_index_sorted = self.perm_unsort[edge_index]
            self._inputs_expected_order = 'original'
        else:
            self.register_buffer('perm_sort', torch.arange(num_nodes), persistent=False)
            self.register_buffer('perm_unsort', torch.arange(num_nodes), persistent=False)
            edge_index_sorted = edge_index
            self._inputs_expected_order = 'already_sorted'
        self.edge_attr = edge_attr
        if precomputed_mask is not None:
            mask_info = precomputed_mask
        else:
            mask_info = self._build_khop_masks_from_pyg(edge_index=edge_index_sorted.to(build_device), num_nodes=num_nodes, k_hop=k_hop, include_leq_k=include_leq_k, device=build_device)
        self.block_size = int(mask_info['block_size'])
        self.num_blocks = int(mask_info['num_blocks'])
        self.pad_nodes = int(mask_info['pad_nodes'])
        self.register_buffer('diag_mask', mask_info['diag_mask'].to(torch.bool))
        self.register_buffer('up_mask', mask_info['up_mask'].to(torch.bool))
        self.register_buffer('low_mask', mask_info['low_mask'].to(torch.bool))

    def _mask_dict(self):
        return dict(block_size=self.block_size, num_blocks=self.num_blocks, pad_nodes=self.pad_nodes, diag_mask=self.diag_mask, up_mask=self.up_mask, low_mask=self.low_mask)

    @staticmethod
    def _sort_nodes_by_lat_from_nodefeat(node_x: torch.Tensor, lat_col: int=0):
        lat_feat = node_x[:, lat_col]
        p = torch.argsort(lat_feat, stable=True)
        pinv = torch.empty_like(p)
        pinv[p] = torch.arange(len(p), device=p.device)
        return (p, pinv)

    @staticmethod
    @torch.no_grad()
    def _build_khop_masks_from_pyg(edge_index: torch.Tensor, num_nodes: int, k_hop: int=16, include_leq_k: bool=True, device: str='cpu'):
        edge_index = edge_index.to(device)
        row, col = edge_index
        rev = torch.stack([col, row], dim=0)
        loop = torch.arange(num_nodes, device=device)
        loop = torch.stack([loop, loop], dim=0)
        edge_index = torch.cat([edge_index, rev, loop], dim=1)
        row, col = edge_index
        A = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).coalesce()
        if include_leq_k:
            V = SparseTensor.eye(num_nodes, num_nodes).coalesce()
            P = A
            for _ in range(k_hop):
                V = (V + P).coalesce().set_value_(None)
                P = P.matmul(A).set_value_(None)
            Ak = V
        else:
            Ak = A
            for _ in range(k_hop - 1):
                Ak = Ak.matmul(A).set_value_(None)
        rows, cols, _ = Ak.coo()
        rows = rows.to(device)
        cols = cols.to(device)
        self_idx = torch.arange(num_nodes, device=device)
        rows_all = torch.cat([rows, self_idx])
        cols_all = torch.cat([cols, self_idx])
        arange_n = torch.arange(num_nodes, device=device)
        row_min = scatter(rows_all, cols_all, dim=0, dim_size=num_nodes, reduce='min')
        row_max = scatter(rows_all, cols_all, dim=0, dim_size=num_nodes, reduce='max')
        lbandwidth = (arange_n - row_min + 1).max().item()
        ubandwidth = (row_max - arange_n + 1).max().item()
        block_size = max(int(lbandwidth), int(ubandwidth))
        num_blocks = (num_nodes + block_size - 1) // block_size
        n_pad = num_blocks * block_size - num_nodes

        def _make_block_mask(selector):
            bs = block_size
            out = []
            for b in range(num_blocks):
                r0, r1 = (b * bs, (b + 1) * bs)
                rr0 = r0
                rr1 = min(r1, num_nodes)
                if rr0 >= rr1:
                    out.append(torch.zeros(bs, bs, dtype=torch.bool, device=device))
                    continue
                c0, c1 = selector(b, bs, num_blocks)
                cc0 = max(c0, 0)
                cc1 = min(c1, num_nodes)
                if cc0 >= cc1:
                    out.append(torch.zeros(bs, bs, dtype=torch.bool, device=device))
                    continue
                sel = (rows >= rr0) & (rows < rr1) & (cols >= cc0) & (cols < cc1)
                r = (rows[sel] - r0).long()
                c = (cols[sel] - c0).long()
                blk = torch.zeros(bs, bs, dtype=torch.bool, device=device)
                if r.numel() > 0:
                    mask_r = (r >= 0) & (r < bs)
                    mask_c = (c >= 0) & (c < bs)
                    keep = mask_r & mask_c
                    if keep.any():
                        blk[r[keep], c[keep]] = True
                out.append(blk)
            return torch.stack(out, dim=0)
        diag_mask = _make_block_mask(lambda b, bs, nb: (b * bs, (b + 1) * bs))
        up_mask = _make_block_mask(lambda b, bs, nb: ((b + 1) * bs, (b + 2) * bs))
        low_mask = _make_block_mask(lambda b, bs, nb: ((b - 1) * bs, (b + 0) * bs))
        return dict(block_size=block_size, num_blocks=num_blocks, pad_nodes=n_pad, diag_mask=diag_mask.to(torch.bool), up_mask=up_mask.to(torch.bool), low_mask=low_mask.to(torch.bool))

    def forward(self, x: torch.Tensor, batch: torch.Tensor, global_cond: torch.Tensor | None=None):
        x_dense, dense_mask = to_dense_batch(x, batch)
        device = x_dense.device
        x_sorted = x_dense.index_select(1, self.perm_sort.to(device))
        mask_info = self._mask_dict()
        for blk in self.layers:
            x_sorted = blk(x_sorted, mask_info, global_cond)
        x_unsorted = x_sorted.index_select(1, self.perm_unsort.to(device))
        x_orig_format = x_unsorted[dense_mask]
        return x_orig_format
