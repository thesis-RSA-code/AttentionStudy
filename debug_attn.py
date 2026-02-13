import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


class DebugVanillaAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads=4, db_precision=False):
        super().__init__()
        self.db_precision = db_precision
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.hidden_channels = hidden_channels
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, debug=False):
        row, col = edge_index
        if self.db_precision:
            Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim).double()
            K = self.k_proj(x).view(-1, self.num_heads, self.head_dim).double()
            V = self.v_proj(x).view(-1, self.num_heads, self.head_dim).double()
        else:
            Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
            K = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
            V = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        attn_scores = (Q[row] * K[col]).sum(dim=-1) / self.head_dim**0.5
        attn_weights = softmax(attn_scores, index=row, num_nodes=x.size(0))

        weighted_v = attn_weights.unsqueeze(-1) * V[col]
        z = scatter_add(weighted_v, row, dim=0, dim_size=x.size(0))

        if debug:
            print("--- DebugVanillaAttention ---")
            print(f"Q mean: {Q.mean().item():.6f}, Q var: {Q.var().item():.6f}")
            print(f"K mean: {K.mean().item():.6f}, K var: {K.var().item():.6f}")
            print(f"V mean: {V.mean().item():.6f}, V var: {V.var().item():.6f}")
            print(f"attn_scores mean: {attn_scores.mean().item():.6f}, var: {attn_scores.var().item():.6f}")
            print(f"attn_weights mean: {attn_weights.mean().item():.6f}, var: {attn_weights.var().item():.6f}")
            print(f"weighted_v mean: {weighted_v.mean().item():.6f}, var: {weighted_v.var().item():.6f}")
            print(f"z mean: {z.mean().item():.6f}, var: {z.var().item():.6f}")

        z_concat = z.view(-1, self.num_heads * self.head_dim).float()
        out = self.o_proj(z_concat)

        if debug:
            print(f"out mean: {out.mean().item():.6f}, var: {out.var().item():.6f}")

        return out, {
            "Q": Q, "K": K, "V": V,
            "attn_scores": attn_scores,
            "attn_weights": attn_weights,
            "weighted_v": weighted_v,
            "z": z,
            "out": out
        }

class DebugFusedAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads=4, db_precision=False):
        super().__init__()
        self.db_precision = db_precision
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.hidden_channels = hidden_channels
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"

        self.qkv_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, debug=False):
        row, col = edge_index

        if self.db_precision:
            qkv = self.qkv_proj(x).double()
        else:
            qkv = self.qkv_proj(x)

        q, k, v = torch.split(qkv, [self.hidden_channels]*3, dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        attn_scores = (q[row] * k[col]).sum(dim=-1) / self.head_dim**0.5
        attn_weights = softmax(attn_scores, index=row, num_nodes=x.size(0))
        
        # attn_weights = torch.zeros_like(attn_scores)
        # for h in range(self.num_heads):
        #     attn_weights[:, h] = softmax(attn_scores[:, h], index=row, num_nodes=x.size(0))

        weighted_v = attn_weights.unsqueeze(-1) * v[col]
        z = scatter_add(weighted_v, row, dim=0, dim_size=x.size(0))

        if debug:
            print("\n--- DebugFusedAttention ---")
            print(f"Q mean: {q.mean().item():.6f}, Q var: {q.var().item():.6f}")
            print(f"K mean: {k.mean().item():.6f}, K var: {k.var().item():.6f}")
            print(f"V mean: {v.mean().item():.6f}, V var: {v.var().item():.6f}")
            print(f"attn_scores mean: {attn_scores.mean().item():.6f}, var: {attn_scores.var().item():.6f}")
            print(f"attn_weights mean: {attn_weights.mean().item():.6f}, var: {attn_weights.var().item():.6f}")
            print(f"weighted_v mean: {weighted_v.mean().item():.6f}, var: {weighted_v.var().item():.6f}")
            print(f"z mean: {z.mean().item():.6f}, var: {z.var().item():.6f}")

        z_concat = z.view(-1, self.num_heads * self.head_dim).float()
        out = self.o_proj(z_concat)

        if debug:
            print(f"out mean: {out.mean().item():.6f}, var: {out.var().item():.6f}")

        return out, {
            "Q": q, "K": k, "V": v,
            "attn_scores": attn_scores,
            "attn_weights": attn_weights,
            "weighted_v": weighted_v,
            "z": z,
            "out": out
        }
