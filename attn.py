import torch
import torch.nn as nn

print("Importing torch_scatter...")
from torch_scatter import scatter_add
print("Importing torch.nn.functional...")
import torch.nn.functional as F

print("Importing torch_geometric.utils.softmax...")
from torch_geometric.utils import softmax

from activation_functions import get_activation


class AttentionLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads=4, activation='relu', db_precision=False, debug=False):
        super().__init__()
        self.db_precision = db_precision
        self.hidden_channels = hidden_channels
        
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        self.debug = debug
        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels)


    def forward(self, x, edge_index):
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
        
        if self.debug:
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
        return out

class AttentionLayerFused(nn.Module):
    """Version avec kernel fusionné pour maximum performance"""
    
    def __init__(self, hidden_channels, num_heads=4, activation='relu', db_precision=False, debug=False):
        super().__init__()
        self.db_precision = db_precision
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        self.debug = debug
        
        self.qkv_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        row, col = edge_index
        
        # Projection fusionnée
        if self.db_precision:
            qkv = self.qkv_proj(x).double()  # (N, 3*hidden_channels)
        else:
            qkv = self.qkv_proj(x)  # (N, 3*hidden_channels)

        q, k, v = torch.split(qkv, [self.hidden_channels]*3, dim=-1) # split cost less overhead than chunk

        # Reshape pour multi-head
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        
        attn_scores = (q[row] * k[col]).sum(dim=-1) / (self.head_dim ** 0.5) # (num_edges, num_heads)
        attn_weights = softmax(attn_scores, index=row, num_nodes=x.size(0)) # vectorized version
        
        # attn_weights = torch.zeros_like(scores) # loop version
        # for h in range(self.num_heads):
        #     attn_weights[:, h] = softmax(scores[:, h], index=row, num_nodes=x.size(0))

        # Agrégation
        # (num_edges, num_heads, 1) * (num_edges, num_heads, head_dim)
        weighted_v = attn_weights.unsqueeze(-1) * v[col]
        
        # Scatter par head
        z = scatter_add(weighted_v, row, dim=0, dim_size=x.size(0))

        if self.debug:
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
        
        return out


class Block(nn.Module):
    """Standard Transformer block."""

    def __init__(self, hidden_channels, num_heads=4, activation='relu', mlp_expansion_factor=2):
        super().__init__()
        self.attn = AttentionLayer(
            hidden_channels, 
            num_heads, 
            activation=activation, 
            mlp_expansion_factor=mlp_expansion_factor
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * mlp_expansion_factor),
            get_activation(activation),
            nn.Linear(hidden_channels * mlp_expansion_factor, hidden_channels),
        )
    
    def forward(self, x, edge_index):
        x = self.norm1(x + self.attn(x, edge_index))
        x = self.norm2(x + self.mlp(x))
        return x


class MultiQueryAttention(nn.Module):
    """MQA: 1 seule paire K,V partagée par toutes les têtes Q"""
    
    def __init__(self, hidden_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0
        
        # Q a num_heads têtes
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # K, V ont UNE SEULE tête (partagée)
        self.k_proj = nn.Linear(hidden_channels, self.head_dim)
        self.v_proj = nn.Linear(hidden_channels, self.head_dim)
        
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        row, col = edge_index
        
        # Q: (N, num_heads, head_dim)
        Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # K, V: (N, head_dim) - UNE SEULE tête
        K = self.k_proj(x)  # (N, head_dim)
        V = self.v_proj(x)  # (N, head_dim)
        
        # Broadcast K, V pour toutes les têtes de Q
        # (num_edges, num_heads, head_dim)
        scores = (Q[row] * K[col].unsqueeze(1)).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # Softmax par tête
        attn_weights = torch.zeros_like(scores)
        for h in range(self.num_heads):
            attn_weights[:, h] = softmax(scores[:, h], index=row, num_nodes=x.size(0))
        
        # Agrégation: V est broadcasté pour toutes les têtes
        # (num_edges, num_heads, head_dim)
        weighted_v = attn_weights.unsqueeze(-1) * V[col].unsqueeze(1)
        
        # Scatter
        z = scatter_add(weighted_v, row, dim=0, dim_size=x.size(0))
        z_concat = z.view(-1, self.num_heads * self.head_dim)
        
        return self.o_proj(z_concat)


class GroupedQueryAttention(nn.Module):
    """GQA: num_kv_groups paires K,V partagées par groupes de têtes Q"""
    
    def __init__(self, hidden_channels, num_heads=8, num_kv_groups=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = hidden_channels // num_heads
        
        assert hidden_channels % num_heads == 0
        assert num_heads % num_kv_groups == 0, "num_heads doit être divisible par num_kv_groups"
        
        self.heads_per_group = num_heads // num_kv_groups
        
        # Q a toutes les têtes
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # K, V ont num_kv_groups têtes
        self.k_proj = nn.Linear(hidden_channels, num_kv_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_channels, num_kv_groups * self.head_dim)
        
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        row, col = edge_index
        
        # Q: (N, num_heads, head_dim)
        Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # K, V: (N, num_kv_groups, head_dim)
        K = self.k_proj(x).view(-1, self.num_kv_groups, self.head_dim)
        V = self.v_proj(x).view(-1, self.num_kv_groups, self.head_dim)
        
        # Répéter K, V pour matcher le nombre de têtes Q
        # (N, num_kv_groups, head_dim) -> (N, num_heads, head_dim)
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)
        
        # Calcul standard (identique à MHA maintenant)
        scores = (Q[row] * K[col]).sum(dim=-1) / (self.head_dim ** 0.5)
        
        attn_weights = torch.zeros_like(scores)
        for h in range(self.num_heads):
            attn_weights[:, h] = softmax(scores[:, h], index=row, num_nodes=x.size(0))
        
        weighted_v = attn_weights.unsqueeze(-1) * V[col]
        z = scatter_add(weighted_v, row, dim=0, dim_size=x.size(0))
        z_concat = z.view(-1, self.num_heads * self.head_dim)
        
        return self.o_proj(z_concat)