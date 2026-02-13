import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

from attn import AttentionLayer, AttentionLayerFused
from debug_attn import DebugVanillaAttention, DebugFusedAttention

def compare_attention_models():
    torch.manual_seed(42)
    hidden_channels = 32
    num_heads = 8
    num_nodes = 50
    db_precision = False  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(num_nodes, hidden_channels, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=device)

    # vanilla_model = DebugVanillaAttention(hidden_channels, num_heads, db_precision=db_precision).to(device)
    # fused_model = DebugFusedAttention(hidden_channels, num_heads, db_precision=db_precision).to(device)
    vanilla_model = AttentionLayer(hidden_channels, num_heads, db_precision=db_precision).to(device)
    fused_model = AttentionLayerFused(hidden_channels, num_heads, db_precision=db_precision).to(device)

    # Copier les poids de q, k, v
    with torch.no_grad():
        fused_model.qkv_proj.weight.data.copy_(torch.cat([
            vanilla_model.q_proj.weight.data,
            vanilla_model.k_proj.weight.data,
            vanilla_model.v_proj.weight.data,
        ], dim=0))
        fused_model.qkv_proj.bias.data.copy_(torch.cat([
            vanilla_model.q_proj.bias.data,
            vanilla_model.k_proj.bias.data,
            vanilla_model.v_proj.bias.data,
        ], dim=0))

    # Copier les poids de o_proj
    with torch.no_grad():
        fused_model.o_proj.weight.data.copy_(vanilla_model.o_proj.weight.data)
        fused_model.o_proj.bias.data.copy_(vanilla_model.o_proj.bias.data)

    # Forward pass avec debug
    out_vanilla, debug_vanilla = vanilla_model(x, edge_index, debug=True)
    out_fused, debug_fused = fused_model(x, edge_index, debug=True)

    # Comparaison des étapes
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    for key in debug_vanilla:
        if key != "out":
            diff = (debug_vanilla[key] - debug_fused[key]).abs()
            print(f"\n{key}:")
            print(f"  Max diff: {diff.max().item():.6f}")
            print(f"  Mean diff: {diff.mean().item():.6f}")
            print(f"  All close: {torch.allclose(debug_vanilla[key], debug_fused[key], atol=1e-5)}")

    # Comparaison des outputs finaux
    diff_out = (out_vanilla - out_fused).abs()
    print(f"\nOutput:")
    print(f"  Max diff: {diff_out.max().item():.6f}")
    print(f"  Mean diff: {diff_out.mean().item():.6f}")
    print(f"  All close: {torch.allclose(out_vanilla, out_fused, atol=1e-5)}")

if __name__ == "__main__":
    compare_attention_models()
