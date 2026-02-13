

print(60 * "=" + " Importing modules. " + 60 * "=")

print(f"Importing torch...")
import torch
print(f"Importing torch nn...")
import torch.nn as nn
import time

print(f"Importing torch_geometric.data...")
from torch_geometric.data import Batch, Data

print(f"Importing torch_geometric.utils...")
from torch_geometric.utils import to_dense_batch

print(f"Importing attn.AttentionLayer...")
from attn import AttentionLayer

print(f"Importing torch_nested_mha_tutorial...")
from torch_nested_mha_tutorial import MultiHeadAttention
print(60 * "=" + " Done importing modules. " + 60 * "=")


def benchmark(model, *args, num_iters=100, **kwargs):
    """Benchmark function to measure time and memory"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(10):
        output = model(*args, **kwargs)
        if isinstance(output, tuple):
            output = output[0]
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iters):
        output = model(*args, **kwargs)
        if isinstance(output, tuple):
            output = output[0]
    
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iters
    peak_memory = torch.cuda.max_memory_allocated()
    
    return output, elapsed_time, peak_memory

def create_graph_batch(num_events, avg_pmts_per_event, hidden_channels, num_edges_per_node, device):
    """
    Crée un batch de graphes avec arêtes aléatoires
    
    Args:
        num_events: nombre d'événements dans le batch
        avg_pmts_per_event: nombre moyen de PMTs par événement
        hidden_channels: dimension des embeddings
        num_edges_per_node: nombre moyen d'arêtes par nœud
    """
    data_list = []
    
    for _ in range(num_events):
        # Nombre variable de PMTs par événement
        num_pmts = torch.randint(
            int(avg_pmts_per_event * 0.8), 
            int(avg_pmts_per_event * 1.2), 
            (1,)
        ).item()
        
        # Features aléatoires
        x = torch.randn(num_pmts, hidden_channels, device=device)
        
        # Génération d'arêtes aléatoires
        num_edges = int(num_pmts * num_edges_per_node)
        
        # Génération aléatoire source -> target
        src = torch.randint(0, num_pmts, (num_edges,), device=device)
        dst = torch.randint(0, num_pmts, (num_edges,), device=device)
        
        # Éviter les self-loops
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        
        edge_index = torch.stack([src, dst], dim=0)
        
        data_list.append(Data(x=x, edge_index=edge_index))
    
    batch = Batch.from_data_list(data_list)
    return batch

def graph_to_dense(batch, max_nodes=None):
    """
    Convertit un batch de graphes en représentation dense pour MHA classique
    
    Returns:
        x_dense: (batch_size, max_nodes, hidden_channels)
        mask: (batch_size, max_nodes) - True pour les vrais nodes
        edge_index: pour votre modèle
    """
    x_dense, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_nodes)
    return x_dense, mask, batch.edge_index

# ============= Configuration =============
num_events = 32  # batch size
avg_pmts_per_event = 10000  # moyenne de PMTs par événement
hidden_channels = 50
num_heads = 8
num_edges_per_node = 50  # nombre moyen d'arêtes par PMT
device = "cuda"

torch.manual_seed(42)

print(f"Configuration:")
print(f"  Batch size: {num_events}")
print(f"  Avg PMTs per event: {avg_pmts_per_event}")
print(f"  Hidden channels: {hidden_channels}")
print(f"  Num heads: {num_heads}")
print(f"  Edges per node: {num_edges_per_node}")
print()

# ============= Création des données =============
batch = create_graph_batch(num_events, avg_pmts_per_event, hidden_channels, 
                          num_edges_per_node, device)

x_dense, mask, edge_index = graph_to_dense(batch)
batch_size, max_seq_len, _ = x_dense.shape

print(f"Batch stats:")
print(f"  Total PMTs: {batch.x.size(0)}")
print(f"  Max sequence length (padded): {max_seq_len}")
print(f"  Total edges: {edge_index.size(1)}")
print(f"  Avg edges per node: {edge_index.size(1) / batch.x.size(0):.1f}")
print(f"  Sparsity vs dense: {edge_index.size(1) / (batch.x.size(0) * max_seq_len):.4f}")
print()

# ============= Modèles =============
torch.manual_seed(42)
your_model = AttentionLayer(hidden_channels, num_heads=num_heads).to(device)

torch.manual_seed(42)
dense_mha = MultiHeadAttention(
    E_q=hidden_channels,
    E_k=hidden_channels, 
    E_v=hidden_channels,
    E_total=hidden_channels,
    nheads=num_heads,
    dropout=0.0,
    bias=True,
    device=device
)

# Wrapper pour MHA dense qui gère le masque
class DenseMHAWrapper(nn.Module):
    def __init__(self, mha, mask):
        super().__init__()
        self.mha = mha
        # Création du masque d'attention (batch_size, max_seq, max_seq)
        self.register_buffer('attn_mask', self._create_attention_mask(mask))
    
    def _create_attention_mask(self, mask):
        """Crée un masque d'attention pour bloquer les positions paddées"""
        batch_size, seq_len = mask.shape
        # mask: (B, L) -> (B, L, L) où attn_mask[b, i, j] = mask[b, i] & mask[b, j]
        attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)
        return attn_mask
    
    def forward(self, x):
        # x: (B, L, C)
        return self.mha(x, x, x, attn_mask=self.attn_mask, is_causal=False)

dense_wrapper = DenseMHAWrapper(dense_mha, mask).to(device)

# Compile les modèles
print("Compiling models...")
your_model_compiled = torch.compile(your_model)
dense_wrapper_compiled = torch.compile(dense_wrapper)

print("=" * 60)
print("BENCHMARKING")
print("=" * 60)

# ============= Benchmark votre modèle (sparse graph attention) =============
print("\n[1] Your Graph Attention (sparse)...")
your_result, your_time, your_memory = benchmark(
    your_model_compiled, 
    batch.x, 
    edge_index,
    num_iters=50
)
print(f"  Time: {your_time:.5f}s")
print(f"  Peak memory: {your_memory/1e9:.2f} GB")

# ============= Benchmark MHA dense =============
print("\n[2] Dense MultiHeadAttention...")
dense_result, dense_time, dense_memory = benchmark(
    dense_wrapper_compiled,
    x_dense,
    num_iters=50
)
print(f"  Time: {dense_time:.5f}s")
print(f"  Peak memory: {dense_memory/1e9:.2f} GB")

# ============= Comparaison =============
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Speedup (dense/sparse): {(dense_time/your_time):.2f}x")
print(f"Memory reduction: {((dense_memory - your_memory)/1e9):.2f} GB")
print(f"Memory ratio (sparse/dense): {(your_memory/dense_memory):.2%}")
print(f"\nTheoretical dense attention complexity: O(N²) = O({max_seq_len}²) = {max_seq_len**2:,} operations per event")
print(f"Your sparse complexity: O(E) = {edge_index.size(1) / num_events:,.0f} operations per event")
print(f"Reduction factor: {(max_seq_len**2) / (edge_index.size(1) / num_events):.1f}x")

print("\nNote: Les résultats numériques diffèrent car sparse et dense")
print("utilisent des patterns de connectivité complètement différents.")