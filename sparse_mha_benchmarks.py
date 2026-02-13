


import os
import time
import random
import numpy as np

def benchmark(model, *args, num_iters=100, **kwargs):
    """Benchmark function to measure time and memory with backpropagation"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(10):
        output = model(*args, **kwargs)
        if isinstance(output, tuple):
            output = output[0]
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    
    # Track forward and backward times separately
    forward_times = []
    backward_times = []
    
    for _ in range(num_iters):
        # Forward pass
        torch.cuda.synchronize()
        fwd_start = time.time()
        output = model(*args, **kwargs)
        if isinstance(output, tuple):
            output = output[0]
        loss = output.sum()
        torch.cuda.synchronize()
        forward_times.append(time.time() - fwd_start)
        
        # Backward pass
        torch.cuda.synchronize()
        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - bwd_start)
        
        model.zero_grad()
    
    avg_forward_time = np.mean(forward_times)
    avg_backward_time = np.mean(backward_times)
    total_time = avg_forward_time + avg_backward_time
    peak_memory = torch.cuda.max_memory_allocated()
    
    return output, avg_forward_time, avg_backward_time, total_time, peak_memory

    
def compare_outputs(output1, output2, name1="Model 1", name2="Model 2", tolerance=1e-5):
    """
    Compare deux outputs et affiche les statistiques de différence
    
    Args:
        output1: Premier output (peut être sparse ou dense)
        output2: Deuxième output (peut être sparse ou dense)
        name1: Nom du premier modèle
        name2: Nom du deuxième modèle
        tolerance: Tolérance pour considérer les valeurs comme égales
    """
    print(f"\n{'='*60}")
    print(f"COMPARING {name1} vs {name2}")
    print(f"{'='*60}")
    
    # Vérifier les shapes
    print(f"{name1} shape: {output1.shape}")
    print(f"{name2} shape: {output2.shape}")
    
    if output1.shape != output2.shape:
        print(f"⚠️  WARNING: Different shapes! Cannot compare directly.")
        return
    
    # Calculer les différences
    diff = (output1 - output2).abs()
    
    print(f"\nDifference statistics:")
    print(f"  Max absolute difference:  {diff.max().item():.2e}")
    print(f"  Mean absolute difference: {diff.mean().item():.2e}")
    print(f"  Median absolute difference: {diff.median().item():.2e}")
    print(f"  Std of difference: {diff.std().item():.2e}")
    
    # Pourcentage de valeurs similaires
    close_mask = diff < tolerance
    pct_close = close_mask.float().mean().item() * 100
    print(f"\nValues within tolerance ({tolerance}):")
    print(f"  {pct_close:.2f}% of values are close")
    
    # Relative error (éviter division par zéro)
    eps = 1e-10
    rel_error = diff / (output1.abs() + eps)
    print(f"\nRelative error:")
    print(f"  Max relative error:  {rel_error.max().item():.2e}")
    print(f"  Mean relative error: {rel_error.mean().item():.2e}")
    
    # Verdict
    if diff.max().item() < tolerance:
        print(f"\n✅ OUTPUTS ARE IDENTICAL (within tolerance {tolerance})")
    elif diff.max().item() < 1e-3:
        print(f"\n✓ Outputs are very similar (max diff < 1e-3)")
    elif diff.max().item() < 1e-1:
        print(f"\n⚠️  Outputs differ moderately (max diff < 1e-1)")
    else:
        print(f"\n❌ OUTPUTS ARE SIGNIFICANTLY DIFFERENT")


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

def main(num_events, avg_pmts_per_event, hidden_channels, num_heads, num_edges_per_node, device, db_precision, compiling, compare_outputs):
    print("Environment variables:")
    print(f"  TRITON_CACHE_DIR: {os.getenv('TRITON_CACHE_DIR')}")
    print(f"  TORCHINDUCTOR_CACHE_DIR: {os.getenv('TORCHINDUCTOR_CACHE_DIR')}")
    print(f"  XDG_CACHE_HOME: {os.getenv('XDG_CACHE_HOME')}")
    print(f"  TORCH_HOME: {os.getenv('TORCH_HOME')}")


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


    max_seq_len = torch.max(torch.unique(batch.batch, return_counts=True)[1]).cpu()
    total_pmts = batch.x.size(0)
    print(f"Batch stats:")
    print(f"  Total PMTs: {total_pmts}")
    print(f"  Max sequence length: {max_seq_len}")
    print()

    # ============= Modèles =============
    print(f"Creating models...")
    print(f"  DB precision: {db_precision}")
    vanilla_attn_layer = AttentionLayer(hidden_channels, num_heads=num_heads, db_precision=db_precision).to(device)
    fused_attn_layer = AttentionLayerFused(hidden_channels, num_heads=num_heads, db_precision=db_precision).to(device)
    
    # Après initialisation des deux modèles
    if compare_outputs:
        with torch.no_grad():
            fused_attn_layer.qkv_proj.weight.data.copy_(torch.cat([
                vanilla_attn_layer.q_proj.weight.data,
                vanilla_attn_layer.k_proj.weight.data,
                vanilla_attn_layer.v_proj.weight.data,
            ], dim=0))
            fused_attn_layer.qkv_proj.bias.data.copy_(torch.cat([
                vanilla_attn_layer.q_proj.bias.data,
                vanilla_attn_layer.k_proj.bias.data,
                vanilla_attn_layer.v_proj.bias.data,
            ], dim=0))

        # Copier les poids de o_proj
        with torch.no_grad():
            fused_attn_layer.o_proj.weight.data.copy_(vanilla_attn_layer.o_proj.weight.data)
            fused_attn_layer.o_proj.bias.data.copy_(vanilla_attn_layer.o_proj.bias.data)

    
    print(f"Done creating models.")

    if compiling:
        print(f"Compiling models...")
        vanilla_attn_layer = torch.compile(
            vanilla_attn_layer, 
            mode="reduce-overhead", # pour moins de variabilité dans les résultats
            fullgraph=True # Force la compilation du graph complet
        )
        fused_attn_layer = torch.compile(
            fused_attn_layer, 
            mode="reduce-overhead", # pour moins de variabilité dans les résultats
            fullgraph=True # Force la compilation du graph complet
        )
        print(f"Done compiling models.")

    # ============= Benchmark sparse graph attention models =============

    vanilla_result, vanilla_fwd, vanilla_bwd, vanilla_total, vanilla_memory = benchmark(
        vanilla_attn_layer, 
        batch.x, 
        batch.edge_index,
        num_iters=50
    )
    print(f"[1] Vanilla sparse attention...")
    print(f"  Forward time:  {vanilla_fwd*1000:.3f}ms")
    print(f"  Backward time: {vanilla_bwd*1000:.3f}ms")
    print(f"  Total time:    {vanilla_total*1000:.3f}ms")
    print(f"  Peak memory:   {vanilla_memory/1e9:.3f} GB")
    print()

    fused_result, fused_fwd, fused_bwd, fused_total, fused_memory = benchmark(
        fused_attn_layer, 
        batch.x, 
        batch.edge_index,
        num_iters=50
    )
    print(f"[2] Fused sparse attention...")
    print(f"  Forward time:  {fused_fwd*1000:.3f}ms")
    print(f"  Backward time: {fused_bwd*1000:.3f}ms")
    print(f"  Total time:    {fused_total*1000:.3f}ms")
    print(f"  Peak memory:   {fused_memory/1e9:.3f} GB")
    print()

    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Memory (GB)':<12}")
    print("-" * 60)
    print(f"{'Vanilla sparse':<20} {vanilla_fwd*1000:<12.3f} {vanilla_bwd*1000:<12.3f} {vanilla_total*1000:<12.3f} {vanilla_memory/1e9:<12.3f}")
    print(f"{'Fused sparse':<20} {fused_fwd*1000:<12.3f} {fused_bwd*1000:<12.3f} {fused_total*1000:<12.3f} {fused_memory/1e9:<12.3f}")
    print("-" * 60)
    print(f"\nSpeedup (vanilla vs fused):")
    print(f"  Forward:  {(vanilla_fwd/fused_fwd):.2f}x")
    print(f"  Backward: {(vanilla_bwd/fused_bwd):.2f}x")
    print(f"  Total:    {(vanilla_total/fused_total):.2f}x")
    print(f"\nMemory reduction: {((vanilla_memory - fused_memory)/1e9):.3f} GB ({(1 - fused_memory/vanilla_memory)*100:.1f}%)")

    if compare_outputs:
        # ============= Comparison of outputs =============
        print("\n" + "=" * 60)
        print("OUTPUT COMPARISON")
        print("=" * 60)
        compare_outputs(vanilla_result, fused_result, name1="Vanilla sparse", name2="Fused sparse")

    config = {
    'num_events': num_events,
    'avg_pmts_per_event': avg_pmts_per_event,
    'hidden_channels': hidden_channels,
    'num_heads': num_heads,
    'num_edges_per_node': num_edges_per_node,
    'db_precision': db_precision,
    'compiled': compiling,
    'device': device,
    'total_pmts': total_pmts,
    'max_seq_length': max_seq_len,
    'seed': seed,
    }
    
    # Prepare results dictionary
    results = {
        'vanilla_sparse': {
            'forward_time': vanilla_fwd,
            'backward_time': vanilla_bwd,
            'total_time': vanilla_total,
            'peak_memory': vanilla_memory,
        },
        'fused_sparse': {
            'forward_time': fused_fwd,
            'backward_time': fused_bwd,
            'total_time': fused_total,
            'peak_memory': fused_memory,
        }
    }
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{timestamp}_bs{num_events}_hc{hidden_channels}_nh{num_heads}_db{db_precision}_comp{compiling}.npz"
    save_benchmark_results(filename, config, results, timestamp)


def is_running_in_container():
    # Check for Singularity/Apptainer
    if os.getenv('SINGULARITY_NAME') or os.getenv('SINGULARITY_CONTAINER') or os.getenv('APPTAINER_CONTAINER'):
        return True
    return False

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Désactive l'auto-optimisation non-déterministe

def save_benchmark_results(filepath, config, results, timestamp=None):
    """
    Save benchmark configuration and results to NPZ file
    
    Args:
        filepath: Path to save the .npz file
        config: Dictionary with configuration parameters
        results: Dictionary with benchmark results for each model
        timestamp: Optional timestamp string
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Prepare data to save
    save_data = {
        # Configuration
        'config/num_events': config['num_events'],
        'config/avg_pmts_per_event': config['avg_pmts_per_event'],
        'config/hidden_channels': config['hidden_channels'],
        'config/num_heads': config['num_heads'],
        'config/num_edges_per_node': config['num_edges_per_node'],
        'config/db_precision': config['db_precision'],
        'config/compiled': config['compiled'],
        'config/device': config['device'],
        
        # Batch stats
        'batch/total_pmts': config['total_pmts'],
        'batch/max_seq_length': config['max_seq_length'],
        
        # Timestamp
        'metadata/timestamp': timestamp,
        'metadata/seed': config.get('seed', 42),
    }
    
    # Add results for each model
    for model_name, model_results in results.items():
        prefix = f'results/{model_name}/'
        save_data[prefix + 'forward_time_ms'] = model_results['forward_time'] * 1000
        save_data[prefix + 'backward_time_ms'] = model_results['backward_time'] * 1000
        save_data[prefix + 'total_time_ms'] = model_results['total_time'] * 1000
        save_data[prefix + 'peak_memory_gb'] = model_results['peak_memory'] / 1e9
    
    # Save to file
    np.savez(filepath, **save_data)
    print(f"\n✓ Benchmark results saved to: {filepath}")


if __name__ == "__main__":

    if is_running_in_container():
        print("Running inside a container. Adjusting paths...")
        user_name = os.getenv('USER', 'elebleve')
        cache_root = f"/tmp/{user_name}/torch_cache"
        print(f"Using user name: {user_name}")
        print(f"Using cache root: {cache_root}")
        os.makedirs(cache_root, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_root, "inductor")
        os.environ["XDG_CACHE_HOME"] = os.path.join(cache_root, "xdg")
        os.environ["TORCH_HOME"] = cache_root
        os.environ["HOME"] = cache_root  # Override HOME
        os.environ["TMPDIR"] = cache_root  # Override TMPDIR

    print(40 * "=" + " Importing torch-related modules. " + 40 * "=")
    print(f"Importing torch...")
    import torch
    print(f"Importing torch_geometric.data...")
    from torch_geometric.data import Batch, Data
    print(f"Importing AttentionLayers...")
    from attn import AttentionLayer, AttentionLayerFused
    print(40 * "=" + " Done importing modules. " + 40 * "=")


    # ============= Configuration =============
    num_events         = 50  # batch size
    avg_pmts_per_event = 5000  # moyenne de PMTs par événement
    hidden_channels    = 50
    num_heads          = 5
    num_edges_per_node = 5  # nombre moyen d'arêtes par PMT
    device             = "cuda"
    db_precision       = True
    compiling          = True
    compare_outputs    = False
    # ============= Seed Management =============
    seed = 42
    set_seeds(seed)

    main(num_events, avg_pmts_per_event, hidden_channels, num_heads, num_edges_per_node, device, db_precision, compiling, compare_outputs)