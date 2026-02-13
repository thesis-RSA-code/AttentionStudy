import numpy as np

def print_benchmark_summary(filepath):
    """
    Load and print benchmark results from NPZ file
    
    Args:
        filepath: Path to the .npz file
    """
    data = np.load(filepath, allow_pickle=True)
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Configuration
    print("\nConfiguration:")
    print(f"  Batch size:          {data['config/num_events']}")
    print(f"  Avg PMTs per event:  {data['config/avg_pmts_per_event']}")
    print(f"  Hidden channels:     {data['config/hidden_channels']}")
    print(f"  Num heads:           {data['config/num_heads']}")
    print(f"  Edges per node:      {data['config/num_edges_per_node']}")
    print(f"  DB precision:        {data['config/db_precision']}")
    print(f"  Compiled:            {data['config/compiled']}")
    print(f"  Device:              {data['config/device']}")
    print(f"  Seed:                {data['metadata/seed']}")
    
    # Batch stats
    print("\nBatch stats:")
    print(f"  Total PMTs:          {data['batch/total_pmts']}")
    print(f"  Max sequence length: {data['batch/max_seq_length']}")
    
    # Metadata
    print(f"\nTimestamp: {data['metadata/timestamp']}")
    
    # Performance results
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Memory (GB)':<12}")
    print("-" * 80)
    
    vanilla_fwd = data['results/vanilla_sparse/forward_time_ms']
    vanilla_bwd = data['results/vanilla_sparse/backward_time_ms']
    vanilla_total = data['results/vanilla_sparse/total_time_ms']
    vanilla_memory = data['results/vanilla_sparse/peak_memory_gb']
    
    fused_fwd = data['results/fused_sparse/forward_time_ms']
    fused_bwd = data['results/fused_sparse/backward_time_ms']
    fused_total = data['results/fused_sparse/total_time_ms']
    fused_memory = data['results/fused_sparse/peak_memory_gb']
    
    print(f"{'Vanilla sparse':<20} {vanilla_fwd:<12.3f} {vanilla_bwd:<12.3f} {vanilla_total:<12.3f} {vanilla_memory:<12.3f}")
    print(f"{'Fused sparse':<20} {fused_fwd:<12.3f} {fused_bwd:<12.3f} {fused_total:<12.3f} {fused_memory:<12.3f}")
    print("-" * 80)
    
    # Speedup calculations
    print(f"\nSpeedup (vanilla vs fused):")
    print(f"  Forward:  {(vanilla_fwd/fused_fwd):.2f}x")
    print(f"  Backward: {(vanilla_bwd/fused_bwd):.2f}x")
    print(f"  Total:    {(vanilla_total/fused_total):.2f}x")
    
    # Memory reduction
    memory_diff = vanilla_memory - fused_memory
    memory_pct = (1 - fused_memory/vanilla_memory) * 100
    print(f"\nMemory reduction: {memory_diff:.3f} GB ({memory_pct:.1f}%)")
    
    print("=" * 80)


if __name__ == "__main__":
    filepath = "/sps/t2k/eleblevec/mini-Caverns-toolsbox/mini-Caverns-benchmarks/2026_02_attn_layers/outputs/benchmark_20260212_115541_bs50_hc50_nh5_dbTrue_compTrue.npz"
    print_benchmark_summary(filepath)
    