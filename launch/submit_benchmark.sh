#!/bin/bash

#SBATCH --account=t2k
#SBATCH --job-name=mha_benchmark
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=/sps/t2k/eleblevec/mini-Caverns-toolsbox/GhostHunter/src/models/utils/logs/mha_benchmark_%j.out
#SBATCH --error=/sps/t2k/eleblevec/mini-Caverns-toolsbox/GhostHunter/src/models/utils/logs/mha_benchmark_%j.err

# Activate conda environment
source /sps/t2k/eleblevec/miniconda3/bin/activate pt28_cuda129

# python src/models/utils/dense_vs_sparse_mha_benchmark.py
python /sps/t2k/eleblevec/mini-Caverns-toolsbox/GhostHunter/src/models/utils/sparse_mha_benchmarks.py