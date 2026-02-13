
#!/bin/bash 

IMAGE_PATH="/sps/t2k/melbaz/env/ml_image.sif"
PYTHON_SCRIPT="compare_debug_layers.py"
# PYTHON_SCRIPT="sparse_mha_benchmarks.py"

BIND_CODE="/sps/t2k/eleblevec/mini-Caverns-toolsbox/mini-Caverns-benchmarks/2026_02_attn_layers:/work/ml"
WORK_DIR="/work/ml"

export TORCHDYNAMO_VERBOSE=1

singularity exec --nv \
    -B $BIND_CODE \
    -W $WORK_DIR \
    $IMAGE_PATH \
    python $PYTHON_SCRIPT
