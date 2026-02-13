
#!/bin/bash 

## CCLyon
# IMAGE_PATH="/sps/t2k/melbaz/env/ml_image.sif"
# PYTHON_SCRIPT="/lustre/work/eleblevec/attn_study/benchmarks/sparse_mhas.py"

## IGPU cluster
IMAGE_PATH="/lustre/work/aatta/containers/container_base_ml_v3.0.0.sif"
PYTHON_SCRIPT="benchmarks/sparse_mhas.py"  # Use container path relative to WORK_DIR
# PYTHON_SCRIPT="sparse_mha_benchmarks.py"

BIND_CODE="/lustre/work/eleblevec/attn_study:/work/ml"
WORK_DIR="/work/ml"

export TORCHDYNAMO_VERBOSE=1

singularity exec --nv \
    -B $BIND_CODE \
    -W $WORK_DIR \
    --env PYTHONPATH=/work/ml \
    $IMAGE_PATH \
    python $PYTHON_SCRIPT
