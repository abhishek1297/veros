#!/bin/bash
# Isolate GPU based on Slurm or OpenMPI local rank
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
echo "MPI rank=${SLURM_PROCID:-0} GPU=$CUDA_VISIBLE_DEVICES"

# Execute the veros command passed by cuda_scaling.py
exec "$@"

