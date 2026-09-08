#! /bin/bash

#SBATCH --job-name=veros-global
#SBATCH --output=.std/veros.out
#SBATCH --error=.std/veros.err
#SBATCH --account=yah@h100
#SBATCH --ntasks=2 -C h100 --gres=gpu:2 --qos=qos_gpu_h100-dev
#SBATCH --time=00:05:00

module purge
module load arch/h100
module load pytorch-gpu/py3/2.13.0

source .venv/bin/activate
export MPI4JAX_USE_CUDA_MPI=1
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# Segfault workaround: without opal_cuda_support, Open MPI/UCX can mishandle
# GPU pointers; without disabling the memtype cache, it can go stale against
# XLA's own pooled CUDA allocator and touch memory as the wrong type.
export OMPI_MCA_opal_cuda_support=1
export UCX_TLS=rc,cuda_copy,cuda_ipc,sm
export UCX_MEMTYPE_CACHE=n
# if this still segfaults, try disabling GPU-direct MPI entirely instead:
#   export MPI4JAX_USE_CUDA_MPI=0

echo "VEROS Asset Dir: $VEROS_ASSET_DIR"
echo "Active Python: $(which python3)"
echo "Active Veros Path: $(which veros 2>/dev/null || echo 'Not found in PATH')"

nvidia-smi
srun veros run veros/setups/global_flexible/global_flexible.py --backend jax --device gpu -n 1 2 --diskless-mode
 
