#! /bin/bash

#SBATCH --job-name=veros-global
#SBATCH --output=.std/veros.out
#SBATCH --error=.std/veros.err
#SBATCH --account=yah@h100
#SBATCH --ntasks=1 -C h100 --gres=gpu:1 --qos=qos_gpu_h100-dev
#SBATCH --time=00:05:00

module purge
module load arch/h100
module load pytorch-gpu/py3/2.13.0

source .venv/bin/activate
export MPI4JAX_USE_CUDA_MPI=1
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

echo "VEROS Asset Dir: $VEROS_ASSET_DIR"
echo "Active Python: $(which python3)"
echo "Active Veros Path: $(which veros 2>/dev/null || echo 'Not found in PATH')"

nvidia-smi
srun veros run veros/setups/global_flexible/global_flexible.py --backend jax --device gpu 
