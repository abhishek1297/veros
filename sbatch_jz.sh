#! /bin/bash

#SBATCH --job-name=veros-global
#SBATCH --output=.std/veros.out
#SBATCH --error=.std/veros.err
#SBATCH --account=yah@h100
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=24 -C h100 --gres=gpu:2 --qos=qos_gpu_h100-dev
#SBATCH --hint=nomultithread 
#SBATCH --time=00:05:00



module purge
module load singularity

export TMPDIR="$SCRATCH/veros_tmp"
mkdir -p "$TMPDIR"


nvidia-smi
srun --mpi=pmix_v3 \
    singularity exec --nv \
    --env-file .jz_env \
    --bind "$TMPDIR:$TMPDIR" \
    --bind "$VEROS_ASSET_DIR:/veros_assets" \
    --bind "$PWD:/workspace/veros" \
    "$SINGULARITY_ALLOWED_DIR/jax-veros2.sif" \
    bash -c '
        export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
        veros run /workspace/veros/veros/setups/global_flexible/global_flexible.py --backend jax --device gpu -n 1 2 --diskless-mode
    ' 
