#! /bin/bash

#SBATCH --job-name=veros-global
#SBATCH --output=.std/veros.out
#SBATCH --error=.std/veros.err
#SBATCH --account=yah@h100
#SBATCH --nodes=1 --ntasks-per-node=2 -C h100 --gres=gpu:2 --qos=qos_gpu_h100-dev
#SBATCH --time=00:05:00

module purge
module load singularity

export TMPDIR="$SCRATCH/veros_tmp"
mkdir -p "$TMPDIR"

# Allow OpenMPI inside the container to run without host PMIx restrictions
# export SINGULARITYENV_OMPI_MCA_orte_keep_fqdn_hostnames=0
# export SINGULARITYENV_OMPI_MCA_btl=self,vader


nvidia-smi

# Execute mpirun directly inside the Singularity context
# singularity exec --nv \
#     --env-file .jz_env \
#     --bind "$TMPDIR:$TMPDIR" \
#     --bind "$VEROS_ASSET_DIR:/veros_assets" \
#     --bind "$PWD:/workspace/veros" \
#     "$SINGULARITY_ALLOWED_DIR/jax-veros2.sif" \
#     mpirun -np 2 --allow-run-as-root \
#     bash -c '
#         export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
#         veros run /workspace/veros/veros/setups/global_flexible/global_flexible.py --backend jax --device gpu -n 1 2 --diskless-mode
#     '


srun --mpi=pmix singularity exec --nv \
    --env-file .jz_env \
    --bind "$TMPDIR:$TMPDIR" \
    --bind "$VEROS_ASSET_DIR:/veros_assets" \
    --bind "$PWD:/workspace/veros" \
    "$SINGULARITY_ALLOWED_DIR/jax-veros2.sif" \
    bash -c '
        export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
        veros run /workspace/veros/veros/setups/global_flexible/global_flexible.py --backend jax --device gpu -n 1 2 --diskless-mode
    '
