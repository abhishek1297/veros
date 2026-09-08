#! /bin/bash

# Changes prior running this
# 1. Applying a patch for relaxing check of hdf5==1.12.0 which causes failure to link h5py against MPI and CUDA modules
# 2. Updated requirement_jax.txt for mpi4py and mpi4jax support

VEROS_DIR=$PWD

rm -rf .venv/
module purge
module load arch/h100
module load pytorch-gpu/py3/2.13.0

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -U --no-cache-dir pip
python3 -m pip install --no-cache-dir setuptools wheel cython

# mpi4jax's CUDA bridge must be built against the same local CUDA toolkit
# that jax[cuda13-local] uses at runtime, or GPU-direct MPI can segfault
# with a CUDA-runtime ABI mismatch. See mpi4jax's README ("jax[cudaXX_local]").
export CUDA_ROOT="${CUDA_ROOT:-$CUDA_HOME}"
if [ -z "$CUDA_ROOT" ]; then
    echo "CUDA_ROOT/CUDA_HOME not set by the loaded modules -- find it with" \
         "'module show cuda/13.2.1' and export it before running this script" >&2
    exit 1
fi

python3 -m pip install --no-cache-dir --no-binary mpi4py,mpi4jax \
        -r requirements.txt -r requirements_jax.txt
python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e .

