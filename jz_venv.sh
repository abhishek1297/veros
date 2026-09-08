#! /bin/bash

# Changes prior running this
# 1. Applying a patch for relaxing check of hdf5==1.12.0 which causes failure to link h5py against MPI and CUDA modules
# 2. Updated requirement_jax.txt for mpi4py and mpi4jax support

rm -rf .venv/
module purge
module load arch/h100
module load pytorch-gpu/py3/2.13.0

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -U pip setuptools wheel cython

export HDF5_DIR=$HDF5_ROOT
export HDF5_MPI="ON"
export CC="$(which mpicc)"
python3 -m pip install -vvv --no-cache-dir --no-binary h5py $WORK/h5py

python3 -m pip install --no-cache-dir --no-binary mpi4py \
        -r requirements.txt -r requirements_jax.txt 2>/dev/null || true
python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e .

