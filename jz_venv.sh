#! /bin/bash

# update jax -> jax[cuda13-local] first

rm -rf .venv/
module purge
module load arch/h100
module load pytorch-gpu/py3/2.13.0

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -U pip setuptools wheel cython
python3 -m pip install --no-binary mpi4py -r requirements.txt -r requirements_jax.txt 2>/dev/null || true
python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e .
