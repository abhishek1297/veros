#! /bin/bash

rm -rf .venv/
module purge
module load arch/h100
module load pytorch-gpu/py3/2.13.0
python3 -v venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install --no-cache-dir  \
  --no-binary mpi4py \
  -r requirements.txt \
  -r requirements_jax.txt  # update jax -> jax[cuda13-local] 
