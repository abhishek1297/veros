# CUDA scaling workflow

This workflow runs `global_flexible` strong- and weak-scaling cases over the
node/GPU topology matrix in `config.yaml`. It launches one `srun` allocation
per case and records timing metadata under `results/scaling/`.

## Run

Load the environment, MPI, CUDA, and MPI-enabled HDF5 modules first. Then run
one case at a time from an allocated Slurm job:

```bash
snakemake --snakefile workflow/Snakefile --cores 1 --printshellcmds
```

For a dry run:

```bash
snakemake --snakefile workflow/Snakefile --cores 1 -n -p
```

Edit `config.yaml` before launching to change dimensions, timesteps, topology,
Nsight options, or the setup file. Keep `--cores 1` unless the cluster profile
is configured to submit independent jobs; each rule invokes `srun` itself.

The workflow uses `SLURM_PROCID` for per-rank Nsight report names and requests
one GPU per task with `--gpu-bind=closest`. Adapt those settings if the site
uses a different scheduler or MPI launcher.

## Plots

The generated `results/scaling/plots/scaling.png` contains:

- strong-scaling elapsed time versus total GPUs, with an ideal reference;
- weak-scaling elapsed time versus total GPUs, where a flat curve is ideal;
- a combined node-placement comparison.

For a fuller study, also plot strong-scaling speedup and efficiency:

```text
speedup(P)   = T(1) / T(P)
efficiency(P) = speedup(P) / P
```

Useful additional plots from Nsight data are GPU utilization versus time,
CUDA kernel duration by routine, MPI wait time versus compute time, and setup
versus steady-state timestep time. Do not use Nsight-instrumented elapsed time
as the final scaling number; use it to explain the non-profiled timing results.
