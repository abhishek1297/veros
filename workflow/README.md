# CUDA scaling workflow

This workflow creates one Slurm script for every `global_flexible` strong- and
weak-scaling case in the node/GPU topology matrix in `config.yaml`. It does not
submit or run the cases automatically.

## Run

Generate all Slurm scripts with:

```bash
snakemake --snakefile workflow/Snakefile --cores 1 --printshellcmds
```

The scripts are written to `results/scaling/jobs/`. After loading the required
environment and modules, submit the desired cases manually, for example:

```bash
sbatch results/scaling/jobs/strong-nodes2-gpus4.slurm
```

For a dry run:

```bash
snakemake --snakefile workflow/Snakefile --cores 1 -n -p
```

Edit `config.yaml` before generating the scripts to change dimensions,
timesteps, topology, Nsight options, or the setup file. Each generated script
contains the matching `#SBATCH` node, task, and GPU directives. Add any
site-specific directives such as account, partition, or walltime before
submitting if your cluster requires them.

The workflow uses `SLURM_PROCID` for per-rank Nsight report names and requests
one GPU per task with `--gpu-bind=closest`. Adapt those settings if the site
uses a different scheduler or MPI launcher.

## Plots

Once every submitted case has completed, generate the plot with:

```bash
snakemake --snakefile workflow/Snakefile --cores 1 results/scaling/plots/scaling.png
```

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
