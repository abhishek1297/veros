# CUDA scaling workflow

This workflow creates one Slurm script for every `global_flexible` strong- and
weak-scaling case in the node/GPU topology matrix in a config file. There is no
default config -- always pass one explicitly with `--configfile`, since setup
differs per cluster:

- `config.odyssey.yaml` -- Odyssey cluster, Conda environment, `mpirun`.
- `config.jz.yaml` -- Jean Zay cluster, modules + venv, `srun`.
- `config.local.yaml` -- single local GPU, Conda environment, `mpirun`.

It does not submit or run the cases automatically.

## Run

Generate all Slurm scripts with:

### Odyssey

```bash
snakemake --snakefile workflow/Snakefile.odyssey --configfile workflow/config.odyssey.yaml --cores 1 --printshellcmds
```

### Jean-zay
```bash
snakemake --snakefile workflow/Snakefile.jz --configfile workflow/config.jz.yaml --cores 1 --printshellcmds
```


The scripts are written to `results/scaling/jobs/`. Submit the desired cases
manually, for example:

```bash
sbatch results/scaling/jobs/strong-nodes2-gpus4.slurm
```

For a dry run:

```bash
snakemake --snakefile workflow/Snakefile.jz --configfile workflow/config.jz.yaml --cores 1 -n -p
```

Edit the relevant config file before generating the scripts to change
dimensions, timesteps, topology, Nsight options, or the setup file. Each
generated script contains the matching `#SBATCH` node, task, and GPU
directives, plus the cluster's `environment.init` block verbatim (module
loads, Conda/venv activation, exported env vars). Add any further
site-specific directives such as account, partition, or walltime before
submitting if your cluster requires them.

The generated jobs launch with the configured `launcher` (`mpirun` or `srun`).
Each rank selects a GPU using the configured `local_rank_env`, while `rank_env`
is used in per-rank Nsight report names. GPU memory use is sampled every
`memory_sample_interval` seconds (five by default) on every rank and embedded
under `gpu_memory_samples` in the case's `scaling.json`.

## Plots

Once every submitted case has completed, generate the plot with:

```bash
snakemake --snakefile workflow/Snakefile --configfile workflow/config.jz.yaml --cores 1 results/scaling/plots/scaling.png
```

The generated `results/scaling/plots/scaling.png` contains:

- strong-scaling elapsed time versus total GPUs, with an ideal reference;
- weak-scaling elapsed time versus total GPUs, where a flat curve is ideal;
- a combined node-placement comparison.

The generated `results/scaling/plots/resource_usage.png` contains a 2-by-3
subplot grid for strong- and weak-scaling GPU memory use, compute utilization,
and synchronized halo-communication share. Time-series lines are means across
GPUs and repetitions; shaded bands span the minimum to maximum GPU value. Raw
per-GPU samples remain available in `scaling.json`.

The generated `results/scaling/plots/communication_compute_ratio.png` plots
the halo-communication time divided by estimated compute time for every
topology. The JSON also records this `communication_to_compute_ratio` and its
per-rank inputs. This is a halo-exchange timing estimate, not a measurement of
all MPI reductions or physical NVLink/InfiniBand bytes.

When `jax_trace: true`, every rank also writes an XProf-compatible JAX trace
under the case's `jax-traces/` directory. Point XProf or TensorBoard at that
directory to inspect device timelines and JAX operations.

JAX tracing cannot be enabled in the same run as `nsys` or `ncu`, because the
profilers compete for GPU tracing facilities. To collect an Nsight report in a
separate run, set `jax_trace: false` and select the desired `profiler`.

For a fuller study, also plot strong-scaling speedup and efficiency:

```text
speedup(P)   = T(1) / T(P)
efficiency(P) = speedup(P) / P
```

Useful additional plots from Nsight data are GPU utilization versus time,
CUDA kernel duration by routine, MPI wait time versus compute time, and setup
versus steady-state timestep time. Do not use Nsight-instrumented elapsed time
as the final scaling number; use it to explain the non-profiled timing results.


## Local test

```bash
# 1. Dry-run against the local config (1 node, 1 GPU, mpirun instead of srun)
snakemake --snakefile workflow/Snakefile --configfile workflow/config.local.yaml --cores 1 -n -p

# 2. Generate the runnable scripts (writes results/scaling/jobs/*.slurm)
snakemake --snakefile workflow/Snakefile --configfile workflow/config.local.yaml --cores 1 -p

# 3. Run them directly with bash — the #SBATCH lines are just comments locally
bash results/scaling/jobs/strong-nodes1-gpus1.slurm
bash results/scaling/jobs/weak-nodes1-gpus1.slurm

# 4. Plot the results
python tools/plot_scaling.py \
  --input-glob 'results/scaling/*/nodes1-gpus1/scaling.json' \
  --output results/scaling/plots/scaling.png
```
