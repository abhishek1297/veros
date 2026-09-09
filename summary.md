# CUDA scaling results guide

This document explains the plots produced by the CUDA scaling workflow, how to
interpret them, and how to inspect the JAX profiles with XProf.

## Generated files

After all scaling jobs have finished, the plotting rule produces:

- `results/scaling/plots/scaling.png`
- `results/scaling/plots/resource_usage.png`
- `results/scaling/plots/communication_compute_ratio.png`

The measurements used by these plots are stored in each case's
`results/scaling/<mode>/nodes<N>-gpus<G>/scaling.json` file.

## Scaling plot

`scaling.png` contains strong scaling, weak scaling, and node-placement panels.

### Strong scaling

Strong scaling keeps the global model size fixed while increasing the total
number of GPUs. Elapsed time should decrease toward the dashed ideal-scaling
line. A curve that flattens indicates that communication, synchronization, or
fixed overhead is becoming large relative to useful computation.

### Weak scaling

Weak scaling keeps the local problem size per MPI rank fixed, so the global
problem grows with the GPU count. Ideal weak scaling has approximately constant
elapsed time. Increasing time indicates growing communication, synchronization,
load imbalance, or multi-node overhead.

### Node placement

The node-placement panel helps distinguish GPU-count scaling from inter-node
effects. A substantial slowdown when work crosses a node boundary suggests that
network communication is more expensive than communication between GPUs on one
node.

## Resource-usage subplots

`resource_usage.png` is a 2-by-3 grid. The top row shows strong scaling and the
bottom row shows weak scaling.

### GPU memory

After the first timestep has completed, memory is sampled every five seconds
and reported in MiB per GPU. Each line is the mean across GPUs and repetitions;
the shaded area spans the minimum and maximum sampled GPUs. A wide band suggests
memory imbalance. A curve approaching the device capacity indicates an
out-of-memory risk.

### GPU compute utilization

Compute utilization is recorded with memory, beginning after the first
timestep, and is the percentage reported by `nvidia-smi`. Sustained high
utilization generally indicates that GPUs remain busy. Low or highly variable
utilization can indicate CPU dispatch delays, communication waits, I/O,
synchronization, or a problem that is too small for the GPU.

This sampled utilization is a coarse signal rather than a kernel-efficiency
measurement. Use XProf for the detailed device timeline.

### Halo communication

Halo communication is the synchronized Veros `boundary_exchange` time as a
percentage of steady-state main-loop time. Growth with GPU count indicates that
domain-boundary exchange is consuming more of the run.

This value does not include every MPI reduction and does not directly measure
NVLink or InfiniBand traffic in bytes. It is a focused estimate of Veros halo
exchange overhead.

## Communication-to-compute ratio

`communication_compute_ratio.png` plots:

```text
halo-exchange time / estimated compute time
```

A ratio of `0.1` means halo exchange takes about 10% as much time as computation.
A ratio of `1.0` means communication and computation take equal time. Values
above `1.0` mean halo communication dominates the measured steady-state loop.

The first timestep is excluded from this ratio, so the initial JAX JIT
compilation normally does not distort it. Use XProf to inspect compilation time.

## Finding the XProf traces

With `jax_trace: true`, each MPI rank writes a separate trace under:

```text
results/scaling/<mode>/nodes<N>-gpus<G>/jax-traces/repeat<R>-rank<RANK>/
```

The actual profile files are below the JAX-created `plugins/profile/`
subdirectory. The generated command passes this location explicitly through
`veros run --jax-trace-dir`. The Slurm log prints `Writing JAX/XProf trace to
...` and `JAX/XProf trace completed at ...` for every rank, making the exact
output directory visible after each run.

List all traces with:

```bash
find results/scaling -path '*/jax-traces/*' -type f
```

For example, the parent log directory for a two-node, four-GPU-per-node strong
scaling case is:

```text
results/scaling/strong/nodes2-gpus4/jax-traces
```

## Opening a trace with XProf

Install XProf in the environment used for analysis:

```bash
python -m pip install -U xprof
```

From the repository root, start XProf and point it at one case:

```bash
xprof --port=8791 results/scaling/strong/nodes2-gpus4/jax-traces
```

Then open `http://localhost:8791` in a browser. When XProf runs on a remote
cluster login node, create an SSH tunnel from the local computer:

```bash
ssh -L 8791:localhost:8791 <user>@<cluster-login-host>
```

Alternatively, use the TensorBoard integration:

```bash
python -m pip install -U tensorboard xprof
tensorboard --logdir results/scaling/strong/nodes2-gpus4/jax-traces --port 8791
```

In XProf, start with the Trace Viewer to inspect JIT compilation, CPU dispatch,
GPU kernels, idle gaps, and communication-related waits. Compare ranks to find
load imbalance or a rank that reaches synchronization points later than others.

Do not enable `jax_trace` in the same run as `nsys` or `ncu`; collect those
profiles in separate jobs.
