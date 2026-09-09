#!/usr/bin/env python3
"""Run Veros scaling experiments with optional Nsight profiling.

The command after ``--`` is treated as a program template. The placeholders
``{ranks}``, ``{px}``, ``{py}``, ``{nx}``, ``{ny}``, and ``{nz}`` are replaced
for every run. For example::

    python tools/cuda_scaling.py --mode weak --local-size 256 256 64 \
        --ranks 1,4,16 --timesteps 30 --profiler nsys -- \
        python benchmarks/acc_benchmark.py -b jax -d gpu \
        --nproc {px} {py} --size {nx} {ny} {nz} --timesteps {timesteps}

For a setup file, use a command such as ``veros run setup.py -b jax -d gpu
-n {px} {py}`` and provide any setup-specific runtime settings in the
template.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


STEP_TIME = "Time step took"


def parse_triplet(values):
    return tuple(int(value) for value in values)


def parse_ranks(value):
    ranks = tuple(int(item) for item in value.split(","))
    if not ranks or any(rank < 1 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must be positive integers")
    return ranks


def process_grid(ranks):
    """Return a balanced (x, y) process grid for a rank count."""
    px = max(factor for factor in range(1, int(math.sqrt(ranks)) + 1) if ranks % factor == 0)
    return px, ranks // px


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("strong", "weak"), required=True)
    parser.add_argument("--ranks", type=parse_ranks, required=True, help="Comma-separated MPI rank counts")
    parser.add_argument("--size", type=int, nargs=3, metavar=("NX", "NY", "NZ"), help="Fixed global size")
    parser.add_argument(
        "--local-size", type=int, nargs=3, metavar=("NX", "NY", "NZ"), help="Fixed per-rank size for weak scaling"
    )
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--launcher", default="mpirun", help="MPI launcher, e.g. mpirun or srun")
    parser.add_argument("--launcher-args", default="", help="Additional arguments passed to the MPI launcher")
    parser.add_argument("--profiler", choices=("none", "nsys", "ncu"), default="none")
    parser.add_argument("--profiler-args", default="", help="Additional arguments passed to nsys or ncu")
    parser.add_argument(
        "--rank-env",
        default="OMPI_COMM_WORLD_RANK",
        help="MPI rank environment variable used in Nsight report names",
    )
    parser.add_argument(
        "--local-rank-env",
        default="OMPI_COMM_WORLD_LOCAL_RANK",
        help="MPI local-rank environment variable used to select the GPU to monitor",
    )
    parser.add_argument(
        "--memory-sample-interval",
        type=float,
        default=5.0,
        help="Seconds between per-GPU memory samples; set to 0 to disable",
    )
    parser.add_argument("--communication-profile", action="store_true", help="Record synchronized halo-exchange timing")
    parser.add_argument("--jax-trace", action="store_true", help="Write a per-rank JAX/XProf trace")
    parser.add_argument("--output-dir", type=Path, default=Path("cuda-scaling-results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Program template, preceded by --")
    return parser


def validate_args(args):
    if args.mode == "strong" and args.size is None:
        raise ValueError("--size is required for strong scaling")
    if args.mode == "weak" and args.local_size is None:
        raise ValueError("--local-size is required for weak scaling")
    if args.repetitions < 1 or args.timesteps < 1:
        raise ValueError("--repetitions and --timesteps must be positive")
    if args.memory_sample_interval < 0:
        raise ValueError("--memory-sample-interval must be non-negative")
    if args.jax_trace and args.profiler != "none":
        raise ValueError("--jax-trace cannot be combined with nsys or ncu; both profilers compete for GPU tracing")
    if not args.command or args.command[0] != "--":
        raise ValueError("the program template must follow --")


def command_for_run(args, ranks, repeat, output_dir):
    px, py = process_grid(ranks)
    if args.mode == "strong":
        nx, ny, nz = args.size
    else:
        local_nx, local_ny, nz = args.local_size
        nx, ny = local_nx * px, local_ny * py

    values = {
        "ranks": ranks,
        "px": px,
        "py": py,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "timesteps": args.timesteps,
    }
    try:
        program = [part.format(**values) for part in args.command[1:]]
    except KeyError as exc:
        raise ValueError(f"unknown command placeholder: {exc.args[0]}") from None

    rank_setup = []
    rank_expression = f'"${{{args.rank_env}:-0}}"'
    if args.communication_profile:
        profile_prefix = output_dir / f"veros-profile-repeat{repeat}-rank"
        rank_setup.append(f"export VEROS_PROFILE_OUTPUT={shlex.quote(str(profile_prefix))}-{rank_expression}.json")
    if args.jax_trace:
        trace_prefix = output_dir / "jax-traces" / f"repeat{repeat}-rank"
        rank_setup.append(f"export VEROS_JAX_PROFILER_TRACE={shlex.quote(str(trace_prefix))}-{rank_expression}")

    if args.memory_sample_interval:
        monitor = Path(__file__).with_name("gpu_memory_monitor.py").resolve()
        output_prefix = output_dir / f"gpu-memory-repeat{repeat}-rank"
        monitor_command = "; ".join(
            rank_setup
            + [
                (
                    f"exec {shlex.quote(sys.executable)} {shlex.quote(str(monitor))} "
                    f"--output {shlex.quote(str(output_prefix))}-{rank_expression}.json "
                    f"--interval {args.memory_sample_interval} "
                    f"--rank-env {shlex.quote(args.rank_env)} "
                    f"--local-rank-env {shlex.quote(args.local_rank_env)} -- \"$@\""
                )
            ]
        )
        program = ["sh", "-c", monitor_command, "veros-memory-monitor", *program]
    elif rank_setup:
        rank_command = "; ".join(rank_setup + ['exec "$@"'])
        program = ["sh", "-c", rank_command, "veros-rank-profile", *program]

    command = shlex.split(args.launcher) + shlex.split(args.launcher_args) + ["-n", str(ranks)]
    if args.profiler == "nsys":
        report_prefix = output_dir / f"nsys-r{ranks}-repeat{repeat}-rank"
        profiler_args = shlex.join(shlex.split(args.profiler_args))
        rank_expression = f'"${{{args.rank_env}:-0}}"'
        profiler_command = (
            f"nsys profile --force-overwrite true -o {shlex.quote(str(report_prefix))}-{rank_expression}"
        )
        if profiler_args:
            profiler_command += f" {profiler_args}"
        command += ["sh", "-c", f"exec {profiler_command} \"$@\"", "veros-nsys"]
    elif args.profiler == "ncu":
        command += ["ncu"] + shlex.split(args.profiler_args)
    command += program
    return command, values


def main():
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ranks in args.ranks:
        for repeat in range(1, args.repetitions + 1):
            command, values = command_for_run(args, ranks, repeat, args.output_dir)
            print("$", shlex.join(command), flush=True)
            if args.dry_run:
                continue

            start = time.perf_counter()
            completed = subprocess.run(command, check=False, env=os.environ.copy())
            elapsed = time.perf_counter() - start
            memory_samples = []
            for memory_file in sorted(args.output_dir.glob(f"gpu-memory-repeat{repeat}-rank*.json")):
                payload = json.loads(memory_file.read_text())
                if payload.get("rank", ranks) < ranks:
                    memory_samples.append(payload)
            memory_samples.sort(key=lambda payload: payload["rank"])
            communication_profiles = []
            for profile_file in sorted(args.output_dir.glob(f"veros-profile-repeat{repeat}-rank*.json")):
                payload = json.loads(profile_file.read_text())
                if payload.get("rank", ranks) < ranks:
                    communication_profiles.append(payload)
            communication_profiles.sort(key=lambda payload: payload["rank"])
            communication_time = sum(payload["halo_exchange_seconds"] for payload in communication_profiles)
            compute_time = sum(payload["estimated_compute_seconds"] for payload in communication_profiles)
            main_time = sum(payload["main_loop_seconds"] for payload in communication_profiles)
            result = {
                "repeat": repeat,
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "gpu_memory_samples": memory_samples,
                "communication_profiles": communication_profiles,
                "communication_fraction_percent": (
                    100 * communication_time / main_time if communication_profiles and main_time else None
                ),
                "communication_to_compute_ratio": (
                    communication_time / compute_time if communication_profiles and compute_time else None
                ),
                **values,
            }
            results.append(result)
            if completed.returncode:
                raise SystemExit(completed.returncode)

    metadata = {
        "mode": args.mode,
        "profiler": args.profiler,
        "launcher": args.launcher,
        "repetitions": args.repetitions,
        "memory_sample_interval_seconds": args.memory_sample_interval,
        "communication_profile": args.communication_profile,
        "jax_trace": args.jax_trace,
        "results": results,
    }
    output_file = args.output_dir / "scaling.json"
    output_file.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
