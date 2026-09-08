#!/usr/bin/env python3
"""Propose a global_flexible grid size (nx, ny, nz) that fits on one GPU.

Queries the free memory on a single GPU (via nvidia-smi, or manual overrides)
and backs out a grid size from a rough per-cell memory budget, so you don't
have to guess-and-check resolutions like we did manually earlier.

Example:

    python tools/estimate_grid_size.py
    python tools/estimate_grid_size.py --free-mb 3700 --nz 20
"""

import argparse
import subprocess

# Rough count of independent (nx+4)(ny+4)nz float64 fields carried by
# global_flexible (temp/salt/u/v/w with multiple time levels, plus dozens of
# single-time-level mixing/diagnostic fields). This is an estimate, not an
# exhaustive count -- validate with nvidia-smi dmon before trusting it blindly.
DEFAULT_FIELDS = 80
BYTES_PER_FLOAT64 = 8

# global_flexible's default domain spans 360 x 160 degrees.
DEFAULT_ASPECT_RATIO = 360 / 160


def query_gpu_memory_mb(gpu_index):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    total_mb, free_mb = (int(value) for value in output.strip().split(","))
    return total_mb, free_mb


def round_down_to_multiple(value, multiple):
    return max(multiple, (int(value) // multiple) * multiple)


def propose_grid(budget_cells, aspect_ratio, nz, xy_multiple):
    """Binary search for the largest ny (and nx = aspect_ratio * ny) such
    that (nx+4)(ny+4)nz fits the cell budget."""

    def cells_for(ny):
        nx = aspect_ratio * ny
        return (nx + 4) * (ny + 4) * nz

    lo, hi = 1, 100_000
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cells_for(mid) <= budget_cells:
            lo = mid
        else:
            hi = mid

    ny = round_down_to_multiple(lo, xy_multiple)
    nx = round_down_to_multiple(aspect_ratio * ny, xy_multiple)
    return nx, ny


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU to query with nvidia-smi")
    parser.add_argument("--free-mb", type=int, default=None, help="Override free GPU memory instead of querying it")
    parser.add_argument("--total-mb", type=int, default=None, help="Override total GPU memory (for display only)")
    parser.add_argument("--nz", type=int, default=15, help="Vertical levels to target (kept fixed while solving)")
    parser.add_argument("--aspect-ratio", type=float, default=DEFAULT_ASPECT_RATIO, help="Target nx/ny ratio")
    parser.add_argument("--fields", type=int, default=DEFAULT_FIELDS, help="Estimated independent 3D fields")
    parser.add_argument("--overhead-mb", type=int, default=600, help="Reserved for CUDA/XLA context and workspace")
    parser.add_argument("--safety-fraction", type=float, default=0.7, help="Fraction of remaining memory to use")
    parser.add_argument("--xy-multiple", type=int, default=2, help="Round nx/ny down to a multiple of this")
    return parser


def main():
    args = build_parser().parse_args()

    if args.free_mb is not None:
        free_mb = args.free_mb
        total_mb = args.total_mb
    else:
        total_mb, free_mb = query_gpu_memory_mb(args.gpu_index)

    usable_mb = max(0, free_mb - args.overhead_mb) * args.safety_fraction
    usable_bytes = usable_mb * 1024**2
    bytes_per_cell = args.fields * BYTES_PER_FLOAT64
    budget_cells = usable_bytes / bytes_per_cell

    nx, ny = propose_grid(budget_cells, args.aspect_ratio, args.nz, args.xy_multiple)
    estimated_bytes = (nx + 4) * (ny + 4) * args.nz * bytes_per_cell

    if total_mb is not None:
        print(f"GPU memory: {total_mb} MiB total, {free_mb} MiB free")
    else:
        print(f"GPU memory: {free_mb} MiB free (total unknown)")
    print(f"Reserved for CUDA/XLA context: {args.overhead_mb} MiB")
    print(f"Usable at {args.safety_fraction:.0%} safety margin: {usable_mb:.0f} MiB")
    print(f"Assumed footprint: ~{args.fields} fields x {BYTES_PER_FLOAT64} bytes = {bytes_per_cell} bytes/cell")
    print()
    print(f"Proposed grid: nx={nx} ny={ny} nz={args.nz}")
    print(f"Estimated resident state: {estimated_bytes / 1024**2:.0f} MiB")
    print()
    print("This is a rough estimate, not an exhaustive field count. Confirm with:")
    print("  nvidia-smi dmon -s um -d 1")


if __name__ == "__main__":
    main()
