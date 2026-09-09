#!/usr/bin/env python3
"""Run one MPI rank and periodically record its node-local GPU usage."""

import argparse
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def query_gpu(gpu_index):
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    index, uuid, used, total, utilization = (item.strip() for item in completed.stdout.strip().split(","))
    return {
        "gpu_index": int(index),
        "gpu_uuid": uuid,
        "memory_used_mib": int(used),
        "memory_total_mib": int(total),
        "compute_utilization_percent": int(utilization),
    }


def gpu_for_local_rank(local_rank):
    visible_devices = [
        item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()
    ]
    if len(visible_devices) == 1:
        return visible_devices[0]
    if local_rank < len(visible_devices):
        return visible_devices[local_rank]
    return local_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--rank-env", default="OMPI_COMM_WORLD_RANK")
    parser.add_argument("--local-rank-env", default="OMPI_COMM_WORLD_LOCAL_RANK")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.command or args.command[0] != "--":
        raise SystemExit("the program command must follow --")
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")

    rank = int(os.environ.get(args.rank_env, 0))
    local_rank = int(os.environ.get(args.local_rank_env, 0))
    gpu_index = gpu_for_local_rank(local_rank)
    process = subprocess.Popen(args.command[1:], env=os.environ.copy())
    start = time.perf_counter()
    samples = []
    sampling_errors = []

    while process.poll() is None:
        sample_time = time.perf_counter()
        try:
            sample = query_gpu(gpu_index)
            sample.update(
                elapsed_seconds=round(sample_time - start, 3),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            samples.append(sample)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
            sampling_errors.append(
                {
                    "elapsed_seconds": round(sample_time - start, 3),
                    "error": str(exc),
                }
            )

        try:
            process.wait(timeout=args.interval)
        except subprocess.TimeoutExpired:
            pass

    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "samples": samples,
    }
    if sampling_errors:
        payload["sampling_errors"] = sampling_errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
