#!/usr/bin/env python3
"""Plot timing and scaling summaries from cuda_scaling.py result files."""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


TOPOLOGY = re.compile(r"nodes(?P<nodes>\d+)-gpus(?P<gpus>\d+)")


def load_results(pattern):
    records = []
    for filename in sorted(glob.glob(pattern)):
        path = Path(filename)
        match = TOPOLOGY.search(str(path))
        if match is None:
            raise ValueError(f"cannot determine topology from {path}")
        payload = json.loads(path.read_text())
        for record in payload["results"]:
            record.update(
                mode=payload["mode"],
                nodes=int(match.group("nodes")),
                gpus_per_node=int(match.group("gpus")),
                total_gpus=int(match.group("nodes")) * int(match.group("gpus")),
            )
            records.append(record)
    return records


def median(records):
    values = sorted(record["elapsed_seconds"] for record in records)
    if not values:
        return float("nan")
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def plot_scaling(records, output):
    grouped = {}
    for record in records:
        key = (record["mode"], record["nodes"], record["gpus_per_node"])
        grouped.setdefault(key, []).append(record)

    times = {
        key: median(group)
        for key, group in grouped.items()
    }
    figures, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for mode, axis in zip(("strong", "weak"), axes[:2]):
        modes = sorted({nodes for this_mode, nodes, _ in times if this_mode == mode})
        if not modes:
            axis.set_visible(False)
            continue
        for nodes in modes:
            points = sorted(
                (nodes * gpus, time)
                for (this_mode, this_nodes, gpus), time in times.items()
                if this_mode == mode and this_nodes == nodes
            )
            gpus, values = zip(*points)
            axis.plot(gpus, values, "o-", label=f"{nodes} node(s)")
            if mode == "strong":
                baseline = values[0]
                axis.plot(gpus, [baseline * gpus[0] / gpu for gpu in gpus], "--", color="0.5")
        axis.set_ylabel("elapsed seconds")
        axis.set_xlabel("GPUs")
        axis.set_title(f"{mode.capitalize()} scaling")
        axis.grid(True, alpha=0.3)
        axis.legend()

    # GPU count alone hides node boundaries, so show node placement explicitly.
    for mode in ("strong", "weak"):
        points = sorted(
            (nodes * gpus, value)
            for (this_mode, nodes, gpus), value in times.items()
            if this_mode == mode
        )
        if points:
            axes[2].plot([point[0] for point in points], [point[1] for point in points], "o-", label=mode)
    axes[2].set_xlabel("GPUs")
    axes[2].set_ylabel("elapsed seconds")
    axes[2].set_title("Node placement")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figures.savefig(output, dpi=160)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="results/scaling/*/scaling.json")
    parser.add_argument("--output", default="results/scaling/plots/scaling.png")
    args = parser.parse_args()
    records = load_results(args.input_glob)
    if not records:
        raise SystemExit("no scaling results found")
    plot_scaling(records, args.output)


if __name__ == "__main__":
    main()
