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
    plt.close(figures)
    print(f"Wrote {output}")


def resource_series(records, value_key):
    """Aggregate corresponding samples across repetitions and GPUs."""
    buckets = {}
    for record in records:
        for gpu in record.get("gpu_memory_samples", []):
            for sample_index, sample in enumerate(gpu.get("samples", [])):
                if value_key in sample:
                    buckets.setdefault(sample_index, []).append(sample)

    series = []
    for sample_index in sorted(buckets):
        samples = buckets[sample_index]
        values = [sample[value_key] for sample in samples]
        series.append(
            {
                "elapsed_seconds": sum(sample["elapsed_seconds"] for sample in samples) / len(samples),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return series


def plot_resource_usage(records, output):
    grouped = {}
    for record in records:
        key = (record["mode"], record["nodes"], record["gpus_per_node"])
        grouped.setdefault(key, []).append(record)

    figure, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    metrics = (
        ("memory", "memory_used_mib", "GPU memory used (MiB / GPU)"),
        ("utilization", "compute_utilization_percent", "GPU compute utilization (%)"),
    )
    for row, mode in enumerate(("strong", "weak")):
        for column, (metric, value_key, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            plotted = False
            for (this_mode, nodes, gpus_per_node), group in sorted(grouped.items()):
                if this_mode != mode:
                    continue
                series = resource_series(group, value_key)
                if not series:
                    continue
                elapsed = [sample["elapsed_seconds"] for sample in series]
                mean = [sample["mean"] for sample in series]
                minimum = [sample["min"] for sample in series]
                maximum = [sample["max"] for sample in series]
                label = f"{nodes} node(s) x {gpus_per_node} GPU(s)"
                line = axis.plot(elapsed, mean, label=label)[0]
                axis.fill_between(elapsed, minimum, maximum, color=line.get_color(), alpha=0.12)
                plotted = True
            axis.set_title(f"{mode.capitalize()} - {metric}")
            axis.set_xlabel("elapsed seconds")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            if metric == "utilization":
                axis.set_ylim(0, 100)
            if plotted:
                axis.legend(fontsize="small")
            else:
                axis.text(0.5, 0.5, "No GPU samples", ha="center", va="center", transform=axis.transAxes)

        communication_axis = axes[row, 2]
        labels = []
        communication = []
        for (this_mode, nodes, gpus_per_node), group in sorted(grouped.items()):
            if this_mode != mode:
                continue
            values = [
                record["communication_fraction_percent"]
                for record in group
                if record.get("communication_fraction_percent") is not None
            ]
            if values:
                labels.append(f"{nodes}x{gpus_per_node}")
                communication.append(sum(values) / len(values))
        if labels:
            positions = list(range(len(labels)))
            communication_axis.plot(positions, communication, "o-")
            communication_axis.set_xticks(positions, labels, rotation=30, ha="right")
        else:
            communication_axis.text(
                0.5, 0.5, "No communication profiles", ha="center", va="center", transform=communication_axis.transAxes
            )
        communication_axis.set_ylim(0, 100)
        communication_axis.set_xlabel("nodes x GPUs per node")
        communication_axis.set_ylabel("main-loop time (%)")
        communication_axis.set_title(f"{mode.capitalize()} - halo communication")
        communication_axis.grid(True, alpha=0.3)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output}")


def plot_communication_compute_ratio(records, output):
    grouped = {}
    for record in records:
        key = (record["mode"], record["nodes"], record["gpus_per_node"])
        grouped.setdefault(key, []).append(record)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, mode in zip(axes, ("strong", "weak")):
        labels = []
        ratios = []
        for (this_mode, nodes, gpus_per_node), group in sorted(grouped.items()):
            if this_mode != mode:
                continue
            ratio_values = [
                record["communication_to_compute_ratio"]
                for record in group
                if record.get("communication_to_compute_ratio") is not None
            ]
            if not ratio_values:
                continue
            labels.append(f"{nodes}x{gpus_per_node}")
            ratios.append(sum(ratio_values) / len(ratio_values))

        if labels:
            positions = list(range(len(labels)))
            axis.plot(positions, ratios, "o-")
            axis.set_xticks(positions, labels, rotation=30, ha="right")
        else:
            axis.text(0.5, 0.5, "No communication profiles", ha="center", va="center", transform=axis.transAxes)
        axis.axhline(1.0, linestyle="--", color="0.5", label="communication = compute")
        axis.set_xlabel("nodes x GPUs per node")
        axis.set_ylabel("halo communication / compute time")
        axis.set_title(f"{mode.capitalize()} communication-to-compute ratio")
        axis.grid(True, alpha=0.3)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output}")


def write_summary(records, output):
    grouped = {}
    for record in records:
        key = (record["mode"], record["nodes"], record["gpus_per_node"])
        grouped.setdefault(key, []).append(record)

    lines = [
        "# CUDA scaling summary",
        "",
        "| Mode | Nodes | GPUs/node | Total GPUs | Median elapsed (s) | Peak memory/GPU (MiB) "
        "| Mean GPU utilization (%) | Halo communication (%) | Communication/compute |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (mode, nodes, gpus_per_node), group in sorted(grouped.items()):
        samples = [
            sample
            for record in group
            for gpu in record.get("gpu_memory_samples", [])
            for sample in gpu.get("samples", [])
        ]
        memory = [sample["memory_used_mib"] for sample in samples if "memory_used_mib" in sample]
        utilization = [
            sample["compute_utilization_percent"]
            for sample in samples
            if "compute_utilization_percent" in sample
        ]
        communication = [
            record["communication_fraction_percent"]
            for record in group
            if record.get("communication_fraction_percent") is not None
        ]
        ratios = [
            record["communication_to_compute_ratio"]
            for record in group
            if record.get("communication_to_compute_ratio") is not None
        ]

        peak_memory = f"{max(memory):.0f}" if memory else "N/A"
        mean_utilization = f"{sum(utilization) / len(utilization):.1f}" if utilization else "N/A"
        mean_communication = f"{sum(communication) / len(communication):.2f}" if communication else "N/A"
        mean_ratio = f"{sum(ratios) / len(ratios):.4f}" if ratios else "N/A"
        lines.append(
            f"| {mode} | {nodes} | {gpus_per_node} | {nodes * gpus_per_node} | {median(group):.3f} "
            f"| {peak_memory} | {mean_utilization} | {mean_communication} | {mean_ratio} |"
        )

    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- Elapsed time is the median wall-clock duration across repetitions.",
            "- Peak memory is the largest sampled per-GPU allocation.",
            "- GPU utilization is averaged across all five-second samples, GPUs, and repetitions.",
            "- Halo communication is synchronized boundary-exchange time as a percentage of main-loop time.",
            "- Communication/compute is halo-exchange time divided by estimated compute time.",
            "- Halo exchange does not include every MPI reduction or measure physical interconnect bytes.",
        ]
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="results/scaling/*/*/scaling.json")
    parser.add_argument("--output", default="results/scaling/plots/scaling.png")
    parser.add_argument("--resource-output", default="results/scaling/plots/resource_usage.png")
    parser.add_argument(
        "--communication-output", default="results/scaling/plots/communication_compute_ratio.png"
    )
    parser.add_argument("--summary-output", default="results/scaling/plots/summary.md")
    args = parser.parse_args()
    records = load_results(args.input_glob)
    if not records:
        raise SystemExit("no scaling results found")
    plot_scaling(records, args.output)
    plot_resource_usage(records, args.resource_output)
    plot_communication_compute_ratio(records, args.communication_output)
    write_summary(records, args.summary_output)


if __name__ == "__main__":
    main()
