#!/usr/bin/env python3
"""Run the repository's Veros CLI inside an optional JAX/XProf trace."""

import os
import sys
from contextlib import nullcontext
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))


def main():
    from veros.cli.veros import cli

    trace_dir = os.environ.pop("VEROS_JAX_PROFILER_TRACE", None)
    if trace_dir:
        import jax

        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        print(f"Writing JAX/XProf trace to {trace_dir}", flush=True)
        trace_context = jax.profiler.trace(trace_dir, create_perfetto_trace=True)
    else:
        print("JAX/XProf tracing is disabled: VEROS_JAX_PROFILER_TRACE is not set", flush=True)
        trace_context = nullcontext()

    with trace_context:
        cli.main(args=sys.argv[1:], prog_name="veros", standalone_mode=False)

    if trace_dir:
        trace_files = [path for path in Path(trace_dir).rglob("*") if path.is_file()]
        if not trace_files:
            raise RuntimeError(f"JAX profiler completed but wrote no trace files to {trace_dir}")
        print(f"JAX/XProf trace completed at {trace_dir} ({len(trace_files)} files)", flush=True)


if __name__ == "__main__":
    main()
