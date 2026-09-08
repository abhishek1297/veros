#!/usr/bin/env python3
"""Convert global_flexible's HDF5 forcing/topography assets to numpy .npz files.

Run this once on a machine with a working h5py/h5netcdf install:

    python tools/convert_global_flexible_assets.py \
        --forcing /path/to/forcing_1deg_global_interpolated.nc \
        --topography /path/to/ETOPO5_Ice_g_gmt4.nc

This writes a .npz file next to each source file. Copy those alongside the
originals on machines where h5py can't be installed; veros/setups/global_flexible
automatically falls back to them when h5netcdf cannot be imported.
"""

import argparse
import os

import h5netcdf
import numpy as np

FORCING_VARS = (
    "xt",
    "yt",
    "zt",
    "temperature",
    "salinity",
    "tau_x",
    "tau_y",
    "q_net",
    "dqdt",
    "swf",
    "sst",
    "sss",
    "tidal_energy",
)
TOPOGRAPHY_VARS = ("x", "y", "z")


def convert(source, variables, destination):
    with h5netcdf.File(source, "r") as dataset:
        arrays = {name: np.asarray(dataset.variables[name]) for name in variables}
    np.savez(destination, **arrays)
    print(f"Wrote {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forcing", required=True, help="Path to forcing_1deg_global_interpolated.nc")
    parser.add_argument("--topography", required=True, help="Path to ETOPO5_Ice_g_gmt4.nc")
    args = parser.parse_args()

    convert(args.forcing, FORCING_VARS, os.path.splitext(args.forcing)[0] + ".npz")
    convert(args.topography, TOPOGRAPHY_VARS, os.path.splitext(args.topography)[0] + ".npz")


if __name__ == "__main__":
    main()
