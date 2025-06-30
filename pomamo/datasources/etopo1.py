from __future__ import print_function

import os
import sys
from math import floor
import threading
import argparse
from typing import Optional

from .interpolate import get_weights
from . import access_data

# Resolution 1/60th of a degree. start = -180, -90
# note the true length of x is one longer because -180 and 180 are featured.
# We just ignore the last
delta = 1.0 / 60
nx = 360 * 60


class ETOPO(object):
    def __init__(self, root: str = "data", lock: Optional[threading.Lock] = None):
        self.lock = lock or threading.Lock()
        with self.lock:
            self.nc = access_data.open_netcdf(
                os.path.join(root, "ETOPO1", "ETOPO1_Bed_g_gmt4.grd")
            )
            self.z = self.nc.variables["z"]
            self.ny = self.z.shape[0]

    def report(self):
        with self.lock:
            x = self.nc.variables["x"][...]
            y = self.nc.variables["y"][...]
        print(
            f"x: {x[0]} - {x[-1]}, {len(x)} elements, step = {(x[-1] - x[0]) / (len(x) - 1)}"
        )
        print(
            f"y: {y[0]} - {y[-1]}, {len(y)} elements, step = {(y[-1] - y[0]) / (len(y) - 1)}"
        )

    def get(self, lat: float, lng: float) -> float:
        assert lat >= -90 and lat <= 90
        ix = ((lng + 180) / delta) % nx
        iy = (lat + 90) / delta
        ixl, iyl = floor(ix), min(floor(iy), self.ny - 2)
        w11, w12, w21, w22 = get_weights(ix, iy, ixl, iyl)
        ixl, iyl = int(ixl), int(iyl)
        z = self.z
        with self.lock:
            z_11 = z[iyl, ixl]
            z_21 = z[iyl, ixl + 1]
            z_12 = z[iyl + 1, ixl]
            z_22 = z[iyl + 1, ixl + 1]
        return w11 * z_11 + w12 * z_12 + w21 * z_21 + w22 * z_22


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data", default=access_data.data_root)
    args = parser.parse_args()

    if args.download:
        url = "https://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/ETOPO1_Bed_g_gmt4.grd.gz"
        root = os.path.join(args.data, "ETOPO1")
        os.makedirs(root, exist_ok=True)
        target = os.path.join(root, os.path.basename(url))
        access_data.download(url, target, "ETOPO1 dataset")
        access_data.gunzip(target)
        sys.exit(0)

    etopo = ETOPO(args.data)
    etopo.report()
    assert etopo.get(90, -180) == etopo.get(90, 180), "Mismatch at top of seam (lat=90)"
    assert etopo.get(0, -180) == etopo.get(0, 180), "Mismatch at middle of seam (lat=0)"
    assert etopo.get(-90, -180) == etopo.get(
        -90, 180
    ), "Mismatch at bottom of seam (lat=-90)"
    assert (
        etopo.get(90, -90) == etopo.get(90, 0) == etopo.get(90, 90)
    ), "Mismatch at North pole"
    assert (
        etopo.get(-90, -90) == etopo.get(-90, 0) == etopo.get(-90, 90)
    ), "Mismatch at South pole"
    test_locations = (
        ("L4", 50.25, -4.2166666),
        ("BATS", 31.6667, -64.1667),
        ("North pole", 90, 0),
        ("South pole", -90, 0),
        ("Equator left", 0, -180),
        ("Equator right", 0, 180),
    )
    for name, lat, lng in test_locations:
        print(
            f"{name}: latitude = {lat}, longitude = {lng}, depth = {etopo.get(lat, lng)} m"
        )
