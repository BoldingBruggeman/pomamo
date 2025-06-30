from __future__ import print_function

import os
from math import floor
import threading
from typing import Optional

from . import access_data
from .interpolate import get_weights

DEFAULT_VERSION = 2023


class GEBCO(object):
    def __init__(
        self,
        root: str = "data",
        lock: Optional[threading.Lock] = None,
        version: int = DEFAULT_VERSION,
    ):
        path = os.path.join(root, "GEBCO", f"GEBCO_{version}.nc")
        print(f"Opening GEBCO ({path})...")
        self.lock = lock or threading.Lock()
        with self.lock:
            self.nc = access_data.open_netcdf(path)
            self.z = self.nc.variables["elevation"]
            self.ny, self.nx = self.z.shape
            self.x0 = self.nc.variables["lon"][0]
            self.y0 = self.nc.variables["lat"][0]
            self.delta_x = self.nc.variables["lon"][1] - self.x0
            self.delta_y = self.nc.variables["lat"][1] - self.y0

    def report(self):
        with self.lock:
            x = self.nc.variables["lon"][...]
            y = self.nc.variables["lat"][...]
        print(f"x: {self.x0} - {x[-1]}, {self.nx} elements, step = {self.delta_x}")
        print(f"y: {self.y0} - {y[-1]}, {self.ny} elements, step = {self.delta_y}")

    def get(self, lat: float, lng: float) -> float:
        assert lat >= -90 and lat <= 90
        ix = ((lng - self.x0) / self.delta_x) % self.nx
        iy = (lat - self.y0) / self.delta_y
        ixl, iyl = floor(ix), max(0, min(floor(iy), self.ny - 2))
        w11, w12, w21, w22 = get_weights(ix, iy, ixl, iyl, extrapolate_j=True)
        ixl, iyl = int(ixl), int(iyl)
        ixh = (ixl + 1) % self.nx
        z = self.z
        with self.lock:
            z_11 = z[iyl, ixl]
            z_21 = z[iyl, ixh]
            z_12 = z[iyl + 1, ixl]
            z_22 = z[iyl + 1, ixh]
        return w11 * z_11 + w12 * z_12 + w21 * z_21 + w22 * z_22


if __name__ == "__main__":
    import argparse
    import sys
    import zipfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data", default=access_data.data_root)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    args = parser.parse_args()

    if args.download:
        url = {
            2019: "https://www.bodc.ac.uk/data/open_download/gebco/GEBCO_15SEC/zip/",
            2020: "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2020/zip/",
            2021: "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2021/zip/",
            2022: "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2022/zip/",
            2023: "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023/zip/",
        }[args.version]
        root = os.path.join(args.data, "GEBCO")
        if not os.path.isdir(root):
            os.makedirs(root)
        path = os.path.join(root, f"GEBCO_{args.version}.zip")
        access_data.download(url, path, f"GEBCO {args.version} dataset")
        print(f"Extracting {os.path.basename(path)}...")
        with zipfile.ZipFile(path, "r") as zipf:
            for name in zipf.namelist():
                if name.endswith(".nc"):
                    print(f"  {name}: writing to {root}")
                    zipf.extract(name, root)
                else:
                    print(f"  {name}: skipping")
        os.remove(path)
        sys.exit(0)

    gebco = GEBCO(root=args.data, version=args.version)
    gebco.report()
    test_locations = (
        ("L4", 50.25, -4.2166666),
        ("BATS", 31.6667, -64.1667),
        ("North pole 1", 90, 0),
        ("North pole 2", 90, 180),
        ("South pole 1", -90, 0),
        ("South pole 2", -90, 180),
        ("Equator left", 0, -180),
        ("Equator right", 0, 180),
    )
    for name, lat, lng in test_locations:
        print(
            f"{name}: latitude = {lat}, longitude = {lng}, depth = {gebco.get(lat, lng)} m"
        )
