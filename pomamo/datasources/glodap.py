from __future__ import print_function

import sys
import os
from math import floor
import threading
import argparse
from typing import Optional

import numpy as np
import netCDF4

from . import access_data


class GLODAP(object):
    def __init__(self, root="data", lock: Optional[threading.Lock] = None):
        path = os.path.join(root, "GLODAPv2", "glodap.nc")
        print(f"Opening GLODAP ({path})...")
        self.lock = lock or threading.Lock()
        with self.lock:
            self.nc = access_data.open_netcdf(path)
            self.TCO2 = self.nc.variables["TCO2"]
            self.TAlk = self.nc.variables["TAlk"]
            self.depth = self.nc.variables["Depth"][:]
            self.lat_start = self.nc.variables["lat"][0]
            self.lng_start = self.nc.variables["lon"][0]
        self.ny, self.nx, self.nz = self.TCO2.shape

    def get(self, lat: float, lng: float):
        assert lat >= -90 and lat <= 90
        ix = (lng - self.lng_start) % 360.0
        iy = max(-89.5, min(89.5, lat)) - self.lat_start
        ix_low = int(floor(ix))
        iy_low = min(int(floor(iy)), self.ny - 2)
        ix_high = (ix_low + 1) % self.nx
        w_yhigh = iy - iy_low
        w_xhigh = ix - ix_low

        w_11 = np.empty((self.nz,))
        w_12 = np.empty((self.nz,))
        w_21 = np.empty((self.nz,))
        w_22 = np.empty((self.nz,))

        def interpolate(ncvar):
            # read annual mean
            with self.lock:
                fill_value = ncvar._FillValue
                v_11 = ncvar[iy_low, ix_low, :]
                v_12 = ncvar[iy_low, ix_high, :]
                v_21 = ncvar[iy_low + 1, ix_low, :]
                v_22 = ncvar[iy_low + 1, ix_high, :]

            # update weights based on masks of all four points
            w_11[:] = (1.0 - w_yhigh) * (1.0 - w_xhigh)
            w_12[:] = (1.0 - w_yhigh) * w_xhigh
            w_21[:] = w_yhigh * (1.0 - w_xhigh)
            w_22[:] = w_yhigh * w_xhigh
            w_11[v_11 == fill_value] = 0
            w_12[v_12 == fill_value] = 0
            w_21[v_21 == fill_value] = 0
            w_22[v_22 == fill_value] = 0
            w_tot = w_11 + w_12 + w_21 + w_22
            mask = w_tot == 0.0
            w_tot[mask] = 1.0
            w_11[:] /= w_tot
            w_12[:] /= w_tot
            w_21[:] /= w_tot
            w_22[:] /= w_tot

            # interpolate
            return np.ma.array(
                w_11 * v_11 + w_12 * v_12 + w_21 * v_21 + w_22 * v_22, mask=mask
            )

        return interpolate(self.TCO2), interpolate(self.TAlk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data", default=access_data.data_root)
    args = parser.parse_args()

    root = os.path.join(args.data, "GLODAPv2")
    if args.download:
        import tarfile
        import shutil
        import glob

        url = "https://www.nodc.noaa.gov/archive/arc0107/0162565/1.1/data/0-data/mapped/GLODAPv2.2016b_MappedClimatologies.tar.gz"
        if not os.path.isdir(root):
            os.mkdir(root)
        target = os.path.join(root, os.path.basename(url))
        access_data.download(url, target, "GLODAPv2")
        print("Extracting data...")
        with tarfile.open(target) as tar:
            tar.extractall(root)
        os.remove(target)
        print("Collecting and transposing data...")
        with netCDF4.Dataset(
            os.path.join(root, "glodap.nc"), "w", format="NETCDF4"
        ) as ncout:
            ncout.set_fill_off()
            for i, path in enumerate(
                glob.glob(os.path.join(root, "GLODAPv2.2016b_MappedClimatologies/*.nc"))
            ):
                varname = os.path.basename(path).rsplit(".", 2)[-2]
                print(f"  - {varname}...")
                with netCDF4.Dataset(path) as nc:
                    nc.set_auto_maskandscale(False)
                    if i == 0:
                        access_data.copyNcVariable(nc["Depth"], ncout)
                        access_data.copyNcVariable(nc["lon"], ncout)
                        access_data.copyNcVariable(nc["lat"], ncout)
                    access_data.copyNcVariable(
                        nc[varname],
                        ncout,
                        dimensions=("lat", "lon", "depth_surface"),
                        copy_data=False,
                    )[...] = np.moveaxis(nc[varname][...], 0, -1)
        shutil.rmtree(os.path.join(root, "GLODAPv2.2016b_MappedClimatologies"))
        sys.exit(0)

    test_locations = (
        ("L4", 50.25, -4.2166666),
        ("BATS", 31.6667, -64.1667),
        (
            "North pole 1",
            90,
            -90,
        ),  # note: North pole never returns the same value for all longitudes, because the max latitude in WOA is not 90, but 89.875
        ("North pole 2", 90, 0),
        ("North pole 3", 90, 90),
    )
    gd = GLODAP(root=args.data)
    print(f"Depths: {gd.depth}")
    for name, lat, lng in test_locations:
        TCO2, TAlk = gd.get(lat, lng)
        depths = gd.depth[np.logical_not(TCO2.mask)]
        if depths.size == 0:
            depths = ("None",)
        print(f"{name}: latitude = {lat}, longitude = {lng} (max depth: {depths[-1]})")
        print(f"  TCO2 range: {TCO2.min():.3f} - {TCO2.max():.3f}")
        print(f"  TAlk range: {TAlk.min():.5f} - {TAlk.max():.5f}")
