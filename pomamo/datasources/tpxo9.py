from __future__ import print_function
import os
import argparse
from math import floor
import threading
import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import netCDF4

import otps2

from . import access_data
from .interpolate import get_weights

url_template = (
    "ftp://ftp.oce.orst.edu/dist/tides/TPXO9_atlas_nc/%s_%s_tpxo9_atlas_30.nc"
)
grid_url = "ftp://ftp.oce.orst.edu/dist/tides/TPXO9_atlas_nc/grid_tpxo9_atlas.nc"
variables = ("h", "u")
COMPONENTS = ("m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1", "m4", "ms4", "mn4", "2n2")


class Result(object):
    def __init__(self, components: Dict[str, Tuple[float, float]], lat: float):
        self.components = components
        self.lat = lat

    def predict(self, start_time: datetime.datetime, n: int, step: float):
        h = otps2.predict_tide(self.components, self.lat, start_time, n, step)
        return h


class TPXO(object):
    def __init__(self, root: str = "data", lock: Optional[threading.Lock] = None):
        self.lock = lock or threading.Lock()
        self.root = os.path.join(root, "TPXO9")
        print(f"Loading TPXO9 from {self.root}")
        filename = "grid_tpxo9_atlas.nc"
        print(f"  - bathymetry ({filename})")
        self.ncs = {}
        self.variables = {}
        with self.lock:
            path = os.path.join(self.root, filename)
            self.ncs[path] = nc = access_data.open_netcdf(path)
            lon, lat = nc["lon_z"], nc["lat_z"]
            self.delta_lng = lon[1] - lon[0]
            self.delta_lat = lat[1] - lat[0]
            self.lng_start = lon[0]
            self.lat_start = lat[0]
            self.nx, self.ny = lon.shape[0], lat.shape[0]
            self.ncvar_hz = nc["hz"]

            for variable in ("u", "v", "h"):
                collection = {"v": "u"}.get(variable, variable)
                self.variables[variable] = []
                for component in COMPONENTS:
                    filename = f"{collection}_{component}_tpxo9_atlas_30.nc"
                    path = os.path.join(self.root, filename)
                    print(f"  - {variable}:{component} ({filename})")
                    self.ncs[path] = nc = access_data.open_netcdf(path)
                    ncvar_re, ncvar_im = (
                        nc.variables[variable + "Re"],
                        nc.variables[variable + "Im"],
                    )
                    self.variables[variable].append((ncvar_re, ncvar_im))

    def get(self, lat: float, lng: float, variable: str = "h"):
        assert lat >= -90 and lat <= 90
        lng = lng % 360
        ix = (lng - self.lng_start) / self.delta_lng
        iy = (lat - self.lat_start) / self.delta_lat
        ix_low = int(floor(ix)) % self.nx
        iy_low = min(int(floor(iy)), self.ny - 2)
        ix_high = (ix_low + 1) % self.nx
        iy_high = iy_low + 1
        with self.lock:
            ncvar = self.ncvar_hz
            mask = (
                np.array(
                    (
                        ncvar[ix_low, iy_low],
                        ncvar[ix_low, iy_high],
                        ncvar[ix_high, iy_low],
                        ncvar[ix_high, iy_high],
                    )
                )
                == 0.0
            )
        if mask.all():
            return
        w_11, w_12, w_21, w_22 = get_weights(
            ix % self.nx, iy, ix_low, iy_low, mask=mask
        )

        def ip(ncvar):
            v_11 = ncvar[ix_low, iy_low]
            v_12 = ncvar[ix_low, iy_high]
            v_21 = ncvar[ix_high, iy_low]
            v_22 = ncvar[ix_high, iy_high]
            return w_11 * v_11 + w_12 * v_12 + w_21 * v_21 + w_22 * v_22

        components = {}
        for component, (ncvar_re, ncvar_im) in zip(
            COMPONENTS, self.variables[variable]
        ):
            with self.lock:
                components[component] = ip(ncvar_re), ip(ncvar_im)

        return Result(components, lat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data", default=access_data.data_root)
    parser.add_argument("--lon", type=float, help="longitude", default=-4 - 10.0 / 60.0)
    parser.add_argument("--lat", type=float, help="latitude", default=50 + 22.0 / 60.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", help="path to store tidal constituents")
    args = parser.parse_args()

    root = os.path.join(args.data, "TPXO9")
    if args.download:
        if not os.path.isdir(root):
            os.makedirs(root)
        for variable in variables:
            for component in COMPONENTS:
                url = url_template % (variable, component)
                target_path = os.path.join(root, os.path.basename(url))
                access_data.download(url, target_path, f"TPXO9 {variable}: {component}")
        target_path = os.path.join(root, os.path.basename(grid_url))
        access_data.download(grid_url, target_path, "TPXO9 grid")
    if args.plot:
        with netCDF4.Dataset(
            os.path.join(root, "h_m2_tpxo9_atlas_30.nc")
        ) as nc, netCDF4.Dataset(os.path.join(root, "grid_tpxo9_atlas.nc")) as ncbath:
            lon = nc["lon_z"][:]
            lat = nc["lat_z"][:]
            delta = 3
            istart, istop = np.searchsorted(lon, (args.lon - delta, args.lon + delta))
            jstart, jstop = np.searchsorted(lat, (args.lat - delta, args.lat + delta))
            dat = ncbath["hz"][istart:istop, jstart:jstop]
            lon = lon[istart:istop]
            lat = lat[jstart:jstop]
            from matplotlib import pyplot

            fig = pyplot.figure()
            ax = fig.gca()
            print(istart, istop, jstart, jstop)
            # cf = ax.contourf(lon, lat, dat.T, levels=20)
            cf = ax.contourf(
                lon, lat, -dat.T, levels=np.linspace(-100, 0, 10), extend="min"
            )
            fig.colorbar(cf)
            pyplot.show()
    tpxo = TPXO()
    start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    times = np.arange(30 * 24 * 6) / (6 * 24)
    print("Extracting components...")
    result = tpxo.get(args.lat, args.lon)
    if args.save:
        print(f"Saving components in json format to {args.save}...")
        import json

        with open(args.save, "w") as f:
            json.dump(result.components, f)
    print("Calculating elevation time series...")
    h = result.predict(start, times.size, 600)
    from matplotlib import pyplot, dates

    fig = pyplot.figure()
    ax = fig.gca()
    dt = dates.date2num(start) + times
    ax.plot_date(dt, h, "-")
    ax.grid(True)
    ax.set_xlabel("time")
    ax.set_ylabel("elevation (mm)")
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.set_title(f"Location: {args.lat} N, {args.lon} E")
    pyplot.show()
