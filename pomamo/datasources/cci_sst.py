import os.path
import os
import sys
import glob
import shutil
import argparse
import datetime
import threading
from math import floor
from typing import Optional

import numpy as np
import netCDF4

from . import access_data
from .interpolate import get_weights


class TimeSlice(object):
    def __init__(self, path):
        self.path = path
        self.nc = access_data.open_netcdf(path)
        for name, ncvar in self.nc.variables.items():
            setattr(self, name, ncvar)
        self.time_units = self.time.units
        self.time_calendar = getattr(self.time, "calendar", "gregorian")
        self.numtime = self.time[:]
        self.time = access_data.num2date(
            self.numtime,
            self.time_units,
            self.time_calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )


class CCI(object):
    def __init__(
        self,
        root="data",
        lock: Optional[threading.Lock] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        self.lock = lock or threading.Lock()
        self.root = os.path.join(root, "CCI-SST")
        self.slices = []
        with self.lock:
            print("Opening CCI-SST...")
            if start is not None and stop is not None:
                paths = [
                    os.path.join(self.root, f"{year}.nc")
                    for year in range(start, stop + 1)
                ]
            else:
                paths = glob.glob(os.path.join(self.root, "????.nc"))
            for path in sorted(paths):
                print(f"- {path}")
                self.slices.append(TimeSlice(path))
            nc = self.slices[0].nc
            lon, lat = nc["lon"], nc["lat"]
            self.delta_lng = lon[1] - lon[0]
            self.delta_lat = lat[1] - lat[0]
            self.lng_start = lon[0]
            self.lat_start = lat[0]
            self.nx, self.ny = lon.shape[0], lat.shape[0]

    def get(
        self,
        lat: float,
        lng: float,
        start: Optional[datetime.datetime] = None,
        stop: Optional[datetime.datetime] = None,
    ):
        assert lat >= -90 and lat <= 90
        ix = (lng - self.lng_start) / self.delta_lng
        iy = (lat - self.lat_start) / self.delta_lat
        ix_low = int(floor(ix)) % self.nx
        iy_low = min(int(floor(iy)), self.ny - 2)
        ix_high = (ix_low + 1) % self.nx
        iy_high = iy_low + 1
        w_11, w_12, w_21, w_22 = get_weights(ix % self.nx, iy, ix_low, iy_low)

        def ip(ncvar):
            fill_value = ncvar._FillValue
            v_11 = ncvar[iy_low, ix_low, :]
            v_12 = ncvar[iy_high, ix_low, :]
            v_21 = ncvar[iy_low, ix_high, :]
            v_22 = ncvar[iy_high, ix_high, :]
            w_11s = np.where(v_11 == fill_value, 0.0, w_11)
            w_12s = np.where(v_12 == fill_value, 0.0, w_12)
            w_21s = np.where(v_21 == fill_value, 0.0, w_21)
            w_22s = np.where(v_22 == fill_value, 0.0, w_22)
            w_tot = w_11s + w_12s + w_21s + w_22s
            mask = w_tot == 0.0
            w_tot[mask] = 1.0
            return np.ma.array(
                (w_11s * v_11 + w_12s * v_12 + w_21s * v_21 + w_22s * v_22) / w_tot,
                mask=mask,
            )

        times, values, sds = [], [], []
        with self.lock:
            for timeslice in self.slices:
                if start is not None and timeslice.time[-1] < start:
                    continue
                if stop is not None and timeslice.time[0] > stop:
                    continue
                times.extend(timeslice.time)
                values.append(ip(timeslice.analysed_sst))
                sds.append(ip(timeslice.analysed_sst_uncertainty))
        if len(values) > 0:
            values, sds = np.ma.concatenate(values), np.ma.concatenate(sds)
            values.data[...] *= 0.01
            sds.data[...] *= 0.01
        return times, values, sds


def download_cds(root: str, year: int, quiet: bool = False, restart: bool = False):
    import zipfile
    import cdsapi
    import io
    import json

    cds_settings = {}
    cds_keyfile = os.path.join(os.path.dirname(__file__), "keys/CDS API.txt")
    if os.path.isfile(cds_keyfile):
        with io.open(cds_keyfile, "r") as f:
            cds_settings = dict([l.rstrip("\n").split(": ") for l in f])
    c = cdsapi.Client(verify=1, quiet=quiet, **cds_settings)
    request = {
        "variable": "all",
        "format": "zip",
        "processinglevel": "level_4",
        "version": "2_1",
        "year": f"{year}",
        "month": [f"{m:02}" for m in range(1, 13)],
        "day": [f"{d:02}" for d in range(1, 32)],
        "sensor_on_satellite": "combined_product",
    }

    def get(months):
        request["month"] = [f"{m:02}" for m in months]
        path = os.path.join(root, f"{year}_{months[0]:02}-{months[-1]:02}.zip")

        print(f"Requesting {os.path.basename(path)}...", end="")
        r = c.retrieve("satellite-sea-surface-temperature", request)
        print("Done")

        print(f"{os.path.basename(path)} is ready - downloading...", end="")
        r.download(path)
        print("Done")

        print(f"Extracting {path} to {root}...", end="")
        with zipfile.ZipFile(path, "r") as zipf:
            zipf.extractall(root)
        print("Done")

        print(f"Deleting {path}...", end="")
        os.remove(path)
        print("Done")

    get(range(1, 7))
    get(range(7, 13))


def download_ceda(root: str, year: int, quiet: bool = False, restart: bool = False):
    for month in range(1, 13):
        lastday = (
            31
            if month == 12
            else (datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)).day
        )
        for day in range(1, lastday + 1):
            url = f"http://dap.ceda.ac.uk/thredds/fileServer/neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.1/{year:04d}/{month:02d}/{day:02d}/{year:04d}{month:02d}{day:02d}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc"
            url = f"ftp://anon-ftp.ceda.ac.uk/neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.1/{year:04d}/{month:02d}/{day:02d}/{year:04d}{month:02d}{day:02d}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc"
            url = f"https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/Analysis/L4/v3.0.1/{year:04d}/{month:02d}/{day:02d}/{year:04d}{month:02d}{day:02d}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR3.0-v02.0-fv01.0.nc?download=1"
            target = os.path.join(root, os.path.basename(url.split("?", 1)[0]))
            if restart and os.path.isfile(target):
                print(f"Skipping {target} (already exists)")
            else:
                access_data.download(url, target, f"CCI-SST {year}-{month:02}-{day:02}")


def transpose(sources, target):
    sources = sorted(glob.glob(sources))
    print(f"Writing {target}...")
    with netCDF4.Dataset(target, "w", format="NETCDF4") as nc:
        nc.createDimension("time", len(sources))
        nc.set_fill_off()
        print("  creating variables...")
        with netCDF4.Dataset(sources[0]) as ncin:
            ncin.set_auto_maskandscale(False)
            sdname = (
                "analysis_uncertainty"
                if "analysis_uncertainty" in ncin.variables
                else "analysed_sst_uncertainty"
            )
            nctime = access_data.copyNcVariable(
                ncin.variables["time"], nc, copy_data=False, name="time"
            )
            access_data.copyNcVariable(
                ncin.variables["lat"], nc, copy_data=True, name="lat"
            )
            access_data.copyNcVariable(
                ncin.variables["lon"], nc, copy_data=True, name="lon"
            )
            ncsst = access_data.copyNcVariable(
                ncin.variables["analysed_sst"],
                nc,
                copy_data=False,
                dimensions=("lat", "lon", "time"),
                name="analysed_sst",
            )
            ncse = access_data.copyNcVariable(
                ncin.variables[sdname],
                nc,
                copy_data=False,
                dimensions=("lat", "lon", "time"),
                name="analysed_sst_uncertainty",
            )
        print(f"  time: setting {len(sources)} values...")
        for i, source in enumerate(sources):
            with netCDF4.Dataset(source) as ncin:
                ncin.set_auto_maskandscale(False)
                nctime[i] = ncin.variables["time"][0]
        data = np.empty((len(sources), 1000, ncsst.shape[1]), dtype=ncsst.dtype)

        def get(ncvar, name):
            print(f"  {name}:")
            jmax = ncvar.shape[0]
            for j in range(0, jmax, 1000):
                jend = min(j + 1000, jmax)
                print(f"  - latitude {j}:{jend}")
                for itime, source in enumerate(sources):
                    with netCDF4.Dataset(source) as ncin:
                        ncin.set_auto_maskandscale(False)
                        data[itime, : jend - j, :] = ncin.variables[name][0, j:jend, :]
                ncvar[j:jend, :, :] = np.moveaxis(data[:, : jend - j, :], 0, -1)

        get(ncsst, "analysed_sst")
        get(ncse, sdname)
        for source in sources:
            os.remove(source)
    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data", default=access_data.data_root)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--year", type=int, action="append", default=[])
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--stop_year", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--source", choices=("ceda", "cds"))
    parser.add_argument("--export")
    parser.add_argument("--lon", type=float, default=-4.2166666)
    parser.add_argument("--lat", type=float, default=50.25)
    args = parser.parse_args()

    if args.download:
        if args.start_year is not None and args.stop_year is not None:
            args.year.extend(range(args.stop_year, args.start_year - 1, -1))
        root = os.path.join(args.data, "CCI-SST")
        if not os.path.isdir(root):
            os.makedirs(root)
        for year in args.year:
            yeardir = os.path.join(root, f"{year}_raw")
            if os.path.isdir(yeardir) and not args.restart:
                shutil.rmtree(yeardir)
            os.makedirs(yeardir, exist_ok=True)
            source = args.source or ("ceda" if year < 2017 else "cds")
            {"ceda": download_ceda, "cds": download_cds}[source](
                yeardir, year, args.quiet, args.restart
            )
            transpose(
                os.path.join(yeardir, f"{year}*fv01.0.nc"),
                os.path.join(root, f"{year}.nc"),
            )
            shutil.rmtree(yeardir)
            print("Done.")
        sys.exit(0)

    cci = CCI(args.data)
    time, sst, sst_se = cci.get(args.lat, args.lon)
    if args.export:
        with open(args.export, "w") as f:
            for tm, mu, se in zip(time, sst, sst_se):
                f.write(f"{tm:%Y-%m-%d %H:%M:%S}\t{mu:.3f}\t{se:.3f}\n")
    if args.plot:
        from matplotlib import pyplot

        fig = pyplot.figure()
        ax = fig.gca()
        ax.fill_between(time, sst - sst_se, sst + sst_se, alpha=0.5)
        ax.plot(time, sst)
        ax.grid(True)
        pyplot.show()
    print(sst)
