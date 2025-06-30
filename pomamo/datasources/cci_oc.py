from __future__ import print_function

import sys
import os
from math import floor
import threading
import argparse
import glob
import datetime
from typing import Optional, Sequence, Tuple

import numpy as np
import netCDF4

from . import access_data
from .interpolate import get_weights

POSTFIXES = {"kd_490": ""}


class TimeSlice(object):
    def __init__(self, path: str):
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
        root: str = "data",
        lock: Optional[threading.Lock] = None,
        variable: str = "kd_490",
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        postfix = POSTFIXES.get(variable, "_" + variable)
        self.lock = lock or threading.Lock()
        self.root = os.path.join(root, "CCI-OC")
        self.slices = []
        with self.lock:
            if start is not None and stop is not None:
                paths = [
                    os.path.join(self.root, f"{year}{postfix}.nc")
                    for year in range(start, stop + 1)
                ]
            else:
                path_exp = os.path.join(self.root, f"????{postfix}.nc")
                paths = glob.glob(path_exp)
                assert paths, f"No paths found matching {path_exp}"
            print(f"Opening CCI-OC ({variable})...")
            for path in sorted(paths):
                year = int(os.path.basename(path)[:4])
                if (start is None or year >= start) and (stop is None or year <= stop):
                    print(f"- {path}")
                    self.slices.append(TimeSlice(path))
            nc = self.slices[0].nc
            lon, lat = nc["lon"], nc["lat"]
            self.delta_lng = lon[1] - lon[0]
            self.delta_lat = lat[1] - lat[0]
            self.lng_start = lon[0]
            self.lat_start = lat[0]
            self.nx, self.ny = lon.shape[0], lat.shape[0]
        self.variable = variable

    def get(
        self,
        lat: float,
        lng: float,
        start: Optional[datetime.datetime] = None,
        stop: Optional[datetime.datetime] = None,
        variable: str = None,
    ) -> Tuple[Sequence[datetime.datetime], np.ma.MaskedArray]:
        if variable is None:
            variable = self.variable
        assert lat >= -90 and lat <= 90
        ix = (lng - self.lng_start) / self.delta_lng
        iy = (lat - self.lat_start) / self.delta_lat
        ix_low = int(floor(ix)) % self.nx
        iy_low = min(int(floor(iy)), self.ny - 2)
        ix_high = (ix_low + 1) % self.nx
        iy_high = iy_low + 1
        w_11_ref, w_12_ref, w_21_ref, w_22_ref = get_weights(
            ix % self.nx, iy, ix_low, iy_low, extrapolate_j=True
        )

        def ip(ncvar):
            fill_value = ncvar._FillValue
            v_11 = ncvar[iy_low, ix_low, :]
            v_12 = ncvar[iy_high, ix_low, :]
            v_21 = ncvar[iy_low, ix_high, :]
            v_22 = ncvar[iy_high, ix_high, :]
            w_11 = np.where(v_11 == fill_value, 0.0, w_11_ref)
            w_12 = np.where(v_12 == fill_value, 0.0, w_12_ref)
            w_21 = np.where(v_21 == fill_value, 0.0, w_21_ref)
            w_22 = np.where(v_22 == fill_value, 0.0, w_22_ref)
            w_tot = w_11 + w_12 + w_21 + w_22
            mask = w_tot == 0.0
            w_tot[mask] = 1.0
            return np.ma.array(
                (w_11 * v_11 + w_12 * v_12 + w_21 * v_21 + w_22 * v_22) / w_tot,
                mask=mask,
            )

        times, values = [], []
        with self.lock:
            for timeslice in self.slices:
                if start is not None and timeslice.time[-1] < start:
                    continue
                if stop is not None and timeslice.time[0] > stop:
                    continue
                times.extend(timeslice.time)
                values.append(ip(getattr(timeslice, variable)))
        if len(values) > 0:
            values = np.ma.concatenate(values)
        else:
            values = np.ma.MaskedArray(())
        return times, values

    def get_log10_stats(
        self,
        lat: float,
        lng: float,
        start: Optional[datetime.datetime] = None,
        stop: Optional[datetime.datetime] = None,
        variable: str = None,
    ) -> Tuple[Sequence[datetime.datetime], np.ma.MaskedArray, np.ma.MaskedArray]:
        if variable is None:
            variable = self.variable
        times, mean = self.get(lat, lng, start, stop, variable)
        _, bias = self.get(lat, lng, start, stop, variable + "_log10_bias")
        _, rmsd = self.get(lat, lng, start, stop, variable + "_log10_rmsd")

        # centred RMSD in log10 units, Eq 2.3 in UG (https://docs.pml.space/share/s/okB2fOuPT7Cj2r4C5sppDg)
        sigma_p = np.ma.sqrt(np.abs(rmsd**2 - bias**2))

        # log10 of unbiased chlorophyll product (mg m-3), Eq 2.5 in UG
        log10_m_p = np.ma.log10(mean) + bias

        # mean of log10-transformed bias-corrected chlorophyll, Eq 2.9 in UG
        mu_p = log10_m_p - 0.5 * np.log(10.0) * sigma_p**2

        return times, mu_p, sigma_p


def download_chlor_a(
    start: int,
    stop: int,
    postfix: str = "",
    restart: bool = False,
    version: str = "5.0",
):
    variable_names = ("chlor_a", "chlor_a_log10_rmsd", "chlor_a_log10_bias")
    for year in range(stop, start - 1, -1):
        ncout_path = os.path.join(root, f"{year}{postfix}.nc")
        with netCDF4.Dataset(ncout_path, "w", format="NETCDF4") as ncout:
            for month in range(1, 13):
                for day in range(1, 32):
                    url = f"ftp://oc-cci-data:ELaiWai8ae@ftp.rsg.pml.ac.uk/occci-v{version}/geographic/netcdf/daily/chlor_a/{year}/ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-{year}{month:02}{day:02}-fv{version}.nc"
                    target_path = os.path.join(root, os.path.basename(url))
                    if os.path.isfile(target_path) and restart:
                        print("Skipping {target_path} as it exists.")
                        continue
                    try:
                        access_data.download(
                            url,
                            target_path,
                            f"OceanColour CCI {year}-{month:02}-{day:02}",
                        )
                    except access_data.urllib_error.URLError as e:
                        if "550" not in str(e.reason):
                            raise
                        print(f"WARNING: {url} not found, skipping")

            print(f"Creating variables for {ncout_path}...")
            paths = os.path.join(
                root,
                f"ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-{year}????-fv{version}.nc",
            )
            with netCDF4.MFDataset(paths, aggdim="time") as nc:
                nc.set_auto_maskandscale(False)
                access_data.copyNcVariable(nc.variables["lon"], ncout)
                nlat = access_data.copyNcVariable(nc.variables["lat"], ncout).size
                ntime = access_data.copyNcVariable(
                    nc.variables["time"], ncout, name="time"
                ).size
                chunksizes = (5, 5, ntime)
                for name in variable_names:
                    access_data.copyNcVariable(
                        nc.variables[name],
                        ncout,
                        dimensions=("lat", "lon", "time"),
                        name=name,
                        copy_data=False,
                        chunksizes=chunksizes,
                        zlib=True,
                    )

            latchunk = 50
            for name in variable_names:
                for ilat in range(0, nlat, latchunk):
                    with netCDF4.MFDataset(paths, aggdim="time") as nc:
                        nc.set_auto_maskandscale(False)
                        ilastlat = min(ilat + latchunk, nlat)
                        print(f"Writing {name} for lat={ilat}:{ilastlat}...")
                        data = nc.variables[name][:, ilat:ilastlat, :]
                        ncout.variables[name][ilat:ilastlat, ...] = np.moveaxis(
                            data, 0, -1
                        )
            for path in glob.glob(paths):
                os.remove(path)


def download_kd_490(
    start: int,
    stop: int,
    postfix: str = "",
    restart: bool = False,
    version: str = "5.0",
):
    for year in range(start, stop + 1):
        for month in range(1, 13):
            url = f"ftp://oc-cci-data:ELaiWai8ae@ftp.rsg.pml.ac.uk/occci-v{version}/geographic/netcdf/monthly/kd/{year}/ESACCI-OC-L3S-K_490-MERGED-1M_MONTHLY_4km_GEO_PML_KD490_Lee-{year}{month:02}-fv{version}.nc"
            access_data.download(
                url,
                os.path.join(root, os.path.basename(url)),
                f"OceanColour CCI {year} month {month:02}",
            )
        paths = os.path.join(
            root,
            f"ESACCI-OC-L3S-K_490-MERGED-1M_MONTHLY_4km_GEO_PML_KD490_Lee-{year}??-fv{version}.nc",
        )
        with netCDF4.MFDataset(paths, aggdim="time") as nc, netCDF4.Dataset(
            os.path.join(root, f"{year}{postfix}.nc"), "w", format="NETCDF4"
        ) as ncout:
            nc.set_auto_maskandscale(False)
            access_data.copyNcVariable(nc.variables["lon"], ncout)
            access_data.copyNcVariable(nc.variables["lat"], ncout)
            access_data.copyNcVariable(nc.variables["time"], ncout, name="time")
            for name in ("kd_490",):
                data = nc.variables[name][...]
                var = access_data.copyNcVariable(
                    nc.variables[name],
                    ncout,
                    dimensions=("lat", "lon", "time"),
                    name=name,
                    copy_data=False,
                )
                var[...] = np.moveaxis(data, 0, -1)
        for path in glob.glob(paths):
            os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--variable", choices=("kd_490", "chlor_a"), default="kd_490")
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--stop_year", type=int, default=None)
    parser.add_argument("--data", default=access_data.data_root)
    parser.add_argument("--lon", type=float, default=-4.2166666)
    parser.add_argument("--lat", type=float, default=50.25)
    parser.add_argument("--version", default="5.0")
    parser.add_argument("--export")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    root = os.path.join(args.data, "CCI-OC")
    if args.download:
        assert (
            args.start_year is not None and args.stop_year is not None
        ), "--start_year and --stop_year must be provided if --download is specified"
        postfix = POSTFIXES.get(args.variable, "_" + args.variable)
        os.makedirs(root, exist_ok=True)
        {"kd_490": download_kd_490, "chlor_a": download_chlor_a}[args.variable](
            args.start_year, args.stop_year, postfix, args.restart, args.version
        )
        sys.exit(0)

    cci = CCI(
        root=args.data,
        variable=args.variable,
        start=args.start_year,
        stop=args.stop_year,
    )
    time, mean = cci.get(args.lat, args.lon)
    if args.variable == "chlor_a":
        _, bias = cci.get(args.lat, args.lon, variable="chlor_a_log10_bias")
        _, rmsd = cci.get(args.lat, args.lon, variable="chlor_a_log10_rmsd")
        sigma_p = np.ma.sqrt(
            np.abs(rmsd**2 - bias**2)
        )  # centred RMSD in log10 units, Eq 2.3 in UG (https://docs.pml.space/share/s/okB2fOuPT7Cj2r4C5sppDg)
        log10_m_p = (
            np.ma.log10(mean) + bias
        )  # log10 of unbiased chlorophyll product (mg m-3), Eq 2.5 in UG
        mu_p = (
            log10_m_p - 0.5 * np.log(10.0) * sigma_p**2
        )  # mean of log10-transformed bias-corrected chlorophyll, Eq 2.9 in UG
        mean = mu_p

    print(
        f"({args.lon}, {args.lat}, {time[0]:%Y-%m-%d} - {time[-1]:%Y-%m-%d}): {args.variable} = {mean.mean():.3f} ({mean.min():.3f} - {mean.max():.3f})"
    )
    if args.export:
        assert (
            args.variable == "chlor_a"
        ), f"Export is currently only supported for chlor_a (not {args.variable})"
        _, se = cci.get(args.lat, args.lon, variable="chlor_a_log10_rmsd")
        with open(args.export, "w") as f:
            for tm, mu, se in zip(time, mean, se):
                if not np.ma.getmask(mu):
                    f.write(f"{tm:%Y-%m-%d %H:%M:%S}\t{mu:.3f}\t{se:.3f}\n")
