from __future__ import print_function

import os
import sys
from math import floor
import threading
import argparse
import glob
import io
import datetime
from typing import Optional, Tuple, Iterable, Sequence
import tempfile
import logging

import numpy as np
import netCDF4

from . import access_data
from .interpolate import get_weights

DEFAULT_SELECTION = ("t2m", "u10", "v10", "d2m", "sp", "tcc", "tp", "ssr")


class ERASlice(object):
    def __init__(self, paths: Sequence[str]):
        self.path = paths[0]
        self.ncs = []
        self.scale_factors = {}
        self.offsets = {}
        for path in paths:
            nc = access_data.open_netcdf(path)
            for name, ncvar in nc.variables.items():
                if hasattr(self, name):
                    assert np.all(
                        getattr(self, name)[...] == ncvar[...]
                    ), f"{path}: values for {name} do not match those from {previous_path}"
                else:
                    setattr(self, name, ncvar)
                    self.scale_factors[name] = getattr(ncvar, "scale_factor", 1.0)
                    self.offsets[name] = getattr(ncvar, "add_offset", 0.0)
                    previous_path = path
            self.ncs.append(nc)
        self.time_units = self.time.units
        if self.time_units.endswith(" 00:00:0.0"):
            self.time_units = self.time_units[:-2] + "0.0"
        self.time_calendar = self.time.calendar
        self.numtime = self.time[:]
        self.numactime = self.numtime - 0.5 * (
            self.numtime[1] - self.numtime[0]
        )  # put accumulation time (e.g., for precipitation) at center of accumulation time window
        assert self.time_units.startswith("hours since")
        self.time = access_data.num2date(
            self.numtime, self.time_units, self.time_calendar
        )
        self.actime = access_data.num2date(
            self.numactime, self.time_units, self.time_calendar
        )
        self.numstart, self.numstop = self.numtime[0], self.numactime[-1]
        self.overlap_with_next = 0


class ERA(object):
    def __init__(
        self,
        root="data",
        lock: Optional[threading.Lock] = None,
        version: str = "ERA-interim",
        start: Optional[int] = None,
        stop: Optional[int] = None,
        selection: Iterable[str] = DEFAULT_SELECTION,
    ):
        self.lock = lock or threading.Lock()
        self.files = []
        root = os.path.join(root, version)
        with self.lock:
            self.time_units = None
            self.time_calendar = None
            prefix = "era5_t2m_" if version == "ERA5" else ""
            if start is not None and stop is not None:
                paths = [
                    os.path.join(root, f"{prefix}{year}.nc")
                    for year in range(start, stop + 1)
                ]
            else:
                paths = glob.glob(os.path.join(root, f"{prefix}????.nc"))
            assert len(paths) > 0, f"No meteo files found at {root}."
            for path in paths:
                if "t2m" in path:
                    # one file per variable
                    all_var_paths = [path.replace("t2m", name) for name in selection]
                else:
                    # one file with all variables
                    all_var_paths = [path]
                file = ERASlice(all_var_paths)
                self.files.append(file)
                assert (
                    self.time_units is None or self.time_units == file.time_units
                ), f"Time units mismatch: {file.time_units} from {path}, but {self.time_units} from {last_path}"
                assert (
                    self.time_calendar is None
                    or self.time_calendar == file.time_calendar
                ), f"Time calendar mismatch: {file.time_calendar} from {path}, but {self.time_calendar} from {last_path}"
                self.time_units, self.time_calendar, last_path = (
                    file.time_units,
                    file.time_calendar,
                    path,
                )
            self.ny, self.nx = file.t2m.shape[:2]
            self.delta_lng = file.longitude[1] - file.longitude[0]
            self.delta_lat = file.latitude[1] - file.latitude[0]
            self.lng_start = file.longitude[0]
            self.lat_start = file.latitude[0]
        self.files.sort(key=lambda file: file.numstart)
        for i, file in enumerate(self.files[:-1]):
            next_file = self.files[i + 1]
            while next_file.numtime[file.overlap_with_next] <= file.numtime[-1]:
                file.overlap_with_next += 1
            if file.overlap_with_next == 0:
                print(
                    f"No overlap between ERA files {file.path} (ends at {file.numtime[-1]}) and {next_file.path} (starts at {next_file.numtime[0]})"
                )
            # assert file.overlap_with_next > 0
        print(f"Found the following ERA sources in {root}:")
        for file in self.files:
            print(
                f"- {os.path.basename(file.path)}: {file.time[0]} - {file.time[-1]} (overlap with next: {file.overlap_with_next} points)"
            )
        self.numstarts = np.array([file.numstart for file in self.files])
        self.numstops = np.array([file.numstop for file in self.files])
        self.interval = file.numtime[1] - file.numtime[0]

    def report(self):
        with self.lock:
            y = self.files[0].latitude[...]
            x = self.files[0].longitude[...]
        print(
            f"lon: {x[0]} - {x[-1]}, {len(x)} elements, step = {(x[-1] - x[0]) / (len(x) - 1)}"
        )
        print(
            f"lat: {y[0]} - {y[-1]}, {len(y)} elements, step = {(y[-1] - y[0]) / (len(y) - 1)}"
        )
        # print(f"T shape: {self.t2m.shape}")

    def get(
        self,
        lat: float,
        lng: float,
        start: Optional[datetime.datetime] = None,
        stop: Optional[datetime.datetime] = None,
        selection: Tuple[str, ...] = DEFAULT_SELECTION,
    ):
        assert lat >= -90 and lat <= 90
        lng = lng % 360
        ix = (lng - self.lng_start) / self.delta_lng
        iy = (lat - self.lat_start) / self.delta_lat
        ix_low = int(floor(ix))
        iy_low = min(int(floor(iy)), self.ny - 2)
        ix_high = (ix_low + 1) % self.nx
        w11, w12, w21, w22 = get_weights(ix, iy, ix_low, iy_low)

        def ip(ncvar, itime_start: int) -> np.ndarray:
            with self.lock:
                v_11 = ncvar[iy_low, ix_low, itime_start:]
                v_12 = ncvar[iy_low + 1, ix_low, itime_start:]
                v_21 = ncvar[iy_low, ix_high, itime_start:]
                v_22 = ncvar[iy_low + 1, ix_high, itime_start:]
            return w11 * v_11 + w12 * v_12 + w21 * v_21 + w22 * v_22

        # Select subset of time slice files to use
        ifirst, ilast, numstart = 0, len(self.files) - 1, 0
        if start is not None:
            numstart = access_data.date2num(start, self.time_units, self.time_calendar)
            if numstart < self.numstarts[0]:
                data_start = access_data.num2date(
                    self.numstarts[0], self.time_units, self.time_calendar
                )
                raise Exception(
                    f"Requested start time {start} precedes start of ERA time series ({data_start})"
                )
            ifirst = self.numstarts.searchsorted(numstart, side="right") - 1
        if stop is not None:
            numstop = access_data.date2num(stop, self.time_units, self.time_calendar)
            if numstop > self.numstops[-1]:
                data_stop = access_data.num2date(
                    self.numstops[-1], self.time_units, self.time_calendar
                )
                raise Exception(
                    f"Requested stop time {stop} lies beyond end of ERA time series ({data_stop})"
                )
            ilast = self.numstops.searchsorted(numstop, side="left")

        # Read data
        itime_start = int((numstart - self.numstarts[ifirst]) // self.interval)
        result = {name: [] for name in selection}
        times, actimes = [], []
        for file in self.files[ifirst : ilast + 1]:
            for name, values in result.items():
                values_ip = ip(getattr(file, name), itime_start)
                values.append(values_ip * file.scale_factors[name] + file.offsets[name])
            times.extend(file.time[itime_start:])
            actimes.extend(file.actime[itime_start:])
            itime_start = file.overlap_with_next
        result = {name: np.concatenate(values) for name, values in result.items()}
        result["time"] = times
        result["actime"] = actimes
        return result


def compare(reference, *results):
    names_to_compare = frozenset(reference).difference(("time", "actime"))
    return all(
        [
            all([(reference[name] == result[name]).all() for name in names_to_compare])
            for result in results
        ]
    )


def download_era5_year(year: int, variable: str, path: str, quiet: bool = False):
    import cdsapi

    cds_settings = {}
    cds_keyfile = os.path.join(os.path.dirname(__file__), "keys/CDS API.txt")
    if os.path.isfile(cds_keyfile):
        with io.open(cds_keyfile, "r") as f:
            cds_settings = dict([l.rstrip("\n").split(": ") for l in f])
    c = cdsapi.Client(verify=1, quiet=quiet, **cds_settings)
    request = {
        "variable": [variable],
        "product_type": "reanalysis",
        "format": "netcdf",
        "year": f"{year}",
        "month": [f"{m:02}" for m in range(1, 13)],
        "day": [f"{d:02}" for d in range(1, 32)],
        "time": [f"{h:02}:00" for h in range(0, 24)],
        "grid": ["0.25/0.25"],
    }
    r = c.retrieve("reanalysis-era5-single-levels", request)
    return (r, path)


def transpose_era5(source: str, target: str, n: int = 10):
    with netCDF4.Dataset(source) as ncin, netCDF4.Dataset(
        target, "w", format="NETCDF4"
    ) as ncout:
        ncin.set_auto_maskandscale(False)
        ncout.set_fill_off()
        print("Copying coordinates...")
        access_data.copyNcVariable(ncin["time"], ncout)
        access_data.copyNcVariable(ncin["longitude"], ncout)
        access_data.copyNcVariable(ncin["latitude"], ncout)
        dims = ("time", "latitude", "longitude")
        for ncvar in ncin.variables.values():
            if ncvar.dimensions == dims:
                print(f"Creating {ncvar.name}...")
                ncvarout = access_data.copyNcVariable(
                    ncvar, ncout, dimensions=dims[1:] + dims[:1], copy_data=False
                )
                print("Copying data...")
                for j in range(ncvar.shape[1]):
                    if j % n == 0:
                        print(f"  reading j = {j}:{j + n} (of {ncvar.shape[1]})")
                        block = ncvar[:, j : j + n, :]
                    print(f"  writing j = {j} (of {ncvar.shape[1]})")
                    ncvarout[j, :, :] = block[:, j % n, :].T


def download_era_interim_year(year: int, era_root: str):
    assert year >= 1979
    start = (
        datetime.datetime(year - 1, 12, 31)
        if year > 1979
        else datetime.datetime(year, 1, 1)
    )
    stop = datetime.datetime(year + 1, 1, 1)
    analysis_path = os.path.join(era_root, f"analysis_{year}.nc")
    forecast_path = os.path.join(era_root, f"forecast_{year}.nc")

    import ecmwfapi
    import json

    with io.open(
        os.path.join(os.path.dirname(__file__), "keys/ECMWF API.txt"), "r"
    ) as f:
        era_settings = json.load(f)
    ecmwf_server = ecmwfapi.ECMWFDataServer(**era_settings)

    def download_from_ecmwf(
        variable_id: Iterable[str],
        path: str,
        start: datetime.datetime,
        stop: datetime.datetime,
        step: str = "0",
        time: str = "00/06/12/18",
        forecast: bool = False,
    ):
        ecmwf_server.retrieve(
            {
                "stream": "oper",
                "levtype": "sfc",
                "param": "/".join(variable_id),
                "dataset": "interim",
                "step": step,
                "grid": "0.75/0.75",
                "time": f"{time}",
                "date": f"{start:%Y-%m-%d}/to/{stop:%Y-%m-%d}",
                #'area'      : '73.5/-27/33/45',
                "type": "fc" if forecast else "an",
                "class": "ei",
                "format": "netcdf",
                "target": path,
            }
        )

    # See also https://software.ecmwf.int/wiki/pages/viewpage.action?pageId=56658233
    print(f"Downloading ERA-interim data for {year}...")
    analysis_variables = {
        "sp": "134.128",
        "tcc": "164.128",
        "u10": "165.128",
        "v10": "166.128",
        "t2m": "167.128",
        "d2m": "168.128",
    }
    forecast_variables = {
        "tp": "228.128",
        "ssr": "176.128",
    }
    download_from_ecmwf(analysis_variables.values(), analysis_path, start, stop)
    download_from_ecmwf(
        forecast_variables.values(),
        forecast_path,
        start,
        stop,
        step="6/12",
        time="00/12",
        forecast=True,
    )

    print(f"Transposing and gathering ERA-interim data for {year}...")
    with netCDF4.Dataset(
        os.path.join(era_root, f"{year}.nc"), "w", format="NETCDF4"
    ) as ncera:
        with netCDF4.Dataset(analysis_path) as nc:
            nc.set_auto_maskandscale(False)
            access_data.copyNcVariable(nc["time"], ncera)
            access_data.copyNcVariable(nc["longitude"], ncera)
            access_data.copyNcVariable(nc["latitude"], ncera)
            for key in analysis_variables:
                access_data.copyNcVariable(
                    nc[key],
                    ncera,
                    dimensions=("latitude", "longitude", "time"),
                    copy_data=False,
                )[...] = np.moveaxis(nc[key][...], 0, -1)
        with netCDF4.Dataset(forecast_path) as nc:
            nc.set_auto_maskandscale(False)
            for key in forecast_variables:
                ncvar = nc[key]
                data = np.array(ncvar[...], dtype=float)
                data[1::2, ...] = (
                    np.maximum(data[1::2, ...] - data[0::2, ...], 0.0)
                    - ncvar.add_offset / ncvar.scale_factor
                )  # every second time point includes accumulation over previous 6 hours - subtract this
                data = data.round().astype(ncvar.dtype)
                access_data.copyNcVariable(
                    ncvar,
                    ncera,
                    dimensions=("latitude", "longitude", "time"),
                    copy_data=False,
                )[...] = np.moveaxis(data, 0, -1)
    os.remove(analysis_path)
    os.remove(forecast_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("source", nargs="?")
    parser.add_argument("target", nargs="?")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--version", choices=("ERA-interim", "ERA5"), default="ERA5")
    parser.add_argument("--start_year", type=int)
    parser.add_argument("--stop_year", type=int)
    parser.add_argument("--variables", default="*")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--data", default=access_data.data_root)
    parser.add_argument("--endpoint_url", default=None)
    parser.add_argument("--temp_dir", default=None)
    args = parser.parse_args()
    if args.download:
        if args.data.startswith("s3://"):
            import s3fs

            client_kwargs = {}
            if args.endpoint_url is not None:
                client_kwargs["endpoint_url"] = args.endpoint_url
            bucket = args.data[5:] + "/" + args.version
            print(f"Opening connection to S3 {bucket}, client_kwargs={client_kwargs}")
            fs = s3fs.S3FileSystem(client_kwargs=client_kwargs)
            fs.makedirs(bucket, exist_ok=True)
            root = tempfile.mkdtemp(dir=args.temp_dir)
        else:
            root = os.path.join(args.data, args.version)
            os.makedirs(root, exist_ok=True)
            fs = None
        if args.version == "ERA-interim":
            for year in range(args.start_year, args.stop_year + 1):
                download_era_interim_year(year, root)
            sys.exit(0)
        import multiprocessing

        if args.start_year is None or args.stop_year is None:
            print("With --download, --start_year and --stop_year must be provided too.")
            sys.exit(2)
        era_variables = {
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "t2m": "2m_temperature",
            "d2m": "2m_dewpoint_temperature",
            "sp": "surface_pressure",
            "tcc": "total_cloud_cover",
            "tp": "total_precipitation",
            "ssr": "surface_net_solar_radiation",
        }
        if args.variables == "*":
            selected_variables = sorted(era_variables.keys())
        else:
            selected_variables = args.variables.split(",")
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        pool = multiprocessing.Pool(
            processes=(args.stop_year - args.start_year + 1) * len(selected_variables)
        )
        results = []
        for key in selected_variables:
            for year in range(args.start_year, args.stop_year + 1):
                path = os.path.join(root, "raw_era5_{}_{}.nc".format(key, year))
                print(f"Queuing download of {path}...")
                results.append(
                    pool.apply_async(
                        download_era5_year,
                        args=(year, era_variables[key], path, args.quiet),
                    )
                )
        for res in results:
            r, path = res.get()
            print(f"Data for {os.path.basename(path)} ready - downloading...")
            r.download(path)
            if args.transpose:
                final_path = os.path.join(root, os.path.basename(path)[4:])
                transpose_era5(path, final_path, args.n)
                os.remove(path)
                path = final_path
            if fs is not None:
                print(f"Copying {os.path.basename(path)} to S3 bucket {bucket}...")
                fs.put_file(path, bucket + "/" + os.path.basename(path))
                os.remove(path)
        sys.exit(0)
    if args.transpose:
        assert args.source is not None and args.target is not None
        transpose_era5(args.source, args.target, args.n)
        sys.exit(0)

    met = ERA(root=args.data, version=args.version)
    met.report()
    assert compare(met.get(90, 0), met.get(90, 360)), "Mismatch at top of seam (lat=90)"
    assert compare(met.get(0, 0), met.get(0, 360)), "Mismatch at middle of seam (lat=0)"
    assert compare(
        met.get(-90, 0), met.get(-90, 360)
    ), "Mismatch at bottom of seam (lat=-90)"
    # assert compare(met.get(90, -90), met.get(90, 0), met.get(90, 90)), 'Mismatch at North pole'
    # assert compare(met.get(-90, -90), met.get(-90, 0), met.get(-90, 90)), 'Mismatch at South pole'
    test_locations = (
        ("L4", 50.25, -4.2166666),
        ("BATS", 31.6667, -64.1667),
        (
            "North Pole 1",
            90,
            -90,
        ),  # Note: wind speeds at North pole at different longitudes mismatch in ERA
        ("North Pole 2", 90, 0),
        ("North Pole 3", 90, 90),
        (
            "South Pole 1",
            -90,
            -90,
        ),  # Note: wind speeds at South pole at different longitudes mismatch in ERA
        ("South Pole 2", -90, 0),
        ("South Pole 3", -90, 90),
    )
    for name, lat, lng in test_locations:
        result = met.get(lat, lng)
        wind_speed = np.ma.sqrt(result["u10"] ** 2 + result["v10"] ** 2)
        print(f"{name}: latitude = {lat}, longitude = {lng}")
        print(
            "  air temperature: mean = %.2f, range = %.2f - %.2f K"
            % (result["t2m"].mean(), result["t2m"].min(), result["t2m"].max())
        )
        print(
            "  wind speed: mean = %.2f, range = %.2f - %.2f m s-1"
            % (wind_speed.mean(), wind_speed.min(), wind_speed.max())
        )
        print(
            "  dew point temperature: mean = %.2f, range = %.2f - %.2f K"
            % (result["d2m"].mean(), result["d2m"].min(), result["d2m"].max())
        )
        print(
            "  surface pressure: mean = %.2f, range = %.2f - %.2f Pa"
            % (result["sp"].mean(), result["sp"].min(), result["sp"].max())
        )
        print(
            "  total cloud cover: mean = %.2f, range = %.2f - %.2f"
            % (result["tcc"].mean(), result["tcc"].min(), result["tcc"].max())
        )
        print(
            "  precipitation: annual mean = %.2f m"
            % (result["tp"].mean() * 24 * 365.25)
        )
        print(
            "  net shortwave solar radiaton: annual mean = %.1f W/m2"
            % (result["ssr"].mean() / 3600.0 / 6.0)
        )
