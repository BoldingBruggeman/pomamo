from __future__ import print_function

import sys
import os.path
from math import floor
import threading
import argparse

import numpy as np
import netCDF4

from . import access_data

DEFAULT_VERSION = "2013v2"
DEFAULT_VERSION = "2018"
NAME2LONG_NAME = {
    "t": "temperature",
    "s": "salinity",
    "n": "nitrate",
    "p": "phosphate",
    "i": "silicate",
    "o": "oxygen",
}


class WOA(object):
    def __init__(
        self, root="data", lock=None, bgc=False, version: str = DEFAULT_VERSION
    ):
        self.lock = lock or threading.Lock()
        self.variables = {}
        filename = "woa.nc" if not bgc else "woa_bgc.nc"
        path = os.path.join(root, f"WOA{version}", filename)
        print(f"Opening World Ocean Atlas {version} {'BGC' if bgc else ''} ({path})...")
        with self.lock:
            self.nc = access_data.open_netcdf(path)

            # Get coordinates
            self.depth = self.nc.variables["depth"][:]
            nclng = self.nc.variables["lon"]
            nclat = self.nc.variables["lat"]
            self.nx = nclng.shape[0]
            self.ny = nclat.shape[0]
            self.nz = self.depth.shape[0]
            self.lng_start = nclng[0]
            self.lat_start = nclat[0]
            self.delta = nclng[1] - self.lng_start
            assert self.lat_start < 0
            assert self.delta > 0
            assert nclat[1] - self.lat_start == self.delta

            # Get variables
            for name in self.nc.variables:
                if name.endswith("_an"):
                    ncvar = self.nc.variables[name]
                    assert ncvar.shape == (self.ny, self.nx, self.nz)
                    self.variables[name[:-3]] = (
                        ncvar,
                        self.nc.variables[name + "_monthly"],
                    )

    def report(self):
        print(f"nx = {self.nx}\nny = {self.ny}\nnz = {self.nz}")
        print(f"lng_start = {self.lng_start}")
        print(f"lat_start = {self.lat_start}")
        print(f"delta = {self.delta}")
        with self.lock:
            y = self.nc.variables["lat"][...]
            x = self.nc.variables["lon"][...]
        print(
            f"lon: {x[0]} - {x[-1]}, {len(x)} elements, step = {(x[-1] - x[0]) / (len(x) - 1)}"
        )
        print(
            f"lat: {y[0]} - {y[-1]}, {len(y)} elements, step = {(y[-1] - y[0]) / (len(y) - 1)}"
        )
        for name, (ncclim, ncmon) in self.variables.items():
            print(
                f"{name}: climatology shape {ncclim.shape}, monthly shape {ncmon.shape}"
            )

    def get(self, lat, lng, monthly=False, selection=("t", "s")):
        assert lat >= -90 and lat <= 90
        lat = max(self.lat_start, min(-self.lat_start, lat))
        ix = ((lng - self.lng_start) % 360.0) / self.delta
        iy = (lat - self.lat_start) / self.delta
        ix_low = int(floor(ix))
        iy_low = min(int(floor(iy)), self.ny - 2)
        ix_high = (ix_low + 1) % self.nx
        w_yhigh = iy - iy_low
        w_xhigh = ix - ix_low

        w_11 = np.empty((self.nz,))
        w_12 = np.empty((self.nz,))
        w_21 = np.empty((self.nz,))
        w_22 = np.empty((self.nz,))
        w_11[:] = (1.0 - w_yhigh) * (1.0 - w_xhigh)
        w_12[:] = (1.0 - w_yhigh) * w_xhigh
        w_21[:] = w_yhigh * (1.0 - w_xhigh)
        w_22[:] = w_yhigh * w_xhigh
        mask = None

        def interpolate(name):
            ncvar, ncvar_monthly = self.variables[name]

            # read annual mean
            with self.lock:
                fill_value = ncvar._FillValue
                v_11 = ncvar[iy_low, ix_low, :]
                v_12 = ncvar[iy_low, ix_high, :]
                v_21 = ncvar[iy_low + 1, ix_low, :]
                v_22 = ncvar[iy_low + 1, ix_high, :]

            # update weights based on masks of all four points
            nonlocal mask
            if mask is None:
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

            # read monthly data (upper 1500 m only) if requested
            if monthly:
                v_11 = v_11.reshape(1, -1).repeat(12, 0)
                v_12 = v_12.reshape(1, -1).repeat(12, 0)
                v_21 = v_21.reshape(1, -1).repeat(12, 0)
                v_22 = v_22.reshape(1, -1).repeat(12, 0)
                with self.lock:
                    n = ncvar_monthly.shape[-1]
                    v_11[:, :n] = ncvar_monthly[iy_low, ix_low, :, :]
                    v_12[:, :n] = ncvar_monthly[iy_low, ix_high, :, :]
                    v_21[:, :n] = ncvar_monthly[iy_low + 1, ix_low, :, :]
                    v_22[:, :n] = ncvar_monthly[iy_low + 1, ix_high, :, :]

            # interpolate
            return np.ma.array(
                w_11 * v_11 + w_12 * v_12 + w_21 * v_21 + w_22 * v_22,
                mask=np.broadcast_to(mask, v_11.shape),
            )

        return [interpolate(name) for name in selection]


def compare(res1, res2):
    if res1[0].count() == 0 and res2[0].count() == 0:
        return True
    return (res1[0] == res2[0]).all() and (res1[1] == res2[1]).all()


def write(root: str, name2source, target: str):
    print(
        "Extracting WOA temperature and salinity fields, transposing, and storing uncompressed..."
    )
    with netCDF4.Dataset(os.path.join(root, target), "w", format="NETCDF4") as ncout:
        ncout.set_fill_off()
        ncout.createDimension("time", 12)
        extra_dims = {}
        for isource, (name, source) in enumerate(name2source.items()):
            print(f"- {name}")
            print("  - mean climatology")
            with netCDF4.Dataset(os.path.join(root, source % 0)) as ncin:
                ncin.set_auto_maskandscale(False)
                if isource == 0:
                    access_data.copyNcVariable(ncin["depth"], ncout)
                    access_data.copyNcVariable(ncin["lon"], ncout)
                    access_data.copyNcVariable(ncin["lat"], ncout)
                ncvar = ncin[f"{name}_an"]
                access_data.copyNcVariable(
                    ncvar, ncout, dimensions=("lat", "lon", "depth"), copy_data=False
                )[...] = np.moveaxis(ncvar[0, ...], 0, -1)
            print("  - monthly data:")
            for imonth in range(1, 13):
                print(f"    - {imonth}")
                with netCDF4.Dataset(os.path.join(root, source % imonth)) as ncin:
                    ncin.set_auto_maskandscale(False)
                    ncvar = ncin[f"{name}_an"]
                    if imonth == 1:
                        if ncvar.shape[1] not in extra_dims:
                            dim = f"depth{len(extra_dims) + 1}"
                            ncout.createDimension(dim, ncvar.shape[1])
                            extra_dims[ncvar.shape[1]] = dim
                        ncvar_mo = access_data.copyNcVariable(
                            ncvar,
                            ncout,
                            dimensions=("lat", "lon", "time", dim),
                            copy_data=False,
                            name=f"{name}_an_monthly",
                        )
                    ncvar_mo[:, :, imonth - 1, :] = np.moveaxis(ncvar[0, ...], 0, -1)
            for imonth in range(13):
                os.remove(os.path.join(root, source % imonth))


def add_pt(root: str, target: str, postfix: str = "_an"):
    import pygsw

    print(f"Calculating potential temperature pt{postfix}...")
    with netCDF4.Dataset(os.path.join(root, target), "r+") as nc:
        nc.set_auto_maskandscale(False)
        nct = nc["t" + postfix]
        ncs = nc["s" + postfix]
        lon = nc["lon"][:]
        lat = nc["lat"][:]
        lon.shape = lon.shape + (1,) * (nct.ndim - 2)
        z = -nc["depth"][: nct.shape[-1]]
        print("- creating NetCDF variable to hold pt...")
        ncvar = access_data.copyNcVariable(
            nct, nc, copy_data=False, name="pt" + postfix
        )
        for ilat, la in enumerate(lat):
            print(f"- processing latitude {ilat + 1} of {lat.size}...")
            t = nct[ilat, ...]
            s = ncs[ilat, ...]
            pt = pygsw.calculate_pt(lon, la, z, t, s)
            pt[np.logical_or(t == nct._FillValue, s == ncs._FillValue)] = nct._FillValue
            ncvar[ilat, ...] = pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--generate_pt", action="store_true")
    parser.add_argument("--data", default=access_data.data_root)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()

    if args.download:
        if args.version == "2013v2":
            url = "https://data.nodc.noaa.gov/thredds/fileServer/woa/WOA13/DATAv2/{long_name}/netcdf/{period}/{resolution}/woa13_{period}_{name}{month}_{grid}}v2.nc"
        elif args.version == "2018":
            url = "https://data.nodc.noaa.gov/thredds/fileServer/ncei/woa/{long_name}/{period}/{resolution}/woa18_{period}_{name}{month}_{grid}.nc"
            url = "https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/{long_name}/{period}/{resolution}/woa18_{period}_{name}{month}_{grid}.nc"
            # url = 'https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA/{long_name}/netcdf/{period}/{resolution}/woa18_{period}_{name}{month}_{grid}.nc'
        else:
            print(f'Unknown WOA version "{args.version}" configured')
            sys.exit(1)
        root = os.path.join(args.data, f"WOA{args.version}")
        if not os.path.isdir(root):
            os.makedirs(root)
        name2source = {}
        for name, long_name in NAME2LONG_NAME.items():
            physics = name in ("t", "s")
            url_base = url.format(
                long_name=long_name,
                name=name,
                resolution="0.25" if physics else "1.00",
                period="decav" if physics else "all",
                grid="04" if physics else "01",
                month="%02i",
            )
            name2source[name] = os.path.basename(url_base)
            for imonth in range(13):
                current_url = url_base % imonth
                access_data.download(
                    current_url,
                    os.path.join(root, os.path.basename(current_url)),
                    f"WOA {args.version} {long_name} month {imonth}",
                )

        write(root, {k: v for k, v in name2source.items() if k in ("t", "s")}, "woa.nc")
        write(
            root,
            {k: v for k, v in name2source.items() if k not in ("t", "s")},
            "woa_bgc.nc",
        )

    if args.generate_pt:
        root = os.path.join(args.data, f"WOA{args.version}")
        add_pt(root, "woa.nc")
        add_pt(root, "woa.nc", postfix="_an_monthly")

    if args.download or args.generate_pt:
        sys.exit(0)

    woa = WOA(root=args.data)
    woa.report()
    woa_bgc = WOA(root=args.data, bgc=True)
    woa_bgc.report()
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
    assert compare(
        woa.get(90, -180), woa.get(90, 180)
    ), "Mismatch at top of seam (lat=90)"
    assert compare(
        woa.get(0, -180), woa.get(0, 180)
    ), "Mismatch at middle of seam (lat=0)"
    assert compare(
        woa.get(-90, -180), woa.get(-90, 180)
    ), "Mismatch at bottom of seam (lat=-90)"
    print(f"Depths: {woa.depth}")
    for name, lat, lng in test_locations:
        tprof, sprof = woa.get(lat, lng)
        print(f"{name}: latitude = {lat}, longitude = {lng}")
        print(f"  temperature range: {tprof.min():.3f} - {tprof.max():.3f}")
        print(f"  salinity range: {sprof.min():.5f} - {sprof.max():.5f}")
        selection = ("n", "p", "i", "o")
        bgc_profs = woa_bgc.get(lat, lng, selection=selection)
        for name, prof in zip(selection, bgc_profs):
            long_name = NAME2LONG_NAME[name]
            print(f"  {long_name} range: {prof.min():.5f} - {prof.max():.5f}")
    if args.plot:
        from matplotlib import pyplot

        fig = pyplot.figure()
        for irow, (name, lat, lng) in enumerate(test_locations):
            tprof, sprof = woa.get(lat, lng, monthly=True)
            ax = fig.add_subplot(len(test_locations), 2, (2 * irow) + 1)
            pc = ax.pcolormesh(tprof[:, ::-1].T)
            fig.colorbar(pc)
            ax = fig.add_subplot(len(test_locations), 2, (2 * irow) + 2)
            pc = ax.pcolormesh(sprof[:, ::-1].T)
            fig.colorbar(pc)
        pyplot.show()
