from __future__ import print_function

import os
import time
from typing import Optional
import gzip

try:
    from urllib import request as urllib_request
    from urllib import error as urllib_error
except ImportError:
    import urllib as urllib_request
    import urllib as urllib_error

import netCDF4
from netCDF4 import num2date, date2num

data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


class download(object):
    def __init__(self, url: str, target: str, title: str, nretries: int = 10):
        print(f"Downloading {title} from {url}...")
        done = False
        while not done:
            self.last = 0.0
            try:
                urllib_request.urlretrieve(
                    url, filename=target, reporthook=self.printStatus
                )
                done = True
            except urllib_error.ContentTooShortError:
                if nretries == 0:
                    raise
                print("ContentTooShortError - retrying after 30 seconds")
                nretries -= 1
                time.sleep(30.0)
        print("Done")
        urllib_request.urlcleanup()

    def printStatus(self, blockcount: int, blocksize: int, totalsize: int):
        current = float(blockcount * blocksize) / totalsize
        if current - self.last > 0.1:
            print(
                f"  downloaded {1e-6 * blockcount * blocksize:.1f} MB ({100 * current:.1f} %)"
            )
            self.last = current


def gunzip(
    source: str, target: Optional[str] = None, blocksize: Optional[int] = 1 << 16
):
    if target is None:
        assert source.endswith(".gz")
        target = source[:-3]
    print(f"Decompressing {source} to {target}...")
    with gzip.open(source, "rb") as f_in, open(target, "wb") as f_out:
        while 1:
            block = f_in.read(blocksize)
            if not block:
                break
            f_out.write(block)


def copyNcVariable(
    ncvar,
    nctarget,
    dimensions=None,
    copy_data=True,
    chunksizes=None,
    name=None,
    zlib=False,
):
    if name is None:
        name = ncvar.name
    if dimensions is None:
        dimensions = ncvar.dimensions
    for dim in dimensions:
        if dim not in nctarget.dimensions:
            length = ncvar.shape[ncvar.dimensions.index(dim)]
            nctarget.createDimension(dim, length)
    fill_value = None if not hasattr(ncvar, "_FillValue") else ncvar._FillValue
    ncvarnew = nctarget.createVariable(
        name,
        ncvar.dtype,
        dimensions,
        fill_value=fill_value,
        chunksizes=chunksizes,
        zlib=zlib,
    )
    ncvarnew.setncatts(
        {att: getattr(ncvar, att) for att in ncvar.ncattrs() if att != "_FillValue"}
    )
    ncvarnew.set_auto_maskandscale(False)
    if copy_data:
        ncvarnew[...] = ncvar[...]
    return ncvarnew


s3 = None


def open_netcdf(path: str) -> netCDF4.Dataset:
    if path.startswith("s3://"):
        import h5netcdf.legacyapi
        import s3fs

        path = path[5:].replace("\\", "/")

        global s3
        if s3 is None:
            client_kwargs = {}
            if "AWS_S3_ENDPOINT" in os.environ:
                client_kwargs["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
            s3 = s3fs.S3FileSystem(
                client_kwargs=client_kwargs,
                default_block_size=int(0.1 * 2**20),
                default_fill_cache=False,
                anon=True,
                use_ssl=False
            )
        f = s3.open(path)
        return h5netcdf.legacyapi.Dataset(f)
    else:
        nc = netCDF4.Dataset(path)
        nc.set_auto_maskandscale(False)
        return nc
