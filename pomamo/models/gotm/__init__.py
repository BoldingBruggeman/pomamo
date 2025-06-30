import os
import shutil
import datetime
from typing import Optional, Iterable
import logging
import time
import enum
import threading
from ruamel import yaml

import numpy as np
import jinja2

from datasources import woa, glodap, era, tpxo9, cci_sst, cci_oc

simulation_root = os.path.join(os.path.dirname(__file__), "simulations")
template_root = os.path.join(os.path.dirname(__file__), "gotm-setup-template")

default_settings = {
    "relax": 1.0e15,
    "dt": 1800,
    "h0b": 0.005,
    "k_min": 1e-6,
    "meteo_source": "ERA5",
    "swr_method": "calculate",
    "light_extinction_method": "Jerlov-I",
    "bgc": "off",
    "cci_sst": "off",
    "cci_chl": "off",
}

env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_root))
gotm_templates = {}
for name in os.listdir(template_root):
    if not name.endswith(".nc"):
        gotm_templates[name] = env.get_template(name)


def initialize_databases(
    root: str,
    netcdf_lock: threading.Lock,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    era_interim: bool = False,
    era5: bool = True,
):
    global meteo_ERA5, meteo_ERA_interim, woa_dataset, woa_bgc_dataset, glodap_dataset, tpxo, cci, cci_oc_Kd_dataset, cci_oc_chl_dataset
    meteo_ERA_interim = None
    meteo_ERA5 = None
    if era_interim:
        meteo_ERA_interim = era.ERA(root, lock=netcdf_lock, start=start, stop=stop)
    if era5:
        meteo_ERA5 = era.ERA(
            root, lock=netcdf_lock, version="ERA5", start=start, stop=stop
        )
    woa_dataset = woa.WOA(root, lock=netcdf_lock)
    woa_bgc_dataset = woa.WOA(root, lock=netcdf_lock, bgc=True)
    glodap_dataset = glodap.GLODAP(root, lock=netcdf_lock)
    tpxo = tpxo9.TPXO(root, lock=netcdf_lock)
    cci = cci_sst.CCI(root, lock=netcdf_lock, start=start, stop=stop)
    cci_oc_Kd_dataset = cci_oc.CCI(root, lock=netcdf_lock, start=start, stop=stop)
    cci_oc_chl_dataset = cci_oc.CCI(
        root, lock=netcdf_lock, variable="chlor_a", start=start, stop=stop
    )


class ProgressReporter:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

    def warn(self, source: str, message: str):
        if self.logger:
            self.logger.warning("%s: %s" % (source, message))

    def error(self, source: str, message: str):
        if self.logger:
            self.logger.error("%s failed: %s" % (source, message))

    def log(self, message: str):
        if self.logger:
            self.logger.info(message)

    def set_status(self, message: str):
        pass

    def set_progress(self, progress: float):
        pass


class Task:
    def __init__(
        self,
        reporter: ProgressReporter,
        path: str = "",
        message: Optional[str] = None,
        offset: float = 0.0,
        fraction: float = 1.0,
        parent: Optional["Task"] = None,
    ):
        self.reporter = reporter
        self.path = path
        self.message = message or path
        self.offset = offset
        self.fraction = fraction
        self.current = offset
        self.parent = parent

    def subtask(self, path: str, message: str, fraction: float) -> "Task":
        return Task(
            self.reporter,
            self.path + "/" + path,
            message,
            offset=self.current,
            fraction=self.fraction * fraction,
            parent=self,
        )

    def __enter__(self):
        self.reporter.set_status(self.message)
        self.start_time = time.perf_counter()
        return self

    def progress(self, fraction: float):
        assert fraction >= 0.0 and fraction <= 1.0
        self.current = self.offset + fraction * self.fraction
        self.reporter.set_progress(self.current)

    def warn(self, message: str):
        self.reporter.warn(self.path, message)

    def __exit__(self, type, value, traceback):
        if type is None:
            self.reporter.log(
                "completed %s (%.3f s)"
                % (self.path, time.perf_counter() - self.start_time)
            )
        else:
            self.reporter.error(self.path, str(value))
        self.progress(1.0)
        if self.parent is not None:
            self.parent.current += self.fraction
        self.reporter = None


def getClimatology(start: datetime.datetime, stop: datetime.datetime, values):
    year, month = start.year, start.month - 1
    if month == 0:
        year, month = year - 1, 12
    last_year, last_month = stop.year, stop.month + 1
    if last_month == 13:
        last_year, last_month = last_year + 1, 1
    while 1:
        yield datetime.datetime(year, month, 15), values[month - 1, :]
        if year == last_year and month == last_month:
            return
        month += 1
        if month == 13:
            year, month = year + 1, 1


def writeProfiles(
    path: str,
    start: datetime.datetime,
    stop: datetime.datetime,
    depth,
    values,
    default: float,
    fmt: str = "%s",
    maxdepth: float = 12000,
):
    assert depth[0] < depth[1]
    masked = np.flatnonzero(
        np.any(np.ma.getmaskarray(values), axis=tuple(range(values.ndim - 1)))
    )
    if masked.size == 0:
        # All depths unmasked
        zs = -depth[:]
    elif masked[0] > 0:
        # At least one depth is unmasked
        n = masked[0]
        zs, values = -depth[:n], values[..., :n]
    else:
        # All depths are masked
        zs, values = np.zeros((1,)), np.array((default,))
    if values.ndim == 1:
        profiles = ((start, values),)
    else:
        profiles = getClimatology(start, stop, values)
    with open(path, "w") as f_prof:
        for dt, prof in profiles:
            f_prof.write(
                "%s\t%i\t2\n" % (dt.strftime("%Y-%m-%d %H:%M:%S"), zs.size + 1)
            )
            for z, value in zip(zs, prof):
                f_prof.write("%s\t%s\n" % (z, fmt % value))
            f_prof.write("%s\t%s\n" % (-maxdepth, fmt % value))
    return masked.size == 0 or masked[0] > 0


def prepare_forcing(
    root_task: Task,
    simulation_dir: str,
    lat: float,
    lng: float,
    depth: float,
    start: datetime.datetime,
    stop: datetime.datetime,
    settings: dict,
    grid_generator,
):
    if not os.path.isdir(simulation_dir):
        os.makedirs(simulation_dir)

    with root_task.subtask(
        "meteorology", "preparing meteorological forcing", 0.3
    ) as meteo_task:
        with meteo_task.subtask("read", "reading meteorology", 0.25):
            met = {"ERA-interim": meteo_ERA_interim, "ERA5": meteo_ERA5}[
                settings["meteo_source"]
            ]
            meteo_data = met.get(lat, lng, start=start, stop=stop)
            meteo_data["t2m"] -= 273.15  # from Kelvin to degrees Celsius
            meteo_data["d2m"] -= 273.15  # from Kelvin to degrees Celsius
            meteo_data["sp"] /= 100  # from Pascal to mbar
            meteo_data["tp"] /= (
                met.interval * 3600
            )  # from water accumulated in one ERA interval to precipition rate in m/s
            np.clip(meteo_data["tcc"], 0.0, 1.0, out=meteo_data["tcc"])
        with meteo_task.subtask("write", "writing meteorology forcing files", 0.75):
            with open(os.path.join(simulation_dir, "meteo.dat"), "w") as f:
                for t, t2m, u10, v10, d2m, sp, tcc in zip(
                    meteo_data["time"],
                    meteo_data["t2m"],
                    meteo_data["u10"],
                    meteo_data["v10"],
                    meteo_data["d2m"],
                    meteo_data["sp"],
                    meteo_data["tcc"],
                ):
                    values = ["%.6g" % v for v in (u10, v10, sp, t2m, d2m, tcc)]
                    f.write(
                        "%s\t%s\n"
                        % (t.strftime("%Y-%m-%d %H:%M:%S"), "\t".join(values))
                    )
            with open(os.path.join(simulation_dir, "precip.dat"), "w") as f:
                for t, tp in zip(meteo_data["actime"], meteo_data["tp"].clip(min=0)):
                    f.write("%s\t%.6g\n" % (t.strftime("%Y-%m-%d %H:%M:%S"), tp))
            if settings["meteo_source"] == "ERA5":
                meteo_data["ssr"] /= (
                    met.interval * 3600
                )  # from shortwave radiation accumulated in one ERA interval to W/m2
                with open(os.path.join(simulation_dir, "ssr.dat"), "w") as f:
                    for t, ssr in zip(
                        meteo_data["actime"], meteo_data["ssr"].clip(min=0)
                    ):
                        f.write("%s\t%.6g\n" % (t.strftime("%Y-%m-%d %H:%M:%S"), ssr))

    with root_task.subtask("tides", "preparing tidal forcing", 0.55) as tide_task:
        with tide_task.subtask("read", "reading tidal constituents", 0.1):
            step = settings["dt"]
            dlat, dlon = 1 / 600.0, 1 / 600.0
            if lat + dlat > 90:
                dlat = -dlat
            tide = tpxo.get(lat, lng)
            tide_plat = tpxo.get(lat + dlat, lng)
            tide_plon = tpxo.get(lat, lng + dlon)
        if tide is None or tide_plat is None or tide_plon is None:
            settings["ext_pressure_method"] = "constant"
            tide_task.warn(
                'The <a href="https://www.tpxo.net/global/tpxo9-atlas" target="_blank">TPXO9-atlas</a> does not contain tidal information for this location. Assuming no tidal influence.'
            )
        else:
            settings["ext_pressure_method"] = "file"
            n = 2 + int((stop - start).total_seconds()) // step
            with tide_task.subtask(
                "predict", "predicting tidal elevation", 0.6
            ) as task:
                h = 0.001 * tide.predict(start, n, step)
                task.progress(0.33)
                h_plat = 0.001 * tide_plat.predict(start, n, step)
                task.progress(0.67)
                h_plng = 0.001 * tide_plon.predict(start, n, step)
                deg2rad = np.pi / 180.0
                earth_radius = 6371e3  # radius of the earth in m
                circumference = 2 * np.pi * earth_radius
                circumference_at_lat = circumference * np.cos(
                    lat * deg2rad
                )  # circumference in East-West direction at specified latitude. Assumes the earth is a perfect sphere.
                dy = dlat / 360.0 * circumference
                dx = dlon / 360.0 * circumference_at_lat
                dh_dy = (h_plat - h) / dy
                dh_dx = (h_plng - h) / dx
            with tide_task.subtask("write", "writing tidal forcing files", 0.3):
                with open(os.path.join(simulation_dir, "zeta.dat"), "w") as fzeta, open(
                    os.path.join(simulation_dir, "ext_press.dat"), "w"
                ) as fpres:
                    for i, (zeta, dzeta_dx, dzeta_dy) in enumerate(
                        zip(h, dh_dx, dh_dy)
                    ):
                        timestring = (
                            start + datetime.timedelta(seconds=i * step)
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        fzeta.write("%s\t%.4f\n" % (timestring, zeta))
                        fpres.write(
                            "%s\t0\t%.3e\t%.3e\n" % (timestring, dzeta_dx, dzeta_dy)
                        )

    with root_task.subtask(
        "temperature and salinity", "preparing temperature and salinity profiles", 0.025
    ) as ts_task:
        tprof, sprof = woa_dataset.get(lat, lng, monthly=True, selection=("pt", "s"))
        min_airt = max(0.0, meteo_data["t2m"].min())
        if not writeProfiles(
            os.path.join(simulation_dir, "sprof.dat"),
            start,
            stop,
            woa_dataset.depth,
            sprof,
            0.0,
            fmt="%.5f",
        ):
            ts_task.warn(
                'The <a href="https://www.nodc.noaa.gov/OC5/woa18/" target="_blank">World Ocean Atlas</a> does not contain salinity profiles for this location. We will treat this as a freshwater basin and initialize the water column with a salinity of 0.'
            )
        if not writeProfiles(
            os.path.join(simulation_dir, "tprof.dat"),
            start,
            stop,
            woa_dataset.depth,
            tprof,
            min_airt,
            fmt="%.3f",
        ):
            ts_task.warn(
                'The <a href="https://www.nodc.noaa.gov/OC5/woa18/" target="_blank">World Ocean Atlas</a> does not contain temperature profiles for this location. We will initialize the water column with a temperature of %.1f \u00b0C (the minimum air temperature, or 0 \u00b0C if that is higher).'
                % min_airt
            )

    bgc = settings.get("bgc", "off")
    if bgc != "off":
        # World Ocean Atlas nutrient profiles
        # Note: World Ocean Database 2018 has units umol kg-1, obtained from umol L-1 by dividing by 1.025 kg L-1 (https://data.nodc.noaa.gov/woa/WOD/DOC/wod_intro.pdf, section 1.1.10)
        # Here we save profiles in umol L-1, so we multiply with 1.025
        with root_task.subtask(
            "biogeochemistry", "preparing biogeochemical profiles", 0.025
        ) as bgc_task:
            selection = "n", "p", "i", "o"
            profs = woa_bgc_dataset.get(lat, lng, monthly=True, selection=selection)
            for name, prof in zip(selection, profs):
                long_name = woa.NAME2LONG_NAME[name]
                default = 0.0
                if not writeProfiles(
                    os.path.join(simulation_dir, "%s.dat" % long_name),
                    start,
                    stop,
                    woa_bgc_dataset.depth,
                    prof * 1.025,
                    default,
                    fmt="%.3f",
                ):
                    bgc_task.warn(
                        'The <a href="https://www.nodc.noaa.gov/OC5/woa18/" target="_blank">World Ocean Atlas</a> does not contain %s values for this location. Using %s = %s.'
                        % (long_name, long_name, default)
                    )

            # GLODAP DIC and alkalinity profiles (note: in umol kg-1!)
            prof_TCO2, prof_TAlk = glodap_dataset.get(lat, lng)
            if not writeProfiles(
                os.path.join(simulation_dir, "TCO2.dat"),
                start,
                stop,
                glodap_dataset.depth,
                prof_TCO2,
                2100.0,
                fmt="%.2f",
            ):
                bgc_task.warn(
                    '<a href="https://www.glodap.info" target="_blank">GLODAP</a> does not contain total dissolved inorganic carbon values for this location. Using TDIC = 2100.'
                )
            if not writeProfiles(
                os.path.join(simulation_dir, "TAlk.dat"),
                start,
                stop,
                glodap_dataset.depth,
                prof_TAlk,
                2300.0,
                fmt="%.2f",
            ):
                bgc_task.warn(
                    '<a href="https://www.glodap.info" target="_blank">GLODAP</a> does not contain total alkalinity values for this location. Using TAlk = 2300.'
                )

    settings["output"] = ["temp", "salt", "nuh"]

    with root_task.subtask(
        "satellite observations", "preparing satellite observations", 0.05
    ) as light_task:
        settings["A"] = 0.6
        settings["g1"] = 1.0
        settings["g2"] = 20.0
        if settings["light_extinction_method"] == "CCI":
            settings["light_extinction_method"] = "custom"
            kd_490s = np.ma.compressed(cci_oc_Kd_dataset.get(lat, lng)[1])
            if kd_490s.size == 0:
                light_task.warn(
                    '<a href="https://climate.esa.int/en/projects/ocean-colour/" target="_blank">OceanColour-CCI</a> does not contain attenuation coefficients for this location. Using attenuation for Jerlov I water types.'
                )
            else:
                settings["g2"] = 1.0 / kd_490s.mean()
        settings["light_extinction_method"] = settings.get(
            "light_extinction_method", "Jerlov-I"
        )

        if settings.get("cci_sst", "off") == "on":
            sst_obs_time, sst_obs, sst_obs_se = cci.get(
                lat, lng, start=start, stop=stop
            )
            if not np.all(np.ma.getmask(sst_obs)):
                with open(os.path.join(simulation_dir, "cci_sst.dat"), "w") as f:
                    f.write(
                        "#time\tSST (degrees Celsius)\tstandard error (degrees Celsius)\n"
                    )
                    for tm, obs, se in zip(sst_obs_time, sst_obs, sst_obs_se):
                        f.write(
                            "%s\t%.3f\t%.3f\n"
                            % (tm.strftime("%Y-%m-%d %H:%M:%S"), obs, se)
                        )
            else:
                light_task.warn(
                    'The <a href="https://climate.esa.int/en/projects/sea-surface-temperature/" target="_blank">SST CCI</a> does not provide sea surface temperatures for this location.'
                )

        if settings.get("cci_chl", "off") == "on":
            time, mu_p, sigma_p = cci_oc_chl_dataset.get_log10_stats(
                lat, lng, start=start, stop=stop
            )
            if not np.all(np.ma.getmask(mu_p)):
                with open(os.path.join(simulation_dir, "cci_chl.dat"), "w") as f:
                    f.write(
                        "#time\tunbiased log10 chlorophyll a (log10 mg m-3)\tunbiased rmsd (log10 mg m-3)\n"
                    )
                    for tm, mu, rmsd in zip(time, mu_p, sigma_p):
                        if not np.ma.getmask(mu):
                            f.write(
                                "%s\t%.3f\t%.3f\n"
                                % (tm.strftime("%Y-%m-%d %H:%M:%S"), mu, rmsd)
                            )
            else:
                light_task.warn(
                    'The <a href="https://climate.esa.int/en/projects/ocean-colour/" target="_blank">Ocean Colour CCI</a> does not provide chlorophyll concentrations for this location.'
                )

    with root_task.subtask("grid", "determining optimal vertical grid", 0.025):
        sigma = grid_generator(depth)
        sigma /= sigma.sum()
        with open(os.path.join(simulation_dir, "grid.dat"), "w") as f:
            f.write("%i\n%s\n" % (sigma.size, "\n".join(map(str, sigma))))
        settings["nlev"] = sigma.size

    settings.setdefault("latitude", lat)
    settings.setdefault("longitude", lng)
    settings.setdefault("depth", depth)
    settings.setdefault("start", start)
    settings.setdefault("stop", stop)


class StandardVariable(enum.Enum):
    CHLOROPHYLL = 1
    CO2_FLUX = 2


def prepare_configuration(
    simulation_dir: str,
    forcing_dir: str,
    settings: dict,
    bgc: Optional[str] = None,
    extra_variables: Iterable[StandardVariable] = (),
):
    available_variables = {}
    if bgc is not None:
        bgc = bgc.lower()
    if bgc == "mops":
        shutil.copyfile(
            os.path.join(
                os.path.dirname(__file__), "lib/fabm/extern/mops/testcases/fabm.yaml"
            ),
            os.path.join(simulation_dir, "fabm.yaml"),
        )
        available_variables = {
            StandardVariable.CHLOROPHYLL: "total_chlorophyll",
            StandardVariable.CO2_FLUX: "carbon_gasex",
        }
    elif bgc == "pisces":
        shutil.copyfile(
            os.path.join(
                os.path.dirname(__file__), "lib/fabm/extern/pisces/testcases/fabm.yaml"
            ),
            os.path.join(simulation_dir, "fabm.yaml"),
        )
        available_variables = {
            StandardVariable.CHLOROPHYLL: "total_chlorophyll",
            StandardVariable.CO2_FLUX: dict(
                name="carbonate_Cflx", scale_factor=86400 * 1000.0
            ),
        }
    elif bgc == "ersem":
        y = yaml.YAML(typ='safe')
        y.default_flow_style = False
        with open(os.path.join(
                os.path.dirname(__file__),
                "lib/fabm/extern/ersem/testcases/fabm-ersem-15.06-L4-ben-docdyn-iop.yaml")) as f:
            conf = y.load(f)
        # conf["instances"]["O3"]["parameters"]["iswtalk"] = 5
        # del conf["instances"]["O3"]["parameters"]["iswbioalk"]
        # conf["instances"]["O3"]["initialization"]["TA"] = 2350.0
        # del conf["instances"]["O3"]["initialization"]["bioalk"]
        with open(os.path.join(simulation_dir, "fabm.yaml"), "w") as f:
            y.dump(conf, f)
        available_variables = {
            StandardVariable.CHLOROPHYLL: "total_chlorophyll",
            StandardVariable.CO2_FLUX: "O3_fair",
        }
    elif bgc == "bfm":
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), "lib/fabm/extern/ogs/fabm_monospectral_4PFTs.yaml"),
            os.path.join(simulation_dir, "fabm.yaml"),
        )
        available_variables = {
            StandardVariable.CHLOROPHYLL: "total_chlorophyll",
            StandardVariable.CO2_FLUX: "O3_fair",
        }
    elif bgc == "ecosmo":
        shutil.copyfile(
            os.path.join(
                os.path.dirname(__file__), "lib/fabm/extern/nersc/ecosmo/fabm.yaml"
            ),
            os.path.join(simulation_dir, "fabm.yaml"),
        )
        available_variables = {
            StandardVariable.CHLOROPHYLL: "total_chlorophyll"
        }
    elif bgc == "ergom":
        shutil.copyfile(
            os.path.join(
                os.path.dirname(__file__), "lib/fabm/testcases/fabm-msi-ergom1.yaml"
            ),
            os.path.join(simulation_dir, "fabm.yaml"),
        )
        available_variables = {StandardVariable.CHLOROPHYLL: "msi_ergom1_tot_chla"}
    elif bgc == "ihamocc":
        shutil.copyfile(
            os.path.join(
                os.path.dirname(__file__), "lib/fabm/extern/ihamocc/testcases/fabm.yaml"
            ),
            os.path.join(simulation_dir, "fabm.yaml"),
        )
        available_variables = {
            StandardVariable.CHLOROPHYLL: "light_chl",
            StandardVariable.CO2_FLUX: dict(
                name="carbon_sco212_sfl", scale_factor=86400 * 1e6
            ),
        }

    settings = dict(settings)
    if forcing_dir != "":
        forcing_dir += "/"
    settings["forcing_dir"] = forcing_dir
    settings.setdefault("temp_offset", 0.0)
    extra_outputs = []
    for v in extra_variables:
        info = available_variables.get(v)
        if isinstance(info, str):
            info = {"name": info}
        extra_outputs.append(info)
    settings["output"] = list(settings["output"]) + [
        v["name"] for v in extra_outputs if v is not None
    ]
    settings["bgc"] = bgc
    for name in os.listdir(template_root):
        if not name.endswith(".nc"):
            with open(os.path.join(simulation_dir, name), "w") as f:
                f.write(gotm_templates[name].render(**settings))

    return extra_outputs, settings["output"]
