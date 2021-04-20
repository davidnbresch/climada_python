"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define TCTracks: IBTracs reader and tracks manager.
"""

__all__ = ['CAT_NAMES', 'SAFFIR_SIM_CAT', 'TCTracks', 'set_category']

# standard libraries
import datetime as dt
import itertools
import logging
import shutil
import warnings
from pathlib import Path

# additional libraries
import cartopy.crs as ccrs
import cftime
import geopandas as gpd
import matplotlib.cm as cm_mp
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import netCDF4 as nc
from numba import jit
import numpy as np
import pandas as pd
import scipy.io.matlab as matlab
from shapely.geometry import Point, LineString, MultiLineString
import shapely.ops
from sklearn.neighbors import DistanceMetric
import statsmodels.api as sm
import xarray as xr

# climada dependencies
from climada.util import ureg
import climada.util.coordinates as u_coord
from climada.util.constants import EARTH_RADIUS_KM, SYSTEM_DIR, DEF_CRS
from climada.util.files_handler import get_file_names, download_ftp
import climada.util.plot as u_plot
import climada.hazard.tc_tracks_synth

LOGGER = logging.getLogger(__name__)

SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 137, 1000]
"""Saffir-Simpson Hurricane Wind Scale in kn based on NOAA"""

CAT_NAMES = {
    -1: 'Tropical Depression',
    0: 'Tropical Storm',
    1: 'Hurricane Cat. 1',
    2: 'Hurricane Cat. 2',
    3: 'Hurricane Cat. 3',
    4: 'Hurricane Cat. 4',
    5: 'Hurricane Cat. 5',
}
"""Saffir-Simpson category names."""

CAT_COLORS = cm_mp.rainbow(np.linspace(0, 1, len(SAFFIR_SIM_CAT)))
"""Color scale to plot the Saffir-Simpson scale."""

IBTRACS_URL = ('https://www.ncei.noaa.gov/data/'
               'international-best-track-archive-for-climate-stewardship-ibtracs/'
               'v04r00/access/netcdf')
"""Site of IBTrACS netcdf file containing all tracks v4.0,
s. https://www.ncdc.noaa.gov/ibtracs/index.php?name=ib-v4-access"""

IBTRACS_FILE = 'IBTrACS.ALL.v04r00.nc'
"""IBTrACS v4.0 file all"""

IBTRACS_AGENCIES = [
    'usa', 'tokyo', 'newdelhi', 'reunion', 'bom', 'nadi', 'wellington',
    'cma', 'hko', 'ds824', 'td9636', 'td9635', 'neumann', 'mlc',
]
"""Names/IDs of agencies in IBTrACS v4.0"""

IBTRACS_USA_AGENCIES = [
    'atcf', 'cphc', 'hurdat_atl', 'hurdat_epa', 'jtwc_cp', 'jtwc_ep', 'jtwc_io',
    'jtwc_sh', 'jtwc_wp', 'nhc_working_bt', 'tcvightals', 'tcvitals'
]
"""Names/IDs of agencies in IBTrACS that correspond to 'usa_*' variables"""


IBTRACS_AGENCY_1MIN_WIND_FACTOR = {
    "usa": [1.0, 0.0],
    "tokyo": [0.60, 23.3],
    "newdelhi": [1.0, 0.0],
    "reunion": [0.88, 0.0],
    "bom": [0.88, 0.0],
    "nadi": [0.88, 0.0],
    "wellington": [0.88, 0.0],
    'cma': [0.871, 0.0],
    'hko': [0.9, 0.0],
    'ds824': [1.0, 0.0],
    'td9636': [1.0, 0.0],
    'td9635': [1.0, 0.0],
    'neumann': [0.88, 0.0],
    'mlc': [1.0, 0.0],
}
"""Scale and shift used by agencies to convert their internal Dvorak 1-minute sustained winds to
the officially reported values that are in IBTrACS. From Table 1 in:

Knapp, K.R., Kruk, M.C. (2010): Quantifying Interagency Differences in Tropical Cyclone Best-Track
Wind Speed Estimates. Monthly Weather Review 138(4): 1459–1473.
https://library.wmo.int/index.php?lvl=notice_display&id=135"""

DEF_ENV_PRESSURE = 1010
"""Default environmental pressure"""

BASIN_ENV_PRESSURE = {
    '': DEF_ENV_PRESSURE,
    'EP': 1010, 'NA': 1010, 'SA': 1010,
    'NI': 1005, 'SI': 1005, 'WP': 1005,
    'SP': 1004,
}
"""Basin-specific default environmental pressure"""

EMANUEL_RMW_CORR_FILES = [
    'temp_ccsm420thcal.mat', 'temp_ccsm4rcp85_full.mat',
    'temp_gfdl520thcal.mat', 'temp_gfdl5rcp85cal_full.mat',
    'temp_hadgem20thcal.mat', 'temp_hadgemrcp85cal_full.mat',
    'temp_miroc20thcal.mat', 'temp_mirocrcp85cal_full.mat',
    'temp_mpi20thcal.mat', 'temp_mpircp85cal_full.mat',
    'temp_mri20thcal.mat', 'temp_mrircp85cal_full.mat',
]
EMANUEL_RMW_CORR_FACTOR = 2.0
"""Kerry Emanuel track files in this list require a correction: The radius of
    maximum wind (rmstore) needs to be multiplied by factor 2."""

class TCTracks():
    """Contains tropical cyclone tracks.

    Attributes
    ----------
    data : list(xarray.Dataset)
        List of tropical cyclone tracks. Each track contains following attributes:
            - time (coords)
            - lat (coords)
            - lon (coords)
            - time_step (in hours)
            - radius_max_wind (in nautical miles)
            - max_sustained_wind
            - central_pressure
            - environmental_pressure
            - max_sustained_wind_unit (attrs)
            - central_pressure_unit (attrs)
            - name (attrs)
            - sid (attrs)
            - orig_event_flag (attrs)
            - data_provider (attrs)
            - basin (attrs)
            - id_no (attrs)
            - category (attrs)
        Computed during processing:
            - on_land
            - dist_since_lf
    """
    def __init__(self, pool=None):
        """Empty constructor. Read csv IBTrACS files if provided."""
        self.data = list()
        if pool:
            self.pool = pool
            LOGGER.debug('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def append(self, tracks):
        """Append tracks to current.

        Parameters
        ----------
        tracks : xarray.Dataset or list(xarray.Dataset)
            tracks to append.
        """
        if not isinstance(tracks, list):
            tracks = [tracks]
        self.data.extend(tracks)

    def get_track(self, track_name=None):
        """Get track with provided name.

        Returns the first matching track based on the assumption that no other track with the same
        name or sid exists in the set.

        Parameters
        ----------
        track_name : str, optional
            Name or sid (ibtracsID for IBTrACS) of track. If None (default), return all tracks.

        Returns
        -------
        result : xarray.Dataset or list of xarray.Dataset
            Usually, a single track is returned. If no track with the specified name is found,
            an empty list `[]` is returned. If called with `track_name=None`, the list of all
            tracks is returned.
        """
        if track_name is None:
            if len(self.data) == 1:
                return self.data[0]
            return self.data

        for track in self.data:
            if track.name == track_name:
                return track
            if hasattr(track, 'sid') and track.sid == track_name:
                return track

        LOGGER.info('No track with name or sid %s found.', track_name)
        return []

    def subset(self, filterdict):
        """Subset tracks based on track attributes.

        Select all tracks matching exactly the given attribute values.

        Parameters
        ----------
        filterdict : dict or OrderedDict
            Keys are attribute names, values are the corresponding attribute values to match.
            In case of an ordered dict, the filters are applied in the given order.

        Returns
        -------
        tc_tracks : TCTracks
            A new instance of TCTracks containing only the matching tracks.
        """
        out = self.__class__(self.pool)
        out.data = self.data

        for key, pattern in filterdict.items():
            out.data = [ds for ds in out.data if ds.attrs[key] == pattern]

        return out

    def tracks_in_exp(self, exposure, buffer=1.0):
        """Select only the tracks that are in the vicinity (buffer) of an exposure.

        Each exposure point/geometry is extended to a disc of radius `buffer`. Each track is
        converted to a line and extended by a radius `buffer`.

        Parameters
        ----------
        exposure : Exposure
            Exposure used to select tracks.
        buffer : float, optional
            Size of buffer around exposure geometries (in the units of `exposure.crs`),
            see `geopandas.distance`. Default: 1.0

        Returns
        -------
        filtered_tracks : climada.hazard.TCTracks()
            TCTracks object with tracks from tc_tracks intersecting the exposure whitin a buffer
            distance.
        """

        if buffer <= 0.0:
            raise ValueError(f"buffer={buffer} is invalid, must be above zero.")
        try:
            exposure.geometry
        except AttributeError:
            exposure.set_geometry_points()

        exp_buffer = exposure.buffer(distance=buffer, resolution=0)
        exp_buffer = exp_buffer.unary_union

        tc_tracks_lines = self.to_geodataframe().buffer(distance=buffer)
        select_tracks = tc_tracks_lines.intersects(exp_buffer)
        tracks_in_exp = [track for j, track in enumerate(self.data) if select_tracks[j]]
        filtered_tracks= TCTracks()
        filtered_tracks.append(tracks_in_exp)

        return filtered_tracks


    def read_ibtracs_netcdf(self, provider=None, rescale_windspeeds=True, storm_id=None,
                            year_range=None, basin=None, interpolate_missing=True,
                            estimate_missing=False, correct_pres=False,
                            discard_single_points=True,
                            file_name='IBTrACS.ALL.v04r00.nc'):
        """Read track data from IBTrACS databse.

        When using data from IBTrACS, make sure to be familiar with the scope and limitations of
        IBTrACS, e.g. by reading the official documentation
        (https://www.ncdc.noaa.gov/ibtracs/pdf/IBTrACS_version4_Technical_Details.pdf). Reading the
        CLIMADA documentation can't replace a thorough understanding of the underlying data. This
        function only provides a (hopefully useful) interface for the data input, but cannot
        provide any guidance or make recommendations about if and how to use IBTrACS data for your
        particular project.

        Resulting tracks are required to have both pressure and wind speed information at all time
        steps. Therefore, all track positions where one of wind speed or pressure are missing are
        discarded unless one of `interpolate_missing` or `estimate_missing` are active.

        Some corrections are automatically applied, such as: `environmental_pressure` is enforced
        to be larger than `central_pressure`.

        Note that the tracks returned by this function might contain irregular time steps since
        that is often the case for the original IBTrACS records. Apply the `equal_timestep`
        function afterwards to enforce regular time steps.

        Parameters
        ----------
        provider : str or list of str, optional
            Either specify an agency, such as "usa", "newdelhi", "bom", "cma", "tokyo", or the
            special values "official" and "official_3h":
            * "official" means using the (usually 6-hourly) officially reported values of the
              officially responsible agencies.
            * "official_3h" means to include (inofficial) 3-hourly data of the officially
              responsible agencies (whenever available).
            If you want to restrict to the officially reported values by the officially responsible
            agencies (`provider="official"`) without any modifications to the original official
            data, make sure to also set `estimate_missing=False` and `interpolate_missing=False`.
            Otherwise, gaps in the official reporting will be filled using interpolation and/or
            statistical estimation procedures (see below).
            If a list is given, the following logic is applied: For each storm, the variables that
            are not reported by the first agency for this storm are taken from the next agency in
            the list that did report this variable for this storm. For different storms, the same
            variable might be taken from different agencies.
            Default: ['official_3h', 'usa', 'tokyo', 'newdelhi', 'reunion', 'bom', 'nadi',
            'wellington', 'cma', 'hko', 'ds824', 'td9636', 'td9635', 'neumann', 'mlc']
        rescale_windspeeds : bool, optional
            If True, all wind speeds are linearly rescaled to 1-minute sustained winds.
            Note however that the IBTrACS documentation (Section 5.2,
            https://www.ncdc.noaa.gov/ibtracs/pdf/IBTrACS_version4_Technical_Details.pdf) includes
            a warning about this kind of conversion: "While a multiplicative factor can describe
            the numerical differences, there are procedural and observational differences between
            agencies that can change through time, which confounds the simple multiplicative
            factor." Default: True
        storm_id : str or list of str, optional
            IBTrACS ID of the storm, e.g. 1988234N13299, [1988234N13299, 1989260N11316].
        year_range : tuple (min_year, max_year), optional
            Year range to filter track selection. Default: (1980, 2018)
        basin : str, optional
            E.g. US, SA, NI, SI, SP, WP, EP, NA. If not provided, consider all basins.
        interpolate_missing : bool, optional
            If True, interpolate temporal reporting gaps within a variable (such as pressure, wind
            speed, or radius) linearly if possible. Temporal interpolation is with respect to the
            time steps defined in IBTrACS for a particular storm. No new time steps are added that
            are not originally defined in IBTrACS.
            For each time step with a missing value, this procedure is only able to fill in that
            value if there are other time steps before and after this time step for which values
            have been reported.
            This procedure will be applied before the statistical estimations referred to
            by `estimate_missing`. It is applied to all variables (eye position, wind speed,
            environmental and central pressure, storm radius and radius of maximum winds).
            Default: True
        estimate_missing : bool, optional
            For each fixed time step, estimate missing pressure, wind speed and radius using other
            variables that are available at that time step.
            The relationships between the variables are purely statistical. In comparison to
            `interpolate_missing`, this procedure is able to estimate values for variables that
            haven't been reported by any agency at any time step, as long as other variables are
            available.
            A typical example are storms before 1950, for which there are often no reported values
            for pressure, but for wind speed. In this case, a rough statistical pressure-wind
            relationship is applied to estimate the missing pressure values from the available
            wind-speed values.
            Make sure to set `rescale_windspeeds=True` when using this option because the
            statistical relationships are calibrated using rescaled wind speeds.
            Default: False
        correct_pres : bool, optional
            For backwards compatibility, alias for `estimate_missing`.
            This is deprecated, use `estimate_missing` instead!
        discard_single_points : bool, optional
            Whether to discard tracks that consists of a single point. Recommended for full
            compatiblity with other functions such as `equal_timesteps`. Default: True.
        file_name : str, optional
            Name of NetCDF file to be dowloaded or located at climada/data/system.
            Default: 'IBTrACS.ALL.v04r00.nc'
        """
        if correct_pres:
            LOGGER.warning("`correct_pres` is deprecated. "
                           "Use `estimate_missing` instead.")
            estimate_missing = True
        if estimate_missing and not rescale_windspeeds:
            LOGGER.warning(
                "Using `estimate_missing` without `rescale_windspeeds` is strongly discouraged!")
        self.data = list()
        fn_nc = SYSTEM_DIR.joinpath(file_name)
        if not fn_nc.is_file():
            try:
                download_ftp(f'{IBTRACS_URL}/{IBTRACS_FILE}', IBTRACS_FILE)
                shutil.move(IBTRACS_FILE, fn_nc)
            except ValueError as err:
                LOGGER.error('Error while downloading %s. Try to download it '
                             'manually and put the file in '
                             'climada_python/data/system/', IBTRACS_URL)
                raise err

        ibtracs_ds = xr.open_dataset(fn_nc)
        match = np.ones(ibtracs_ds.sid.shape[0], dtype=bool)
        if storm_id:
            if not isinstance(storm_id, list):
                storm_id = [storm_id]
            match &= ibtracs_ds.sid.isin([i.encode() for i in storm_id])
            if np.count_nonzero(match) == 0:
                LOGGER.info('No tracks with given IDs %s.', storm_id)
        else:
            year_range = year_range if year_range else (1980, 2018)
        if year_range:
            years = ibtracs_ds.sid.str.slice(0, 4).astype(int)
            match &= (years >= year_range[0]) & (years <= year_range[1])
            if np.count_nonzero(match) == 0:
                LOGGER.info('No tracks in time range (%s, %s).', *year_range)
        if basin:
            match &= (ibtracs_ds.basin == basin.encode()).any(dim='date_time')
            if np.count_nonzero(match) == 0:
                LOGGER.info('No tracks in basin %s.', basin)

        if np.count_nonzero(match) == 0:
            LOGGER.info('There are no tracks matching the specified requirements.')
            self.data = []
            return

        ibtracs_ds = ibtracs_ds.sel(storm=match)
        ibtracs_ds['valid_t'] = ibtracs_ds.time.notnull()
        valid_storms_mask = ibtracs_ds.valid_t.any(dim="date_time")
        invalid_storms_idx = np.nonzero(~valid_storms_mask.data)[0]
        if invalid_storms_idx.size > 0:
            invalid_sids = ', '.join(ibtracs_ds.sid.sel(storm=invalid_storms_idx).astype(str).data)
            LOGGER.warning('No valid timestamps found for %s.', invalid_sids)
            ibtracs_ds = ibtracs_ds.sel(storm=valid_storms_mask)

        if rescale_windspeeds:
            for agency in IBTRACS_AGENCIES:
                scale, shift = IBTRACS_AGENCY_1MIN_WIND_FACTOR[agency]
                ibtracs_ds[f'{agency}_wind'] -= shift
                ibtracs_ds[f'{agency}_wind'] /= scale

        if provider is None:
            provider = ["official_3h"] + IBTRACS_AGENCIES
        elif isinstance(provider, str):
            provider = [provider]

        for tc_var in ['lat', 'lon', 'wind', 'pres', 'rmw', 'poci', 'roci']:
            if "official" in provider or "official_3h" in provider:
                ibtracs_add_official_variable(
                    ibtracs_ds, tc_var, add_3h=("official_3h" in provider))

            # set up dimension of agency-reported values in order of preference, including the
            # newly created `official` and `official_3h` data if specified
            ag_vars = [f'{ag}_{tc_var}' for ag in provider]
            ag_vars = [ag_var for ag_var in ag_vars if ag_var in ibtracs_ds.data_vars.keys()]
            all_vals = ibtracs_ds[ag_vars].to_array(dim='agency')
            # argmax returns the first True (i.e. valid) along the 'agency' dimension
            preferred_idx = all_vals.notnull().any(dim="date_time").argmax(dim='agency')
            ibtracs_ds[tc_var] = all_vals.isel(agency=preferred_idx)

            # Usually, if an agency reports about a track that crosses the antimeridian, the
            # longitude is always chosen positive. However, it can happen, that the TC crosses the
            # antimeridian according to one agency, but not according to another. When mixing
            # agency data, this can yield inconsistent sign changes in longitude. We remove those:
            if tc_var == 'lon':
                # By IBTrACS default, no longitude should be <= -180, but this is not true for some
                # agencies, so we have to manually enforce this policy:
                ibtracs_ds[tc_var].values[(ibtracs_ds[tc_var] <= -180).values] += 360
                crossing_mask = ((ibtracs_ds[tc_var] > 170).any(dim="date_time")
                                 & (ibtracs_ds[tc_var] < -170).any(dim="date_time")
                                 & (ibtracs_ds[tc_var] < 0)).values
                ibtracs_ds[tc_var].values[crossing_mask] += 360

            if interpolate_missing:
                with warnings.catch_warnings():
                    # Upstream issue, see https://github.com/pydata/xarray/issues/4167
                    warnings.simplefilter(action="ignore", category=FutureWarning)

                    # don't interpolate if there is only a single record for this variable
                    nonsingular_mask = (
                        ibtracs_ds[tc_var].notnull().sum(dim="date_time") > 1).values
                    if nonsingular_mask.sum() > 0:
                        ibtracs_ds[tc_var].values[nonsingular_mask] = (
                            ibtracs_ds[tc_var].sel(storm=nonsingular_mask).interpolate_na(
                                dim="date_time", method="linear"))
        ibtracs_ds = ibtracs_ds[['sid', 'name', 'basin', 'lat', 'lon', 'time', 'valid_t',
                                 'wind', 'pres', 'rmw', 'roci', 'poci']]

        if estimate_missing:
            ibtracs_ds['pres'][:] = _estimate_pressure(
                ibtracs_ds.pres, ibtracs_ds.lat, ibtracs_ds.lon, ibtracs_ds.wind)
            ibtracs_ds['wind'][:] = _estimate_vmax(
                ibtracs_ds.wind, ibtracs_ds.lat, ibtracs_ds.lon, ibtracs_ds.pres)

        ibtracs_ds['valid_t'] &= (ibtracs_ds.lat.notnull() & ibtracs_ds.lon.notnull()
                                  & ibtracs_ds.wind.notnull() & ibtracs_ds.pres.notnull())
        valid_storms_mask = ibtracs_ds.valid_t.any(dim="date_time")
        invalid_storms_idx = np.nonzero(~valid_storms_mask.data)[0]
        if invalid_storms_idx.size > 0:
            invalid_sids = list(ibtracs_ds.sid.sel(storm=invalid_storms_idx).astype(str).data)
            LOGGER.warning('%d storm events are discarded because no valid wind/pressure values '
                           'have been found: %s%s', len(invalid_sids), ", ".join(invalid_sids[:5]),
                           ", ..." if len(invalid_sids) > 5  else ".")
            ibtracs_ds = ibtracs_ds.sel(storm=valid_storms_mask)
            
        if discard_single_points:
            valid_storms_mask = ibtracs_ds.valid_t.sum(dim="date_time") > 1
            invalid_storms_idx = np.nonzero(~valid_storms_mask.data)[0]
            if invalid_storms_idx.size > 0:
                invalid_sids = list(ibtracs_ds.sid.sel(storm=invalid_storms_idx).astype(str).data)
                LOGGER.warning('%d storm events are discarded because only one valid timestep '
                               'has been found: %s%s', len(invalid_sids), ", ".join(invalid_sids[:5]),
                               ", ..." if len(invalid_sids) > 5  else ".")
                ibtracs_ds = ibtracs_ds.sel(storm=valid_storms_mask)

        max_wind = ibtracs_ds.wind.max(dim="date_time").data.ravel()
        category_test = (max_wind[:, None] < np.array(SAFFIR_SIM_CAT)[None])
        category = np.argmax(category_test, axis=1) - 1
        basin_map = {b.encode("utf-8"): v for b, v in BASIN_ENV_PRESSURE.items()}
        basin_fun = lambda b: basin_map[b]

        ibtracs_ds['id_no'] = (ibtracs_ds.sid.str.replace(b'N', b'0')
                               .str.replace(b'S', b'1')
                               .astype(float))
        provider_str = f"ibtracs_{provider[0]}" + ("" if len(provider) == 1 else "_mixed")

        last_perc = 0
        all_tracks = []
        for i_track, t_msk in enumerate(ibtracs_ds.valid_t.data):
            perc = 100 * len(all_tracks) / ibtracs_ds.sid.size
            if perc - last_perc >= 10:
                LOGGER.info("Progress: %d%%", perc)
                last_perc = perc
            track_ds = ibtracs_ds.sel(storm=i_track, date_time=t_msk)
            st_penv = xr.apply_ufunc(basin_fun, track_ds.basin, vectorize=True)

            # a track that crosses the antimeridian in IBTrACS might be truncated by `t_msk` in
            # such a way that the remaining part is not crossing the antimeridian:
            if (track_ds.lon.values > 180).all():
                track_ds['lon'] -= 360

            # set time_step in hours
            track_ds['time_step'] = xr.ones_like(track_ds.time, dtype=float)
            if track_ds.time.size > 1:
                track_ds.time_step.values[1:] = (track_ds.time.diff(dim="date_time")
                                                 / np.timedelta64(1, 'h'))
                track_ds.time_step.values[0] = track_ds.time_step[1]

            with warnings.catch_warnings():
                # See https://github.com/pydata/xarray/issues/4167
                warnings.simplefilter(action="ignore", category=FutureWarning)

                track_ds['rmw'] = track_ds.rmw \
                    .ffill(dim='date_time', limit=1) \
                    .bfill(dim='date_time', limit=1) \
                    .fillna(0)
                track_ds['roci'] = track_ds.roci \
                    .ffill(dim='date_time', limit=1) \
                    .bfill(dim='date_time', limit=1) \
                    .fillna(0)
                track_ds['poci'] = track_ds.poci \
                    .ffill(dim='date_time', limit=4) \
                    .bfill(dim='date_time', limit=4)
                # this is the most time consuming line in the processing:
                track_ds['poci'] = track_ds.poci.fillna(st_penv)

            if estimate_missing:
                track_ds['rmw'][:] = estimate_rmw(track_ds.rmw.values, track_ds.pres.values)
                track_ds['roci'][:] = estimate_roci(track_ds.roci.values, track_ds.pres.values)
                track_ds['roci'][:] = np.fmax(track_ds.rmw.values, track_ds.roci.values)

            # ensure environmental pressure >= central pressure
            # this is the second most time consuming line in the processing:
            track_ds['poci'][:] = np.fmax(track_ds.poci, track_ds.pres)

            all_tracks.append(xr.Dataset({
                'time_step': ('time', track_ds.time_step.data),
                'radius_max_wind': ('time', track_ds.rmw.data),
                'radius_oci': ('time', track_ds.roci.data),
                'max_sustained_wind': ('time', track_ds.wind.data),
                'central_pressure': ('time', track_ds.pres.data),
                'environmental_pressure': ('time', track_ds.poci.data),
            }, coords={
                'time': track_ds.time.dt.round('s').data,
                'lat': ('time', track_ds.lat.data),
                'lon': ('time', track_ds.lon.data),
            }, attrs={
                'max_sustained_wind_unit': 'kn',
                'central_pressure_unit': 'mb',
                'name': track_ds.name.astype(str).item(),
                'sid': track_ds.sid.astype(str).item(),
                'orig_event_flag': True,
                'data_provider': provider_str,
                'basin': track_ds.basin.values[0].astype(str).item(),
                'id_no': track_ds.id_no.item(),
                'category': category[i_track],
            }))
        if last_perc != 100:
            LOGGER.info("Progress: 100%")
        self.data = all_tracks

    def read_processed_ibtracs_csv(self, file_names):
        """Fill from processed ibtracs csv file(s).

        Parameters
        ----------
        file_names : str or list of str
            Absolute file name(s) or folder name containing the files to read.
        """
        self.data = list()
        all_file = get_file_names(file_names)
        for file in all_file:
            self._read_ibtracs_csv_single(file)

    def read_simulations_emanuel(self, file_names, hemisphere='S'):
        """Fill from Kerry Emanuel tracks.

        Parameters
        ----------
        file_names : str or list of str
            Absolute file name(s) or folder name containing the files to read.
        hemisphere : str, optional
            'S', 'N' or 'both'. Default: 'S'
        """
        self.data = []
        for path in get_file_names(file_names):
            rmw_corr = Path(path).name in EMANUEL_RMW_CORR_FILES
            self._read_file_emanuel(path, hemisphere=hemisphere,
                                    rmw_corr=rmw_corr)

    def _read_file_emanuel(self, path, hemisphere='S', rmw_corr=False):
        """Append tracks from file containing Kerry Emanuel simulations.

        Parameters
        ----------
        path : str
            absolute path of file to read.
        hemisphere : str, optional
            'S', 'N' or 'both'. Default: 'S'
        rmw_corr : str, optional
            If True, multiply the radius of maximum wind by factor 2. Default: False.
        """
        if hemisphere == 'S':
            hem_min, hem_max = -90, 0
        elif hemisphere == 'N':
            hem_min, hem_max = 0, 90
        else:
            hem_min, hem_max = -90, 90

        LOGGER.info('Reading %s.', path)
        data_mat = matlab.loadmat(path)
        lat = data_mat['latstore']
        ntracks, nnodes = lat.shape
        years_uniq = np.unique(data_mat['yearstore'])
        LOGGER.info("File contains %s tracks (at most %s nodes each), "
                    "representing %s years (%s-%s).", ntracks, nnodes,
                    years_uniq.size, years_uniq[0], years_uniq[-1])

        # filter according to chosen hemisphere
        hem_mask = (lat >= hem_min) & (lat <= hem_max) | (lat == 0)
        hem_idx = np.all(hem_mask, axis=1).nonzero()[0]
        data_hem = lambda keys: [data_mat[f'{k}store'][hem_idx] for k in keys]

        lat, lon = data_hem(['lat', 'long'])
        months, days, hours = data_hem(['month', 'day', 'hour'])
        months, days, hours = [np.int8(ar) for ar in [months, days, hours]]
        tc_rmw, tc_maxwind, tc_pressure = data_hem(['rm', 'v', 'p'])
        years = data_mat['yearstore'][0, hem_idx]

        ntracks, nnodes = lat.shape
        LOGGER.info("Loading %s tracks on %s hemisphere.", ntracks, hemisphere)

        # change lon format to -180 to 180
        lon[lon > 180] = lon[lon > 180] - 360

        # change units from kilometers to nautical miles
        tc_rmw = (tc_rmw * ureg.kilometer).to(ureg.nautical_mile).magnitude
        if rmw_corr:
            LOGGER.info("Applying RMW correction.")
            tc_rmw *= EMANUEL_RMW_CORR_FACTOR

        for i_track in range(lat.shape[0]):
            valid_idx = (lat[i_track, :] != 0).nonzero()[0]
            nnodes = valid_idx.size
            time_step = np.abs(np.diff(hours[i_track, valid_idx])).min()

            # deal with change of year
            year = np.full(valid_idx.size, years[i_track])
            year_change = (np.diff(months[i_track, valid_idx]) < 0)
            year_change = year_change.nonzero()[0]
            if year_change.size > 0:
                year[year_change[0] + 1:] += 1

            try:
                datetimes = map(dt.datetime, year,
                                months[i_track, valid_idx],
                                days[i_track, valid_idx],
                                hours[i_track, valid_idx])
                datetimes = list(datetimes)
            except ValueError as err:
                # dates are known to contain invalid February 30
                date_feb = (months[i_track, valid_idx] == 2) \
                         & (days[i_track, valid_idx] > 28)
                if np.count_nonzero(date_feb) == 0:
                    # unknown invalid date issue
                    raise err
                step = time_step if not date_feb[0] else -time_step
                reference_idx = 0 if not date_feb[0] else -1
                reference_date = dt.datetime(
                    year[reference_idx],
                    months[i_track, valid_idx[reference_idx]],
                    days[i_track, valid_idx[reference_idx]],
                    hours[i_track, valid_idx[reference_idx]],)
                datetimes = [reference_date + dt.timedelta(hours=int(step * i))
                             for i in range(nnodes)]
            datetimes = [cftime.DatetimeProlepticGregorian(d.year, d.month, d.day, d.hour)
                         for d in datetimes]

            max_sustained_wind = tc_maxwind[i_track, valid_idx]
            max_sustained_wind_unit = 'kn'
            env_pressure = np.full(nnodes, DEF_ENV_PRESSURE)
            category = set_category(max_sustained_wind,
                                    max_sustained_wind_unit,
                                    SAFFIR_SIM_CAT)
            tr_ds = xr.Dataset({
                'time_step': ('time', np.full(nnodes, time_step)),
                'radius_max_wind': ('time', tc_rmw[i_track, valid_idx]),
                'max_sustained_wind': ('time', max_sustained_wind),
                'central_pressure': ('time', tc_pressure[i_track, valid_idx]),
                'environmental_pressure': ('time', env_pressure),
            }, coords={
                'time': datetimes,
                'lat': ('time', lat[i_track, valid_idx]),
                'lon': ('time', lon[i_track, valid_idx]),
            }, attrs={
                'max_sustained_wind_unit': max_sustained_wind_unit,
                'central_pressure_unit': 'mb',
                'name': str(hem_idx[i_track]),
                'sid': str(hem_idx[i_track]),
                'orig_event_flag': True,
                'data_provider': 'Emanuel',
                'basin': hemisphere,
                'id_no': hem_idx[i_track],
                'category': category,
            })
            self.data.append(tr_ds)

    def read_one_gettelman(self, nc_data, i_track):
        """Fill from Andrew Gettelman tracks.

        Parameters
        ----------
        nc_data : str
            netCDF4.Dataset Objekt
        i_tracks : int
            track number
        """
        scale_to_10m = (10. / 60.)**.11
        mps2kts = 1.94384
        basin_dict = {0: 'NA - North Atlantic',
                      1: 'SA - South Atlantic',
                      2: 'WP - West Pacific',
                      3: 'EP - East Pacific',
                      4: 'SP - South Pacific',
                      5: 'NI - North Indian',
                      6: 'SI - South Indian',
                      7: 'AS - Arabian Sea',
                      8: 'BB - Bay of Bengal',
                      9: 'EA - Eastern Australia',
                      10: 'WA - Western Australia',
                      11: 'CP - Central Pacific',
                      12: 'CS - Carribbean Sea',
                      13: 'GM - Gulf of Mexico',
                      14: 'MM - Missing'}

        val_len = nc_data.variables['numObs'][i_track]
        sid = str(i_track)
        times = nc_data.variables['source_time'][i_track, :][:val_len]

        datetimes = list()
        for time in times:
            try:
                datetimes.append(
                    dt.datetime.strptime(
                        str(nc.num2date(time, 'days since {}'.format('1858-11-17'),
                                        calendar='standard')),
                        '%Y-%m-%d %H:%M:%S'))
            except ValueError:
                # If wrong t, set t to previous t plus 3 hours
                if datetimes:
                    datetimes.append(datetimes[-1] + dt.timedelta(hours=3))
                else:
                    pos = list(times).index(time)
                    time = times[pos + 1] - 1 / 24 * 3
                    datetimes.append(
                        dt.datetime.strptime(
                            str(nc.num2date(time, 'days since {}'.format('1858-11-17'),
                                            calendar='standard')),
                            '%Y-%m-%d %H:%M:%S'))
        time_step = []
        for i_time, time in enumerate(datetimes[1:], 1):
            time_step.append((time - datetimes[i_time - 1]).total_seconds() / 3600)
        time_step.append(time_step[-1])

        basins = list()
        for basin in nc_data.variables['basin'][i_track, :][:val_len]:
            try:
                basins.extend([basin_dict[basin]])
            except KeyError:
                basins.extend([np.nan])

        lon = nc_data.variables['lon'][i_track, :][:val_len]
        lon[lon > 180] = lon[lon > 180] - 360  # change lon format to -180 to 180
        lat = nc_data.variables['lat'][i_track, :][:val_len]
        cen_pres = nc_data.variables['pres'][i_track, :][:val_len]
        av_prec = nc_data.variables['precavg'][i_track, :][:val_len]
        max_prec = nc_data.variables['precmax'][i_track, :][:val_len]

        # m/s to kn
        wind = nc_data.variables['wind'][i_track, :][:val_len] * mps2kts * scale_to_10m
        if not all(wind.data):  # if wind is empty
            wind = np.ones(wind.size) * -999.9

        tr_df = pd.DataFrame({'time': datetimes, 'lat': lat, 'lon': lon,
                              'max_sustained_wind': wind,
                              'central_pressure': cen_pres,
                              'environmental_pressure': np.ones(lat.size) * 1015.,
                              'radius_max_wind': np.ones(lat.size) * 65.,
                              'maximum_precipitation': max_prec,
                              'average_precipitation': av_prec,
                              'basins': basins,
                              'time_step': time_step})

        # construct xarray
        tr_ds = xr.Dataset.from_dataframe(tr_df.set_index('time'))
        tr_ds.coords['lat'] = ('time', tr_ds.lat)
        tr_ds.coords['lon'] = ('time', tr_ds.lon)
        tr_ds.attrs = {'max_sustained_wind_unit': 'kn',
                       'central_pressure_unit': 'mb',
                       'sid': sid,
                       'name': sid, 'orig_event_flag': False,
                       'basin': basins[0],
                       'id_no': i_track,
                       'category': set_category(wind, 'kn')}
        self.data.append(tr_ds)

    def read_simulations_chaz(self, file_names, year_range=None, ensemble_nums=None):
        """Read track output from CHAZ simulations

            Lee, C.-Y., Tippett, M.K., Sobel, A.H., Camargo, S.J. (2018): An Environmentally
            Forced Tropical Cyclone Hazard Model. J Adv Model Earth Sy 10(1): 223–241.

        Parameters
        ----------
        file_names : str or list of str
            Absolute file name(s) or folder name containing the files to read.
        year_range : tuple (min_year, max_year), optional
            Filter by year, if given.
        ensemble_nums : list, optional
            Filter by ensembleNum, if given.
        """
        self.data = []
        for path in get_file_names(file_names):
            LOGGER.info('Reading %s.', path)
            chaz_ds = xr.open_dataset(path)
            chaz_ds.time.attrs["units"] = "days since 1950-1-1"
            chaz_ds.time.attrs["missing_value"] = -54786.0
            chaz_ds = xr.decode_cf(chaz_ds)
            chaz_ds['id_no'] = chaz_ds.stormID * 1000 + chaz_ds.ensembleNum
            for var in ['time', 'longitude', 'latitude']:
                chaz_ds[var] = chaz_ds[var].expand_dims(ensembleNum=chaz_ds.ensembleNum)
            chaz_ds = chaz_ds.stack(id=("ensembleNum", "stormID"))
            years_uniq = chaz_ds.time.dt.year.data
            years_uniq = np.unique(years_uniq[~np.isnan(years_uniq)])
            LOGGER.info("File contains %s tracks (at most %s nodes each), "
                        "representing %s years (%d-%d).",
                        chaz_ds.id_no.size, chaz_ds.lifelength.size,
                        years_uniq.size, years_uniq[0], years_uniq[-1])

            # filter by year range if given
            if year_range:
                match = ((chaz_ds.time.dt.year >= year_range[0])
                         & (chaz_ds.time.dt.year <= year_range[1])).sel(lifelength=0)
                if np.count_nonzero(match) == 0:
                    LOGGER.info('No tracks in time range (%s, %s).', *year_range)
                    self.data = []
                    continue
                chaz_ds = chaz_ds.sel(id=match)

            # filter by ensembleNum if given
            if ensemble_nums is not None:
                match = np.isin(chaz_ds.ensembleNum.values, ensemble_nums)
                if np.count_nonzero(match) == 0:
                    LOGGER.info('No tracks with specified ensemble numbers.')
                    self.data = []
                    continue
                chaz_ds = chaz_ds.sel(id=match)

            # remove invalid tracks from selection
            chaz_ds['valid_t'] = chaz_ds.time.notnull() & chaz_ds.Mwspd.notnull()
            valid_st = chaz_ds.valid_t.any(dim="lifelength")
            invalid_st = np.nonzero(~valid_st.data)[0]
            if invalid_st.size > 0:
                LOGGER.info('No valid Mwspd values found for %d out of %d storm tracks.',
                            invalid_st.size, valid_st.size)
                chaz_ds = chaz_ds.sel(id=valid_st)

            # estimate central pressure from location and max wind
            chaz_ds['pres'] = xr.full_like(chaz_ds.Mwspd, -1, dtype=float)
            chaz_ds['pres'][:] = _estimate_pressure(
                chaz_ds.pres, chaz_ds.latitude, chaz_ds.longitude, chaz_ds.Mwspd)

            # compute time stepsizes
            chaz_ds['time_step'] = xr.zeros_like(chaz_ds.time, dtype=float)
            chaz_ds['time_step'][1:, :] = (chaz_ds.time.diff(dim="lifelength")
                                            / np.timedelta64(1, 'h'))
            chaz_ds['time_step'][0, :] = chaz_ds.time_step[1, :]

            # determine Saffir-Simpson category
            max_wind = chaz_ds.Mwspd.max(dim="lifelength").data.ravel()
            category_test = (max_wind[:, None] < np.array(SAFFIR_SIM_CAT)[None])
            chaz_ds['category'] = ("id", np.argmax(category_test, axis=1) - 1)

            fname = Path(path).name
            chaz_ds.time[:] = chaz_ds.time.dt.round('s').data
            chaz_ds['radius_max_wind'] = xr.full_like(chaz_ds.pres, np.nan)
            chaz_ds['environmental_pressure'] = xr.full_like(chaz_ds.pres, DEF_ENV_PRESSURE)
            chaz_ds["track_name"] = ("id", [f"{fname}-{track_id.item()[1]}-{track_id.item()[0]}"
                                            for track_id in chaz_ds.id])

            # add tracks one by one
            last_perc = 0
            for i_track in chaz_ds.id_no:
                perc = 100 * len(self.data) / chaz_ds.id_no.size
                if perc - last_perc >= 10:
                    LOGGER.info("Progress: %d%%", perc)
                    last_perc = perc
                track_ds = chaz_ds.sel(id=i_track.id.item())
                track_ds = track_ds.sel(lifelength=track_ds.valid_t.data)
                self.data.append(xr.Dataset({
                    'time_step': ('time', track_ds.time_step),
                    'max_sustained_wind': ('time', track_ds.Mwspd.data),
                    'central_pressure': ('time', track_ds.pres.data),
                    'radius_max_wind': ('time', track_ds.radius_max_wind.data),
                    'environmental_pressure': ('time', track_ds.environmental_pressure.data),
                }, coords={
                    'time': track_ds.time.data,
                    'lat': ('time', track_ds.latitude.data),
                    'lon': ('time', track_ds.longitude.data),
                }, attrs={
                    'max_sustained_wind_unit': 'kn',
                    'central_pressure_unit': 'mb',
                    'name': track_ds.track_name.item(),
                    'sid': track_ds.track_name.item(),
                    'orig_event_flag': True,
                    'data_provider': "CHAZ",
                    'basin': "global",
                    'id_no': track_ds.id_no.item(),
                    'category': track_ds.category.item(),
                }))
            if last_perc != 100:
                LOGGER.info("Progress: 100%")

    def read_simulations_storm(self, path, years=None):
        """Read track output from STORM simulations

            Bloemendaal et al. (2020): Generation of a global synthetic tropical cyclone hazard
            dataset using STORM. Scientific Data 7(1): 40.

        Track data available for download from

            https://doi.org/10.4121/uuid:82c1dc0d-5485-43d8-901a-ce7f26cda35d

        Parameters
        ----------
        path : str
            Full path to a txt-file as contained in the `data.zip` archive from the official source
            linked above.
        years : list of int, optional
            If given, only read the specified "years" from the txt-File. Note that a "year" refers
            to one ensemble of tracks in the data set that represents one sample year.
        """
        self.data = []
        basins = ["EP", "NA", "NI", "SI", "SP", "WP"]
        tracks_df = pd.read_csv(path, names=['year', 'time_start', 'tc_num', 'time_delta',
                                             'basin', 'lat', 'lon', 'pres', 'wind',
                                             'rmw', 'category', 'landfall', 'dist_to_land'],
                                converters={
                                    "time_start": lambda d: dt.datetime(1980, int(float(d)), 1, 0),
                                    "time_delta": lambda d: dt.timedelta(hours=3 * float(d)),
                                    "basin": lambda d: basins[int(float(d))],
                                },
                                dtype={
                                    "year": int,
                                    "tc_num": int,
                                    "category": int,
                                })

        # filter specified years
        if years is not None:
            tracks_df = tracks_df[np.isin(tracks_df['year'], years)]

        # a bug in the data causes some storm tracks to be double-listed:
        tracks_df = tracks_df.drop_duplicates(subset=["year", "tc_num", "time_delta"])

        # conversion of units and time
        tracks_df['rmw'] *= (1 * ureg.kilometer).to(ureg.nautical_mile).magnitude
        tracks_df['wind'] *= (1 * ureg.meter / ureg.second).to(ureg.knot).magnitude
        tracks_df['time'] = tracks_df['time_start'] + tracks_df['time_delta']
        tracks_df = tracks_df.drop(
            labels=['time_start', 'time_delta', 'landfall', 'dist_to_land'], axis=1)

        # add tracks one by one
        last_perc = 0
        fname = Path(path).name
        groups = tracks_df.groupby(by=["year", "tc_num"])
        for idx, group in groups:
            perc = 100 * len(self.data) / len(groups)
            if perc - last_perc >= 10:
                LOGGER.info("Progress: %d%%", perc)
                last_perc = perc
            track_name = f"{fname}-{idx[0]}-{idx[1]}"
            basin = group['basin'].values[0]
            env_pressure = DEF_ENV_PRESSURE
            if basin in BASIN_ENV_PRESSURE:
                env_pressure = BASIN_ENV_PRESSURE[basin]
            env_pressure = np.full_like(group['pres'].values, env_pressure)
            self.data.append(xr.Dataset({
                'time_step': ('time', np.full(group['time'].shape, 3)),
                'max_sustained_wind': ('time', group['wind'].values),
                'central_pressure': ('time', group['pres'].values),
                'radius_max_wind': ('time', group['rmw'].values),
                'environmental_pressure': ('time', env_pressure),
            }, coords={
                'time': ('time', group['time'].values),
                'lat': ('time', group['lat'].values),
                'lon': ('time', group['lon'].values),
            }, attrs={
                'max_sustained_wind_unit': 'kn',
                'central_pressure_unit': 'mb',
                'name': track_name,
                'sid': track_name,
                'orig_event_flag': True,
                'data_provider': "STORM",
                'basin': basin,
                'id_no': idx[0] * 1000 + idx[1],
                'category': group['category'].max(),
            }))
        if last_perc != 100:
            LOGGER.info("Progress: 100%")

    def equal_timestep(self, time_step_h=1, land_params=False):
        """Generate interpolated track values to time steps of time_step_h.

        Parameters
        ----------
        time_step_h : float or int, optional
            Temporal resolution in hours (positive, may be non-integer-valued). Default: 1.
        land_params : bool, optional
            If True, recompute `on_land` and `dist_since_lf` at each node. Default: False.
        """
        if time_step_h <= 0:
            raise ValueError(f"time_step_h is not a positive number: {time_step_h}")
        LOGGER.info('Interpolating %s tracks to %sh time steps.', self.size, time_step_h)

        if land_params:
            extent = self.get_extent()
            land_geom = u_coord.get_land_geometry(extent, resolution=10)
        else:
            land_geom = None

        if self.pool:
            chunksize = min(self.size // self.pool.ncpus, 1000)
            self.data = self.pool.map(self._one_interp_data, self.data,
                                      itertools.repeat(time_step_h, self.size),
                                      itertools.repeat(land_geom, self.size),
                                      chunksize=chunksize)
        else:
            last_perc = 0
            new_data = []
            for track in self.data:
                # progress indicator
                perc = 100 * len(new_data) / len(self.data)
                if perc - last_perc >= 10:
                    LOGGER.debug("Progress: %d%%", perc)
                    last_perc = perc
                new_data.append(
                    self._one_interp_data(track, time_step_h, land_geom))
            self.data = new_data

    def calc_random_walk(self, **kwargs):
        """Deprecated. Use `TCTracks.calc_perturbed_trajectories` instead."""
        LOGGER.warning("The use of TCTracks.calc_random_walk is deprecated."
                       "Use TCTracks.calc_perturbed_trajectories instead.")
        if kwargs.get('ens_size'):
            kwargs['nb_synth_tracks'] = kwargs.pop('ens_size')
        return self.calc_perturbed_trajectories(**kwargs)

    def calc_perturbed_trajectories(self, **kwargs):
        """See function in `climada.hazard.tc_tracks_synth`."""
        climada.hazard.tc_tracks_synth.calc_perturbed_trajectories(self, **kwargs)

    @property
    def size(self):
        """Get longitude from coord array."""
        return len(self.data)

    def get_bounds(self, deg_buffer=0.1):
        """Get bounds as (lon_min, lat_min, lon_max, lat_max) tuple.

        Parameters
        ----------
        deg_buffer : float
            A buffer to add around the bounding box

        Returns
        -------
        bounds : tuple (lon_min, lat_min, lon_max, lat_max)
        """
        bounds = u_coord.latlon_bounds(
            np.concatenate([t.lat.values for t in self.data]),
            np.concatenate([t.lon.values for t in self.data]),
            buffer=deg_buffer)
        return bounds

    @property
    def bounds(self):
        """Exact bounds of trackset as tuple, no buffer."""
        return self.get_bounds(deg_buffer=0.0)

    def get_extent(self, deg_buffer=0.1):
        """Get extent as (lon_min, lon_max, lat_min, lat_max) tuple.

        Parameters
        ----------
        deg_buffer : float
            A buffer to add around the bounding box

        Returns
        -------
        extent : tuple (lon_min, lon_max, lat_min, lat_max)
        """
        bounds = self.get_bounds(deg_buffer=deg_buffer)
        return (bounds[0], bounds[2], bounds[1], bounds[3])

    @property
    def extent(self):
        """Exact extent of trackset as tuple, no buffer."""
        return self.get_extent(deg_buffer=0.0)

    def plot(self, axis=None, figsize=(9, 13), legend=True, **kwargs):
        """Track over earth. Historical events are blue, probabilistic black.

        Parameters
        ----------
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: (float, float), optional
            figure size for plt.subplots
            The default is (9, 13)
        legend : bool, optional
            whether to display a legend of Tropical Cyclone categories.
            Default: True.
        kwargs : optional
            arguments for LineCollection matplotlib, e.g. alpha=0.5

        Returns
        -------
        axis : matplotlib.axes._subplots.AxesSubplot
        """
        if 'lw' not in kwargs:
            kwargs['lw'] = 2
        if 'transform' not in kwargs:
            kwargs['transform'] = ccrs.PlateCarree()

        if not self.size:
            LOGGER.info('No tracks to plot')
            return None

        extent = self.get_extent(deg_buffer=1)
        mid_lon = 0.5 * (extent[1] + extent[0])

        if not axis:
            proj = ccrs.PlateCarree(central_longitude=mid_lon)
            _, axis = u_plot.make_map(proj=proj, figsize=figsize)
        axis.set_extent(extent, crs=kwargs['transform'])
        u_plot.add_shapes(axis)

        synth_flag = False
        cmap = ListedColormap(colors=CAT_COLORS)
        norm = BoundaryNorm([0] + SAFFIR_SIM_CAT, len(SAFFIR_SIM_CAT))
        for track in self.data:
            lonlat = np.stack([track.lon.values, track.lat.values], axis=-1)
            lonlat[:, 0] = u_coord.lon_normalize(lonlat[:, 0], center=mid_lon)
            segments = np.stack([lonlat[:-1], lonlat[1:]], axis=1)
            # remove segments which cross 180 degree longitude boundary
            segments = segments[segments[:, 0, 0] * segments[:, 1, 0] >= 0, :, :]
            if track.orig_event_flag:
                track_lc = LineCollection(segments, cmap=cmap, norm=norm,
                                          linestyle='solid', **kwargs)
            else:
                synth_flag = True
                track_lc = LineCollection(segments, cmap=cmap, norm=norm,
                                          linestyle=':', **kwargs)
            track_lc.set_array(track.max_sustained_wind.values)
            axis.add_collection(track_lc)

        if legend:
            leg_lines = [Line2D([0], [0], color=CAT_COLORS[i_col], lw=2)
                         for i_col in range(len(SAFFIR_SIM_CAT))]
            leg_names = [CAT_NAMES[i_col] for i_col in sorted(CAT_NAMES.keys())]
            if synth_flag:
                leg_lines.append(Line2D([0], [0], color='grey', lw=2, ls='solid'))
                leg_lines.append(Line2D([0], [0], color='grey', lw=2, ls=':'))
                leg_names.append('Historical')
                leg_names.append('Synthetic')
            axis.legend(leg_lines, leg_names, loc=0)

        return axis

    def write_netcdf(self, folder_name):
        """Write a netcdf file per track with track.sid name in given folder.

        Parameters
        ----------
        folder_name : str
            Folder name where to write files.
        """
        list_path = [Path(folder_name, track.sid + '.nc') for track in self.data]
        LOGGER.info('Writting %s files.', self.size)
        for track in self.data:
            track.attrs['orig_event_flag'] = int(track.orig_event_flag)
        xr.save_mfdataset(self.data, list_path)

    def read_netcdf(self, folder_name):
        """Read all netcdf files contained in folder and fill a track per file.

        Parameters
        ----------
        folder_name : str
            Folder name where to write files.
        """
        file_tr = get_file_names(folder_name)
        LOGGER.info('Reading %s files.', len(file_tr))
        self.data = list()
        for file in file_tr:
            if Path(file).suffix != '.nc':
                continue
            track = xr.open_dataset(file)
            track.attrs['orig_event_flag'] = bool(track.orig_event_flag)
            self.data.append(track)


    def to_geodataframe(self, as_points=False, split_lines_antimeridian=True):
        """Transform this TCTracks instance into a GeoDataFrame.

        Parameters
        ----------
        as_points : bool, optional
            If False (default), one feature (row) per track with a LineString or MultiLineString
            as geometry (or Point geometry for tracks of length one) and all track attributes
            (sid, name, orig_event_flag, etc) as dataframe columns. If True, one feature (row)
            per track time step, with variable values per time step (radius_max_wind,
            max_sustained_wind, etc) as columns in addition to attributes.
        split_lines_antimeridian : bool, optional
            If True, tracks that cross the antimeridian are split into multiple Lines as a
            MultiLineString, with each Line on either side of the meridian. This ensures all Lines
            are within (-180, +180) degrees longitude. Note that lines might be split at more
            locations than strictly necessary, due to the underlying splitting algorithm
            (https://github.com/Toblerity/Shapely/issues/572).

        Returns
        -------
        gdf : GeoDataFrame
        """
        gdf = gpd.GeoDataFrame(
            [dict(track.attrs) for track in self.data]
        )

        if as_points:
            gdf_long = pd.concat([track.to_dataframe().assign(idx=i)
                                  for i, track in enumerate(self.data)])
            gdf_long['lon'] = u_coord.lon_normalize(gdf_long['lon'].values.copy())
            gdf_long['geometry'] = gdf_long.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
            gdf_long = gdf_long.drop(columns=['lon', 'lat'])
            gdf_long = gpd.GeoDataFrame(gdf_long.reset_index().set_index('idx'),
                                        geometry='geometry', crs=DEF_CRS)
            gdf = gdf_long.join(gdf)

        elif split_lines_antimeridian:
            # enforce longitudes to be within [-180, 180] range
            t_lons = [u_coord.lon_normalize(t.lon.values.copy()) for t in self.data]
            t_lats = [t.lat.values for t in self.data]

            # LineString only works with more than one lat/lon pair
            gdf.geometry = gpd.GeoSeries([
                LineString(np.c_[lons, lats]) if lons.size > 1
                else Point(lons, lats)
                for lons, lats in zip(t_lons, t_lats)
            ])
            gdf.crs = DEF_CRS

            # for splitting, restrict to tracks that come close to the antimeridian
            t_split_mask = np.asarray([
                (lon > 170).any() and (lon < -170).any() and lon.size > 1
                for lon in t_lons])

            # note that tracks might be splitted at self-intersections as well:
            # https://github.com/Toblerity/Shapely/issues/572
            antimeridian = LineString([(180, -90), (180, 90)])
            gdf.loc[t_split_mask, "geometry"] = gdf.geometry[t_split_mask] \
                .to_crs({"proj": "longlat", "lon_wrap": 180}) \
                .apply(lambda line: MultiLineString([
                    LineString([(x - 360, y) for x, y in segment.coords])
                    if any(x > 180 for x, y in segment.coords) else segment
                    for segment in shapely.ops.split(line, antimeridian).geoms
                ]))
        else:
            # LineString only works with more than one lat/lon pair
            gdf.geometry = gpd.GeoSeries([
                LineString(np.c_[track.lon, track.lat]) if track.lon.size > 1
                else Point(track.lon.data, track.lat.data)
                for track in self.data
            ])
            gdf.crs = DEF_CRS

        return gdf

    @staticmethod
    @jit(parallel=True, forceobj=True)
    def _one_interp_data(track, time_step_h, land_geom=None):
        """Interpolate values of one track.

        Parameters
        ----------
        track : xr.Dataset
            Track data.
        time_step_h : int or float
            Desired temporal resolution in hours (may be non-integer-valued).
        land_geom : shapely.geometry.multipolygon.MultiPolygon, optional
            Land geometry. If given, recompute `dist_since_lf` and `on_land` property.

        Returns
        -------
        track_int : xr.Dataset
        """
        if track.time.size >= 2:
            method = ['linear', 'quadratic', 'cubic'][min(2, track.time.size - 2)]

            # handle change of sign in longitude
            lon = track.lon.copy()
            if (lon < -170).any() and (lon > 170).any():
                # crosses 180 degrees east/west -> use positive degrees east
                lon[lon < 0] += 360

            time_step = pd.tseries.frequencies.to_offset(pd.Timedelta(hours=time_step_h)).freqstr
            track_int = track.resample(time=time_step, keep_attrs=True, skipna=True)\
                             .interpolate('linear')
            track_int['time_step'][:] = time_step_h
            lon_int = lon.resample(time=time_step).interpolate(method)
            lon_int[lon_int > 180] -= 360
            track_int.coords['lon'] = lon_int
            track_int.coords['lat'] = track.lat.resample(time=time_step)\
                                               .interpolate(method)
            track_int.attrs['category'] = set_category(
                track_int.max_sustained_wind.values,
                track_int.max_sustained_wind_unit)
            # restrict to time steps within original bounds
            track_int = track_int.sel(
                time=(track.time[0] <= track_int.time) & (track_int.time <= track.time[-1]))
        else:
            LOGGER.warning('Track interpolation not done. '
                           'Not enough elements for %s', track.name)
            track_int = track

        if land_geom:
            track_land_params(track_int, land_geom)
        return track_int

    def _read_ibtracs_csv_single(self, file_name):
        """Read IBTrACS track file in CSV format.

        Parameters
        ----------
        file_name : str
            File name of CSV file.
        """
        LOGGER.info('Reading %s', file_name)
        dfr = pd.read_csv(file_name)
        name = dfr['ibtracsID'].values[0]

        datetimes = list()
        for time in dfr['isotime'].values:
            year = np.fix(time / 1e6)
            time = time - year * 1e6
            month = np.fix(time / 1e4)
            time = time - month * 1e4
            day = np.fix(time / 1e2)
            hour = time - day * 1e2
            datetimes.append(dt.datetime(int(year), int(month), int(day),
                                         int(hour)))

        lat = dfr['cgps_lat'].values.astype('float')
        lon = dfr['cgps_lon'].values.astype('float')
        cen_pres = dfr['pcen'].values.astype('float')
        max_sus_wind = dfr['vmax'].values.astype('float')
        max_sus_wind_unit = 'kn'
        if np.any(cen_pres <= 0):
            # Warning: If any pressure value is invalid, this enforces to use
            # estimated pressure values everywhere!
            cen_pres[:] = -999
            cen_pres = _estimate_pressure(cen_pres, lat, lon, max_sus_wind)

        tr_ds = xr.Dataset()
        tr_ds.coords['time'] = ('time', datetimes)
        tr_ds.coords['lat'] = ('time', lat)
        tr_ds.coords['lon'] = ('time', lon)
        tr_ds['time_step'] = ('time', dfr['tint'].values)
        tr_ds['radius_max_wind'] = ('time', dfr['rmax'].values.astype('float'))
        tr_ds['max_sustained_wind'] = ('time', max_sus_wind)
        tr_ds['central_pressure'] = ('time', cen_pres)
        tr_ds['environmental_pressure'] = ('time',
                                           dfr['penv'].values.astype('float'))
        tr_ds.attrs['max_sustained_wind_unit'] = max_sus_wind_unit
        tr_ds.attrs['central_pressure_unit'] = 'mb'
        tr_ds.attrs['name'] = name
        tr_ds.attrs['sid'] = name
        tr_ds.attrs['orig_event_flag'] = bool(dfr['original_data']. values[0])
        tr_ds.attrs['data_provider'] = dfr['data_provider'].values[0]
        tr_ds.attrs['basin'] = dfr['gen_basin'].values[0]
        try:
            tr_ds.attrs['id_no'] = float(name.replace('N', '0').
                                         replace('S', '1'))
        except ValueError:
            tr_ds.attrs['id_no'] = float(str(datetimes[0].date()).
                                         replace('-', ''))
        tr_ds.attrs['category'] = set_category(max_sus_wind, max_sus_wind_unit)

        self.data.append(tr_ds)


def track_land_params(track, land_geom):
    """Compute parameters of land for one track.

    Parameters
    ----------
    track : xr.Dataset
        tropical cyclone track
    land_geom : shapely.geometry.multipolygon.MultiPolygon
        land geometry
    """
    track['on_land'] = ('time',
                        u_coord.coord_on_land(track.lat.values, track.lon.values, land_geom))
    track['dist_since_lf'] = ('time', _dist_since_lf(track))

def _dist_since_lf(track):
    """Compute the distance to landfall in km point for every point on land.

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.

    Returns
    -------
    dist : np.arrray
        Distances in km, points on water get nan values.
    """
    dist_since_lf = np.zeros(track.time.values.shape)

    # Index in sea that follows a land index
    sea_land_idx, land_sea_idx = _get_landfall_idx(track, True)
    if not sea_land_idx.size:
        return (dist_since_lf + 1) * np.nan
    
    orig_lf = np.empty((sea_land_idx.size, 2))
    for i_lf, lf_point in enumerate(sea_land_idx):
        if lf_point > 0:
            # Assume the landfall started between this and the previous point
            orig_lf[i_lf][0] = track.lat[lf_point - 1] + \
                (track.lat[lf_point] - track.lat[lf_point - 1]) / 2
            orig_lf[i_lf][1] = track.lon[lf_point - 1] + \
                (track.lon[lf_point] - track.lon[lf_point - 1]) / 2
        else:
            # track starts over land, assume first 'landfall' starts here
            orig_lf[i_lf][0] = track.lat[lf_point]
            orig_lf[i_lf][1] = track.lon[lf_point]


    dist = DistanceMetric.get_metric('haversine')
    nodes1 = np.radians(np.array([track.lat.values[1:],
                                  track.lon.values[1:]]).transpose())
    nodes0 = np.radians(np.array([track.lat.values[:-1],
                                  track.lon.values[:-1]]).transpose())
    dist_since_lf[1:] = dist.pairwise(nodes1, nodes0).diagonal()
    dist_since_lf[~track.on_land.values] = 0.0
    nodes1 = np.array([track.lat.values[sea_land_idx],
                       track.lon.values[sea_land_idx]]).transpose() / 180 * np.pi
    dist_since_lf[sea_land_idx] = \
        dist.pairwise(nodes1, orig_lf / 180 * np.pi).diagonal()
    for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
        dist_since_lf[sea_land:land_sea] = \
            np.cumsum(dist_since_lf[sea_land:land_sea])

    dist_since_lf *= EARTH_RADIUS_KM
    dist_since_lf[~track.on_land.values] = np.nan

    return dist_since_lf

def _get_landfall_idx(track, include_starting_landfall=False):
    """Get the position of the start and end of landfalls for a TC track.

    Parameters
    ----------
    track : xr.Dataset
        track (variable 'on_land' must exist, if not present they can be added with
               climada.hazard.tc_tracks.track_land_params(track, land_geom))
    include_starting_landfall : bool
        If the track starts over land, whether to include the track segment before
        reaching the ocean as a landfall. Default: False.
        

    Returns
    -------
    sea_land_idx : numpy.ndarray of dtype int
        Indexes of the first point over land for each landfall
    land_sea_idx : numpy.ndarrat of dtype int
        Indexes of first point over the ocean after each landfall. If the track
        ends over land, the last value is set to track.time.size.
    """
    # Index in land that comes from previous sea index
    sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0] + 1
    # Index in sea that comes from previous land index
    land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
    if track.on_land[-1]:
        # track ends over land: add last track point as the end of that landfall
        land_sea_idx = np.append(land_sea_idx, track.time.size)
    if track.on_land[0]:
        # track starts over land: remove first land-to-sea transition (not a landfall)?
        if include_starting_landfall:
            sea_land_idx = np.append(1, sea_land_idx)
        else:
            land_sea_idx = land_sea_idx[1:]
    if land_sea_idx.size != sea_land_idx.size:
        raise ValueError('Mismatch')
    return sea_land_idx,land_sea_idx

def _estimate_pressure(cen_pres, lat, lon, v_max):
    """Replace missing pressure values with statistical estimate.

    In addition to NaNs, negative values and zeros in `cen_pres` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('pres', ['lat', 'lon', 'wind'], year_range=(1980, 2020))
    >>> r^2: 0.8726728075520206

    Parameters
    ----------
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).
    lat : array-like
        Latitudinal coordinates of eye location.
    lon : array-like
        Longitudinal coordinates of eye location.
    v_max : array-like
        Maximum wind speed along track in knots.

    Returns
    -------
    cen_pres_estimated : np.array
        Estimated central pressure values in hPa (mbar).
    """
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    v_max = np.where(np.isnan(v_max), -1, v_max)
    lat, lon = [np.where(np.isnan(ar), -999, ar) for ar in [lat, lon]]
    msk = (cen_pres <= 0) & (v_max > 0) & (lat > -999) & (lon > -999)
    c_const, c_lat, c_lon, c_vmax = 1026.3401, -0.05504, -0.03536, -0.7357
    cen_pres[msk] = c_const + c_lat * lat[msk] \
                            + c_lon * lon[msk] \
                            + c_vmax * v_max[msk]
    return np.where(cen_pres <= 0, np.nan, cen_pres)

def _estimate_vmax(v_max, lat, lon, cen_pres):
    """Replace missing wind speed values with a statistical estimate.

    In addition to NaNs, negative values and zeros in `v_max` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('wind', ['lat', 'lon', 'pres'], year_range=(1980, 2020))
    >>> r^2: 0.8683725434617979

    Parameters
    ----------
    v_max : array-like
        Maximum wind speed along track in knots.
    lat : array-like
        Latitudinal coordinates of eye location.
    lon : array-like
        Longitudinal coordinates of eye location.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    v_max_estimated : np.array
        Estimated maximum wind speed values in knots.
    """
    v_max = np.where(np.isnan(v_max), -1, v_max)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    lat, lon = [np.where(np.isnan(ar), -999, ar) for ar in [lat, lon]]
    msk = (v_max <= 0) & (cen_pres > 0) & (lat > -999) & (lon > -999)
    c_const, c_lat, c_lon, c_pres = 1216.5223, -0.04086, -0.04190, -1.1797
    v_max[msk] = c_const + c_lat * lat[msk] \
                         + c_lon * lon[msk] \
                         + c_pres * cen_pres[msk]
    return np.where(v_max <= 0, np.nan, v_max)

def estimate_roci(roci, cen_pres):
    """Replace missing radius (ROCI) values with statistical estimate.

    In addition to NaNs, negative values and zeros in `roci` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('roci', ['pres'],
    ...                   order=[(872, 950, 985, 1005, 1021)],
    ...                   year_range=(1980, 2019))
    >>> r^2: 0.9148320406675339

    Parameters
    ----------
    roci : array-like
        ROCI values along track in km.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    roci_estimated : np.array
        Estimated ROCI values in km.
    """
    roci = np.where(np.isnan(roci), -1, roci)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    msk = (roci <= 0) & (cen_pres > 0)
    pres_l = [872, 950, 985, 1005, 1021]
    roci_l = [210.711487, 215.897110, 198.261520, 159.589508, 90.900116]
    roci[msk] = 0
    for i, pres_l_i in enumerate(pres_l):
        slope_0 = 1. / (pres_l_i - pres_l[i - 1]) if i > 0 else 0
        slope_1 = 1. / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0
        roci[msk] += roci_l[i] * np.fmax(0, (1 - slope_0 * np.fmax(0, pres_l_i - cen_pres[msk])
                                             - slope_1 * np.fmax(0, cen_pres[msk] - pres_l_i)))
    return np.where(roci <= 0, np.nan, roci)

def estimate_rmw(rmw, cen_pres):
    """Replace missing radius (RMW) values with statistical estimate.

    In addition to NaNs, negative values and zeros in `rmw` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('rmw', ['pres'], order=[(872, 940, 980, 1021)], year_range=(1980, 2019))
    >>> r^2: 0.7905970811843872

    Parameters
    ----------
    rmw : array-like
        RMW values along track in km.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    rmw : np.array
        Estimated RMW values in km.
    """
    rmw = np.where(np.isnan(rmw), -1, rmw)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    msk = (rmw <= 0) & (cen_pres > 0)
    pres_l = [872, 940, 980, 1021]
    rmw_l = [14.907318, 15.726927, 25.742142, 56.856522]
    rmw[msk] = 0
    for i, pres_l_i in enumerate(pres_l):
        slope_0 = 1. / (pres_l_i - pres_l[i - 1]) if i > 0 else 0
        slope_1 = 1. / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0
        rmw[msk] += rmw_l[i] * np.fmax(0, (1 - slope_0 * np.fmax(0, pres_l_i - cen_pres[msk])
                                           - slope_1 * np.fmax(0, cen_pres[msk] - pres_l_i)))
    return np.where(rmw <= 0, np.nan, rmw)

def ibtracs_fit_param(explained, explanatory, year_range=(1980, 2019), order=1):
    """Statistically fit an ibtracs parameter to other ibtracs variables.

    A linear ordinary least squares fit is done using the statsmodels package.

    Parameters
    ----------
    explained : str
        Name of explained variable.
    explanatory : iterable
        Names of explanatory variables.
    year_range : tuple
        First and last year to include in the analysis.
    order : int or tuple
        The maximal order of the explanatory variables.

    Returns
    -------
    result : OLSResults
    """
    wmo_vars = ['wind', 'pres', 'rmw', 'roci', 'poci']
    all_vars = ['lat', 'lon'] + wmo_vars
    explanatory = list(explanatory)
    variables = explanatory + [explained]
    for var in variables:
        if var not in all_vars:
            LOGGER.error("Unknown ibtracs variable: %s", var)
            raise KeyError

    # load ibtracs dataset
    fn_nc = SYSTEM_DIR.joinpath('IBTrACS.ALL.v04r00.nc')
    ibtracs_ds = xr.open_dataset(fn_nc)

    # choose specified year range
    years = ibtracs_ds.sid.str.slice(0, 4).astype(int)
    match = (years >= year_range[0]) & (years <= year_range[1])
    ibtracs_ds = ibtracs_ds.sel(storm=match)

    if "wind" in variables:
        for agency in IBTRACS_AGENCIES:
            scale, shift = IBTRACS_AGENCY_1MIN_WIND_FACTOR[agency]
            ibtracs_ds[f'{agency}_wind'] -= shift
            ibtracs_ds[f'{agency}_wind'] /= scale

    # fill values
    agency_pref, track_agency_ix = ibtracs_track_agency(ibtracs_ds)
    for var in wmo_vars:
        if var not in variables:
            continue
        # array of values in order of preference
        cols = [f'{a}_{var}' for a in agency_pref]
        cols = [col for col in cols if col in ibtracs_ds.data_vars.keys()]
        all_vals = ibtracs_ds[cols].to_array(dim='agency')
        preferred_ix = all_vals.notnull().argmax(dim='agency')
        if var in ['wind', 'pres']:
            # choice: wmo -> wmo_agency/usa_agency -> preferred
            ibtracs_ds[var] = ibtracs_ds['wmo_' + var] \
                .fillna(all_vals.isel(agency=track_agency_ix)) \
                .fillna(all_vals.isel(agency=preferred_ix))
        else:
            ibtracs_ds[var] = all_vals.isel(agency=preferred_ix)
    fit_df = pd.DataFrame({var: ibtracs_ds[var].values.ravel() for var in variables})
    fit_df = fit_df.dropna(axis=0, how='any').reset_index(drop=True)
    if 'lat' in explanatory:
        fit_df['lat'] = fit_df['lat'].abs()

    # prepare explanatory variables
    d_explanatory = fit_df[explanatory]
    if isinstance(order, int):
        order = (order,) * len(explanatory)
    add_const = False
    for ex, max_o in zip(explanatory, order):
        if isinstance(max_o, tuple):
            if fit_df[ex].min() > max_o[0]:
                print(f"Minimum data value is {fit_df[ex].min()} > {max_o[0]}.")
            if fit_df[ex].max() < max_o[-1]:
                print(f"Maximum data value is {fit_df[ex].max()} < {max_o[-1]}.")
            # piecewise linear with given break points
            d_explanatory = d_explanatory.drop(labels=[ex], axis=1)
            for i, max_o_i in enumerate(max_o):
                col = f'{ex}{max_o_i}'
                slope_0 = 1. / (max_o_i - max_o[i - 1]) if i > 0 else 0
                slope_1 = 1. / (max_o[i + 1] - max_o_i) if i + 1 < len(max_o) else 0
                d_explanatory[col] = np.fmax(0, (1 - slope_0 * np.fmax(0, max_o_i - fit_df[ex])
                                                 - slope_1 * np.fmax(0, fit_df[ex] - max_o_i)))
        elif max_o < 0:
            d_explanatory = d_explanatory.drop(labels=[ex], axis=1)
            for order in range(1, abs(max_o) + 1):
                d_explanatory[f'{ex}^{-order}'] = fit_df[ex]**(-order)
            add_const = True
        else:
            for order in range(2, max_o + 1):
                d_explanatory[f'{ex}^{order}'] = fit_df[ex]**order
            add_const = True
    d_explained = fit_df[[explained]]
    if add_const:
        d_explanatory['const'] = 1.0

    # run statistical fit
    sm_results = sm.OLS(d_explained, d_explanatory).fit()

    # print results
    print(sm_results.params)
    print("r^2:", sm_results.rsquared)

    return sm_results

def ibtracs_track_agency(ds_sel):
    """Get preferred IBTrACS agency for each entry in the dataset.

    Parameters
    ----------
    ds_sel : xarray.Dataset
        Subselection of original IBTrACS NetCDF dataset.

    Returns
    -------
    agency_pref : list of str
        Names of IBTrACS agencies in order of preference.
    track_agency_ix : xarray.DataArray of ints
        For each entry in `ds_sel`, the agency to use, given as an index into `agency_pref`.
    """
    agency_pref = ["wmo"] + IBTRACS_AGENCIES
    agency_map = {a.encode('utf-8'): i for i, a in enumerate(agency_pref)}
    agency_map.update({
        a.encode('utf-8'): agency_map[b'usa'] for a in IBTRACS_USA_AGENCIES
    })
    agency_map[b''] = agency_map[b'wmo']
    agency_fun = lambda x: agency_map[x]
    if "track_agency" not in ds_sel.data_vars.keys():
        ds_sel['track_agency'] = ds_sel.wmo_agency.where(ds_sel.wmo_agency != b'',
                                                         ds_sel.usa_agency)
    track_agency_ix = xr.apply_ufunc(agency_fun, ds_sel.track_agency, vectorize=True)
    return agency_pref, track_agency_ix

def ibtracs_add_official_variable(ibtracs_ds, tc_var, add_3h=False):
    """Add variables for the officially responsible agencies to an IBTrACS dataset

    This function adds new variables to the xarray.Dataset `ibtracs_ds` that contain values of the
    specified TC variable `var` that have been reported by the officially responsible agencies.
    For example, if `tc_var` is "wind", there will be a new variable "official_wind" and, if
    `add_3h` is True, an additional variable "official_3h_wind".

    Parameters
    ----------
    ibtracs_ds : xarray.Dataset
        Subselection of original IBTrACS NetCDF dataset.
    tc_var : str
        Name of variable for which to add an "official" version, e.g. "lat", "wind", "pres".
    add_3h : bool, optional
        Optionally, add an "official_3h" version where also 3-hourly data by the officially
        reporting agencies is included (if available). Default: False
    """
    if "nan_var" not in ibtracs_ds.data_vars.keys():
        # add an array full of NaN as a fallback value in the procedure
        ibtracs_ds['nan_var'] = xr.full_like(ibtracs_ds.lat, np.nan)

    # determine which of the official agencies report this variable at all
    available_agencies = [a for a in IBTRACS_AGENCIES
                          if f'{a}_{tc_var}' in ibtracs_ds.data_vars.keys()]

    # map all non-reporting agency variables to the 'nan_var' (0)
    agency_map = {
        a.encode("utf-8"): available_agencies.index(a) + 1 if a in available_agencies else 0
        for a in [''] + IBTRACS_AGENCIES
    }
    agency_map.update({
        a.encode('utf-8'): agency_map[b'usa'] for a in IBTRACS_USA_AGENCIES
    })

    # read from officially responsible agencies that report this variable, but only
    # at official reporting times (usually 6-hourly)
    official_agency_ix = xr.apply_ufunc(
        lambda x: agency_map[x], ibtracs_ds.wmo_agency, vectorize=True)
    available_cols = ['nan_var'] + [f'{a}_{tc_var}' for a in available_agencies]
    all_vals = ibtracs_ds[available_cols].to_array(dim='agency')
    ibtracs_ds[f'official_{tc_var}'] = all_vals.isel(agency=official_agency_ix)

    if add_3h:
        # create a copy in float for NaN interpolation
        official_agency_ix_interp = official_agency_ix.astype(np.float16)

        # extrapolate track agency for tracks with only a single record
        mask_singular = ((official_agency_ix_interp > 0).sum(dim="date_time") == 1).values
        official_agency_ix_interp.values[mask_singular,:] = \
            official_agency_ix_interp.sel(storm=mask_singular).max(dim="date_time").values[:,None]

        with warnings.catch_warnings():
            # See https://github.com/pydata/xarray/issues/4167
            warnings.simplefilter(action="ignore", category=FutureWarning)

            # interpolate responsible agencies using nearest neighbor interpolation
            official_agency_ix_interp.values[official_agency_ix_interp.values == 0.0] = np.nan
            official_agency_ix_interp = official_agency_ix_interp.interpolate_na(
                dim="date_time", method="nearest", fill_value="extrapolate")

        # read from officially responsible agencies that report this variable, including
        # 3-hour time steps if available
        official_agency_ix_interp.values[official_agency_ix_interp.isnull().values] = 0.0
        ibtracs_ds[f'official_3h_{tc_var}'] = all_vals.isel(
            agency=official_agency_ix_interp.astype(int))

def _change_max_wind_unit(wind, unit_orig, unit_dest):
    """Compute maximum wind speed in unit_dest.

    Parameters
    ----------
    wind : np.array
        Wind speed values in original units.
    unit_orig : str
        Original units of wind speed.
    unit_dest : str
        New units of the computed maximum wind speed.

    Returns
    -------
    maxwind : double
        Maximum wind speed in specified wind speed units.
    """
    if unit_orig in ('kn', 'kt'):
        ur_orig = ureg.knot
    elif unit_orig == 'mph':
        ur_orig = ureg.mile / ureg.hour
    elif unit_orig == 'm/s':
        ur_orig = ureg.meter / ureg.second
    elif unit_orig == 'km/h':
        ur_orig = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Unit not recognised %s.', unit_orig)
        raise ValueError
    if unit_dest in ('kn', 'kt'):
        ur_dest = ureg.knot
    elif unit_dest == 'mph':
        ur_dest = ureg.mile / ureg.hour
    elif unit_dest == 'm/s':
        ur_dest = ureg.meter / ureg.second
    elif unit_dest == 'km/h':
        ur_dest = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Unit not recognised %s.', unit_dest)
        raise ValueError
    return (np.nanmax(wind) * ur_orig).to(ur_dest).magnitude

def set_category(max_sus_wind, wind_unit='kn', saffir_scale=None):
    """Add storm category according to Saffir-Simpson hurricane scale.

    Parameters
    ----------
    max_sus_wind : np.array
        Maximum sustained wind speed records for a single track.
    wind_unit : str, optional
        Units of wind speed. Default: 'kn'.
    saffir_scale : list, optional
        Saffir-Simpson scale in same units as wind (default scale valid for knots).

    Returns
    -------
    category : int
        Intensity of given track according to the Saffir-Simpson hurricane scale:
          * -1 : tropical depression
          *  0 : tropical storm
          *  1 : Hurricane category 1
          *  2 : Hurricane category 2
          *  3 : Hurricane category 3
          *  4 : Hurricane category 4
          *  5 : Hurricane category 5
    """
    if saffir_scale is None:
        saffir_scale = SAFFIR_SIM_CAT
        if wind_unit != 'kn':
            max_sus_wind = _change_max_wind_unit(max_sus_wind, wind_unit, 'kn')
    max_wind = np.nanmax(max_sus_wind)
    try:
        return (np.argwhere(max_wind < saffir_scale) - 1)[0][0]
    except IndexError:
        return -1

