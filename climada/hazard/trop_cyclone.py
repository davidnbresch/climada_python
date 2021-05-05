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

Define TropCyclone class.
"""

__all__ = ['TropCyclone']

import itertools
import logging
import copy
import time
import datetime as dt
import numpy as np
from scipy import sparse
import matplotlib.animation as animation
from tqdm import tqdm

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.tc_tracks import TCTracks, estimate_rmw
from climada.hazard.tc_clim_change import get_knutson_criterion, calc_scale_knutson
from climada.hazard.centroids.centr import Centroids
from climada.util import ureg
import climada.util.coordinates as u_coord
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TC'
"""Hazard type acronym for Tropical Cyclone"""

INLAND_MAX_DIST_KM = 1000
"""Maximum inland distance of the centroids in km"""

CENTR_NODE_MAX_DIST_KM = 300
"""Maximum distance between centroid and TC track node in km"""

CENTR_NODE_MAX_DIST_DEG = 5.5
"""Maximum distance between centroid and TC track node in degrees"""

MODEL_VANG = {'H08': 0}
"""Enumerate different symmetric wind field calculation."""

KMH_TO_MS = (1.0 * ureg.km / ureg.hour).to(ureg.meter / ureg.second).magnitude
KN_TO_MS = (1.0 * ureg.knot).to(ureg.meter / ureg.second).magnitude
NM_TO_KM = (1.0 * ureg.nautical_mile).to(ureg.kilometer).magnitude
"""Unit conversion factors for JIT functions that can't use ureg"""

class TropCyclone(Hazard):
    """
    Contains tropical cyclone events.

    Attributes
    ----------
    category : np.array(int)
        for every event, the TC category using the
        Saffir-Simpson scale:
            -1 tropical depression
             0 tropical storm
             1 Hurrican category 1
             2 Hurrican category 2
             3 Hurrican category 3
             4 Hurrican category 4
             5 Hurrican category 5
    basin : list(str)
        basin where every event starts
        'NA' North Atlantic
        'EP' Eastern North Pacific
        'WP' Western North Pacific
        'NI' North Indian
        'SI' South Indian
        'SP' Southern Pacific
        'SA' South Atlantic
    """
    intensity_thres = 17.5
    """intensity threshold for storage in m/s"""

    vars_opt = Hazard.vars_opt.union({'category'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self, pool=None):
        """Empty constructor."""
        Hazard.__init__(self, HAZ_TYPE)
        self.category = np.array([], int)
        self.basin = list()
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def set_from_tracks(self, tracks, centroids=None, description='',
                        model='H08', ignore_distance_to_coast=False,
                        store_windfields=False, metric="equirect"):
        """
        Clear and fill with windfields from specified tracks.

        Parameters
        ----------
        tracks : TCTracks
            Tracks of storm events.
        centroids : Centroids, optional
            Centroids where to model TC. Default: global centroids at 360 arc-seconds resolution.
        description : str, optional
            Description of the event set. Default: "".
        model : str, optional
            Model to compute gust. Currently only 'H08' is supported for the one implemented in
            `_stat_holland` according to Greg Holland. Default: "H08".
        ignore_distance_to_coast : boolean, optional
            If True, centroids far from coast are not ignored. Default: False.
        store_windfields : boolean, optional
            If True, the Hazard object gets a list `windfields` of sparse matrices. For each track,
            the full velocity vectors at each centroid and track position are stored in a sparse
            matrix of shape (npositions,  ncentroids * 2) that can be reshaped to a full ndarray
            of shape (npositions, ncentroids, 2). Default: False.
        metric : str, optional
            Specify an approximation method to use for earth distances:
            * "equirect": Distance according to sinusoidal projection. Fast, but inaccurate for
              large distances and high latitudes.
            * "geosphere": Exact spherical distance. Much more accurate at all distances, but slow.
            Default: "equirect".

        Raises
        ------
        ValueError
        """
        num_tracks = tracks.size
        if centroids is None:
            centroids = Centroids.from_base_grid(res_as=360, land=False)

        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        if ignore_distance_to_coast:
            # Select centroids with lat < 61
            coastal_idx = (np.abs(centroids.lat) < 61).nonzero()[0]
        else:
            # Select centroids which are inside INLAND_MAX_DIST_KM and lat < 61
            if not centroids.dist_coast.size:
                centroids.set_dist_coast()
            coastal_idx = ((centroids.dist_coast < INLAND_MAX_DIST_KM * 1000)
                           & (np.abs(centroids.lat) < 61)).nonzero()[0]

        # Restrict to coastal centroids within reach of any of the tracks
        t_lon_min, t_lat_min, t_lon_max, t_lat_max = tracks.get_bounds(
            deg_buffer=CENTR_NODE_MAX_DIST_DEG)
        t_mid_lon = 0.5 * (t_lon_min + t_lon_max)
        coastal_centroids = centroids.coord[coastal_idx]
        u_coord.lon_normalize(coastal_centroids[:, 1], center=t_mid_lon)
        coastal_idx = coastal_idx[((t_lon_min <= coastal_centroids[:, 1])
                                   & (coastal_centroids[:, 1] <= t_lon_max)
                                   & (t_lat_min <= coastal_centroids[:, 0])
                                   & (coastal_centroids[:, 0] <= t_lat_max))]

        LOGGER.info('Mapping %s tracks to %s coastal centroids.', str(tracks.size),
                    str(coastal_idx.size))
        if self.pool:
            chunksize = min(num_tracks // self.pool.ncpus, 1000)
            tc_haz = self.pool.map(
                self._tc_from_track, tracks.data,
                itertools.repeat(centroids, num_tracks),
                itertools.repeat(coastal_idx, num_tracks),
                itertools.repeat(model, num_tracks),
                itertools.repeat(store_windfields, num_tracks),
                itertools.repeat(metric, num_tracks),
                chunksize=chunksize)
        else:
            last_perc = 0
            tc_haz = []
            for track in tracks.data:
                perc = 100 * len(tc_haz) / len(tracks.data)
                if perc - last_perc >= 10:
                    LOGGER.info("Progress: %d%%", perc)
                    last_perc = perc
                tc_haz.append(
                    self._tc_from_track(track, centroids, coastal_idx,
                                        model=model, store_windfields=store_windfields,
                                        metric=metric))
            if last_perc < 100:
                LOGGER.info("Progress: 100%")
        LOGGER.debug('Append events.')
        self.concatenate(tc_haz)
        LOGGER.debug('Compute frequency.')
        self.frequency_from_tracks(tracks.data)
        self.tag.description = description

    def set_climate_scenario_knu(self, ref_year=2050, rcp_scenario=45):
        """
        Compute future events for a given RCP scenario and year based on the
        parametrized values derived from Table 3 in Knutson et al 2015.
        https://doi.org/10.1175/JCLI-D-15-0129.1 . The scaling for different
        years and RCP scenarios is obtained by linear interpolation.ß

        Note: The parametrized values are derived from the overall changes
        in statistical ensemble of tracks. Hence, this method should only be
        applied to sufficiently large tropical cyclone event sets that
        approximate the reference years 1981 - 2008 used in Knutson et. al.

        The frequency and intensity changes are applied independently from
        one another. The mean intensity factors can thus slightly deviate
        from the Knutson value (deviation was found to be less than 1%
        for default IBTrACS event sets 1980-2020 for each basin).

        Parameters
        ----------
        ref_year : int
            year between 2000 ad 2100. Default: 2050
        rcp_scenario : int
            26 for RCP 2.6, 45 for RCP 4.5, 60 for RCP 6.0 and 85 for RCP 8.5.
            The default is 45.

        Returns
        -------
        haz_cc : climada.hazard.TropCyclone
            Tropical cyclone with frequencies and intensity scaled according
            to the Knutson criterion for the given year and RCP. Returns
            a new instance of climada.hazard.TropCyclone, self is not
            modified.
        """
        chg_int_freq = get_knutson_criterion()
        scale_rcp_year  = calc_scale_knutson(ref_year, rcp_scenario)
        haz_cc = self._apply_knutson_criterion(chg_int_freq, scale_rcp_year)
        haz_cc.tag.description = 'climate change scenario for year %s and RCP %s '\
        'from Knutson et al 2015.' % (str(ref_year), str(rcp_scenario))
        return haz_cc

    @staticmethod
    def video_intensity(track_name, tracks, centroids, file_name=None,
                        writer=animation.PillowWriter(bitrate=500),
                        figsize=(9, 13), **kwargs):
        """
        Generate video of TC wind fields node by node and returns its
        corresponding TropCyclone instances and track pieces.

        Parameters
        ----------
        track_name : str
            name of the track contained in tracks to record
        tracks : climada.hazard.TCTracks
            tropical cyclone tracks
        centroids : climada.hazard.Centroids
            centroids where wind fields are mapped
        file_name : str, optional
            file name to save video (including full path and file extension)
        writer : matplotlib.animation.*, optional
            video writer. Default is pillow with bitrate=500
        figsize : tuple, optional
            figure size for plt.subplots
        kwargs : optional
            arguments for pcolormesh matplotlib function used in event plots

        Returns
        -------
        tc_list, tc_coord : list(TropCyclone), list(np.array)

        Raises
        ------
        ValueError

        """
        # initialization
        track = tracks.get_track(track_name)
        if not track:
            raise ValueError('%s not found in track data.' % track_name)
        idx_plt = np.argwhere(
            (track.lon.values < centroids.total_bounds[2] + 1)
            & (centroids.total_bounds[0] - 1 < track.lon.values)
            & (track.lat.values < centroids.total_bounds[3] + 1)
            & (centroids.total_bounds[1] - 1 < track.lat.values)
        ).reshape(-1)

        tc_list = []
        tr_coord = {'lat': [], 'lon': []}
        for node in range(idx_plt.size - 2):
            tr_piece = track.sel(
                time=slice(track.time.values[idx_plt[node]],
                           track.time.values[idx_plt[node + 2]]))
            tr_piece.attrs['n_nodes'] = 2  # plot only one node
            tr_sel = TCTracks()
            tr_sel.append(tr_piece)
            tr_coord['lat'].append(tr_sel.data[0].lat.values[:-1])
            tr_coord['lon'].append(tr_sel.data[0].lon.values[:-1])

            tc_tmp = TropCyclone()
            tc_tmp.set_from_tracks(tr_sel, centroids)
            tc_tmp.event_name = [
                track.name + ' ' + time.strftime(
                    "%d %h %Y %H:%M",
                    time.gmtime(tr_sel.data[0].time[1].values.astype(int)
                                / 1000000000)
                )
            ]
            tc_list.append(tc_tmp)

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'Greys'
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.array([tc_.intensity.min() for tc_ in tc_list]).min()
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.array([tc_.intensity.max() for tc_ in tc_list]).max()

        def run(node):
            tc_list[node].plot_intensity(1, axis=axis, **kwargs)
            axis.plot(tr_coord['lon'][node], tr_coord['lat'][node], 'k')
            axis.set_title(tc_list[node].event_name[0])
            pbar.update()

        if file_name:
            LOGGER.info('Generating video %s', file_name)
            fig, axis = u_plot.make_map(figsize=figsize)
            pbar = tqdm(total=idx_plt.size - 2)
            ani = animation.FuncAnimation(fig, run, frames=idx_plt.size - 2,
                                          interval=500, blit=False)
            ani.save(file_name, writer=writer)
            pbar.close()
        return tc_list, tr_coord

    def frequency_from_tracks(self, tracks):
        """
        Set hazard frequency from tracks data.

        Parameters
        ----------
        tracks : list of xarray.Dataset
        """
        if not tracks:
            return
        year_max = np.amax([t.time.dt.year.values.max() for t in tracks])
        year_min = np.amin([t.time.dt.year.values.min() for t in tracks])
        year_delta = year_max - year_min + 1
        num_orig = np.count_nonzero(self.orig)
        ens_size = (self.event_id.size / num_orig) if num_orig > 0 else 1
        self.frequency = np.ones(self.event_id.size) / (year_delta * ens_size)

    def _tc_from_track(self, track, centroids, coastal_idx, model='H08',
                       store_windfields=False, metric="equirect"):
        """
        Generate windfield hazard from a single track dataset

        Parameters
        ----------
        track : xr.Dataset
            Single tropical cyclone track.
        centroids : Centroids
            Centroids instance.
        coastal_idx : np.array
            Indices of centroids close to coast.
        model : str, optional
            Windfield model. Default: H08.
        store_windfields : boolean, optional
            If True, store windfields. Default: False.
        metric : str, optional
            Specify an approximation method to use for earth distances: "equirect" (faster) or
            "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
            Default: "equirect".

        Raises
        ------
        ValueError, KeyError

        Returns
        -------
        haz : TropCyclone
        """
        try:
            mod_id = MODEL_VANG[model]
        except KeyError as err:
            raise ValueError(f'Model not implemented: {model}.') from err
        ncentroids = centroids.coord.shape[0]
        coastal_centr = centroids.coord[coastal_idx]
        windfields, reachable_centr_idx = compute_windfields(track, coastal_centr, mod_id,
                                                             metric=metric)
        reachable_coastal_centr_idx = coastal_idx[reachable_centr_idx]
        npositions = windfields.shape[0]

        intensity = np.linalg.norm(windfields, axis=-1).max(axis=0)
        intensity[intensity < self.intensity_thres] = 0
        intensity_sparse = sparse.csr_matrix(
            (intensity, reachable_coastal_centr_idx, [0, intensity.size]),
            shape=(1, ncentroids))
        intensity_sparse.eliminate_zeros()

        new_haz = TropCyclone()
        new_haz.tag = TagHazard(HAZ_TYPE, 'Name: ' + track.name)
        new_haz.intensity = intensity_sparse
        if store_windfields:
            n_reachable_coastal_centr = reachable_coastal_centr_idx.size
            indices = np.zeros((npositions, n_reachable_coastal_centr, 2), dtype=np.int64)
            indices[:, :, 0] = 2 * reachable_coastal_centr_idx[None]
            indices[:, :, 1] = 2 * reachable_coastal_centr_idx[None] + 1
            indices = indices.ravel()
            indptr = np.arange(npositions + 1) * n_reachable_coastal_centr * 2
            windfields_sparse = sparse.csr_matrix((windfields.ravel(), indices, indptr),
                                                  shape=(npositions, ncentroids * 2))
            windfields_sparse.eliminate_zeros()
            new_haz.windfields = [windfields_sparse]
        new_haz.units = 'm/s'
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.event_name = [track.sid]
        new_haz.fraction = new_haz.intensity.copy()
        new_haz.fraction.data.fill(1)
        # store first day of track as date
        new_haz.date = np.array([
            dt.datetime(track.time.dt.year[0],
                        track.time.dt.month[0],
                        track.time.dt.day[0]).toordinal()
        ])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.category = np.array([track.category])
        new_haz.basin = [str(track.basin.values[0])]
        return new_haz

    def _apply_knutson_criterion(self, chg_int_freq, scaling_rcp_year):
        """
        Apply changes to intensities and cumulative frequencies.

        Parameters
        ----------
        criterion : list(dict))
            list of criteria from climada.hazard.tc_clim_change
        scale : float
            scale parameter because of chosen year and RCP
        Returns
        -------
        tc_cc : climada.hazard.TropCyclone
            Tropical cyclone with frequency and intensity scaled according
            to the Knutson criterion. Returns a new instance of TropCyclone.
        """

        tc_cc = copy.deepcopy(self)

        # Criterion per basin
        for basin in np.unique(tc_cc.basin):

            bas_sel = (np.array(tc_cc.basin) == basin)

            # Apply intensity change
            inten_chg = [chg
                         for chg in chg_int_freq
                         if (chg['variable'] == 'intensity' and
                             chg['basin'] == basin)
                         ]
            for chg in inten_chg:
                sel_cat_chg = np.isin(tc_cc.category, chg['category']) & bas_sel
                inten_scaling = 1 + (chg['change'] - 1) * scaling_rcp_year
                tc_cc.intensity = sparse.diags(
                    np.where(sel_cat_chg, inten_scaling, 1)
                    ).dot(tc_cc.intensity)

            # Apply frequency change
            freq_chg = [chg
                        for chg in chg_int_freq
                        if (chg['variable'] == 'frequency' and
                            chg['basin'] == basin)
                        ]
            freq_chg.sort(reverse=False, key=lambda x: len(x['category']))

            # Iteratively scale frequencies for each category such that
            # cumulative frequencies are scaled according to Knutson criterion.

            cat_larger_list = []
            for chg in freq_chg:
                cat_chg_list = [cat
                                for cat in chg['category']
                                if cat not in cat_larger_list
                                ]
                sel_cat_chg = np.isin(tc_cc.category, cat_chg_list) & bas_sel
                if sel_cat_chg.any():
                    freq_scaling = 1 + (chg['change'] - 1) * scaling_rcp_year
                    sel_cat_all = (np.isin(tc_cc.category, chg['category'])
                                   & bas_sel)
                    sel_cat_larger = (np.isin(tc_cc.category, cat_larger_list)
                                      & bas_sel)
                    freq_scaling_cor = (
                        (np.sum(self.frequency[sel_cat_all]) * freq_scaling
                         - np.sum(tc_cc.frequency[sel_cat_larger]))
                        / np.sum(self.frequency[sel_cat_chg])
                    )
                    tc_cc.frequency[sel_cat_chg] *= freq_scaling_cor
                cat_larger_list += cat_chg_list

        if (tc_cc.frequency < 0).any():
            raise ValueError("The application of the given climate scenario"
                             "resulted in at least one negative frequenciy."
                             "This is likely due to the use of a"
                             "non-representative event set (too small, "
                             "incorrect reference period, ...)")

        return tc_cc


def compute_windfields(track, centroids, model, metric="equirect"):
    """Compute 1-minute sustained winds (in m/s) at 10 meters above ground

    In a first step, centroids within reach of the track are determined so that wind fields will
    only be computed and returned for those centroids.

    Parameters
    ----------
    track : xr.Dataset
        Track infomation.
    centroids : 2d np.array
        Each row is a centroid [lat, lon].
        Centroids that are not within reach of the track are ignored.
    model : int
        Holland model selection according to MODEL_VANG.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".

    Returns
    -------
    windfields : np.array of shape (npositions, nreachable, 2)
        Directional wind fields for each track position on those centroids within reach
        of the TC track.
    reachable_centr_idx : np.array of shape (nreachable,)
        List of indices of input centroids within reach of the TC track.
    """
    # copies of track data
    # Note that max wind records are not used in the Holland wind field models!
    t_lat, t_lon, t_tstep, t_rad, t_env, t_cen = [
        track[ar].values.copy() for ar in ['lat', 'lon', 'time_step', 'radius_max_wind',
                                           'environmental_pressure', 'central_pressure']
    ]

    # start with the assumption that no centroids are within reach
    npositions = t_lat.shape[0]
    reachable_centr_idx = np.zeros((0,), dtype=np.int64)
    windfields = np.zeros((npositions, 0, 2), dtype=np.float64)

    # the wind field model requires at least two track positions because translational speed
    # as well as the change in pressure are required
    if npositions < 2:
        return windfields, reachable_centr_idx

    # normalize longitude values (improves performance of `dist_approx` and `_close_centroids`)
    mid_lon = 0.5 * sum(u_coord.lon_bounds(t_lon))
    u_coord.lon_normalize(t_lon, center=mid_lon)
    u_coord.lon_normalize(centroids[:, 1], center=mid_lon)

    # restrict to centroids within rectangular bounding boxes around track positions
    track_centr_msk = _close_centroids(t_lat, t_lon, centroids)
    track_centr = centroids[track_centr_msk]
    nreachable = track_centr.shape[0]
    if nreachable == 0:
        return windfields, reachable_centr_idx

    # compute distances and vectors to all centroids
    [d_centr], [v_centr] = u_coord.dist_approx(t_lat[None], t_lon[None],
                                               track_centr[None, :, 0], track_centr[None, :, 1],
                                               log=True, normalize=False, method=metric)

    # exclude centroids that are too far from or too close to the eye
    close_centr_msk = (d_centr < CENTR_NODE_MAX_DIST_KM) & (d_centr > 1e-2)
    if not np.any(close_centr_msk):
        return windfields, reachable_centr_idx
    v_centr_normed = np.zeros_like(v_centr)
    v_centr_normed[close_centr_msk] = v_centr[close_centr_msk] / d_centr[close_centr_msk, None]

    # make sure that central pressure never exceeds environmental pressure
    pres_exceed_msk = (t_cen > t_env)
    t_cen[pres_exceed_msk] = t_env[pres_exceed_msk]

    # extrapolate radius of max wind from pressure if not given
    t_rad[:] = estimate_rmw(t_rad, t_cen) * NM_TO_KM

    # translational speed of track at every node
    v_trans = _vtrans(t_lat, t_lon, t_tstep, metric=metric)
    v_trans_norm = v_trans[0]

    # adjust pressure at previous track point
    prev_pres = t_cen[:-1].copy()
    msk = (prev_pres < 850)
    prev_pres[msk] = t_cen[1:][msk]

    # compute b-value
    if model == 0:
        hol_b = _bs_hol08(v_trans_norm[1:], t_env[1:], t_cen[1:], prev_pres,
                          t_lat[1:], t_tstep[1:])
    else:
        raise NotImplementedError

    # derive angular velocity
    v_ang_norm = _stat_holland(d_centr[1:], t_rad[1:], hol_b, t_env[1:],
                               t_cen[1:], t_lat[1:], close_centr_msk[1:])
    hemisphere = 'N'
    if np.count_nonzero(t_lat < 0) > np.count_nonzero(t_lat > 0):
        hemisphere = 'S'
    v_ang_rotate = [1.0, -1.0] if hemisphere == 'N' else [-1.0, 1.0]
    v_ang_dir = np.array(v_ang_rotate)[..., :] * v_centr_normed[1:, :, ::-1]
    v_ang = np.zeros_like(v_ang_dir)
    v_ang[close_centr_msk[1:]] = v_ang_norm[close_centr_msk[1:], None] \
                                 * v_ang_dir[close_centr_msk[1:]]

    # Influence of translational speed decreases with distance from eye.
    # The "absorbing factor" is according to the following paper (see Fig. 7):
    #
    #   Mouton, F., & Nordbeck, O. (1999). Cyclone Database Manager. A tool
    #   for converting point data from cyclone observations into tracks and
    #   wind speed profiles in a GIS. UNED/GRID-Geneva.
    #   https://unepgrid.ch/en/resource/19B7D302
    #
    t_rad_bc = np.broadcast_arrays(t_rad[:, None], d_centr)[0]
    v_trans_corr = np.zeros_like(d_centr)
    v_trans_corr[close_centr_msk] = np.fmin(
        1, t_rad_bc[close_centr_msk] / d_centr[close_centr_msk])

    # add angular and corrected translational velocity vectors
    v_full = v_trans[1][1:, None, :] * v_trans_corr[1:, :, None] + v_ang
    v_full[np.isnan(v_full)] = 0

    windfields = np.zeros((npositions, nreachable, 2), dtype=np.float64)
    windfields[1:, :, :] = v_full
    [reachable_centr_idx] = track_centr_msk.nonzero()
    return windfields, reachable_centr_idx

def _close_centroids(t_lat, t_lon, centroids, buffer=CENTR_NODE_MAX_DIST_DEG):
    """Check whether centroids lay within a rectangular buffer around track positions

    The longitudinal coordinates are assumed to be normalized around a central longitude. This
    makes sure that the buffered bounding box around the track doesn't cross the antimeridian.

    The only hypothetical problem occurs when a TC track is travelling more than 349 degrees in
    longitude because that's when adding a buffer of 5.5 degrees might cross the antimeridian.
    Of course, this case is physically impossible.

    Parameters
    ----------
    t_lat : np.array of shape (npositions,)
        Latitudinal coordinates of track positions.
    t_lon : np.array of shape (npositions,)
        Longitudinal coordinates of track positions, normalized around a central longitude.
    centroids : np.array of shape (ncentroids, 2)
        Coordinates of centroids, each row is a pair [lat, lon].
    buffer : float (optional)
        Size of the buffer. Default: CENTR_NODE_MAX_DIST_DEG.

    Returns
    -------
    mask : np.array of shape (ncentroids,)
        Mask that is True for close centroids and False for other centroids.
    """
    centr_lat, centr_lon = centroids[:, 0], centroids[:, 1]
    # check for each track position which centroids are within buffer, uses NumPy's broadcasting
    mask = ((t_lat[:, None] - buffer < centr_lat[None])
            & (centr_lat[None] < t_lat[:, None] + buffer)
            & (t_lon[:, None] - buffer < centr_lon[None])
            & (centr_lon[None] < t_lon[:, None] + buffer))
    # for each centroid, check whether it is in the buffer for any of the track positions
    return mask.any(axis=0)

def _vtrans(t_lat, t_lon, t_tstep, metric="equirect"):
    """Translational vector and velocity at each track node.

    Parameters
    ----------
    t_lat : np.array
        track latitudes
    t_lon : np.array
        track longitudes
    t_tstep : np.array
        track time steps
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".

    Returns
    -------
    v_trans_norm : np.array
        Same shape as input, the first velocity is always 0.
    v_trans : np.array
        Directional vectors of velocity.
    """
    v_trans = np.zeros((t_lat.size, 2))
    v_trans_norm = np.zeros((t_lat.size,))
    norm, vec = u_coord.dist_approx(t_lat[:-1, None], t_lon[:-1, None],
                                    t_lat[1:, None], t_lon[1:, None],
                                    log=True, normalize=False, method=metric)
    v_trans[1:, :] = vec[:, 0, 0]
    v_trans[1:, :] *= KMH_TO_MS / t_tstep[1:, None]
    v_trans_norm[1:] = norm[:, 0, 0]
    v_trans_norm[1:] *= KMH_TO_MS / t_tstep[1:]

    # limit to 30 nautical miles per hour
    msk = (v_trans_norm > 30 * KN_TO_MS)
    fact = 30 * KN_TO_MS / v_trans_norm[msk]
    v_trans[msk, :] *= fact[:, None]
    v_trans_norm[msk] *= fact
    return v_trans_norm, v_trans

def _bs_hol08(v_trans, penv, pcen, prepcen, lat, tint):
    """
    Holland's 2008 b-value computation for sustained surface winds

    The parameter applies to 1-minute sustained winds at 10 meters above ground.
    It is taken from equation (11) in the following paper:

    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
    Weather Review, 136(9), 3432–3445. https://doi.org/10.1175/2008MWR2395.1

    For reference, it reads

    b_s = -4.4 * 1e-5 * (penv - pcen)^2 + 0.01 * (penv - pcen)
          + 0.03 * (dp/dt) - 0.014 * |lat| + 0.15 * (v_trans)^hol_xx + 1.0

    where `dp/dt` is the time derivative of central pressure and `hol_xx` is Holland's x
    parameter: hol_xx = 0.6 * (1 - (penv - pcen) / 215)

    The equation for b_s has been fitted statistically using hurricane best track records for
    central pressure and maximum wind. It therefore performs best in the North Atlantic.

    Furthermore, b_s has been fitted under the assumption of a "cyclostrophic" wind field which
    means that the influence from Coriolis forces is assumed to be small. This is reasonable close
    to the radius of maximum wind where the Coriolis term (r*f/2) is small compared to the rest
    (see `_stat_holland`). More precisely: At the radius of maximum wind speeds, the typical order
    of the Coriolis term is 1 while wind speed is 50 (which changes away from the
    radius of maximum winds and as the TC moves away from the equator).

    Parameters
    ----------
    v_trans : float
        translational wind (m/s)
    penv : float
        environmental pressure (hPa)
    pcen : float
        central pressure (hPa)
    prepcen : float
        previous central pressure (hPa)
    lat : float
        latitude (degrees)
    tint : float
        time step (h)

    Returns
    -------
    b_s : float
    """
    hol_xx = 0.6 * (1. - (penv - pcen) / 215)
    hol_b = -4.4e-5 * (penv - pcen)**2 + 0.01 * (penv - pcen) + \
        0.03 * (pcen - prepcen) / tint - 0.014 * abs(lat) + \
        0.15 * v_trans**hol_xx + 1.0
    return np.clip(hol_b, 1, 2.5)

def _stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, close_centr):
    """
    Holland symmetric and static wind field (in m/s)

    This function applies the gradient wind model expressed in equation (4) (combined with
    equation (6)) from

    Holland, G.J. (1980): An Analytic Model of the Wind and Pressure Profiles in Hurricanes.
    Monthly Weather Review 108(8): 1212–1218.

    More precisely, this function implements the following equation:

    V(r) = sqrt[(B/rho) * (r_max/r)^B * (penv - pcen) * e^(-(r_max/r)^B) + (r*f/2)^2] + (r*f/2)

    In terms of this function's arguments, B is `hol_b` and r is `d_centr`.
    The air density rho is assumed to be constant while the Coriolis parameter f is computed
    from the latitude `lat` using the constant rotation rate of the earth.

    Even though the equation has been derived originally for gradient winds, it can be used for
    surface winds by adjusting the parameter `hol_b` (see function `_bs_hol08`).

    Parameters
    ----------
    d_centr : np.array of shape (nnodes, ncentroids)
        Distance between centroids and track nodes.
    r_max : np.array of shape (nnodes,)
        Radius of maximum winds at each track node.
    hol_b : np.array of shape (nnodes,)
        Holland's b parameter at each track node.
    penv : np.array of shape (nnodes,)
        Environmental pressure at each track node.
    pcen : np.array of shape (nnodes,)
        Central pressure at each track node.
    lat : np.array of shape (nnodes,)
        Latitudinal coordinate of each track node.
    close_centr : np.array of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.

    Returns
    -------
    v_ang : np.array (nnodes, ncentroids)
        Absolute values of wind speeds in angular direction.
    """
    v_ang = np.zeros_like(d_centr)
    r_max, hol_b, lat, penv, pcen, d_centr = [
        ar[close_centr] for ar in np.broadcast_arrays(
            r_max[:, None], hol_b[:, None], lat[:, None],
            penv[:, None], pcen[:, None], d_centr)
    ]

    # air density
    rho = 1.15

    # Coriolis parameter with earth rotation rate 7.29e-5
    f_coriolis = 2 * 0.0000729 * np.sin(np.radians(np.abs(lat)))

    # d_centr is in km, convert to m (factor 1000) and apply Coriolis parameter
    r_coriolis = 0.5 * 1000 * d_centr * f_coriolis

    # the factor 100 is from conversion between mbar and pascal
    r_max_norm = (r_max / d_centr)**hol_b
    sqrt_term = 100 * hol_b / rho * r_max_norm * (penv - pcen) \
                * np.exp(-r_max_norm) + r_coriolis**2

    v_ang[close_centr] = np.sqrt(np.fmax(0, sqrt_term)) - r_coriolis
    return v_ang
