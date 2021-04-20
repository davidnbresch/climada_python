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

Generate synthetic tropical cyclone tracks from real ones
"""

import array
import itertools
import logging
import matplotlib.cm as cm_mp
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from numba import jit
import numpy as np

from climada import CONFIG
import climada.util.coordinates
import climada.hazard.tc_tracks

LOGGER = logging.getLogger(__name__)

LANDFALL_DECAY_V = {
    -1: 0.00012859077693295416,
    0: 0.0017226346292718126,
    1: 0.002309772914350468,
    2: 0.0025968221565522698,
    3: 0.002626252944053856,
    4: 0.002550639312763181,
    5: 0.003788695795963695
}
"""Global landfall decay parameters for wind speed by TC category.
Keys are TC categories with -1='TD', 0='TS', 1='Cat 1', ..., 5='Cat 5'.
It is v_rel as derived from:
tracks = TCTracks()
tracks.read_ibtracs_netcdf(year_range=(1980,2019),
                           estimate_missing=True)
extent = tracks.get_extent()
land_geom = climada.util.coordinates.get_land_geometry(
    extent=extent, resolution=10
)
v_rel, p_rel = _calc_land_decay(tracks.data, land_geom,
                                pool=tracks.pool)"""

LANDFALL_DECAY_P = {
    -1: (1.0088807492745373, 0.002117478217863062),
    0: (1.0192813768091684, 0.003068578025845065),
    1: (1.0362982218631644, 0.003620816186262243),
    2: (1.0468630800617038, 0.004067381088015585),
    3: (1.0639055205005432, 0.003708174876364079),
    4: (1.0828373148889825, 0.003997492773076179),
    5: (1.1088615145002092, 0.005224331234796362)}
"""Global landfall decay parameters for pressure by TC category.
Keys are TC categories with -1='TD', 0='TS', 1='Cat 1', ..., 5='Cat 5'.
It is p_rel as derived from:
tracks = TCTracks()
tracks.read_ibtracs_netcdf(year_range=(1980,2019),
                           estimate_missing=True)
extent = tracks.get_extent()
land_geom = climada.util.coordinates.get_land_geometry(
    extent=extent, resolution=10
)
v_rel, p_rel = _calc_land_decay(tracks.data, land_geom,
                                pool=tracks.pool)"""

def calc_perturbed_trajectories(tracks,
                                nb_synth_tracks=9,
                                max_shift_ini=0.75,
                                max_dspeed_rel=0.3,
                                max_ddirection=np.pi / 360,
                                autocorr_dspeed=0.85,
                                autocorr_ddirection=0.5,
                                seed=CONFIG.hazard.trop_cyclone.random_seed.int(),
                                decay=True,
                                use_global_decay_params=True):
    """
    Generate synthetic tracks based on directed random walk. An ensemble of nb_synth_tracks
    synthetic tracks is computed for every track contained in self.

    The methodology perturbs the tracks locations, and if decay is True it additionally
    includes decay of wind speed and central pressure drop after landfall. No other track
    parameter is perturbed.
    The track starting point location is perturbed by random uniform values of
    magnitude up to max_shift_ini in both longitude and latitude. Then, each segment
    between two consecutive points is perturbed in direction and distance (i.e.,
    translational speed). These perturbations can be correlated in time, i.e.,
    the perturbation in direction applied to segment i is correlated with the perturbation
    in direction applied to segment i-1 (and similarly for the perturbation in translational
    speed).
    Perturbations in track direction and temporal auto-correlations in perturbations are
    on an hourly basis, and the perturbations in translational speed is relative.
    Hence, the parameter values are relatively insensitive to the temporal
    resolution of the tracks. Note however that all tracks should be at the same
    temporal resolution, which can be achieved using equal_timestep().
    max_dspeed_rel and autocorr_dspeed control the spread along the track ('what distance
    does the track run for'), while max_ddirection and autocorr_ddirection control the spread
    perpendicular to the track movement ('how does the track diverge in direction').
    max_dspeed_rel and max_ddirection control the amplitude of perturbations at each track
    timestep but perturbations may tend to compensate each other over time, leading to
    a similar location at the end of the track, while autocorr_dspeed and autocorr_ddirection
    control how these perturbations persist in time and hence the amplitude of the
    perturbations towards the end of the track.

    Note that the default parameter values have been only roughly calibrated so that
    the frequency of tracks in each 5x5degree box remains approximately constant.
    This is not an in-depth calibration and should be treated as such.
    The object is mutated in-place.

    Parameters
    ----------
    tracks : climada.hazard.TCTracks
        Tracks data.
    nb_synth_tracks : int, optional
        Number of ensemble members per track. Default: 9.
    max_shift_ini : float, optional
        Amplitude of max random starting point shift in decimal degree
        (up to +/-max_shift_ini for longitude and latitude). Default: 0.75.
    max_dspeed_rel : float, optional
        Amplitude of translation speed perturbation in relative terms
        (e.g., 0.2 for +/-20%). Default: 0.3.
    max_ddirection : float, optional
        Amplitude of track direction (bearing angle) perturbation
        per hour, in radians. Default: pi/360.
    autocorr_dspeed : float, optional
        Temporal autocorrelation in translation speed perturbation
        at a lag of 1 hour. Default: 0.85.
    autocorr_ddirection : float, optional
        Temporal autocorrelation of translational direction perturbation
        at a lag of 1 hour. Default: 0.5.
    seed : int, optional
        Random number generator seed for replicability of random walk.
        Put negative value if you don't want to use it. Default: configuration file.
    decay : bool, optional
        Whether to apply landfall decay in probabilistic tracks. Default: True.
    use_global_decay_params : bool, optional
        Whether to use precomputed global parameter values for landfall decay
        obtained from IBTrACS (1980-2019). If False, parameters are fitted
        using historical tracks in input parameter 'tracks', in which case the
        landfall decay applied depends on the tracks passed as an input and may
        not be robust if few historical tracks make landfall in this object.
        Default: True.
    """
    
    LOGGER.info('Computing %s synthetic tracks.', nb_synth_tracks * tracks.size)

    if seed >= 0:
        np.random.seed(seed)

    # ensure tracks have constant time steps
    time_step_h = np.unique([np.unique(x['time_step']) for x in tracks.data if x.time.size > 1])
    if not np.allclose(time_step_h, time_step_h[0]):
        LOGGER.error('Tracks have different temporal resolution. '
                     'Please ensure constant time steps by applying equal_timestep beforehand')
        raise ValueError('Tracks have different temporal resolution.')
    time_step_h = time_step_h[0]

    # number of random value per synthetic track:
    # 2*nb_synth_tracks for starting points (lon, lat)
    # nb_synth_tracks*(track.time.size-1) for angle and same for translation perturbation
    # hence sum is nb_synth_tracks * (2 + 2*(size-1)) = nb_synth_tracks * 2 * size
    # https://stats.stackexchange.com/questions/48086/algorithm-to-produce-autocorrelated-uniformly-distributed-number
    if autocorr_ddirection == 0 and autocorr_dspeed == 0:
        random_vec = [np.random.uniform(size=nb_synth_tracks * (2 * track.time.size))
                      for track in tracks.data]
    else:
        random_vec = [np.concatenate((np.random.uniform(size=nb_synth_tracks * 2),
                                      _random_uniform_ac(nb_synth_tracks * (track.time.size - 1),
                                                         autocorr_ddirection, time_step_h),
                                      _random_uniform_ac(nb_synth_tracks * (track.time.size - 1),
                                                         autocorr_dspeed, time_step_h)))
                      if track.time.size > 1 else np.random.uniform(size=nb_synth_tracks * 2)
                      for track in tracks.data]

    if tracks.pool:
        chunksize = min(tracks.size // tracks.pool.ncpus, 1000)
        new_ens = tracks.pool.map(_one_rnd_walk, tracks.data,
                                  itertools.repeat(nb_synth_tracks, tracks.size),
                                  itertools.repeat(max_shift_ini, tracks.size),
                                  itertools.repeat(max_dspeed_rel, tracks.size),
                                  itertools.repeat(max_ddirection, tracks.size),
                                  random_vec, chunksize=chunksize)
    else:
        new_ens = [_one_rnd_walk(track, nb_synth_tracks, max_shift_ini,
                                 max_dspeed_rel, max_ddirection, rand)
                   for track, rand in zip(tracks.data, random_vec)]

    cutoff_track_ids_tc = [x[1] for x in new_ens]
    cutoff_track_ids_tc = sum(cutoff_track_ids_tc, [])
    cutoff_track_ids_ts = [x[2] for x in new_ens]
    cutoff_track_ids_ts = sum(cutoff_track_ids_ts, [])
    if len(cutoff_track_ids_tc) > 0:
        LOGGER.info('The following generated synthetic tracks moved beyond '
                    'the range of [-70, 70] degrees latitude. Cut out '
                    'at TC category >1: %s.',
                    ', '.join(cutoff_track_ids_tc))
    if len(cutoff_track_ids_ts) > 0:
        LOGGER.debug('The following generated synthetic tracks moved beyond '
                     'the range of [-70, 70] degrees latitude. Cut out '
                     'at TC category <= 1: %s.',
                     ', '.join(cutoff_track_ids_ts))
    new_ens = [x[0] for x in new_ens]
    tracks.data = sum(new_ens, [])

    if decay:
        extent = tracks.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        if use_global_decay_params:
            tracks.data = _apply_land_decay(tracks.data, LANDFALL_DECAY_V,
                                            LANDFALL_DECAY_P,
                                            land_geom, pool=tracks.pool)
        else:
            # fit land decay coefficients based on historical tracks
            hist_tracks = [track for track in tracks.data if track.orig_event_flag]
            if hist_tracks:
                try:
                    v_rel, p_rel = _calc_land_decay(hist_tracks, land_geom,
                                                    pool=tracks.pool)
                    tracks.data = _apply_land_decay(tracks.data, v_rel, p_rel,
                                                    land_geom, pool=tracks.pool)
                except ValueError as verr:
                    raise ValueError('Landfall decay could not be applied.') from verr
            else:
               raise ValueError('No historical tracks found. Historical'
                                'tracks are needed for land decay calibration'
                                'if use_global_decay_params=False.')


def _one_rnd_walk(track, nb_synth_tracks, max_shift_ini, max_dspeed_rel, max_ddirection, rnd_vec):
    """
    Apply random walk to one track.

    Parameters
    ----------
    track : xr.Dataset
        Track data.
    nb_synth_tracks : int, optional
        Number of ensemble members per track. Default: 9.
    max_shift_ini : float, optional
        Amplitude of max random starting point shift in decimal degree
        (up to +/-max_shift_ini for longitude and latitude). Default: 0.75.
    max_dspeed_rel : float, optional
        Amplitude of translation speed perturbation in relative terms
        (e.g., 0.2 for +/-20%). Default: 0.3.
    max_ddirection : float, optional
        Amplitude of track direction (bearing angle) perturbation
        per hour, in radians. Default: pi/180.
    rnd_vec : np.ndarray of shape (2 * nb_synth_tracks * track.time.size),)
        Vector of random perturbations.

    Returns
    -------
    ens_track : list(xr.Dataset)
        List of the track and the generated synthetic tracks.
    cutoff_track_ids_tc : List of str
        List containing information about the tracks that were cut off at high
        latitudes with wind speed of TC category 2-5.
    curoff_track_ids_ts : List of str
        List containing information about the tracks that were cut off at high
        latitudes with a wind speed up to TC category 1.
    """
    ens_track = list()
    n_dat = track.time.size
    n_seg = n_dat - 1
    xy_ini = max_shift_ini * (2 * rnd_vec[:2 * nb_synth_tracks].reshape((2, nb_synth_tracks)) - 1)
    [dt] = np.unique(track['time_step'])

    ens_track.append(track)
    cutoff_track_ids_ts = []
    cutoff_track_ids_tc = []
    for i_ens in range(nb_synth_tracks):
        i_track = track.copy(True)

        # select angular perturbation for that synthetic track
        i_start_ang = 2 * nb_synth_tracks + i_ens * n_seg
        i_end_ang = i_start_ang + track.time.size - 1
        # scale by maximum perturbation and time step in hour (temporal-resolution independent)
        ang_pert = dt * np.degrees(max_ddirection * (2 * rnd_vec[i_start_ang:i_end_ang] - 1))
        ang_pert_cum = np.cumsum(ang_pert)

        # select translational speed perturbation for that synthetic track
        i_start_trans = 2 * nb_synth_tracks + nb_synth_tracks * n_seg + i_ens * n_seg
        i_end_trans = i_start_trans + track.time.size - 1
        # scale by maximum perturbation and time step in hour (temporal-resolution independent)
        trans_pert = 1 + max_dspeed_rel * (2 * rnd_vec[i_start_trans:i_end_trans] - 1)

        # get bearings and angular distance for the original track
        bearings = _get_bearing_angle(i_track.lon.values, i_track.lat.values)
        angular_dist = climada.util.coordinates.dist_approx(i_track.lat.values[:-1, None],
                                                            i_track.lon.values[:-1, None],
                                                            i_track.lat.values[1:, None],
                                                            i_track.lon.values[1:, None],
                                                            method="geosphere",
                                                            units="degree")[:, 0, 0]

        # apply perturbation to lon / lat
        new_lon = np.zeros_like(i_track.lon.values)
        new_lat = np.zeros_like(i_track.lat.values)
        new_lon[0] = i_track.lon.values[0] + xy_ini[0, i_ens]
        new_lat[0] = i_track.lat.values[0] + xy_ini[1, i_ens]
        last_idx = i_track.time.size
        for i in range(0, len(new_lon) - 1):
            new_lon[i + 1], new_lat[i + 1] = \
                _get_destination_points(new_lon[i], new_lat[i],
                                        bearings[i] + ang_pert_cum[i],
                                        trans_pert[i] * angular_dist[i])
            # if track crosses latitudinal thresholds (+-70°),
            # keep up to this segment (i+1), set i+2 as last point,
            # and discard all further points > i+2.
            if i+2 < last_idx and (new_lat[i + 1] > 70 or new_lat[i + 1] < -70):
                last_idx = i + 2
                # end the track here
                max_wind_end = i_track.max_sustained_wind.values[last_idx]
                ss_scale_end = climada.hazard.tc_tracks.set_category(max_wind_end,
                                                                     i_track.max_sustained_wind_unit)
                # TC category at ending point should not be higher than 1
                cutoff_txt = i_track.attrs['name'] + '_gen' + str(i_ens + 1)
                cutoff_txt = cutoff_txt + ' (%s)' % climada.hazard.tc_tracks.CAT_NAMES[ss_scale_end]
                if ss_scale_end > 1:
                    cutoff_track_ids_tc = cutoff_track_ids_tc + [cutoff_txt]
                else:
                    cutoff_track_ids_ts = cutoff_track_ids_ts + [cutoff_txt]
                break
        # make sure longitude values are within (-180, 180)
        climada.util.coordinates.lon_normalize(new_lon, center=0.0)

        i_track.lon.values = new_lon
        i_track.lat.values = new_lat
        i_track.attrs['orig_event_flag'] = False
        i_track.attrs['name'] = i_track.attrs['name'] + '_gen' + str(i_ens + 1)
        i_track.attrs['sid'] = i_track.attrs['sid'] + '_gen' + str(i_ens + 1)
        i_track.attrs['id_no'] = i_track.attrs['id_no'] + (i_ens + 1) / 100
        i_track = i_track.isel(time=slice(None, last_idx))

        ens_track.append(i_track)

    return ens_track, cutoff_track_ids_tc, cutoff_track_ids_ts


def _random_uniform_ac(n_ts, autocorr, time_step_h):
    """
    Generate a series of autocorrelated uniformly distributed random numbers.

    This implements the algorithm described here to derive a uniformly distributed
    series with specified autocorrelation (here at a lag of 1 hour):
    https://stats.stackexchange.com/questions/48086/
        algorithm-to-produce-autocorrelated-uniformly-distributed-number
    Autocorrelation is specified at a lag of 1 hour. To get a time series at a
    different temporal resolution (time_step_h), an hourly time series is generated
    and resampled (using linear interpolation) to the target resolution.

    Parameters
    ----------
    n_ts : int
        Length of the series.
    autocorr : float
        Autocorrelation (between -1 and 1) at hourly time scale.
    time_step_h : float
        Temporal resolution of the time series, in hour.

    Returns
    -------
    x_ts : numpy.ndarray of shape (n_ts,)
        n values at time_step_h intervals that are uniformly distributed and with
            the requested temporal autocorrelation at a scale of 1 hour.
    """
    # generate autocorrelated 1-hourly perturbations, so first create hourly
    #   time series of perturbations
    n_ts_hourly_exact = n_ts * time_step_h
    n_ts_hourly = int(np.ceil(n_ts_hourly_exact))
    x = np.random.normal(size=n_ts_hourly)
    theta = np.arccos(autocorr)
    for i in range(1, len(x)):
        x[i] = _h_ac(x[i - 1], x[i], theta)
    # scale x to have magnitude [0,1]
    x = (x + np.sqrt(3)) / (2 * np.sqrt(3))
    # resample at target time step
    x_ts = np.interp(np.arange(start=0, stop=n_ts_hourly_exact, step=time_step_h),
                     np.arange(n_ts_hourly), x)
    return x_ts


@jit
def _h_ac(x, y, theta):
    """
    Generate next random number from current number for autocorrelated uniform series

    Implements function h defined in:
    https://stats.stackexchange.com/questions/48086/
        algorithm-to-produce-autocorrelated-uniformly-distributed-number

    Parameters
    ----------
    x : float
        Previous random number.
    y : float
        Random Standard Normal.
    theta : float
        arccos of autocorrelation.

    Returns
    -------
    x_next : float
        Next value in the series.
    """
    gamma = np.abs(np.mod(theta, np.pi) - \
                   np.floor((np.mod(theta, np.pi) / (np.pi / 2)) + 0.5) * np.pi / 2)
    x_next = 2 * np.sqrt(3) * (_f_ac(np.cos(theta) * x + np.sin(theta) * y, gamma) - 1 / 2)
    return x_next


@jit
def _f_ac(z, theta):
    """
    F transform for autocorrelated random uniform series generation

    Implements function F defined in:
    https://stats.stackexchange.com/questions/48086/
        algorithm-to-produce-autocorrelated-uniformly-distributed-number
    i.e., the CDF of Y.

    Parameters
    ----------
    z : float
        Value.
    theta : float
        arccos of autocorrelation.

    Returns
    -------
        res : float
            CDF at value z
    """
    c = np.cos(theta)
    s = np.sin(theta)
    if z >= np.sqrt(3) * (c + s):
        res = 1
    elif z > np.sqrt(3) * (c - s):
        res = 1 / 12 / np.sin(2 * theta) * \
              (-3 - z ** 2 + 2 * np.sqrt(3) * z * (c + s) + 9 * np.sin(2 * theta))
    elif z > np.sqrt(3) * (-c + s):
        res = 1 / 6 * (3 + np.sqrt(3) * z / c)
    elif z > -np.sqrt(3) * (c + s):
        res = 1 / 12 / np.sin(2 * theta) * \
              (z ** 2 + 2 * np.sqrt(3) * z * (c + s) + 3 * (1 + np.sin(2 * theta)))
    else:
        res = 0
    return res


@jit
def _get_bearing_angle(lon, lat):
    """
    Compute bearing angle of great circle paths defined by consecutive points

    Returns initial bearing (also called forward azimuth) of the n-1 great circle
    paths define by n consecutive longitude/latitude points. The bearing is the angle
    (clockwise from North) which if followed in a straight line along a great-circle
    arc will take you from the start point to the end point. See also:
    http://www.movable-type.co.uk/scripts/latlong.html
    Here, the bearing of each pair of consecutive points is computed.

    Parameters
    ----------
    lon : numpy.ndarray of shape (n,)
        Longitude coordinates of consecutive point, in decimal degrees.
    lat : numpy.ndarray of shape (n,)
        Latitude coordinates of consecutive point, in decimal degrees.

    Returns
    -------
        earth_ang_fix : numpy.ndarray of shape (n-1,)
            Bearing angle for each segment, in decimal degrees
    """
    lon, lat = map(np.radians, [lon, lat])
    # Segments between all point (0 -> 1, 1 -> 2, ..., n-1 -> n)
    # starting points
    lat_1 = lat[:-1]
    lon_1 = lon[:-1]
    # ending points
    lat_2 = lat[1:]
    lon_2 = lon[1:]
    delta_lon = lon_2 - lon_1
    # what to do with the points that don't move?
    #   i.e. where lat_2=lat_1 and lon_2=lon_1? The angle does not matter in
    # that case because angular distance will be 0.
    earth_ang_fix = np.arctan2(np.sin(delta_lon) * np.cos(lat_2),
                               np.cos(lat_1) * np.sin(lat_2) - \
                               np.sin(lat_1) * np.cos(lat_2) * np.cos(delta_lon))
    return np.degrees(earth_ang_fix)


@jit
def _get_destination_points(lon, lat, bearing, angular_distance):
    """
    Get coordinates of endpoints from a given locations with the provided bearing and distance

    Parameters
    ----------
    lon : numpy.ndarray of shape (n,)
        Longitude coordinates of each starting point, in decimal degrees.
    lat : numpy.ndarray of shape (n,)
        Latitude coordinates of each starting point, in decimal degrees.
    bearing : numpy.ndarray of shape (n,)
        Bearing to follow for each starting point (direction Northward, clockwise).
    angular_distance : numpy.ndarray of shape (n,)
        Angular distance to travel for each starting point, in decimal degrees.

    Returns
    -------
        lon_2 : numpy.ndarray of shape (n,)
            Longitude coordinates of each ending point, in decimal degrees.
        lat_2 : numpy.ndarray of shape (n,)
            Latitude coordinates of each ending point, in decimal degrees.
    """
    lon, lat = map(np.radians, [lon, lat])
    bearing = np.radians(bearing)
    angular_distance = np.radians(angular_distance)
    lat_2 = np.arcsin(np.sin(lat) * np.cos(angular_distance) + np.cos(lat) * \
                      np.sin(angular_distance) * np.cos(bearing))
    lon_2 = lon + np.arctan2(np.sin(bearing) * np.sin(angular_distance) * np.cos(lat),
                             np.cos(angular_distance) - np.sin(lat) * np.sin(lat_2))
    return np.degrees(lon_2), np.degrees(lat_2)


def _calc_land_decay(hist_tracks, land_geom, s_rel=True, check_plot=False,
                     pool=None):
    """Compute wind and pressure decay coefficients from historical events

    Decay is calculated for every TC category according to the formulas:

        - wind decay = exp(-x*A)
        - pressure decay = S-(S-1)*exp(-x*B)

    Parameters:
        hist_tracks (list): List of xarray Datasets describing TC tracks.
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
        s_rel (bool, optional): use environmental presure to calc S value
            (true) or central presure (false)
        check_plot (bool, optional): visualize computed coefficients.
            Default: False

    Returns:
        v_rel (dict(category: A)), p_rel (dict(category: (S, B)))
    """
    
    if len(hist_tracks) < 100:
        LOGGER.warning('For the calibration of the landfall decay '
                       'it is recommended to provide as many historical '
                       'tracks as possible, but only %s historical tracks '
                       'were provided. '
                       'For a more robust calculation consider using '
                       'a larger number of tracks or set '
                       '`use_global_decay_params` to True', len(hist_tracks))

    # Key is Saffir-Simpson scale
    # values are lists of wind/wind at landfall
    v_lf = dict()
    # values are tuples with first value the S parameter, second value
    # list of central pressure/central pressure at landfall
    p_lf = dict()
    # x-scale values to compute landfall decay
    x_val = dict()

    if pool:
        dec_val = pool.map(_decay_values, hist_tracks, itertools.repeat(land_geom),
                           itertools.repeat(s_rel),
                           chunksize=min(len(hist_tracks) // pool.ncpus, 1000))
    else:
        dec_val = [_decay_values(track, land_geom, s_rel) for track in hist_tracks]

    for (tv_lf, tp_lf, tx_val) in dec_val:
        for key in tv_lf.keys():
            v_lf.setdefault(key, []).extend(tv_lf[key])
            p_lf.setdefault(key, ([], []))
            p_lf[key][0].extend(tp_lf[key][0])
            p_lf[key][1].extend(tp_lf[key][1])
            x_val.setdefault(key, []).extend(tx_val[key])

    v_rel, p_rel = _decay_calc_coeff(x_val, v_lf, p_lf)
    if check_plot:
        _check_decay_values_plot(x_val, v_lf, p_lf, v_rel, p_rel)

    return v_rel, p_rel


def _apply_land_decay(tracks, v_rel, p_rel, land_geom, s_rel=True,
                      check_plot=False, pool=None):
    """Compute wind and pressure decay due to landfall in synthetic tracks.

    Parameters:
        v_rel (dict): {category: A}, where wind decay = exp(-x*A)
        p_rel (dict): (category: (S, B)}, where pressure decay
            = S-(S-1)*exp(-x*B)
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
        s_rel (bool, optional): use environmental presure to calc S value
            (true) or central presure (false)
        check_plot (bool, optional): visualize computed changes
    """
    sy_tracks = [track for track in tracks if not track.orig_event_flag]
    if not sy_tracks:
        LOGGER.error('No synthetic tracks contained. Synthetic tracks'
                     ' are needed.')
        raise ValueError

    if not v_rel or not p_rel:
        LOGGER.info('No decay coefficients.')
        return

    if check_plot:
        orig_wind, orig_pres = [], []
        for track in sy_tracks:
            orig_wind.append(np.copy(track.max_sustained_wind.values))
            orig_pres.append(np.copy(track.central_pressure.values))

    if pool:
        chunksize = min(len(tracks) // pool.ncpus, 1000)
        tracks = pool.map(_apply_decay_coeffs, tracks,
                          itertools.repeat(v_rel), itertools.repeat(p_rel),
                          itertools.repeat(land_geom), itertools.repeat(s_rel),
                          chunksize=chunksize)
    else:
        tracks = [_apply_decay_coeffs(track, v_rel, p_rel, land_geom, s_rel)
                  for track in tracks]

    for track in tracks:
        if track.orig_event_flag:
            climada.hazard.tc_tracks.track_land_params(track, land_geom)
    if check_plot:
        _check_apply_decay_plot(tracks, orig_wind, orig_pres)
    return tracks


def _decay_values(track, land_geom, s_rel):
    """Compute wind and pressure relative to landafall values.

    Parameters
    ----------
    track : xr.Dataset
        track
    land_geom : shapely.geometry.multipolygon.MultiPolygon
        land geometry
    s_rel : bool
        use environmental presure for S value (true) or central presure (false)

    Returns
    -------
    v_lf : dict
        key is Saffir-Simpson scale, values are arrays of wind/wind at landfall
    p_lf : dict
        key is Saffir-Simpson scale, values are tuples with first value array
        of S parameter, second value array of central pressure/central pressure
        at landfall
    x_val : dict
        key is Saffir-Simpson scale, values are arrays with the values used as
        "x" in the coefficient fitting, the distance since landfall
    """
    v_lf = dict()
    p_lf = dict()
    x_val = dict()

    climada.hazard.tc_tracks.track_land_params(track, land_geom)
    sea_land_idx, land_sea_idx = climada.hazard.tc_tracks._get_landfall_idx(track)
    if sea_land_idx.size:
        for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
            v_landfall = track.max_sustained_wind[sea_land - 1].values
            ss_scale = climada.hazard.tc_tracks.set_category(v_landfall,
                                                             track.max_sustained_wind_unit)

            v_land = track.max_sustained_wind[sea_land - 1:land_sea].values
            if v_land[0] > 0:
                v_land = (v_land[1:] / v_land[0]).tolist()
            else:
                v_land = v_land[1:].tolist()

            p_landfall = float(track.central_pressure[sea_land - 1].values)
            p_land = track.central_pressure[sea_land - 1:land_sea].values
            p_land = (p_land[1:] / p_land[0]).tolist()

            p_land_s = _calc_decay_ps_value(
                track, p_landfall, land_sea - 1, s_rel)
            p_land_s = len(p_land) * [p_land_s]

            if ss_scale not in v_lf:
                v_lf[ss_scale] = array.array('f', v_land)
                p_lf[ss_scale] = (array.array('f', p_land_s),
                                      array.array('f', p_land))
                x_val[ss_scale] = array.array('f',
                                                  track.dist_since_lf[sea_land:land_sea])
            else:
                v_lf[ss_scale].extend(v_land)
                p_lf[ss_scale][0].extend(p_land_s)
                p_lf[ss_scale][1].extend(p_land)
                x_val[ss_scale].extend(track.dist_since_lf[sea_land:land_sea])
    return v_lf, p_lf, x_val


def _decay_calc_coeff(x_val, v_lf, p_lf):
    """From track's relative velocity and pressure, compute the decay
    coefficients.
    - wind decay = exp(-x*A)
    - pressure decay = S-(S-1)*exp(-x*A)

    Parameters:
        x_val (dict): key is Saffir-Simpson scale, values are lists with
            the values used as "x" in the coefficient fitting, the
            distance since landfall
        v_lf (dict): key is Saffir-Simpson scale, values are lists of
            wind/wind at landfall
        p_lf (dict): key is Saffir-Simpson scale, values are tuples with
            first value the S parameter, second value list of central
            pressure/central pressure at landfall

    Returns:
        v_rel (dict()), p_rel (dict())
    """
    np.warnings.filterwarnings('ignore')
    v_rel = dict()
    p_rel = dict()
    for ss_scale, val_lf in v_lf.items():
        x_val_ss = np.array(x_val[ss_scale])

        y_val = np.array(val_lf)
        v_coef = _solve_decay_v_function(y_val, x_val_ss)
        v_coef = v_coef[np.isfinite(v_coef)]
        v_coef = np.mean(v_coef)

        ps_y_val = np.array(p_lf[ss_scale][0])
        y_val = np.array(p_lf[ss_scale][1])
        y_val[ps_y_val <= y_val] = np.nan
        y_val[ps_y_val <= 1] = np.nan
        valid_p = np.isfinite(y_val)
        ps_y_val = ps_y_val[valid_p]
        y_val = y_val[valid_p]
        p_coef = _solve_decay_p_function(ps_y_val, y_val, x_val_ss[valid_p])
        ps_y_val = np.mean(ps_y_val)
        p_coef = np.mean(p_coef)

        if np.isfinite(v_coef) and np.isfinite(ps_y_val) and np.isfinite(ps_y_val):
            v_rel[ss_scale] = v_coef
            p_rel[ss_scale] = (ps_y_val, p_coef)

    scale_fill = np.array(list(p_rel.keys()))
    if not scale_fill.size:
        LOGGER.info('No historical track with landfall.')
        return v_rel, p_rel
    for ss_scale in climada.hazard.tc_tracks.CAT_NAMES.keys():
        if ss_scale not in p_rel:
            close_scale = scale_fill[np.argmin(np.abs(scale_fill - ss_scale))]
            LOGGER.debug('No historical track of category %s with landfall. '
                         'Decay parameters from category %s taken.',
                         climada.hazard.tc_tracks.CAT_NAMES[ss_scale],
                         climada.hazard.tc_tracks.CAT_NAMES[close_scale])
            v_rel[ss_scale] = v_rel[close_scale]
            p_rel[ss_scale] = p_rel[close_scale]
        elif v_rel[ss_scale] < 0:
            raise ValueError('The calibration of landfall decay for wind '
                             'speed resulted in a wind speed increase for '
                             'TC category %s.'
                             'This behavious in unphysical.'
                             'Please use a larger number of tracks or '
                             'use global paramaters by setting '
                             '`use_global_decay_params` to True',
                             climada.hazard.tc_tracks.CAT_NAMES[ss_scale])
        elif p_rel[ss_scale][0] < 0 or p_rel[ss_scale][1] < 0:
            raise ValueError('The calibration of landfall decay for central '
                             'pressure resulted in a pressure decrease for '
                             'TC category %s.'
                             'This behavious in unphysical.'
                             'Please use a larger number of tracks or '
                             'use global paramaters by setting '
                             '`use_global_decay_params` to True',
                             climada.hazard.tc_tracks.CAT_NAMES[ss_scale])

    return v_rel, p_rel


def _check_decay_values_plot(x_val, v_lf, p_lf, v_rel, p_rel):
    """Generate one graph with wind decay and an other with central pressure
    decay, true and approximated."""
    # One graph per TC category
    for track_cat, color in zip(v_lf.keys(),
                                cm_mp.rainbow(np.linspace(0, 1, len(v_lf)))):
        _, axes = plt.subplots(2, 1)
        x_eval = np.linspace(0, np.max(x_val[track_cat]), 20)

        axes[0].set_xlabel('Distance from landfall (km)')
        axes[0].set_ylabel('Max sustained wind relative to landfall')
        axes[0].set_title('Wind, TC cat %s' %
                          climada.hazard.tc_tracks.CAT_NAMES[track_cat])
        axes[0].plot(x_val[track_cat], v_lf[track_cat], '*', c=color,
                     label=climada.hazard.tc_tracks.CAT_NAMES[track_cat])
        axes[0].plot(x_eval, _decay_v_function(v_rel[track_cat], x_eval),
                     '-', c=color)

        axes[1].set_xlabel('Distance from landfall (km)')
        axes[1].set_ylabel('Central pressure relative to landfall')
        axes[1].set_title('Pressure, TC cat %s' %
                          climada.hazard.tc_tracks.CAT_NAMES[track_cat])
        axes[1].plot(x_val[track_cat], p_lf[track_cat][1], '*', c=color,
                     label=climada.hazard.tc_tracks.CAT_NAMES[track_cat])
        axes[1].plot(
            x_eval,
            _decay_p_function(p_rel[track_cat][0], p_rel[track_cat][1], x_eval),
            '-', c=color)


def _apply_decay_coeffs(track, v_rel, p_rel, land_geom, s_rel):
    """Change track's max sustained wind and central pressure using the land
    decay coefficients.

    Parameters:
        track (xr.Dataset): TC track
        v_rel (dict): {category: A}, where wind decay = exp(-x*A)
        p_rel (dict): (category: (S, B)},
            where pressure decay = S-(S-1)*exp(-x*B)
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
        s_rel (bool): use environmental presure for S value (true) or
            central presure (false)

    Returns:
        xr.Dataset
    """
    # return if historical track
    if track.orig_event_flag:
        return track

    climada.hazard.tc_tracks.track_land_params(track, land_geom)
    sea_land_idx, land_sea_idx = climada.hazard.tc_tracks._get_landfall_idx(track)
    if not sea_land_idx.size:
        return track
    for idx, (sea_land, land_sea) \
            in enumerate(zip(sea_land_idx, land_sea_idx)):
        v_landfall = track.max_sustained_wind[sea_land - 1].values
        p_landfall = float(track.central_pressure[sea_land - 1].values)
        ss_scale = climada.hazard.tc_tracks.set_category(v_landfall,
                                                         track.max_sustained_wind_unit)
        if land_sea - sea_land == 1:
            continue
        S = _calc_decay_ps_value(track, p_landfall, land_sea - 1, s_rel)
        if S <= 1:
            # central_pressure at start of landfall > env_pres after landfall:
            # set central_pressure to environmental pressure during whole lf
            track.central_pressure[sea_land:land_sea] = track.environmental_pressure[sea_land:land_sea]
        else:
            p_decay = _decay_p_function(S, p_rel[ss_scale][1],
                                        track.dist_since_lf[sea_land:land_sea].values)
            # dont apply decay if it would decrease central pressure
            if np.any(p_decay < 1):
                LOGGER.info('Landfall decay would decrease pressure for '
                            'track id %s, leading to an intensification '
                            'of the Tropical Cyclone. This behaviour is '
                            'unphysical and therefore landfall decay is not '
                            'applied in this case.',
                            track.sid)
                p_decay[p_decay < 1] = track.central_pressure[sea_land:land_sea][p_decay < 1] / p_landfall
            track.central_pressure[sea_land:land_sea] = p_landfall * p_decay

        v_decay = _decay_v_function(v_rel[ss_scale],
                                    track.dist_since_lf[sea_land:land_sea].values)
        # dont apply decay if it would increase wind speeds
        if np.any(v_decay > 1):
            # should not happen unless v_rel is negative
            LOGGER.info('Landfall decay would increase wind speed for '
                        'track id %s. This behavious in unphysical and '
                        'therefore landfall decay is not applied in this '
                        'case.',
                        track.sid)
            v_decay[v_decay > 1] = (track.max_sustained_wind[sea_land:land_sea][v_decay > 1]
                                    / v_landfall)
        track.max_sustained_wind[sea_land:land_sea] = v_landfall * v_decay

        # correct values of sea after a landfall (until next landfall, if any)
        if land_sea < track.time.size:
            if idx + 1 < sea_land_idx.size:
                # if there is a next landfall, correct until last point before
                # reaching land again
                end_cor = sea_land_idx[idx + 1]
            else:
                # if there is no further landfall, correct until the end of
                # the track
                end_cor = track.time.size
            rndn = 0.1 * float(np.abs(np.random.normal(size=1) * 5) + 6)
            r_diff = track.central_pressure[land_sea].values - \
                     track.central_pressure[land_sea - 1].values + rndn
            track.central_pressure[land_sea:end_cor] += - r_diff

            rndn = rndn * 10  # mean value 10
            r_diff = track.max_sustained_wind[land_sea].values - \
                     track.max_sustained_wind[land_sea - 1].values - rndn
            track.max_sustained_wind[land_sea:end_cor] += - r_diff

        # correct limits
        np.warnings.filterwarnings('ignore')
        cor_p = track.central_pressure.values > track.environmental_pressure.values
        track.central_pressure[cor_p] = track.environmental_pressure[cor_p]
        track.max_sustained_wind[track.max_sustained_wind < 0] = 0

    track.attrs['category'] = climada.hazard.tc_tracks.set_category(
        track.max_sustained_wind.values, track.max_sustained_wind_unit)
    return track


def _check_apply_decay_plot(all_tracks, syn_orig_wind, syn_orig_pres):
    """Plot wind and presure before and after correction for synthetic tracks.
    Plot wind and presure for unchanged historical tracks."""
    # Plot synthetic tracks
    sy_tracks = [track for track in all_tracks if not track.orig_event_flag]
    graph_v_b, graph_v_a, graph_p_b, graph_p_a, graph_pd_a, graph_ped_a = \
        _check_apply_decay_syn_plot(sy_tracks, syn_orig_wind,
                                    syn_orig_pres)

    # Plot historic tracks
    hist_tracks = [track for track in all_tracks if track.orig_event_flag]
    graph_hv, graph_hp, graph_hpd_a, graph_hped_a = \
        _check_apply_decay_hist_plot(hist_tracks)

    # Put legend and fix size
    scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
    leg_lines = [Line2D([0], [0], color=climada.hazard.tc_tracks.CAT_COLORS[i_col], lw=2)
                 for i_col in range(len(scale_thresholds))]
    leg_lines.append(Line2D([0], [0], color='k', lw=2))
    leg_names = [climada.hazard.tc_tracks.CAT_NAMES[i_col]
                 for i_col in sorted(climada.hazard.tc_tracks.CAT_NAMES.keys())]
    leg_names.append('Sea')
    all_gr = [graph_v_a, graph_v_b, graph_p_a, graph_p_b, graph_ped_a,
              graph_pd_a, graph_hv, graph_hp, graph_hpd_a, graph_hped_a]
    for graph in all_gr:
        graph.axs[0].legend(leg_lines, leg_names)
        fig, _ = graph.get_elems()
        fig.set_size_inches(18.5, 10.5)


def _calc_decay_ps_value(track, p_landfall, pos, s_rel):
    if s_rel:
        p_land_s = track.environmental_pressure[pos].values
    else:
        p_land_s = track.central_pressure[pos].values
    return float(p_land_s / p_landfall)


def _decay_v_function(a_coef, x_val):
    """Decay function used for wind after landfall."""
    return np.exp(-a_coef * x_val)


def _solve_decay_v_function(v_y, x_val):
    """Solve decay function used for wind after landfall. Get A coefficient."""
    return -np.log(v_y) / x_val


def _decay_p_function(s_coef, b_coef, x_val):
    """Decay function used for pressure after landfall."""
    return s_coef - (s_coef - 1) * np.exp(-b_coef * x_val)


def _solve_decay_p_function(ps_y, p_y, x_val):
    """Solve decay function used for pressure after landfall.
    Get B coefficient."""
    return -np.log((ps_y - p_y) / (ps_y - 1.0)) / x_val


def _check_apply_decay_syn_plot(sy_tracks, syn_orig_wind,
                                syn_orig_pres):
    """Plot winds and pressures of synthetic tracks before and after
    correction."""
    _, graph_v_b = plt.subplots()
    graph_v_b.set_title('Wind before land decay correction')
    graph_v_b.set_xlabel('Node number')
    graph_v_b.set_ylabel('Max sustained wind (kn)')

    _, graph_v_a = plt.subplots()
    graph_v_a.set_title('Wind after land decay correction')
    graph_v_a.set_xlabel('Node number')
    graph_v_a.set_ylabel('Max sustained wind (kn)')

    _, graph_p_b = plt.subplots()
    graph_p_b.set_title('Pressure before land decay correctionn')
    graph_p_b.set_xlabel('Node number')
    graph_p_b.set_ylabel('Central pressure (mb)')

    _, graph_p_a = plt.subplots()
    graph_p_a.set_title('Pressure after land decay correctionn')
    graph_p_a.set_xlabel('Node number')
    graph_p_a.set_ylabel('Central pressure (mb)')

    _, graph_pd_a = plt.subplots()
    graph_pd_a.set_title('Relative pressure after land decay correction')
    graph_pd_a.set_xlabel('Distance from landfall (km)')
    graph_pd_a.set_ylabel('Central pressure relative to landfall')

    _, graph_ped_a = plt.subplots()
    graph_ped_a.set_title(
        'Environmental - central pressure after land decay correction')
    graph_ped_a.set_xlabel('Distance from landfall (km)')
    graph_ped_a.set_ylabel('Environmental pressure - Central pressure (mb)')

    for track, orig_wind, orig_pres in \
            zip(sy_tracks, syn_orig_wind, syn_orig_pres):
        sea_land_idx, land_sea_idx = climada.hazard.tc_tracks._get_landfall_idx(track)
        if sea_land_idx.size:
            for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
                v_lf = track.max_sustained_wind[sea_land - 1].values
                p_lf = track.central_pressure[sea_land - 1].values
                scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
                ss_scale_idx = np.where(v_lf < scale_thresholds)[0][0]
                on_land = np.arange(track.time.size)[sea_land:land_sea]

                graph_v_a.plot(on_land, track.max_sustained_wind[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_v_b.plot(on_land, orig_wind[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_p_a.plot(on_land, track.central_pressure[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_p_b.plot(on_land, orig_pres[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_pd_a.plot(track.dist_since_lf[on_land],
                                track.central_pressure[on_land] / p_lf,
                                'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_ped_a.plot(track.dist_since_lf[on_land],
                                 track.environmental_pressure[on_land] -
                                 track.central_pressure[on_land],
                                 'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])

            on_sea = np.arange(track.time.size)[~track.on_land]
            graph_v_a.plot(on_sea, track.max_sustained_wind[on_sea],
                           'o', c='k', markersize=5)
            graph_v_b.plot(on_sea, orig_wind[on_sea],
                           'o', c='k', markersize=5)
            graph_p_a.plot(on_sea, track.central_pressure[on_sea],
                           'o', c='k', markersize=5)
            graph_p_b.plot(on_sea, orig_pres[on_sea],
                           'o', c='k', markersize=5)

    return graph_v_b, graph_v_a, graph_p_b, graph_p_a, graph_pd_a, graph_ped_a


def _check_apply_decay_hist_plot(hist_tracks):
    """Plot winds and pressures of historical tracks."""
    _, graph_hv = plt.subplots()
    graph_hv.set_title('Historical wind')
    graph_hv.set_xlabel('Node number')
    graph_hv.set_ylabel('Max sustained wind (kn)')

    _, graph_hp = plt.subplots()
    graph_hp.set_title('Historical pressure')
    graph_hp.set_xlabel('Node number')
    graph_hp.set_ylabel('Central pressure (mb)')

    _, graph_hpd_a = plt.subplots()
    graph_hpd_a.set_title('Historical relative pressure')
    graph_hpd_a.set_xlabel('Distance from landfall (km)')
    graph_hpd_a.set_ylabel('Central pressure relative to landfall')

    _, graph_hped_a = plt.subplots()
    graph_hped_a.set_title('Historical environmental - central pressure')
    graph_hped_a.set_xlabel('Distance from landfall (km)')
    graph_hped_a.set_ylabel('Environmental pressure - Central pressure (mb)')

    for track in hist_tracks:
        sea_land_idx, land_sea_idx = climada.hazard.tc_tracks._get_landfall_idx(track)
        if sea_land_idx.size:
            for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
                scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
                ss_scale_idx = np.where(track.max_sustained_wind[sea_land - 1].values
                                 < scale_thresholds)[0][0]
                on_land = np.arange(track.time.size)[sea_land:land_sea]

                graph_hv.add_curve(on_land, track.max_sustained_wind[on_land],
                                   'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_hp.add_curve(on_land, track.central_pressure[on_land],
                                   'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_hpd_a.plot(track.dist_since_lf[on_land],
                                 track.central_pressure[on_land]
                                 / track.central_pressure[sea_land - 1].values,
                                 'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_hped_a.plot(track.dist_since_lf[on_land],
                                  track.environmental_pressure[on_land] -
                                  track.central_pressure[on_land],
                                  'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])

            on_sea = np.arange(track.time.size)[~track.on_land]
            graph_hp.plot(on_sea, track.central_pressure[on_sea],
                          'o', c='k', markersize=5)
            graph_hv.plot(on_sea, track.max_sustained_wind[on_sea],
                          'o', c='k', markersize=5)

    return graph_hv, graph_hp, graph_hpd_a, graph_hped_a
