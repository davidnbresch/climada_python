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

Define WildFire class.
"""

__all__ = ['WildFire']

import logging
import warnings
import itertools
from datetime import date
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import numba
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from climada.hazard.centroids.centr import Centroids
from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.util.constants import ONE_LAT_KM, DEF_CRS
from climada.util.dates_times import str_to_date

from climada.util.alpha_shape import alpha_shape, plot_polygon
warnings.simplefilter(action='ignore', category=FutureWarning)

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'WF'
""" Hazard type acronym for Wild Fire, might be changed to WFseason or WFsingle """

CLEAN_THRESH = 30
""" Minimal confidence value for the data from MODIS instrument to be use as input"""

RES_DATA = 1.0
""" Resolution of the data if no data origin provided (km) """

BLURR_STEPS = 4
""" steps with exponential decay for fire propagation matrix """

class WildFire(Hazard):

    """Contains wild fires.

    Wildfires comprise the challange that the definition of an event is unclear.
    Reporting standards vary accross regions and over time. Hence, to have
    consistency, we consider an event as a whole fire season. This allows
    consistent risk assessment across the global and over time. Events that
    refer to a fire season have the tag 'WFseason'.

    In order to perform concrete case studies or calibrate impact functions,
    events can be displayed as single fires. In that case they have the tag
    'WFsingle'.


    Attributes:
        date_end ((np.array): integer date corresponding to the proleptic
            Gregorian ordinal, where January 1 of year 1 has ordinal 1
            (ordinal format of datetime library))

        n_fires (np.array): number of single fires in a fire season

    """

    days_thres_firms = 2
    """ Minimum number of days to consider different fires """

    clus_thres_firms = 15
    """ Clustering factor which multiplies instrument resolution """

    remove_minor_fires_firms = True
    """ removes FIRMS fires below defined theshold of entries"""

    minor_fire_thres_firms = 3
    """ number of FIRMS entries required to be considered a fire """

    prop_proba = 0.21
    """ probability of fire propagation """

    max_it_propa = 500000
    """ maximum propgation iterations for a probabilistic fire """

    def __init__(self, pool=None):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def set_hist_fire_FIRMS(self, csv_firms, centr_res_factor=1, centroids=None):
        """ Parse FIRMS data and generate historical fires by temporal and spatial
        clustering.

        Parameters:
            csv_firms: csv file of the FIRMS data (https://firms.modaps.eosdis.nasa.gov/download/)
                or pd.DataFrame of FIRMS data
            centr_res_factor (int, optional): resolution factor with respect to
                the satellite data to use for centroids creation. Default: 1
            centroids (Centroids, optional): centroids in degrees to map data
        """
        self.clear()

        # read and initialize data
        firms = self._clean_firms_csv(csv_firms)
        # compute centroids
        res_data = self._firms_resolution(firms)
        if not centroids:
            centroids = self._firms_centroids_creation(firms, res_data, centr_res_factor)
        else:
            if not centroids.coord.size:
                centroids.set_meta_to_lat_lon()
        res_centr = centroids._centroids_resolution(centroids)

        # fire identification
        while firms.iter_ev.any():
            # Compute cons_id: consecutive fires in current iteration
            self._firms_cons_days(firms)
            # Compute clus_id: cluster identifier inside cons_id
            self._firms_clustering(firms, res_data, self.clus_thres_firms)
            # compute event_id
            self._firms_fire(self.days_thres_firms, firms.cons_id.values, \
                             firms.clus_id.values, firms.event_id.values, \
                             firms.iter_ev.values, firms.datenum.values)
            LOGGER.info('Remaining fires to identify: %s.', str(np.argwhere(\
            firms.iter_ev.values).size))

        # remove minor fires
        if self.remove_minor_fires_firms:
            firms = self._firms_remove_minor_fires(firms,
                                                   self.minor_fire_thres_firms)

        # compute brightness and fill class attributes
        LOGGER.info('Computing intensity of %s fires.',
                    np.unique(firms.event_id).size)
        self._calc_brightness(firms, centroids, res_centr)


    def hull_burned_area(self, ev_id, alpha=100.87, return_plot=False):
        """Compute the burned area for a given fire.
        Please note: NASA advises against calculating burned area using the
        FIRMS data.

        Algorithm used: https://pypi.org/project/alphashape/

        Parameters:
            ev_id: id of the selected fire
            alpha (float, optional): parameter used to compute the concave hull
            return_plot (bool, optional): indicate if the output plot of the
                concave hull algorithm should be returned
            info

        Returns:
            float
        """
        ev_idx = np.argwhere(self.event_id == ev_id).reshape(-1)[0]
        if not ev_idx.size:
            LOGGER.error('No event with id %s', str(ev_id))
            raise ValueError

        if not self.centroids.lat.size:
            self.centroids.set_meta_to_lat_lon()

        # Extract coordinates where intensity > 0 (for a given fire)
        fire_lat = self.centroids.lat[self.intensity[ev_idx, :].nonzero()[1]]
        fire_lon = self.centroids.lon[self.intensity[ev_idx, :].nonzero()[1]]

        # Creation of the geodataframe
        fire = gpd.GeoDataFrame(crs=DEF_CRS)
        fire['geometry'] = list(zip(fire_lon, fire_lat))
        fire['geometry'] = fire['geometry'].apply(Point)
        points = fire.geometry.values

        # Compute concave hull
        concave_hull, _ = alpha_shape(points, alpha=alpha)

        # Compute area concave hull in right projection
        project = partial(
            pyproj.transform,
            pyproj.Proj(init=DEF_CRS['init']), # source coordinate system
            pyproj.Proj(init='epsg:3310')) # destination coordinate system: albers california

        concave_hull_m = transform(project, concave_hull)  # apply projection
        area_hull_one_fire = concave_hull_m.area/10000

        # Plot the polygone around the fire
        if return_plot:
            plot_polygon(concave_hull)
            plt.plot(fire_lon, fire_lat, 'o', color='red', markersize=0.5)

        return area_hull_one_fire

    def set_hist_fire_seasons_FIRMS(self, csv_firms, centr_res_factor=1,
                                    centroids=None, hemisphere=None,
                                    year_start=None, year_end=None,
                                    keep_all_fires=False):

        """ Parse FIRMS data and generate historical fire seasons. fires
        are created using temporal and spatial clustering according to the
        'set_hist_fire' method. fires are then summarized using max
        intensity at each centroid for each year.

        Parameters:
            csv_firms: csv file of the FIRMS data (https://firms.modaps.eosdis.nasa.gov/download/)
                or pd.DataFrame of FIRMS data
            centr_res_factor (int, optional): resolution factor with respect to
                the satellite data to use for centroids creation. Default: 1
            centroids (Centroids, optional): centroids in degrees to map data
            hemisphere (str, optional): 'SHS' or 'NHS' to define fire seasons
            year_start (int, optional): start year; FIRMS fires before that
                are cut; no cut if not specified
            year_end (int, optional): end year; FIRMS fires after that are cut;
                no cut if not specified
            keep_all_fires (bool, optional): keep detailed list of all fires;
                default is False to save memory.
        """

        LOGGER.info('Setting up historical fires for year set.')
        self.clear()

        # read and initialize data
        firms = self._clean_firms_csv(csv_firms)
        # compute centroids
        res_data = self._firms_resolution(firms)
        if not centroids:
            centroids = self._firms_centroids_creation(firms, res_data, centr_res_factor)
        else:
            if not centroids.coord.size:
                centroids.set_meta_to_lat_lon()

        # define hemisphere
        if hemisphere is None:
            if centroids.lat[0] > 0:
                hemisphere = 'NHS'
            elif centroids.lat[0] < 0:
                hemisphere = 'SHS'

        # define years
        years = np.arange(date.fromordinal(firms.datenum.min()).year,
                          date.fromordinal(firms.datenum.max()).year+1)
        if year_start is not None:
            years = np.delete(years, np.argwhere(years < year_start))
        if year_end is not None:
            years = np.delete(years, np.argwhere(years > year_end))

        # make fire seasons
        hist_fire_seasons = [] # list to save fire seasons

        for year in years:
            LOGGER.info('Setting up historical fire seasons %s.', str(year))
            firms_temp = self._select_fire_season(firms, year, hemisphere=hemisphere)
            # calculate historic fire seasons
            wf_year = WildFire()
            wf_year.set_hist_fire_FIRMS(firms_temp, centroids=centroids)
            hist_fire_seasons.append(wf_year)

        # fires season (used to define distribution of n fire for the
        # probabilistic fire seasons)
        n_fires = np.zeros(len(years))
        for idx, wf in enumerate(hist_fire_seasons):
            n_fires[idx] = len(wf.event_id)

        if keep_all_fires:
            self.hist_fire_seasons = hist_fire_seasons

        # save
        self.tag = TagHazard('WFseason')
        self.centroids = centroids
        self.n_fires = n_fires
        self.units = 'K' # Kelvin brightness

        # Following values are defined for each fire
        self.event_id = np.arange(1, len(years)+1).astype(int)
        self.event_name = list(map(str, years))
        self.date = np.zeros(len(years), int)
        self.date_end = np.zeros(len(years), int)
        if hemisphere == 'NHS':
            for y_idx, year in enumerate(years):
                self.date[y_idx] = date.toordinal(date(year, 1, 1))
                self.date_end[y_idx] = date.toordinal(date(year, 12, 31))
        elif hemisphere == 'SHS':
            for y_idx, year in enumerate(years):
                self.date[y_idx] = date.toordinal(date(year, 7, 1))
                self.date_end[y_idx] = date.toordinal(date(year+1, 6, 30))
        self.orig = np.ones(len(years), bool)
        self._set_frequency()

        # Following values are defined for each fire and centroid
        self.intensity = sparse.lil_matrix(np.zeros((len(years), len(centroids.lat))))
        for idx, wf in enumerate(hist_fire_seasons):
            self.intensity[idx] = wf.intensity.max(axis=0)
        self.intensity = self.intensity.tocsr()
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    def set_proba_fire_seasons(self, n_fire_seasons=1, n_ignitions=None,
                               keep_all_fires=False):
        """ Generate probabilistic fire seasons. Fire seasons
        are created by running n probabilistic fires per year which are then
        summarized into a probabilistic fire season by calculating the max
        intensity at each centroid for each probabilistic fire season.
        Probabilistic fires are created using the logic described in the
        method 'run_one_bushfire'.

        The fire propagation matrix can be assigned separately, if that is not
        done it will be generated on the available historic fire (seasons).

        Intensities are drawn randomly from historic events. Thus, this method
        requires at least one fire to draw from.

        Parameters:
            self: must have calculated historic fire seasons before
            n_event_years (int, optional): number of fire seasons to be generated
            n_ignitions (array, optional) -> [min, max]: min/max of uniform
                distribution to sample from, in order to determin n_fire per
                probabilistic year set. If none, min/max is taken from hist.
            keep_all_fires (bool, optional): keep detailed list of all fires;
                default is False to save memory.
        """
        # min/max for uniform distribtion to sample for n_fires per year
        if n_ignitions is None:
            ign_min = np.min(self.n_fires)
            ign_max = np.max(self.n_fires)
        else:
            ign_min = n_ignitions[0]
            ign_max = n_ignitions[1]

        prob_fire_seasons = [] # list to save probabilistic fire seasons
        # create probabilistic fire seasons
        for i in range(n_fire_seasons):
            n_ign = np.random.randint(ign_min, ign_max)
            LOGGER.info('Setting up probabilistic fire season with %s fires.',\
                        str(n_ign))
            prob_fire_seasons.append(self._set_one_proba_fire_season(n_ign, seed=i))

        if keep_all_fires:
            self.prob_fire_seasons = prob_fire_seasons

        # save
        # Following values are defined for each fire
        new_event_id = np.arange(np.max(self.event_id)+1, np.max(self.event_id)+n_fire_seasons+1)
        self.event_id = np.concatenate((self.event_id, new_event_id), axis=None)
        new_event_name = list(map(str, new_event_id))
        self.event_name = np.append(self.event_name, new_event_name)
        new_orig = np.zeros(len(new_event_id), bool)
        self.orig = np.concatenate((self.orig, new_orig))
        self._set_frequency()

        # Following values are defined for each event and centroid
        new_intensity = sparse.lil_matrix((np.zeros([n_fire_seasons, len(self.centroids.lat)])))
        for idx, wf in enumerate(prob_fire_seasons):
            new_intensity[idx] = sparse.csr_matrix(wf).max(0)
        new_intensity = new_intensity.tocsr()
        self.intensity = sparse.vstack([self.intensity, new_intensity],
                                       format='csr')
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    def combine_fires(self, event_id_merge=None, remove_rest=False,
                      probabilistic=False):
        """ Combine events that are identified as different fires by the
        clustering algorithms but need to be treated as one (i.e. due to impact
        data reporting or for case studies). Orig fires are removed and a new
        fire id created; max intensity at overlapping centroids is assigned.

        Parameters:
            event_id_merge (array of int, optional): events to be merged
            remove_rest (bool, optional): if set to true, only the merged event
                is returned.
            probabilistic(bool, optional): differentiate, because probabilistic
                events do not come with a date.
        """

        if probabilistic is False:
            if event_id_merge is not None:
                # get index of events to merge
                evt_idx_merge = []
                for i in event_id_merge:
                    evt_idx_merge.append(np.argwhere(self.event_id == i).reshape(-1)[0])
                # get dates
                date_start = np.min(self.date[evt_idx_merge])
                date_end = np.max(self.date_end[evt_idx_merge])

                if remove_rest:
                    self.intensity = sparse.csr_matrix( \
                        np.amax(self.intensity[evt_idx_merge], 0))
                    self.event_id = np.array([np.max(self.event_id)+1])
                    self.event_name = list(map(str, self.event_id))
                    self.date = np.array([date_start])
                    self.date_end = np.array([date_end])
                    self.orig = np.ones(1, bool)
                    self._set_frequency()
                    self.fraction = self.intensity.copy()
                    self.fraction.data.fill(1.0)
                else:
                    # merge event & append
                    self.intensity = sparse.vstack([self.intensity, \
                        np.amax(self.intensity[evt_idx_merge], 0)], format='csr')
                    self.event_id = np.append(self.event_id, np.max(self.event_id)+1)
                    self.event_name = list(map(str, self.event_id))
                    self.date = np.append(self.date, date_start)
                    self.date_end = np.append(self.date_end, date_end)
                    self.orig = np.append(self.orig, np.ones(1, bool))

                    # remove initial events
                    self.intensity = self.intensity[np.delete(np.arange(0, \
                        self.intensity.get_shape()[0]), evt_idx_merge)]
                    self.event_id = np.delete(self.event_id, evt_idx_merge)
                    self.event_name = list(map(str, self.event_id))
                    self.date = np.delete(self.date, evt_idx_merge)
                    self.date_end = np.delete(self.date_end, evt_idx_merge)
                    self.orig = np.delete(self.orig, evt_idx_merge)
                    self._set_frequency()
                    self.fraction = self.intensity.copy()
                    self.fraction.data.fill(1.0)

            else:
                self.intensity = sparse.csr_matrix(np.amax(self.intensity, 0))
                self.event_id = np.array([np.max(self.event_id)+1])
                self.event_name = list(map(str, self.event_id))
                self.date = np.array([np.min(self.date)])
                self.date_end = np.array([np.max(self.date_end)])
                self.orig = np.ones(1, bool)
                self._set_frequency()
                self.fraction = self.intensity.copy()
                self.fraction.data.fill(1.0)
            LOGGER.info('The merged event has event_id %s', self.event_id[-1])

        else:
            self.intensity = sparse.csr_matrix(np.amax(self.intensity, 0))
            self.event_id = np.array([np.max(self.event_id)+1])
            self.orig = np.zeros(1, bool)
            self._set_frequency()
            self.fraction = self.intensity.copy()
            self.fraction.data.fill(1.0)

    def summarize_fires_to_seasons(self, year_start=None, year_end=None,
                                   hemisphere=None):
        """ Summarize historic fires into fire seasons. Fires are summarized
        by taking the maximum intensity at each grid point.

        Parameters:
            year_start (int, optional): start year; fires before that
                are cut; no cut if not specified
            year_end (int, optional): end year; fires after that are cut;
                no cut if not specified
            hemisphere (str, optional): 'SHS' or 'NHS' to define fire seasons
        """

        # define hemisphere
        if hemisphere is None:
            if self.centroids.lat[0] > 0:
                hemisphere = 'NHS'
            elif self.centroids.lat[0] < 0:
                hemisphere = 'SHS'

        # define years
        fire_years = np.array([date.fromordinal(i).year for i in self.date])
        years = np.arange(np.min(fire_years), np.max(fire_years)+1)
        if year_start is not None:
            years = np.delete(years, np.argwhere(years < year_start))
        if year_end is not None:
            years = np.delete(years, np.argwhere(years > year_end))

        # summarize to fire season
        date_new = np.zeros(len(years), int)
        date_end_new = np.zeros(len(years), int)
        n_fires = np.zeros(len(years), int)
        intensity_new = sparse.lil_matrix(np.zeros((len(years), len(self.centroids.lat))))

        for i, year in enumerate(years):
            if hemisphere == 'NHS':
                start = date.toordinal(date(year, 1, 1))
                end = date.toordinal(date(year+1, 1, 1))
            elif hemisphere == 'SHS':
                start = date.toordinal(date(year, 7, 1))
                end = date.toordinal(date(year+1, 7, 1))

            date_new[i] = start
            date_end_new[i] = end
            idx = np.where((self.date > start-1) & \
                           (self.date < end + 1))
            n_fires[i] = len(idx)
            intensity_new[i] = sparse.csr_matrix( \
                                    np.amax(self.intensity[idx], 0))

        # save
        self.tag = TagHazard('WFseason')
        self.units = 'K' # Kelvin brightness

        # Following values are defined for each fire season
        self.event_id = np.arange(1, len(years)+1).astype(int)
        self.event_name = list(map(str, years))
        self.date = date_new
        self.date_end = date_end_new
        self.n_fires = n_fires
        self.orig = np.ones(len(years), bool)
        self._set_frequency()

        # Following values are defined for each fire and centroid
        self.intensity = intensity_new.tocsr()
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)


    @staticmethod
    def _clean_firms_csv(csv_firms):
        """Read and remove low confidence data from firms:
            - MODIS: remove data where confidence values are lower than CLEAN_THRESH
            - VIIRS: remove data where confidence values are set to low (keep
                nominal and high values)

        Parameters:
            csv_firms: csv file of the FIRMS data or pd.DataFrame of FIRMS data

        Returns:
            pd.DataFrame
        """
        if isinstance(csv_firms, pd.DataFrame):
            firms = csv_firms
        else:
            firms = pd.read_csv(csv_firms)
        # Check for the type of instrument (MODIS vs VIIRS)
        # Remove data with low confidence interval
        # Uniformize the name of the birghtness columns between VIIRS and MODIS
        temp = pd.DataFrame()
        if 'instrument' in firms.columns:
            if firms.instrument.any() == 'MODIS' or firms.instrument.any() == 'VIIRS':
                firms_modis = firms.drop(firms[firms.instrument == 'VIIRS'].index)
                firms_modis.confidence = np.array(
                    list(map(int, firms_modis.confidence.values.tolist())))
                firms_modis = firms_modis.drop(firms_modis[ \
                    firms_modis.confidence < CLEAN_THRESH].index)
                temp = firms_modis
                firms_viirs = firms.drop(firms[firms.instrument == 'MODIS'].index)
                if firms_viirs.size:
                    firms_viirs = firms_viirs.drop(firms_viirs[firms_viirs.confidence == 'l'].index)
                    firms_viirs = firms_viirs.rename(columns={'bright_ti4':'brightness'})
                    temp = temp.append(firms_viirs, sort=True)
                    temp = temp.drop(columns=['bright_ti4'])

            firms = temp
            firms = firms.reset_index()
            firms = firms.drop(columns=['index'])

        firms['iter_ev'] = np.ones(len(firms), bool)
        firms['cons_id'] = np.zeros(len(firms), int) - 1
        firms['event_id'] = np.zeros(len(firms), int)
        firms['clus_id'] = np.zeros(len(firms), int) - 1
        firms['datenum'] = np.array(str_to_date(firms['acq_date'].values))
        return firms

    @staticmethod
    def _firms_resolution(firms):
        """ Returns resolution of satellite used in FIRMS in degrees

        Parameters:
            firms (pd.DataFrame): FIRMS data

        Returns:
            float
        """
        # Resolution in km of the centroids depends on the data origin.
        if 'instrument' in firms.columns:
            if firms['instrument'].any() == 'MODIS':
                res_data = 1.0
            else:
                res_data = 0.375 # For VIIRS data
        else:
            res_data = RES_DATA # For undefined data origin, defined by user
        return res_data/ONE_LAT_KM

    @staticmethod
    def _firms_centroids_creation(firms, res_data, centr_res_factor):
        """ Get centroids from the firms dataset and refactor them.

        Parameters:
            firms (DataFrame): dataset obtained from FIRMS data
            res_data (float): FIRMS instrument resolution in degrees
            centr_res_factor (float): the factor applied to voluntarly decrease/increase
                the centroids resolution

        Returns:
            centroids (Centroids)
        """
        centroids = Centroids()
        centroids.set_raster_from_pnt_bounds((firms['longitude'].min(), \
            firms['latitude'].min(), firms['longitude'].max(), \
            firms['latitude'].max()), res=res_data/centr_res_factor)
        centroids.set_meta_to_lat_lon()
        centroids.set_area_approx()
        centroids.set_on_land()
        centroids.empty_geometry_points()

        return centroids

    def _firms_cons_days(self, firms):
        """ Compute clusters of consecutive days (temporal clusters).
            An interruption of days_thresh is necessary to be set in two
            different temporal clusters.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """
        LOGGER.debug('Computing clusters of consecutive days.')
        firms_iter = firms[firms['iter_ev']][['datenum', 'cons_id', 'event_id']]
        max_cons_id = firms.cons_id.max() + 1
        for event_id in np.unique(firms_iter.event_id.values):

            firms_cons = firms_iter[firms_iter.event_id == event_id].reset_index()

            # Order firms dataframe per ascending acq_date order
            firms_cons = firms_cons.sort_values('datenum')
            sort_idx = firms_cons.index
            firms_cons = firms_cons.reset_index()
            firms_cons = firms_cons.drop(columns=['index'])

            # Check if there is more than 1 day interruption between data
            firms_cons.at[0, 'cons_id'] = max_cons_id
            max_cons_id += 1
            for index in range(1, len(firms_cons)):
                if abs((firms_cons.at[index, 'datenum'] - firms_cons.at[index-1, 'datenum'])) \
                >= self.days_thres_firms:
                    firms_cons.at[index, 'cons_id'] = max_cons_id
                    max_cons_id += 1
                else:
                    firms_cons.at[index, 'cons_id'] = firms_cons.at[(index-1), 'cons_id']

            re_order = np.zeros(len(firms_cons), int)
            for data, order in zip(firms_cons.cons_id.values, sort_idx):
                re_order[order] = data
            firms_iter.cons_id.values[firms_iter.event_id == event_id] = re_order

        firms.cons_id.values[firms['iter_ev'].values] = firms_iter.cons_id.values
        return firms

    @staticmethod
    def _firms_clustering(firms, res_data, clus_thres):
        """Compute geographic clusters and sort firms with ascending clus_id
        for each cons_id.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            res_data (float): FIRMS instrument resolution in degrees

        Returns:
            firms
        """

        LOGGER.debug('Computing geographic clusters in consecutive fires.')
        firms_iter = firms[firms['iter_ev']][['latitude', 'longitude', 'cons_id',
                                              'clus_id', 'event_id']]

        for event_id in np.unique(firms_iter.event_id.values):

            firms_cons = firms_iter[firms_iter.event_id == event_id]

            # Creation of an identifier for geographical clustering
            # For each temporal cluster, perform geographical clustering with DBSCAN algo
            for cons_id in np.unique(firms_cons['cons_id'].values):
                temp = np.argwhere(firms_cons['cons_id'].values == cons_id).reshape(-1,)
                lat_lon = firms_cons.iloc[temp][['latitude', 'longitude']].values
                lat_lon_uni, lat_lon_cpy = np.unique(lat_lon, return_inverse=True, axis=0)
                cluster_id = DBSCAN(eps=res_data*clus_thres, min_samples=1).\
                                    fit(lat_lon_uni).labels_
                cluster_id = cluster_id[lat_lon_cpy]
                firms_cons.clus_id.values[temp] = cluster_id
                firms_iter.clus_id.values[firms_iter.event_id == event_id] = \
                    firms_cons.clus_id.values

        firms.clus_id.values[firms['iter_ev'].values] = firms_iter.clus_id.values

        return firms

    @staticmethod
    @numba.njit(parallel=True)
    def _firms_fire(days_thres, fir_cons_id, fir_clus_id, fir_ev_id, fir_iter,
                    fir_date):
        """Creation of event_id for each dataset point.
        A fire is characterized by a unique combination of 'cons_id' and 'clus_id'.

        Parameters:
            firms (dataframe)

        Returns:
            firms
        """
        ev_id = 0
        for cons_id in np.unique(fir_cons_id):
            firms_cons = fir_clus_id[fir_cons_id == cons_id]
            for clus_id in np.unique(firms_cons):
                fir_ev_id[np.logical_and(fir_cons_id == cons_id, \
                fir_clus_id == clus_id)] = ev_id
                ev_id += 1

        for ev_id in np.unique(fir_ev_id):
            date_ord = np.sort(fir_date[fir_ev_id == ev_id])
            if (np.diff(date_ord) < days_thres).all():
                fir_iter[fir_ev_id == ev_id] = False
            else:
                fir_iter[fir_ev_id == ev_id] = True

    @staticmethod
    def _firms_remove_minor_fires(firms, minor_fires_thres):
        """Remove fires containg fewer FIRMS entries than threshold.
        A fire is characterized by a unique combination of 'cons_id' and 'clus_id'.

        Parameters:
            firms (dataframe)
            minor_fires_thres(int)

        Returns:
            firms
        """
        for i in range(np.unique(firms.event_id).size):
            if (firms.event_id == i).sum() < minor_fires_thres:
                firms = firms.drop(firms[firms.event_id == i].index)
        # assign new event IDs
        event_id_new = 1
        for i in np.unique(firms.event_id):
            firms.event_id[firms.event_id == i] = event_id_new
            event_id_new = event_id_new + 1

        firms = firms.reset_index()

        return firms

    def _calc_brightness(self, firms, centroids, res_centr):
        """ Compute intensity matrix per fire with the maximum brightness at
        each centroid and al other hazard attributes.

        Parameters:
            firms (dataframe)
            centroids (Centroids): centroids for the dataset
            res_centr (float): centroids resolution in centroids unit

        Returns:
            brightness (Hazard)
        """
        uni_ev = np.unique(firms['event_id'].values)
        num_ev = uni_ev.size
        num_centr = centroids.size

        # For one fire, if more than one points of firms dataframe have the
        # same coordinates, take the maximum brightness value
        # of these points (maximal damages).
        tree_centr = BallTree(centroids.coord, metric='chebyshev')
        if self.pool:
            chunksize = min(num_ev//self.pool.ncpus, 1000)
            bright_list = self.pool.map(self._brightness_one_fire,
                                        itertools.repeat(firms, num_ev),
                                        itertools.repeat(tree_centr, num_ev),
                                        uni_ev, itertools.repeat(res_centr),
                                        itertools.repeat(num_centr),
                                        chunksize=chunksize)
        else:
            bright_list = []
            for ev_id in uni_ev:
                bright_list.append(self._brightness_one_fire(firms, tree_centr, \
                ev_id, res_centr, num_centr))

        bright_list, firms = self._remove_empty_fires(bright_list, firms)

        uni_ev = np.unique(firms['event_id'].values)
        num_ev = uni_ev.size

        # save
        self.tag = TagHazard('WFsingle')
        self.centroids = centroids
        self.units = 'K' # Kelvin brightness

        # Following values are defined for each fire
        self.event_id = np.arange(1, num_ev+1).astype(int)
        self.event_name = list(map(str, self.event_id))
        self.date = np.zeros(num_ev, int)
        self.date_end = np.zeros(num_ev, int)
        for ev_idx, ev_id in enumerate(uni_ev):
            self.date[ev_idx] = firms[firms.event_id == ev_id].datenum.min()
            self.date_end[ev_idx] = firms[firms.event_id == ev_id].datenum.max()
        self.orig = np.ones(num_ev, bool)
        self._set_frequency()

        # Following values are defined for each fire and centroid
        self.intensity = sparse.lil_matrix(np.zeros((num_ev, num_centr)))
        for idx, ev_bright in enumerate(bright_list):
            self.intensity[idx] = ev_bright
        self.intensity = self.intensity.tocsr()
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    @staticmethod
    def _brightness_one_fire(firms, tree_centr, ev_id, res_centr, num_centr):
        """ For a given fire, fill in an intensity np.array with the maximum brightness
        at each centroid.

        Parameters:
            firms (dataframe)
            centroids (Centroids): centroids for the dataset
            ev_id (int): id of the selected event

        Returns:
            brightness_ev (np.array): maximum brightness at each centroids

        """
        LOGGER.debug('Brightness corresponding to FIRMS event %s.', str(ev_id))
        temp_firms = firms.reindex(
            index=(np.argwhere(firms['event_id'].values == ev_id).reshape(-1,)),
            columns=['latitude', 'longitude', 'brightness', 'datenum'])

        # Identifies the unique (lat,lon) points of the firms dataframe -> lat_lon_uni
        # Set the same index value for each duplicate (lat,lon) points -> lat_lon_cpy
        lat_lon_uni, lat_lon_cpy = np.unique(temp_firms[['latitude', 'longitude']].values,
                                             return_inverse=True, axis=0)
        index_uni = np.unique(lat_lon_cpy, axis=0)

        # Search closest centroid for each firms point
        ind, _ = tree_centr.query_radius(lat_lon_uni, r=res_centr/2, count_only=False,
                                         return_distance=True, sort_results=True)
        ind = np.array([ind_i[0] if ind_i.size else -1 for ind_i in ind])

        brightness_ev = _fill_intensity_max(num_centr, ind, index_uni, lat_lon_cpy,
                                            temp_firms['brightness'].values)

        return sparse.lil_matrix(brightness_ev)

    @staticmethod
    def _remove_empty_fires(bright_list, firms):
        bright_list_nonzero = []
        event_id_new = 1
        for i, br_list in enumerate(bright_list):
            if br_list.count_nonzero() > 0:
                bright_list_nonzero.append(br_list)
                firms.event_id.values[firms.event_id == i+1] = event_id_new
                event_id_new = event_id_new + 1
            else:
                firms = firms.drop(firms[firms.event_id == i].index)

        firms = firms.reset_index()
        LOGGER.info('Returning %s fires that impacted the defined centroids.',
                    len(bright_list_nonzero))

        return bright_list_nonzero, firms

    def _set_one_proba_fire_season(self, n_ignitions, seed=8):
        """ Generate a probabilistic fire season.

        Parameters:
            n_ignitions (int): number of wild fires
            seed (int)

        Returns:
            proba_fires (lil_matrix)
        """
        np.random.seed(seed)
        proba_fires = sparse.lil_matrix(np.zeros((n_ignitions, self.centroids.size)))
        for i in range(n_ignitions):
            if np.mod(i, 10) == 0:
                LOGGER.info('Created %s fires', str(i))
            centr_burned = self._run_one_fire()
            proba_fires[i, :] = self._set_proba_intensity(centr_burned)

        return proba_fires

    def _run_one_fire(self):
        """ Run one bushfire on a fire propagation probability matrix.
            If the matrix is not defined, it is constructed using past fire
            experience -> a fire can only propagate on centroids that burned
            in the past including a exponentially blurred range around the
            historic fires.
            The ignition point of a fire can be on any centroid, on which
            the propagation probability equals 1. The fire is then propagated
            with a cellular automat.
            If the fire has not stoped bruning after a defined number of
            iterations (self.max_it_propa, default=500'000), the propagation
            is interrupted.

            Propagation rules:
                1. select a burning centroid (at the start there is only one,
                    afterwards select one randomly)
                2. every adjunct centroid to the selected centroid can start
                    burning with a probability of the overall propagation
                    probability (self.prop_proba) times the centroid specific
                    propagation probability (which is defined on the
                    fire_propa_matrix).
                3. the selected burning centroid becomes an ember centroid
                    which can not start burning again and thus no longer
                    propagate any fire.
            Properties from centr_burned:
                0 = unburned centroid
                1 = burning centroid
                2 = ember centroid
            Stop criteria: the propagation stops when no centroid is burning.
            The initial version of this code was inspired by
            https://scipython.com/blog/the-forest-fire-model/


        Returns:
            centr_burned
        """
        # set fire propagation matrix if not already defined
        if not hasattr(self.centroids, 'fire_propa_matrix'):
            self._set_fire_propa_matrix()

        # Ignation only at centroids that burned in the past
        pos_centr = np.argwhere(self.centroids.fire_propa_matrix.reshape( \
            len(self.centroids.lat)) == 1)[:, 1]

        LOGGER.debug('Start ignition.')
        # Random selection of ignition centroid
        for _ in range(self.centroids.size):
            centr = np.random.choice(pos_centr)
            centr_ix = int(centr/self.centroids.shape[1])
            centr_iy = centr%self.centroids.shape[1]
            centr_ix = max(0, centr_ix)
            centr_ix = min(self.centroids.shape[0]-1, centr_ix)
            centr_iy = max(0, centr_iy)
            centr_iy = min(self.centroids.shape[1]-1, centr_iy)
            centr = centr_ix*self.centroids.shape[1] + centr_iy
            if 1 <= centr_ix < self.centroids.shape[0] - 1 and \
            1 <= centr_iy < self.centroids.shape[1] - 1:
                break

        LOGGER.debug('Propagate fire.')
        centr_burned = np.zeros((self.centroids.shape), int)
        centr_burned[centr_ix, centr_iy] = 1
        # Iterate the fire according to the propagation rules
        count_it = 0
        while np.any(centr_burned == 1) and count_it < self.max_it_propa:
            count_it += 1
            # Select randomly one of the burning centroids
            # and propagate throught its neighborhood
            burned = np.argwhere(centr_burned == 1)
            if len(burned) > 1:
                centr_ix, centr_iy = burned[np.random.randint(0, len(burned))]
            elif len(burned) == 1:
                centr_ix, centr_iy = burned[0]
            if not count_it % (self.max_it_propa-1):
                LOGGER.warning('Fire propagation not converging at iteration %s.',
                               count_it)
            if 1 <= centr_ix < self.centroids.shape[0]-1 and \
            1 <= centr_iy < self.centroids.shape[1]-1 and \
            self.centroids.on_land[(centr_ix*self.centroids.shape[1] + centr_iy)]:
                centr_burned = self._fire_propagation_on_matrix(self.centroids.shape, \
                    self.centroids.fire_propa_matrix, self.prop_proba, \
                    centr_ix, centr_iy, centr_burned, np.random.random(500))

        return centr_burned

    @staticmethod
    @numba.njit
    def _fire_propagation_on_matrix(centr_shape, fire_propa_matrix, prop_proba,
                                    centr_ix, centr_iy, centr_burned, prob_array):
        """ Propagation of the fire in the 8 neighbouring cells around
        (centr_ix, centr_iy) according to propagation rules.

        Parameters:
            centr_shape(np.array): shape of centroids
            fire_propa_matrix(np.array): fire proagation matrix
            prop_proba(float): global propagation probability
            centr_ix (int): x coordinates of the burning centroid in the
                centroids matrix
            centr_iy (int): y coordinates of the burning centroid in the
                centroids matrix
            centr_burned (np.array): array containing burned centroids
            prob_array(np.array): array of random numbers to draw from for
                random fire propagation

        Returns:
            centr_burned(np.array): updated centr_burned matrix
        """
        # Neighbourhood
        hood = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        # Creation of the temporary centroids grids
        burned_one_step = np.zeros(centr_shape, dtype=numba.int32)
        burned_one_step[centr_ix, centr_iy] = 1

        # Displacements from a centroid to its eight neighbours
        for i_neig, (delta_x, delta_y) in enumerate(hood):
            if prob_array[i_neig] <= prop_proba * \
                fire_propa_matrix[centr_ix+delta_x, centr_iy+delta_y] and \
            centr_burned[centr_ix+delta_x, centr_iy+delta_y] == 0:
                burned_one_step[centr_ix+delta_x, centr_iy+delta_y] = 1

        # Calculate burned area
        centr_burned += burned_one_step

        return centr_burned

    def _set_proba_intensity(self, centr_burned):
        """ The intensity values are chosen randomly at every burned centroid
        from the intensity values of the historical fire

        Parameters:
            self (bf): contains info of historical fires
            centr_burned (np.array): array containing burned centroids

        Returns:
            proba_intensity(lil_matrix): intensity of probabilistic fire
        """
        # The brightness values are chosen randomly at every burned centroids
        # from the brightness values of the historical fire
        ev_proba_uni = centr_burned.nonzero()[0] * self.centroids.shape[1] + \
            centr_burned.nonzero()[1]
        proba_intensity = sparse.lil_matrix(np.zeros((1, self.centroids.size)))
        for ev_prob in ev_proba_uni:
            proba_intensity[0, ev_prob] = np.random.choice( \
                self.intensity.data)

        return proba_intensity

    def _set_fire_propa_matrix(self, n_blurr=BLURR_STEPS):

        """ sets fire propagation matrix which is used to propagate
        probabilistic fires. The matrix is set so that burn probability on
        centroids which burned historically is set to 1. A blurr with
        exponential decay of burn probabilities is set around these
        centroids.
        Alternatively, the fire propagation probability matrix can be any
        matrix that coresponds to the shape of the centroids and thus not have
        to be set this way.

        Parameters:
            self (bf): contains info of historical fires
            n_blurr (int): blurr with around historical fires

        Assigns:
            self.centroids.fire_propa_matrix (np.array)
        """
        # historically burned centroids
        hist_burned = np.zeros(self.centroids.lat.shape, dtype=bool)
        hist_burned = self.intensity.sum(0) > 0.
        self.centroids.hist_burned = hist_burned

        fire_propa_matrix = hist_burned.reshape(self.centroids.shape).astype(float)
        blurr_lvl = np.ones(n_blurr)
        # exponential decay of fire propagation on cell level
        for i in range(n_blurr):
            blurr_lvl[i] = 2.**(-i)
        # Neighbourhood
        hood = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        # loop over matrix to create blurr around historical fires
        for blurr in range(n_blurr-1):
            for i in range(fire_propa_matrix.shape[0]):
                for j in range(fire_propa_matrix.shape[1]):
                    if fire_propa_matrix[i, j] == blurr_lvl[blurr]:
                        for _, (delta_x, delta_y) in enumerate(hood):
                            try:
                                if fire_propa_matrix[i+delta_x, j+delta_y] == 0:
                                    fire_propa_matrix[i+delta_x, j+delta_y] = blurr_lvl[blurr+1]
                            except IndexError:
                                pass

        self.centroids.fire_propa_matrix = fire_propa_matrix

    def plot_fire_prob_matrix(self):
        """ Plots fire propagation probability matrix as contour plot.
        At this point just to check the matrix but could easily be improved to
        normal map.

        Returns.
            contour plot of fire_propa_matrix
        """

        lon = np.reshape(self.centroids.lon, self.centroids.fire_propa_matrix.shape)
        lat = np.reshape(self.centroids.lat, self.centroids.fire_propa_matrix.shape)
        plt.contourf(lon, lat, self.centroids.fire_propa_matrix)

    @staticmethod
    def _select_fire_season(firms, year, hemisphere='SHS'):
        """ Selects data to create historic fire season. Need to
        differentiate between Northern & Souther hemisphere

        Parameters:
            firms (pd.dataframe)
            year (int)
            hemisphere (str, optional): 'NHS' or 'SHS'
        Returns:
            firms (pd.dataframe): all fire of the specified fire season
        """

        firms['date'] = firms['acq_date'].apply(pd.to_datetime)
        if hemisphere == 'NHS':
            start = pd.Timestamp(year, 1, 1)
            end = pd.Timestamp(year+1, 1, 1)
        elif hemisphere == 'SHS':
            start = pd.Timestamp(year, 7, 1)
            end = pd.Timestamp(year+1, 7, 1)

        firms = firms[(firms['date'] > start) & (firms['date'] < end)]
        firms = firms.drop('date', axis=1)
        return firms

    def _set_frequency(self):
        """Set hazard frequency from intensity matrix. """
        delta_time = date.fromordinal(int(np.max(self.date))).year - \
            date.fromordinal(int(np.min(self.date))).year + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

@numba.njit
def _fill_intensity_max(num_centr, ind, index_uni, lat_lon_cpy, fir_bright):
    brightness_ev = np.zeros((1, num_centr), dtype=numba.float64)
    for idx in range(index_uni.size):
        if ind[idx] != -1:
            brightness_ev[0, ind[idx]] = max(brightness_ev[0, ind[idx]], \
                         np.max(fir_bright[lat_lon_cpy == index_uni[idx]]))
    return brightness_ev
