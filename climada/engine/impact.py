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

Define Impact and ImpactFreqCurve classes.
"""

__all__ = ['ImpactFreqCurve', 'Impact']

import logging
import copy
import csv
import warnings
import datetime as dt
from itertools import zip_longest
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import xlsxwriter
from tqdm import tqdm


from climada.entity import Exposures, Tag
from climada.entity.exposures import INDICATOR_CENTR
from climada.hazard import Tag as TagHaz
import climada.util.plot as u_plot
from climada import CONFIG
from climada.util.constants import DEF_CRS, CMAP_CONTINUOUS2
import climada.util.coordinates as u_coord
import climada.util.dates_times as u_dt
from climada.util.select import get_attributes_with_matching_dimension

LOGGER = logging.getLogger(__name__)

class Impact():
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes:
        tag (dict): dictionary of tags of exposures, impact functions set and
            hazard: {'exp': Tag(), 'impf_set': Tag(), 'haz': TagHazard()}
        event_id (np.array): id (>0) of each hazard event
        event_name (list): name of each hazard event
        date (np.array): date if events as integer date corresponding to the
            proleptic Gregorian ordinal, where January 1 of year 1 has
            ordinal 1 (ordinal format of datetime library)
        coord_exp (np.ndarray): exposures coordinates [lat, lon] (in degrees)
        eai_exp (np.array): expected annual impact for each exposure
        at_event (np.array): impact for each hazard event
        frequency (np.arrray): annual frequency of event
        tot_value (float): total exposure value affected
        aai_agg (float): average annual impact (aggregated)
        unit (str): value unit used (given by exposures unit)
        imp_mat (sparse.csr_matrix): matrix num_events x num_exp with impacts.
            only filled if save_mat is True in calc()
    """

    def __init__(self):
        """Empty initialization."""
        self.tag = dict()
        self.event_id = np.array([], int)
        self.event_name = list()
        self.date = np.array([], int)
        self.coord_exp = np.ndarray([], float)
        self.crs = DEF_CRS
        self.eai_exp = np.array([])
        self.at_event = np.array([])
        self.frequency = np.array([])
        self.tot_value = 0
        self.aai_agg = 0
        self.unit = ''
        self.imp_mat = sparse.csr_matrix(np.empty((0, 0)))

    def calc_freq_curve(self, return_per=None):
        """Compute impact exceedance frequency curve.

        Parameters:
            return_per (np.array, optional): return periods where to compute
                the exceedance impact. Use impact's frequencies if not provided

        Returns:
            ImpactFreqCurve
        """
        ifc = ImpactFreqCurve()
        ifc.tag = self.tag
        # Sort descendingly the impacts per events
        sort_idxs = np.argsort(self.at_event)[::-1]
        # Calculate exceedence frequency
        exceed_freq = np.cumsum(self.frequency[sort_idxs])
        # Set return period and imact exceeding frequency
        ifc.return_per = 1 / exceed_freq[::-1]
        ifc.impact = self.at_event[sort_idxs][::-1]
        ifc.unit = self.unit
        ifc.label = 'Exceedance frequency curve'

        if return_per is not None:
            interp_imp = np.interp(return_per, ifc.return_per, ifc.impact)
            ifc.return_per = return_per
            ifc.impact = interp_imp

        return ifc

    def calc(self, exposures, impact_funcs, hazard, save_mat=False):
        """Compute impact of an hazard to exposures.

        Parameters:
            exposures (Exposures): exposures
            impact_funcs (ImpactFuncSet): impact functions
            hazard (Hazard): hazard
            self_mat (bool): self impact matrix: events x exposures

        Examples:
            Use Entity class:

            >>> haz = Hazard('TC') # Set hazard
            >>> haz.read_mat(HAZ_DEMO_MAT)
            >>> haz.check()
            >>> ent = Entity() # Load entity with default values
            >>> ent.read_excel(ENT_TEMPLATE_XLS) # Set exposures
            >>> ent.check()
            >>> imp = Impact()
            >>> imp.calc(ent.exposures, ent.impact_funcs, haz)
            >>> imp.calc_freq_curve().plot()

            Specify only exposures and impact functions:

            >>> haz = Hazard('TC') # Set hazard
            >>> haz.read_mat(HAZ_DEMO_MAT)
            >>> haz.check()
            >>> funcs = ImpactFuncSet()
            >>> funcs.read_excel(ENT_TEMPLATE_XLS) # Set impact functions
            >>> funcs.check()
            >>> exp = Exposures(pd.read_excel(ENT_TEMPLATE_XLS)) # Set exposures
            >>> exp.check()
            >>> imp = Impact()
            >>> imp.calc(exp, funcs, haz)
            >>> imp.aai_agg
        """
        # 1. Assign centroids to each exposure if not done
        assign_haz = INDICATOR_CENTR + hazard.tag.haz_type
        if assign_haz not in exposures.gdf:
            exposures.assign_centroids(hazard)
        else:
            LOGGER.info('Exposures matching centroids found in %s', assign_haz)

        # 2. Initialize values
        self.unit = exposures.value_unit
        self.event_id = hazard.event_id
        self.event_name = hazard.event_name
        self.date = hazard.date
        self.coord_exp = np.stack([exposures.gdf.latitude.values,
                                   exposures.gdf.longitude.values], axis=1)
        self.frequency = hazard.frequency
        self.at_event = np.zeros(hazard.intensity.shape[0])
        self.eai_exp = np.zeros(exposures.gdf.value.size)
        self.tag = {'exp': exposures.tag, 'impf_set': impact_funcs.tag,
                    'haz': hazard.tag}
        self.crs = exposures.crs

        # Select exposures with positive value and assigned centroid
        exp_idx = np.where((exposures.gdf.value > 0) & (exposures.gdf[assign_haz] >= 0))[0]
        if exp_idx.size == 0:
            LOGGER.warning("No affected exposures.")

        num_events = hazard.intensity.shape[0]
        LOGGER.info('Calculating damage for %s assets (>0) and %s events.',
                    exp_idx.size, num_events)

        # Get damage functions for this hazard
        impf_haz = exposures.get_impf_column(hazard.tag.haz_type)
        haz_imp = impact_funcs.get_func(hazard.tag.haz_type)

        # Check if deductible and cover should be applied
        insure_flag = False
        if ('deductible' in exposures.gdf) and ('cover' in exposures.gdf) \
        and exposures.gdf.cover.max():
            insure_flag = True

        if save_mat:
            # (data, (row_ind, col_ind))
            self.imp_mat = ([], ([], []))

        # 3. Loop over exposures according to their impact function
        tot_exp = 0
        for imp_fun in haz_imp:
            # get indices of all the exposures with this impact function
            exp_iimp = np.where(exposures.gdf[impf_haz].values[exp_idx] == imp_fun.id)[0]
            tot_exp += exp_iimp.size
            exp_step = CONFIG.max_matrix_size.int() // num_events
            if not exp_step:
                raise ValueError('Increase max_matrix_size configuration parameter to > %s'
                                 % str(num_events))
            # separte in chunks
            chk = -1
            for chk in range(int(exp_iimp.size / exp_step)):
                self._exp_impact(
                    exp_idx[exp_iimp[chk * exp_step:(chk + 1) * exp_step]],
                    exposures, hazard, imp_fun, insure_flag)
            self._exp_impact(exp_idx[exp_iimp[(chk + 1) * exp_step:]],
                             exposures, hazard, imp_fun, insure_flag)

        if not tot_exp:
            LOGGER.warning('No impact functions match the exposures.')
        self.aai_agg = sum(self.at_event * hazard.frequency)

        if save_mat:
            shape = (self.date.size, exposures.gdf.value.size)
            self.imp_mat = sparse.csr_matrix(self.imp_mat, shape=shape)

    def calc_risk_transfer(self, attachment, cover):
        """Compute traaditional risk transfer over impact. Returns new impact
        with risk transfer applied and the insurance layer resulting Impact metrics.

        Parameters:
            attachment (float): attachment (deductible)
            cover (float): cover

        Returns:
            Impact, Impact
        """
        new_imp = copy.deepcopy(self)
        if attachment or cover:
            imp_layer = np.minimum(np.maximum(new_imp.at_event - attachment, 0), cover)
            new_imp.at_event = np.maximum(new_imp.at_event - imp_layer, 0)
            new_imp.aai_agg = np.sum(new_imp.at_event * new_imp.frequency)
            # next values are no longer valid
            new_imp.eai_exp = np.array([])
            new_imp.coord_exp = np.array([])
            new_imp.imp_mat = sparse.csr_matrix(np.empty((0, 0)))
            # insurance layer metrics
            risk_transfer = copy.deepcopy(new_imp)
            risk_transfer.at_event = imp_layer
            risk_transfer.aai_agg = np.sum(imp_layer * new_imp.frequency)
            return new_imp, risk_transfer

        return new_imp, Impact()

    def plot_hexbin_eai_exposure(self, mask=None, ignore_zero=True,
                                 pop_name=True, buffer=0.0, extend='neither',
                                 axis=None, **kwargs):
        """Plot hexbin expected annual impact of each exposure.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates.
                Default: 1.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for hexbin matplotlib function

        Returns:
            cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_CONTINUOUS2

        eai_exp = self._build_exp()
        axis = eai_exp.plot_hexbin(mask, ignore_zero, pop_name, buffer,
                                   extend, axis=axis, **kwargs)
        axis.set_title('Expected annual impact')
        return axis

    def plot_scatter_eai_exposure(self, mask=None, ignore_zero=True,
                                  pop_name=True, buffer=0.0, extend='neither',
                                  axis=None, **kwargs):
        """Plot scatter expected annual impact of each exposure.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates.
                Default: 1.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for hexbin matplotlib function

        Returns:
            cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_CONTINUOUS2

        eai_exp = self._build_exp()
        axis = eai_exp.plot_scatter(mask, ignore_zero, pop_name, buffer,
                                    extend, axis=axis, **kwargs)
        axis.set_title('Expected annual impact')
        return axis

    def plot_raster_eai_exposure(self, res=None, raster_res=None, save_tiff=None,
                                 raster_f=lambda x: np.log10((np.fmax(x + 1, 1))),
                                 label='value (log10)', axis=None, **kwargs):
        """Plot raster expected annual impact of each exposure.

        Parameters:
            res (float, optional): resolution of current data in units of latitude
                and longitude, approximated if not provided.
            raster_res (float, optional): desired resolution of the raster
            save_tiff (str, optional): file name to save the raster in tiff
                format, if provided
            raster_f (lambda function): transformation to use to data. Default:
                log10 adding 1.
            label (str): colorbar label
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for imshow matplotlib function

        Returns:
            cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        eai_exp = self._build_exp()
        axis = eai_exp.plot_raster(res, raster_res, save_tiff, raster_f,
                                   label, axis=axis, **kwargs)
        axis.set_title('Expected annual impact')
        return axis

    def plot_basemap_eai_exposure(self, mask=None, ignore_zero=False, pop_name=True,
                                  buffer=0.0, extend='neither', zoom=10,
                                  url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png',
                                  axis=None, **kwargs):
        """Plot basemap expected annual impact of each exposure.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates. Default: 0.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            zoom (int, optional): zoom coefficient used in the satellite image
            url (str, optional): image source, e.g. ctx.sources.OSM_C
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for scatter matplotlib function, e.g.
                cmap='Greys'. Default: 'Wistia'

        Returns:
            cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_CONTINUOUS2
        eai_exp = self._build_exp()
        axis = eai_exp.plot_basemap(mask, ignore_zero, pop_name, buffer,
                                    extend, zoom, url, axis=axis, **kwargs)
        axis.set_title('Expected annual impact')
        return axis

    def plot_hexbin_impact_exposure(self, event_id=1, mask=None, ignore_zero=True,
                                    pop_name=True, buffer=0.0, extend='neither',
                                    axis=None, **kwargs):
        """Plot hexbin impact of an event at each exposure.
        Requires attribute imp_mat.

        Parameters:
            event_id (int, optional): id of the event for which to plot the impact.
                Default: 1.
            mask (np.array, optional): mask to apply to impact plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates.
                Default: 1.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            kwargs (optional): arguments for hexbin matplotlib function
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use

        Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
            """
        if not hasattr(self.imp_mat, "shape") or self.imp_mat.shape[1] == 0:
            raise ValueError('attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True')
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_CONTINUOUS2
        impact_at_events_exp = self._build_exp_event(event_id)
        axis = impact_at_events_exp.plot_hexbin(mask, ignore_zero, pop_name,
                                                buffer, extend, axis=axis, **kwargs)

        return axis

    def plot_basemap_impact_exposure(self, event_id=1, mask=None, ignore_zero=True,
                                     pop_name=True, buffer=0.0, extend='neither', zoom=10,
                                     url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png',
                                     axis=None, **kwargs):
        """Plot basemap impact of an event at each exposure.
        Requires attribute imp_mat.

        Parameters:
            event_id (int, optional): id of the event for which to plot the impact.
                Default: 1.
            mask (np.array, optional): mask to apply to impact plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates. Default: 0.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            zoom (int, optional): zoom coefficient used in the satellite image
            url (str, optional): image source, e.g. ctx.sources.OSM_C
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for scatter matplotlib function, e.g.
                cmap='Greys'. Default: 'Wistia'

        Returns:
            cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if not hasattr(self.imp_mat, "shape") or self.imp_mat.shape[1] == 0:
            raise ValueError('attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True')

        if event_id not in self.event_id:
            raise ValueError(f'Event ID {event_id} not found')
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_CONTINUOUS2
        impact_at_events_exp = self._build_exp_event(event_id)
        axis = impact_at_events_exp.plot_basemap(mask, ignore_zero, pop_name,
                                                 buffer, extend, zoom, url, axis=axis, **kwargs)

        return axis

    def write_csv(self, file_name):
        """Write data into csv file. imp_mat is not saved.

        Parameters:
            file_name (str): absolute path of the file
        """
        LOGGER.info('Writing %s', file_name)
        with open(file_name, "w") as imp_file:
            imp_wr = csv.writer(imp_file)
            imp_wr.writerow(["tag_hazard", "tag_exposure", "tag_impact_func",
                             "unit", "tot_value", "aai_agg", "event_id",
                             "event_name", "event_date", "event_frequency",
                             "at_event", "eai_exp", "exp_lat", "exp_lon", "exp_crs"])
            csv_data = [[[self.tag['haz'].haz_type], [self.tag['haz'].file_name],
                         [self.tag['haz'].description]],
                        [[self.tag['exp'].file_name], [self.tag['exp'].description]],
                        [[self.tag['impf_set'].file_name], [self.tag['impf_set'].description]],
                        [self.unit], [self.tot_value], [self.aai_agg],
                        self.event_id, self.event_name, self.date,
                        self.frequency, self.at_event,
                        self.eai_exp, self.coord_exp[:, 0], self.coord_exp[:, 1],
                        [str(self.crs)]]
            for values in zip_longest(*csv_data):
                imp_wr.writerow(values)

    def write_excel(self, file_name):
        """Write data into Excel file. imp_mat is not saved.

        Parameters:
            file_name (str): absolute path of the file
        """
        LOGGER.info('Writing %s', file_name)
        def write_col(i_col, imp_ws, xls_data):
            """Write one measure"""
            row_ini = 1
            for dat_row in xls_data:
                imp_ws.write(row_ini, i_col, dat_row)
                row_ini += 1

        imp_wb = xlsxwriter.Workbook(file_name)
        imp_ws = imp_wb.add_worksheet()

        header = ["tag_hazard", "tag_exposure", "tag_impact_func",
                  "unit", "tot_value", "aai_agg", "event_id",
                  "event_name", "event_date", "event_frequency",
                  "at_event", "eai_exp", "exp_lat", "exp_lon", "exp_crs"]
        for icol, head_dat in enumerate(header):
            imp_ws.write(0, icol, head_dat)
        data = [self.tag['haz'].haz_type, str(self.tag['haz'].file_name),
                str(self.tag['haz'].description)]
        write_col(0, imp_ws, data)
        data = [str(self.tag['exp'].file_name), str(self.tag['exp'].description)]
        write_col(1, imp_ws, data)
        data = [str(self.tag['impf_set'].file_name), str(self.tag['impf_set'].description)]
        write_col(2, imp_ws, data)
        write_col(3, imp_ws, [self.unit])
        write_col(4, imp_ws, [self.tot_value])
        write_col(5, imp_ws, [self.aai_agg])
        write_col(6, imp_ws, self.event_id)
        write_col(7, imp_ws, self.event_name)
        write_col(8, imp_ws, self.date)
        write_col(9, imp_ws, self.frequency)
        write_col(10, imp_ws, self.at_event)
        write_col(11, imp_ws, self.eai_exp)
        write_col(12, imp_ws, self.coord_exp[:, 0])
        write_col(13, imp_ws, self.coord_exp[:, 1])
        write_col(14, imp_ws, [str(self.crs)])

        imp_wb.close()

    def write_sparse_csr(self, file_name):
        """Write imp_mat matrix in numpy's npz format."""
        LOGGER.info('Writing %s', file_name)
        np.savez(file_name, data=self.imp_mat.data, indices=self.imp_mat.indices,
                 indptr=self.imp_mat.indptr, shape=self.imp_mat.shape)

    def calc_impact_year_set(self, all_years=True, year_range=None):
        """Calculate yearly impact from impact data.

        Parameters:
            all_years (boolean): return values for all years between first and
            last year with event, including years without any events.
            year_range (tuple or list with integers): start and end year

        Returns:
             Impact year set of type numpy.ndarray with summed impact per year.
        """
        if year_range is None:
            year_range = []

        orig_year = np.array([dt.datetime.fromordinal(date).year
                              for date in self.date])
        if orig_year.size == 0 and len(year_range) == 0:
            return dict()
        if orig_year.size == 0 or (len(year_range) > 0 and all_years):
            years = np.arange(min(year_range), max(year_range) + 1)
        elif all_years:
            years = np.arange(min(orig_year), max(orig_year) + 1)
        else:
            years = np.array(sorted(np.unique(orig_year)))
        if not len(year_range) == 0:
            years = years[years >= min(year_range)]
            years = years[years <= max(year_range)]

        year_set = dict()

        for year in years:
            year_set[year] = sum(self.at_event[orig_year == year])
        return year_set

    def local_exceedance_imp(self, return_periods=(25, 50, 100, 250)):
        """Compute exceedance impact map for given return periods.
        Requires attribute imp_mat.

        Parameters:
            return_periods (np.array): return periods to consider

        Returns:
            np.array
        """
        LOGGER.info('Computing exceedance impact map for return periods: %s',
                    return_periods)
        try:
            self.imp_mat.shape[1]
        except AttributeError as err:
            raise ValueError('attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True') from err
        num_cen = self.imp_mat.shape[1]
        imp_stats = np.zeros((len(return_periods), num_cen))
        cen_step = CONFIG.max_matrix_size.int() // self.imp_mat.shape[0]
        if not cen_step:
            raise ValueError('Increase max_matrix_size configuration parameter to > %s'
                             % str(self.imp_mat.shape[0]))
        # separte in chunks
        chk = -1
        for chk in range(int(num_cen / cen_step)):
            self._loc_return_imp(np.array(return_periods),
                                 self.imp_mat[:, chk * cen_step:(chk + 1) * cen_step].toarray(),
                                 imp_stats[:, chk * cen_step:(chk + 1) * cen_step])
        self._loc_return_imp(np.array(return_periods),
                             self.imp_mat[:, (chk + 1) * cen_step:].toarray(),
                             imp_stats[:, (chk + 1) * cen_step:])

        return imp_stats

    def plot_rp_imp(self, return_periods=(25, 50, 100, 250),
                    log10_scale=True, smooth=True, axis=None, **kwargs):
        """Compute and plot exceedance impact maps for different return periods.
        Calls local_exceedance_imp.

        Parameters:
            return_periods (tuple(int), optional): return periods to consider
            log10_scale (boolean, optional): plot impact as log10(impact)
            smooth (bool, optional): smooth plot to plot.RESOLUTIONxplot.RESOLUTION
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots

        Returns:
            matplotlib.axes._subplots.AxesSubplot,
            np.ndarray (return_periods.size x num_centroids)
        """
        imp_stats = self.local_exceedance_imp(np.array(return_periods))
        if imp_stats == []:
            raise ValueError('Error: Attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True')
        if log10_scale:
            if np.min(imp_stats) < 0:
                imp_stats_log = np.log10(abs(imp_stats) + 1)
                colbar_name = 'Log10(abs(Impact)+1) (' + self.unit + ')'
            elif np.min(imp_stats) < 1:
                imp_stats_log = np.log10(imp_stats + 1)
                colbar_name = 'Log10(Impact+1) (' + self.unit + ')'
            else:
                imp_stats_log = np.log10(imp_stats)
                colbar_name = 'Log10(Impact) (' + self.unit + ')'
        else:
            imp_stats_log = imp_stats
            colbar_name = 'Impact (' + self.unit + ')'
        title = list()
        for ret in return_periods:
            title.append('Return period: ' + str(ret) + ' years')
        axis = u_plot.geo_im_from_array(imp_stats_log, self.coord_exp,
                                        colbar_name, title, smooth=smooth, axes=axis, **kwargs)

        return axis, imp_stats

    @staticmethod
    def read_sparse_csr(file_name):
        """Read imp_mat matrix from numpy's npz format.

        Parameters:
            file_name (str): file name

        Returns:
            sparse.csr_matrix
        """
        LOGGER.info('Reading %s', file_name)
        loader = np.load(file_name)
        return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                 shape=loader['shape'])

    def read_csv(self, file_name):
        """Read csv file containing impact data generated by write_csv.

        Parameters:
            file_name (str): absolute path of the file
        """
        LOGGER.info('Reading %s', file_name)
        imp_df = pd.read_csv(file_name)
        self.__init__()
        self.unit = imp_df.unit[0]
        self.tot_value = imp_df.tot_value[0]
        self.aai_agg = imp_df.aai_agg[0]
        self.event_id = imp_df.event_id[~np.isnan(imp_df.event_id)].values
        num_ev = self.event_id.size
        self.event_name = imp_df.event_name[:num_ev].values.tolist()
        self.date = imp_df.event_date[:num_ev].values
        self.at_event = imp_df.at_event[:num_ev].values
        self.frequency = imp_df.event_frequency[:num_ev].values
        self.eai_exp = imp_df.eai_exp[~np.isnan(imp_df.eai_exp)].values
        num_exp = self.eai_exp.size
        self.coord_exp = np.zeros((num_exp, 2))
        self.coord_exp[:, 0] = imp_df.exp_lat[:num_exp]
        self.coord_exp[:, 1] = imp_df.exp_lon[:num_exp]
        try:
            self.crs = u_coord.to_crs_user_input(imp_df.exp_crs.values[0])
        except AttributeError:
            self.crs = DEF_CRS
        self.tag['haz'] = TagHaz(str(imp_df.tag_hazard[0]),
                                 str(imp_df.tag_hazard[1]),
                                 str(imp_df.tag_hazard[2]))
        self.tag['exp'] = Tag(str(imp_df.tag_exposure[0]),
                              str(imp_df.tag_exposure[1]))
        self.tag['impf_set'] = Tag(str(imp_df.tag_impact_func[0]),
                                 str(imp_df.tag_impact_func[1]))

    def read_excel(self, file_name):
        """Read excel file containing impact data generated by write_excel.

        Parameters:
            file_name (str): absolute path of the file
        """
        LOGGER.info('Reading %s', file_name)
        dfr = pd.read_excel(file_name)
        self.__init__()
        self.tag['haz'] = TagHaz()
        self.tag['haz'].haz_type = dfr['tag_hazard'][0]
        self.tag['haz'].file_name = dfr['tag_hazard'][1]
        self.tag['haz'].description = dfr['tag_hazard'][2]
        self.tag['exp'] = Tag()
        self.tag['exp'].file_name = dfr['tag_exposure'][0]
        self.tag['exp'].description = dfr['tag_exposure'][1]
        self.tag['impf_set'] = Tag()
        self.tag['impf_set'].file_name = dfr['tag_impact_func'][0]
        self.tag['impf_set'].description = dfr['tag_impact_func'][1]

        self.unit = dfr.unit[0]
        self.tot_value = dfr.tot_value[0]
        self.aai_agg = dfr.aai_agg[0]

        self.event_id = dfr.event_id[~np.isnan(dfr.event_id.values)].values
        self.event_name = dfr.event_name[:self.event_id.size].values
        self.date = dfr.event_date[:self.event_id.size].values
        self.frequency = dfr.event_frequency[:self.event_id.size].values
        self.at_event = dfr.at_event[:self.event_id.size].values

        self.eai_exp = dfr.eai_exp[~np.isnan(dfr.eai_exp.values)].values
        self.coord_exp = np.zeros((self.eai_exp.size, 2))
        self.coord_exp[:, 0] = dfr.exp_lat.values[:self.eai_exp.size]
        self.coord_exp[:, 1] = dfr.exp_lon.values[:self.eai_exp.size]
        try:
            self.crs = u_coord.to_csr_user_input(dfr.exp_crs.values[0])
        except AttributeError:
            self.crs = DEF_CRS

    @staticmethod
    def video_direct_impact(exp, impf_set, haz_list, file_name='',
                            writer=animation.PillowWriter(bitrate=500),
                            imp_thresh=0, args_exp=None, args_imp=None):
        """
        Computes and generates video of accumulated impact per input events
        over exposure.

        Parameters:
            exp (Exposures): exposures instance, constant during all video
            impf_set (ImpactFuncSet): impact functions
            haz_list (list(Hazard)): every Hazard contains an event; all hazards
                use the same centroids
            file_name (str, optional): file name to save video, if provided
            writer = (matplotlib.animation.*, optional): video writer. Default:
                pillow with bitrate=500
            imp_thresh (float): represent damages greater than threshold
            args_exp (optional): arguments for scatter (points) or hexbin (raster)
                matplotlib function used in exposures
            args_imp (optional): arguments for scatter (points) or hexbin (raster)
                matplotlib function used in impact

        Returns:
            list(Impact)
        """
        if args_exp is None:
            args_exp = dict()
        if args_imp is None:
            args_imp = dict()
        imp_list = []
        exp_list = []
        imp_arr = np.zeros(len(exp.gdf))
        for i_time, _ in enumerate(haz_list):
            imp_tmp = Impact()
            imp_tmp.calc(exp, impf_set, haz_list[i_time])
            imp_arr = np.maximum(imp_arr, imp_tmp.eai_exp)
            # remove not impacted exposures
            save_exp = imp_arr > imp_thresh
            imp_tmp.coord_exp = imp_tmp.coord_exp[save_exp, :]
            imp_tmp.eai_exp = imp_arr[save_exp]
            imp_list.append(imp_tmp)
            exp_list.append(~save_exp)

        v_lim = [np.array([haz.intensity.min() for haz in haz_list]).min(),
                 np.array([haz.intensity.max() for haz in haz_list]).max()]

        if 'vmin' not in args_exp:
            args_exp['vmin'] = exp.gdf.value.values.min()

        if 'vmin' not in args_imp:
            args_imp['vmin'] = np.array([imp.eai_exp.min() for imp in imp_list
                                         if imp.eai_exp.size]).min()

        if 'vmax' not in args_exp:
            args_exp['vmax'] = exp.gdf.value.values.max()

        if 'vmax' not in args_imp:
            args_imp['vmax'] = np.array([imp.eai_exp.max() for imp in imp_list
                                         if imp.eai_exp.size]).max()

        if 'cmap' not in args_exp:
            args_exp['cmap'] = 'winter_r'

        if 'cmap' not in args_imp:
            args_imp['cmap'] = 'autumn_r'


        plot_raster = False
        if exp.meta:
            plot_raster = True

        def run(i_time):
            haz_list[i_time].plot_intensity(1, axis=axis, cmap='Greys', vmin=v_lim[0],
                                            vmax=v_lim[1], alpha=0.8)
            if plot_raster:
                exp.plot_hexbin(axis=axis, mask=exp_list[i_time], ignore_zero=True,
                                pop_name=False, **args_exp)
                if imp_list[i_time].coord_exp.size:
                    imp_list[i_time].plot_hexbin_eai_exposure(axis=axis, pop_name=False,
                                                              **args_imp)
                    fig.delaxes(fig.axes[1])
            else:
                exp.plot_scatter(axis=axis, mask=exp_list[i_time], ignore_zero=True,
                                 pop_name=False, **args_exp)
                if imp_list[i_time].coord_exp.size:
                    imp_list[i_time].plot_scatter_eai_exposure(axis=axis, pop_name=False,
                                                               **args_imp)
                    fig.delaxes(fig.axes[1])
            fig.delaxes(fig.axes[1])
            fig.delaxes(fig.axes[1])
            axis.set_xlim(haz_list[-1].centroids.lon.min(), haz_list[-1].centroids.lon.max())
            axis.set_ylim(haz_list[-1].centroids.lat.min(), haz_list[-1].centroids.lat.max())
            axis.set_title(haz_list[i_time].event_name[0])
            pbar.update()

        if file_name:
            LOGGER.info('Generating video %s', file_name)
            fig, axis = u_plot.make_map()
            ani = animation.FuncAnimation(fig, run, frames=len(haz_list),
                                          interval=500, blit=False)
            pbar = tqdm(total=len(haz_list))
            ani.save(file_name, writer=writer)
            pbar.close()

        return imp_list

    def _loc_return_imp(self, return_periods, imp, exc_imp):
        """Compute local exceedence impact for given return period.

        Parameters:
            return_periods (np.array): return periods to consider
            cen_pos (int): centroid position

        Returns:
            np.array
        """
        # sorted impacts
        sort_pos = np.argsort(imp, axis=0)[::-1, :]
        columns = np.ones(imp.shape, int)
        # pylint: disable=unsubscriptable-object  # pylint/issues/3139
        columns *= np.arange(columns.shape[1])
        imp_sort = imp[sort_pos, columns]
        # cummulative frequency at sorted intensity
        freq_sort = self.frequency[sort_pos]
        np.cumsum(freq_sort, axis=0, out=freq_sort)

        for cen_idx in range(imp.shape[1]):
            exc_imp[:, cen_idx] = self._cen_return_imp(
                imp_sort[:, cen_idx], freq_sort[:, cen_idx],
                0, return_periods)

    def _exp_impact(self, exp_iimp, exposures, hazard, imp_fun, insure_flag):
        """Compute impact for inpute exposure indexes and impact function.

        Parameters:
            exp_iimp (np.array): exposures indexes
            exposures (Exposures): exposures instance
            hazard (Hazard): hazard instance
            imp_fun (ImpactFunc): impact function instance
            insure_flag (bool): consider deductible and cover of exposures
        """
        if not exp_iimp.size:
            return

        # get assigned centroids
        icens = exposures.gdf[INDICATOR_CENTR + hazard.tag.haz_type].values[exp_iimp]

        # get affected intensities
        inten_val = hazard.intensity[:, icens]
        # get affected fractions
        fract = hazard.fraction[:, icens]
        # impact = fraction * mdr * value
        inten_val.data = imp_fun.calc_mdr(inten_val.data)
        impact = fract.multiply(inten_val).multiply(exposures.gdf.value.values[exp_iimp])

        if insure_flag and impact.nonzero()[0].size:
            inten_val = hazard.intensity[:, icens].toarray()
            paa = np.interp(inten_val, imp_fun.intensity, imp_fun.paa)
            impact = impact.toarray()
            impact -= exposures.gdf.deductible.values[exp_iimp] * paa
            impact = np.clip(impact, 0, exposures.gdf.cover.values[exp_iimp])
            self.eai_exp[exp_iimp] += np.einsum('ji,j->i', impact, hazard.frequency)
            impact = sparse.coo_matrix(impact)
        else:
            self.eai_exp[exp_iimp] += np.squeeze(np.asarray(np.sum(
                impact.multiply(hazard.frequency.reshape(-1, 1)), axis=0)))

        self.at_event += np.squeeze(np.asarray(np.sum(impact, axis=1)))
        self.tot_value += np.sum(exposures.gdf.value.values[exp_iimp])
        if isinstance(self.imp_mat, tuple):
            row_ind, col_ind = impact.nonzero()
            self.imp_mat[0].extend(list(impact.data))
            self.imp_mat[1][0].extend(list(row_ind))
            self.imp_mat[1][1].extend(list(exp_iimp[col_ind]))

    def _build_exp(self):
        return Exposures(
            data={
                'value': self.eai_exp,
                'latitude': self.coord_exp[:, 0],
                'longitude': self.coord_exp[:, 1],
            },
            crs=self.crs,
            value_unit=self.unit,
            ref_year=0,
            tag=Tag(),
            meta=None
        )

    def _build_exp_event(self, event_id):
        """Write impact of an event as Exposures

        Parameters:
            event_id(int): id of the event
        """
        [[ix]] = (self.event_id == event_id).nonzero()
        return Exposures(
            data={
                'value': self.imp_mat[ix].toarray().ravel(),
                'latitude': self.coord_exp[:, 0],
                'longitude': self.coord_exp[:, 1],
            },
            crs=self.crs,
            value_unit=self.unit,
            ref_year=0,
            tag=Tag(),
            meta=None
        )

    @staticmethod
    def _cen_return_imp(imp, freq, imp_th, return_periods):
        """From ordered impact and cummulative frequency at centroid, get
        exceedance impact at input return periods.

        Parameters:
            imp (np.array): sorted impact at centroid
            freq (np.array): cummulative frequency at centroid
            imp_th (float): impact threshold
            return_periods (np.array): return periods

        Returns:
            np.array
        """
        imp_th = np.asarray(imp > imp_th).squeeze()
        imp_cen = imp[imp_th]
        freq_cen = freq[imp_th]
        if not imp_cen.size:
            return np.zeros((return_periods.size,))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pol_coef = np.polyfit(np.log(freq_cen), imp_cen, deg=1)
        except ValueError:
            pol_coef = np.polyfit(np.log(freq_cen), imp_cen, deg=0)
        imp_fit = np.polyval(pol_coef, np.log(1 / return_periods))
        wrong_inten = (return_periods > np.max(1 / freq_cen)) & np.isnan(imp_fit)
        imp_fit[wrong_inten] = 0.

        return imp_fit


    def select(self,
               event_ids=None, event_names=None, dates=None,
               coord_exp=None):
        """
        Select a subset of events and/or exposure points from the impact.
        If multiple input variables are not None, it returns all the impacts
        matching at least one of the conditions.

        Note:
            the frequencies are NOT adjusted. Method to adjust frequencies
        and obtain correct eai_exp:
            1- Select subset of impact according to your choice
            imp = impact.select(...)
            2- Adjust manually the frequency of the subset of impact
            imp.frequency = [...]
            3- Use select without arguments to select all events and recompute
            the eai_exp with the updated frequencies.
            imp = imp.select()

        Parameters
        ----------
        event_ids : list[int], optional
            Selection of events by their id. The default is None.
        event_names : list[str], optional
            Selection of events by their name. The default is None.
        dates : tuple(), optional
            (start-date, end-date), events are selected if they are >=
            than start-date and <= than end-date. Dates in same format
            as impact.date (ordinal format of datetime library)
            The default is None.
        coord_exp : (np.ndarray), optional]
            Selection of exposures coordinates [lat, lon] (in degrees)
            The default is None.

        Raises
        ------
        ValueError
            If the impact matrix is missing, the eai_exp and aai_agg cannot
            be updated for a selection of events and/or exposures.

        Returns
        -------
        imp : climada.engine.Impact
            A new impact object with a selection of events and/or exposures

        """

        nb_events = self.event_id.size
        nb_exp = len(self.coord_exp)

        if self.imp_mat.shape != (nb_events, nb_exp):
            raise ValueError("The impact matrix is missing or incomplete. "
                             "The eai_exp and aai_agg cannot be computed. "
                             "Please recompute impact.calc() with save_mat=True "
                             "before using impact.select()")

        if nb_events == nb_exp:
            LOGGER.warning("The number of events is equal to the number of "
                           "exposure points. It is not possible to "
                           "differentiate events and exposures attributes. "
                           "Please add/remove one event/exposure point. "
                           "This is a purely technical limitation of this "
                           "method.")
            return None

        imp = copy.deepcopy(self)

        # apply event selection to impact attributes
        sel_ev = self._selected_events_idx(event_ids, event_names, dates, nb_events)
        if sel_ev is not None:
            # set all attributes that are 'per event', i.e. have a dimension
            # of length equal to the number of events (=nb_events)
            for attr in get_attributes_with_matching_dimension(imp, [nb_events]):
                value = imp.__getattribute__(attr)
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        setattr(imp, attr, value[sel_ev])
                    else:
                        LOGGER.warning("Found a multidimensional numpy array "
                                       "with one dimension matching the number of events. "
                                       "But multidimensional numpy arrays are not handled "
                                       "in impact.select")
                elif isinstance(value, sparse.csr_matrix):
                    setattr(imp, attr, value[sel_ev, :])
                elif isinstance(value, list) and value:
                    setattr(imp, attr, [value[idx] for idx in sel_ev])
                else:
                    pass

            LOGGER.info("The eai_exp and aai_agg are computed for the "
                        "selected subset of events WITHOUT modification of "
                        "the frequencies.")

        # apply exposure selection to impact attributes
        if coord_exp is not None:
            sel_exp = self._selected_exposures_idx(coord_exp)
            imp.coord_exp = imp.coord_exp[sel_exp]
            imp.imp_mat = imp.imp_mat[:, sel_exp]

            # .A1 reduce 1d matrix to 1d array
            imp.at_event = imp.imp_mat.sum(axis=1).A1
            imp.tot_value = None
            LOGGER.info("The total value cannot be re-computed for a "
                        "subset of exposures and is set to None.")

        # cast frequency vector into 2d array for sparse matrix multiplication
        freq_mat = imp.frequency.reshape(len(imp.frequency), 1)
        # .A1 reduce 1d matrix to 1d array
        imp.eai_exp = imp.imp_mat.multiply(freq_mat).sum(axis=0).A1
        imp.aai_agg = imp.eai_exp.sum()

        return imp

    def _selected_exposures_idx(self, coord_exp):
        assigned_idx = u_coord.assign_coordinates(self.coord_exp, coord_exp, threshold=0)
        sel_exp = (assigned_idx >= 0).nonzero()[0]
        if sel_exp.size == 0:
            LOGGER.warning("No exposure coordinates match the selection.")
        return sel_exp

    def _selected_events_idx(self, event_ids, event_names, dates, nb_events):
        if all(var is None for var in [dates, event_ids, event_names]):
            return None

        # filter events by date
        if dates is None:
            mask_dt = np.zeros(nb_events, dtype=bool)
        else:
            mask_dt = np.ones(nb_events, dtype=bool)
            date_ini, date_end = dates
            if isinstance(date_ini, str):
                date_ini = u_dt.str_to_date(date_ini)
                date_end = u_dt.str_to_date(date_end)
            mask_dt &= (date_ini <= self.date)
            mask_dt &= (self.date <= date_end)
            if not np.any(mask_dt):
                LOGGER.info('No impact event in given date range %s.', dates)

        sel_dt = mask_dt.nonzero()[0]  # Convert bool to indices

        # filter events by id
        if event_ids is None:
            sel_id = np.array([], dtype=int)
        else:
            sel_id = np.isin(self.event_id, event_ids).nonzero()[0]
            if sel_id.size == 0:
                LOGGER.info('No impact event with given ids %s found.', event_ids)

        # filter events by name
        if event_names is None:
            sel_na = np.array([], dtype=int)
        else:
            sel_na = np.isin(self.event_name, event_names).nonzero()[0]
            if sel_na.size == 0:
                LOGGER.info('No impact event with given names %s found.', event_names)

        # select events with machting id, name or date field.
        sel_ev = np.unique(np.concatenate([sel_dt, sel_id, sel_na]))

        # if no event found matching ids, names or dates, warn the user
        if sel_ev.size == 0:
            LOGGER.warning("No event matches the selection. ")

        return sel_ev

class ImpactFreqCurve():
    """Impact exceedence frequency curve.

    Attributes:
        tag (dict): dictionary of tags of exposures, impact functions set and
            hazard: {'exp': Tag(), 'impf_set': Tag(), 'haz': TagHazard()}
        return_per (np.array): return period
        impact (np.array): impact exceeding frequency
        unit (str): value unit used (given by exposures unit)
        label (str): string describing source data
    """
    def __init__(self):
        self.tag = dict()
        self.return_per = np.array([])
        self.impact = np.array([])
        self.unit = ''
        self.label = ''

    def plot(self, axis=None, log_frequency=False, **kwargs):
        """Plot impact frequency curve.

        Parameters:
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            log_frequency (boolean): plot logarithmioc exceedance frequency on x-axis
            kwargs (optional): arguments for plot matplotlib function, e.g. color='b'

        Returns:
            matplotlib.axes._subplots.AxesSubplot
        """
        if not axis:
            _, axis = plt.subplots(1, 1)
        axis.set_title(self.label)
        axis.set_ylabel('Impact (' + self.unit + ')')
        if log_frequency:
            axis.set_xlabel('Exceedance frequency (1/year)')
            axis.set_xscale('log')
            axis.plot(self.return_per**-1, self.impact, **kwargs)
        else:
            axis.set_xlabel('Return period (year)')
            axis.plot(self.return_per, self.impact, **kwargs)
        return axis
