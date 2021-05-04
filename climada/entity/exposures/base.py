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

Define Exposures class.
"""

__all__ = ['Exposures', 'add_sea', 'INDICATOR_IMPF', 'INDICATOR_CENTR']

import logging
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from geopandas import GeoDataFrame
import rasterio
from rasterio.warp import Resampling
import contextily as ctx
import cartopy.crs as ccrs

from climada.entity.tag import Tag
import climada.util.hdf5_handler as u_hdf5
from climada.util.constants import ONE_LAT_KM, DEF_CRS, CMAP_DIVERGING
import climada.util.coordinates as u_coord
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

INDICATOR_IMPF_OLD = 'if_'
"""Previously used name of the column containing the impact functions id of specified hazard"""

INDICATOR_IMPF = 'impf_'
"""Name of the column containing the impact functions id of specified hazard"""

INDICATOR_CENTR = 'centr_'
"""Name of the column containing the centroids id of specified hazard"""

DEF_REF_YEAR = 2018
"""Default reference year"""

DEF_VALUE_UNIT = 'USD'
"""Default value unit"""

DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'assets',
               'var_name': {'lat': 'lat',
                            'lon': 'lon',
                            'val': 'Value',
                            'ded': 'Deductible',
                            'cov': 'Cover',
                            'impf': 'DamageFunID',
                            'cat': 'Category_ID',
                            'reg': 'Region_ID',
                            'uni': 'Value_unit',
                            'ass': 'centroid_index',
                            'ref': 'reference_year'
                            }
               }
"""MATLAB variable names"""

class Exposures():
    """geopandas GeoDataFrame with metada and columns (pd.Series) defined in
    Attributes.

    Attributes:
        tag (Tag): metada - information about the source data
        ref_year (int): metada - reference year
        value_unit (str): metada - unit of the exposures values
        latitude (pd.Series): latitude
        longitude (pd.Series): longitude
        crs (dict or crs): CRS information inherent to GeoDataFrame.
        value (pd.Series): a value for each exposure
        impf_ (pd.Series, optional): e.g. impf_TC. impact functions id for hazard TC.
            There might be different hazards defined: impf_TC, impf_FL, ...
            If not provided, set to default 'impf_' with ids 1 in check().
        geometry (pd.Series, optional): geometry of type Point of each instance.
            Computed in method set_geometry_points().
        meta (dict): dictionary containing corresponding raster properties (if any):
            width, height, crs and transform must be present at least (transform needs
            to contain upper left corner!). Exposures might not contain all the points
            of the corresponding raster. Not used in internal computations.
        deductible (pd.Series, optional): deductible value for each exposure
        cover (pd.Series, optional): cover value for each exposure
        category_id (pd.Series, optional): category id for each exposure
        region_id (pd.Series, optional): region id for each exposure
        centr_ (pd.Series, optional): e.g. centr_TC. centroids index for hazard
            TC. There might be different hazards defined: centr_TC, centr_FL, ...
            Computed in method assign_centroids().
    """
    _metadata = ['tag', 'ref_year', 'value_unit', 'meta']

    vars_oblig = ['value', 'latitude', 'longitude']
    """Name of the variables needed to compute the impact."""

    vars_def = [INDICATOR_IMPF, INDICATOR_IMPF_OLD]
    """Name of variables that can be computed."""

    vars_opt = [INDICATOR_CENTR, 'deductible', 'cover', 'category_id',
                'region_id', 'geometry']
    """Name of the variables that aren't need to compute the impact."""

    @property
    def crs(self):
        """Coordinate Reference System, refers to the crs attribute of the inherent GeoDataFrame"""
        try:
            return self.gdf.crs
        except AttributeError:
            return self.meta.get('crs')

    def __init__(self, *args, **kwargs):
        """Creates an Exposures object from a GeoDataFrame

        Parameters
        ----------
        *args :
            Arguments of the GeoDataFrame constructor
        **kwargs :
            Named arguments of the GeoDataFrame constructor, additionally
        tag : climada.entity.exposures.tag.Tag
            Exopusres tag
        ref_year : int
            Reference Year
        value_unit : str
            Unit of the exposed value
        meta : dict
            Metadata dictionary
        """
        # meta data
        try:
            self.meta = kwargs.pop('meta')
            if self.meta is None:
                self.meta = {}
            if not isinstance(self.meta, dict):
                raise ValueError("meta must be a dictionary")
        except KeyError:
            self.meta = {}
            LOGGER.info('meta set to default value %s', self.meta)

        # tag
        try:
            self.tag = kwargs.pop('tag')
        except KeyError:
            self.tag = self.meta.get('tag', Tag())
            if 'tag' not in self.meta:
                LOGGER.info('tag set to default value %s', self.tag)

        # reference year
        try:
            self.ref_year = kwargs.pop('ref_year')
        except KeyError:
            self.ref_year = self.meta.get('ref_year', DEF_REF_YEAR)
            if 'ref_year' not in self.meta:
                LOGGER.info('ref_year set to default value %s', self.ref_year)

        # value unit
        try:
            self.value_unit = kwargs.pop('value_unit')
        except KeyError:
            self.value_unit = self.meta.get('ref_year', DEF_VALUE_UNIT)
            if 'value_unit' not in self.meta:
                LOGGER.info('value_unit set to default value %s', self.value_unit)

        # remaining generic attributes
        for mda in type(self)._metadata:
            if mda not in Exposures._metadata:
                if mda in kwargs:
                    setattr(self, mda, kwargs.pop(mda))
                elif mda in self.meta:
                    setattr(self, mda, self.meta[mda])
                else:
                    setattr(self, mda, None)

        # make the data frame
        self.gdf = GeoDataFrame(*args, **kwargs)

        # align crs from gdf and meta data
        if self.gdf.crs:
            crs = self.gdf.crs
        # With geopandas 3.1, the crs attribute is not conserved by the constructor
        # without a geometry column. Therefore the conservation is done 'manually':
        elif len(args) > 0:
            try:
                crs = args[0].crs
            except AttributeError:
                crs = None
        elif 'data' in kwargs:
            try:
                crs = kwargs['data'].crs
            except AttributeError:
                crs = None
        else:
            crs = None
        # store the crs in the meta dictionary
        if crs:
            if self.meta.get('crs') and not u_coord.equal_crs(self.meta.get('crs'), crs):
                LOGGER.info('crs from `meta` argument ignored and overwritten by GeoDataFrame'
                            ' crs: %s', self.gdf.crs)
            self.meta['crs'] = crs
            if not self.gdf.crs:
                self.gdf.crs = crs
        else:
            if 'crs' not in self.meta:
                LOGGER.info('crs set to default value: %s', DEF_CRS)
                self.meta['crs'] = DEF_CRS
            self.gdf.crs = self.meta['crs']

    def __str__(self):
        return '\n'.join(
            [f"{md}: {self.__dict__[md]}" for md in type(self)._metadata] +
            [f"crs: {self.crs}", "data:", str(self.gdf)]
        )

    def _access_item(self, *args):
        raise TypeError("Since CLIMADA 2.0, Exposures objects are not subscriptable. Data "
                        "fields of Exposures objects are accessed using the `gdf` attribute. "
                        "For example, `expo['value']` is replaced by `expo.gdf['value']`.")
    __getitem__ = _access_item
    __setitem__ = _access_item
    __delitem__ = _access_item

    def check(self):
        """Check Exposures consistency.

        Reports missing columns in log messages.
        If no impf_* column is present in the dataframe, a default column 'impf_' is added with
        default impact function id 1.
        """
        # mandatory columns
        for var in self.vars_oblig:
            if var not in self.gdf.columns:
                raise ValueError(f"{var} missing in gdf")

        # computable columns except impf_*
        for var in sorted(set(self.vars_def).difference([INDICATOR_IMPF, INDICATOR_IMPF_OLD])):
            if not var in self.gdf.columns:
                LOGGER.info("%s not set.", var)

        # special treatment for impf_*
        default_impf_present = False
        for var in [INDICATOR_IMPF, INDICATOR_IMPF_OLD]:
            if var in self.gdf.columns:
                LOGGER.info("Hazard type not set in %s", var)
                default_impf_present = True

        if not default_impf_present and not [
                col for col in self.gdf.columns
                if col.startswith(INDICATOR_IMPF) or col.startswith(INDICATOR_IMPF_OLD)
            ]:
            LOGGER.info("Setting %s to default impact functions ids 1.", INDICATOR_IMPF)
            self.gdf[INDICATOR_IMPF] = 1

        # optional columns except centr_*
        for var in sorted(set(self.vars_opt).difference([INDICATOR_CENTR])):
            if not var in self.gdf.columns:
                LOGGER.info("%s not set.", var)

        # special treatment for centr_*
        if INDICATOR_CENTR in self.gdf.columns:
            LOGGER.info("Hazard type not set in %s", INDICATOR_CENTR)

        elif not any([col.startswith(INDICATOR_CENTR) for col in self.gdf.columns]):
            LOGGER.info("%s not set.", INDICATOR_CENTR)

        # check whether geometry corresponds to lat/lon
        try:
            if (self.gdf.geometry.values[0].x != self.gdf.longitude.values[0] or
                self.gdf.geometry.values[0].y != self.gdf.latitude.values[0]):
                raise ValueError("Geometry values do not correspond to latitude and" +
                                 " longitude. Use set_geometry_points() or set_lat_lon().")
        except AttributeError:  # no geometry column
            pass

    def get_impf_column(self, haz_type=''):
        """Find the best matching column name in the exposures dataframe for a given hazard type,

        Parameters
        ----------
        haz_type : str or None
            hazard type, as in the hazard's tag.haz_type
            which is the HAZ_TYPE constant of the hazard's module

        Returns
        -------
        str
            a column name, the first of the following that is present in the exposures' dataframe:
            - impf_[haz_type]
            - if_[haz_type]
            - impf_
            - if_

        Raises
        ------
        ValueError
            if none of the above is found in the dataframe.
        """
        if INDICATOR_IMPF + haz_type in self.gdf.columns:
            return INDICATOR_IMPF + haz_type
        if INDICATOR_IMPF_OLD + haz_type in self.gdf.columns:
            LOGGER.info("Impact function column name 'if_%s' is not according to current"
                        " naming conventions. It's suggested to use 'impf_%s' instead.",
                        haz_type, haz_type)
            return INDICATOR_IMPF_OLD + haz_type
        if INDICATOR_IMPF in self.gdf.columns:
            LOGGER.info("No specific impact function column found for hazard %s."
                        " Using the anonymous 'impf_' column.", haz_type)
            return INDICATOR_IMPF
        if INDICATOR_IMPF_OLD in self.gdf.columns:
            LOGGER.info("No specific impact function column found for hazard %s. Using the"
                        " anonymous 'if_' column, which is not according to current naming"
                        " conventions. It's suggested to use 'impf_' instead.", haz_type)
            return INDICATOR_IMPF_OLD
        raise ValueError(f"Missing exposures impact functions {INDICATOR_IMPF}.")

    def assign_centroids(self, hazard, method='NN', distance='haversine',
                         threshold=100):
        """Assign for each exposure coordinate closest hazard coordinate.
        -1 used for disatances > threshold in point distances. If raster hazard,
        -1 used for centroids outside raster.

        Parameters
        ----------
        hazard : Hazard
            Hazard to match (with raster or vector centroids).
        method : str, optional
            Interpolation method to use in case of vector centroids. Currently, "NN" (nearest
            neighbor) is the only supported value, see `climada.util.interpolation.interpol_index`.
        distance : str, optional
            Distance to use in case of vector centroids. Possible values are "haversine" and
            "approx", see `climada.util.interpolation.interpol_index`. Default: "haversine"
        threshold : float
            If the distance to the nearest neighbor exceeds `threshold`, the index `-1` is
            assigned. Set `threshold` to 0, to disable nearest neighbor matching. Default: 100 (km)
        """
        LOGGER.info('Matching %s exposures with %s centroids.',
                    str(self.gdf.shape[0]), str(hazard.centroids.size))
        if not u_coord.equal_crs(self.crs, hazard.centroids.crs):
            raise ValueError('Set hazard and exposure to same CRS first!')
        if hazard.centroids.meta:
            assigned = u_coord.assign_grid_points(
                self.gdf.longitude.values, self.gdf.latitude.values,
                hazard.centroids.meta['width'], hazard.centroids.meta['height'],
                hazard.centroids.meta['transform'])
        else:
            assigned = u_coord.assign_coordinates(
                np.stack([self.gdf.latitude.values, self.gdf.longitude.values], axis=1),
                hazard.centroids.coord, method=method, distance=distance, threshold=threshold)
        self.gdf[INDICATOR_CENTR + hazard.tag.haz_type] = assigned

    def set_geometry_points(self, scheduler=None):
        """Set geometry attribute of GeoDataFrame with Points from latitude and
        longitude attributes.

        Parameters:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
        """
        u_coord.set_df_geometry_points(self.gdf, scheduler)

    def set_lat_lon(self):
        """Set latitude and longitude attributes from geometry attribute."""
        LOGGER.info('Setting latitude and longitude attributes.')
        self.gdf['latitude'] = self.gdf.geometry[:].y
        self.gdf['longitude'] = self.gdf.geometry[:].x

    def set_from_raster(self, file_name, band=1, src_crs=None, window=False,
                        geometry=False, dst_crs=False, transform=None,
                        width=None, height=None, resampling=Resampling.nearest):
        """Read raster data and set latitude, longitude, value and meta

        Parameters:
            file_name (str): file name containing values
            band (int, optional): bands to read (starting at 1)
            src_crs (crs, optional): source CRS. Provide it if error without it.
            window (rasterio.windows.Windows, optional): window where data is
                extracted
            geometry (shapely.geometry, optional): consider pixels only in shape
            dst_crs (crs, optional): reproject to given crs
            transform (rasterio.Affine): affine transformation to apply
            wdith (float): number of lons for transform
            height (float): number of lats for transform
            resampling (rasterio.warp,.Resampling optional): resampling
                function used for reprojection to dst_crs
        """
        self.tag = Tag()
        self.tag.file_name = str(file_name)
        meta, value = u_coord.read_raster(file_name, [band], src_crs, window,
                                          geometry, dst_crs, transform, width,
                                          height, resampling)
        ulx, xres, _, uly, _, yres = meta['transform'].to_gdal()
        lrx = ulx + meta['width'] * xres
        lry = uly + meta['height'] * yres
        x_grid, y_grid = np.meshgrid(np.arange(ulx + xres / 2, lrx, xres),
                                     np.arange(uly + yres / 2, lry, yres))
        try:
            self.gdf.crs = meta['crs'].to_dict()
        except AttributeError:
            self.gdf.crs = meta['crs']
        self.gdf['longitude'] = x_grid.flatten()
        self.gdf['latitude'] = y_grid.flatten()
        self.gdf['value'] = value.reshape(-1)
        self.meta = meta

    def plot_scatter(self, mask=None, ignore_zero=False, pop_name=True,
                     buffer=0.0, extend='neither', axis=None, figsize=(9, 13),
                     **kwargs):
        """Plot exposures geometry's value sum scattered over Earth's map.
        The plot will we projected according to the current crs.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name : bool or str, optional
                add names of the populated places, by default True. If set to 'markers' only the points
                will be shown without a description.
            buffer (float, optional): border to add to coordinates. Default: 0.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            figsize (tuple, optional): figure size for plt.subplots
            kwargs (optional): arguments for scatter matplotlib function, e.g.
                cmap='Greys'. Default: 'Wistia'
         Returns:
            cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        crs_epsg, _ = u_plot.get_transformation(self.crs)
        title = self.tag.description
        cbar_label = 'Value (%s)' % self.value_unit
        if mask is None:
            mask = np.ones((self.gdf.shape[0],), dtype=bool)
        if ignore_zero:
            pos_vals = self.gdf.value[mask].values > 0
        else:
            pos_vals = np.ones((self.gdf.value[mask].values.size,), dtype=bool)
        value = self.gdf.value[mask][pos_vals].values
        coord = np.stack([self.gdf.latitude[mask][pos_vals].values,
                          self.gdf.longitude[mask][pos_vals].values], axis=1)
        return u_plot.geo_scatter_from_array(value, coord, cbar_label, title,
                                             pop_name, buffer, extend,
                                             proj=crs_epsg, axes=axis,
                                             figsize=figsize, **kwargs)

    def plot_hexbin(self, mask=None, ignore_zero=False, pop_name=True,
                    buffer=0.0, extend='neither', axis=None, figsize=(9, 13),
                    **kwargs):
        """Plot exposures geometry's value sum binned over Earth's map.
        An other function for the bins can be set through the key reduce_C_function.
        The plot will we projected according to the current crs.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name : bool or str, optional
                add names of the populated places, by default True. If set to 'markers' only the points
                will be shown without a description.
            buffer (float, optional): border to add to coordinates. Default: 0.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            figsize (tuple): figure size for plt.subplots
            kwargs (optional): arguments for hexbin matplotlib function, e.g.
                reduce_C_function=np.average. Default: reduce_C_function=np.sum
         Returns:
            cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        crs_epsg, _ = u_plot.get_transformation(self.crs)
        title = self.tag.description
        cbar_label = 'Value (%s)' % self.value_unit
        if 'reduce_C_function' not in kwargs:
            kwargs['reduce_C_function'] = np.sum
        if mask is None:
            mask = np.ones((self.gdf.shape[0],), dtype=bool)
        if ignore_zero:
            pos_vals = self.gdf.value[mask].values > 0
        else:
            pos_vals = np.ones((self.gdf.value[mask].values.size,), dtype=bool)
        value = self.gdf.value[mask][pos_vals].values
        coord = np.stack([self.gdf.latitude[mask][pos_vals].values,
                          self.gdf.longitude[mask][pos_vals].values], axis=1)
        return u_plot.geo_bin_from_array(value, coord, cbar_label, title,
                                         pop_name, buffer, extend, proj=crs_epsg,
                                         axes=axis, figsize=figsize, **kwargs)

    def plot_raster(self, res=None, raster_res=None, save_tiff=None,
                    raster_f=lambda x: np.log10((np.fmax(x + 1, 1))),
                    label='value (log10)', scheduler=None, axis=None,
                    figsize=(9, 13), fill=True, **kwargs):
        """Generate raster from points geometry and plot it using log10 scale:
        np.log10((np.fmax(raster+1, 1))).

        Parameters:
            res (float, optional): resolution of current data in units of latitude
                and longitude, approximated if not provided.
            raster_res (float, optional): desired resolution of the raster
            save_tiff (str, optional): file name to save the raster in tiff
                format, if provided
            raster_f (lambda function): transformation to use to data. Default:
                log10 adding 1.
            label (str): colorbar label
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            figsize (tuple, optional): figure size for plt.subplots
            fill(bool, optional): If false, the areas with no data will be plotted
                in white.
            kwargs (optional): arguments for imshow matplotlib function

        Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if self.meta and self.meta.get('height', 0) * self.meta.get('height', 0) == len(self.gdf):
            raster = self.gdf.value.values.reshape((self.meta['height'],
                                                    self.meta['width']))
            # check raster starts by upper left corner
            if self.gdf.latitude.values[0] < self.gdf.latitude.values[-1]:
                raster = np.flip(raster, axis=0)
            if self.gdf.longitude.values[0] > self.gdf.longitude.values[-1]:
                raise ValueError('Points are not ordered according to meta raster.')
        else:
            raster, meta = u_coord.points_to_raster(self.gdf, ['value'], res, raster_res, scheduler)
            raster = raster.reshape((meta['height'], meta['width']))
        # save tiff
        if save_tiff is not None:
            with rasterio.open(save_tiff, 'w', driver='GTiff',
                               height=meta['height'], width=meta['width'], count=1,
                               dtype=np.float32, crs=self.crs, transform=meta['transform']
                              ) as ras_tiff:
                ras_tiff.write(raster.astype(np.float32), 1)
        # make plot
        proj_data, _ = u_plot.get_transformation(self.crs)
        proj_plot = proj_data
        if isinstance(proj_data, ccrs.PlateCarree):
            # use different projections for plot and data to shift the central lon in the plot
            xmin, ymin, xmax, ymax = u_coord.latlon_bounds(
                self.gdf.latitude.values, self.gdf.longitude.values)
            proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (xmin + xmax))
        else:
            xmin, ymin, xmax, ymax = (self.gdf.longitude.min(), self.gdf.latitude.min(),
                                      self.gdf.longitude.max(), self.gdf.latitude.max())

        if not axis:
            _, axis = u_plot.make_map(proj=proj_plot, figsize=figsize)

        cbar_ax = make_axes_locatable(axis).append_axes('right', size="6.5%",
                                                        pad=0.1, axes_class=plt.Axes)
        axis.set_extent((xmin, xmax, ymin, ymax), crs=proj_data)
        u_plot.add_shapes(axis)
        if not fill:
            raster = np.where(raster == 0, np.nan, raster)
            raster_f = lambda x: np.log10((np.maximum(x + 1, 1)))
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_DIVERGING
        imag = axis.imshow(raster_f(raster), **kwargs, origin='upper',
                           extent=(xmin, xmax, ymin, ymax), transform=proj_data)
        plt.colorbar(imag, cax=cbar_ax, label=label)
        plt.draw()
        return axis

    def plot_basemap(self, mask=None, ignore_zero=False, pop_name=True,
                     buffer=0.0, extend='neither', zoom=10,
                     url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png',
                     axis=None, **kwargs):
        """Scatter points over satellite image using contextily

         Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted. Same
                size of the exposures, only the selected indexes will be plot.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name : bool or str, optional
                add names of the populated places, by default True. If set to 'markers' only the points
                will be shown without a description.
            buffer (float, optional): border to add to coordinates. Default: 0.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            zoom (int, optional): zoom coefficient used in the satellite image
            url (str, optional): image source, e.g. ctx.sources.OSM_C
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for scatter matplotlib function, e.g.
                cmap='Greys'. Default: 'Wistia'

         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if 'geometry' not in self.gdf.columns:
            self.set_geometry_points()
        crs_ori = self.crs
        self.to_crs(epsg=3857, inplace=True)
        axis = self.plot_scatter(mask, ignore_zero, pop_name, buffer,
                                 extend, shapes=False, axis=axis, **kwargs)
        ctx.add_basemap(axis, zoom, url, origin='upper')
        axis.set_axis_off()
        self.to_crs(crs_ori, inplace=True)
        return axis

    def write_hdf5(self, file_name):
        """Write data frame and metadata in hdf5 format

        Parameters:
            file_name (str): (path and) file name to write to.
        """
        LOGGER.info('Writting %s', file_name)
        store = pd.HDFStore(file_name)
        pandas_df = pd.DataFrame(self.gdf)
        for col in pandas_df.columns:
            if str(pandas_df[col].dtype) == "geometry":
                pandas_df[col] = np.asarray(self.gdf[col])
        store.put('exposures', pandas_df)
        var_meta = {}
        for var in type(self)._metadata:
            var_meta[var] = getattr(self, var)

        store.get_storer('exposures').attrs.metadata = var_meta
        store.close()

    def read_hdf5(self, file_name):
        """Read data frame and metadata in hdf5 format

        Parameters:
            file_name (str): (path and) file name to read from.

        Optional Parameters:
            additional_vars (list): list of additional variable names to read that
                are not in exposures.base._metadata
        """
        LOGGER.info('Reading %s', file_name)
        with pd.HDFStore(file_name) as store:
            self.__init__(store['exposures'])
            metadata = store.get_storer('exposures').attrs.metadata
            for key, val in metadata.items():
                if key in type(self)._metadata:
                    setattr(self, key, val)
                if key == 'crs':
                    self.gdf.crs = val

    def read_mat(self, file_name, var_names=None):
        """Read MATLAB file and store variables in exposures.

        Parameters:
            file_name (str): absolute path file
            var_names (dict, optional): dictionary containing the name of the
                MATLAB variables. Default: DEF_VAR_MAT.
        """
        LOGGER.info('Reading %s', file_name)
        if not var_names:
            var_names = DEF_VAR_MAT

        data = u_hdf5.read(file_name)
        try:
            data = data[var_names['sup_field_name']]
        except KeyError:
            pass

        try:
            data = data[var_names['field_name']]
            exposures = dict()

            _read_mat_obligatory(exposures, data, var_names)
            _read_mat_optional(exposures, data, var_names)
        except KeyError as var_err:
            raise KeyError(f"Variable not in MAT file: {var_names.get('field_name')}")\
                from var_err

        self.gdf = GeoDataFrame(data=exposures, crs=self.crs)
        _read_mat_metadata(self, data, file_name, var_names)

    #
    # Extends the according geopandas method
    #
    def to_crs(self, crs=None, epsg=None, inplace=False):
        """Wrapper of the GeoDataFrame.to_crs method.

        Transform geometries to a new coordinate reference system.
        Transform all geometries in a GeoSeries to a different coordinate reference system.
        The crs attribute on the current GeoSeries must be set. Either crs in string or dictionary
        form or an EPSG code may be specified for output.
        This method will transform all points in all objects. It has no notion or projecting entire
        geometries. All segments joining points are assumed to be lines in the current projection,
        not geodesics. Objects crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters:
            crs : dict or str
                Output projection parameters as string or in dictionary form.
            epsg : int
                EPSG code specifying output projection.
            inplace : bool, optional, default: False
                Whether to return a new GeoDataFrame or do the transformation in
                place.

        Returns:
            None if inplace is True
            else a transformed copy of the exposures object
        """
        if inplace:
            self.gdf.to_crs(crs, epsg, True)
            self.meta['crs'] = crs
            self.set_lat_lon()
            return None

        exp = self.copy()
        exp.to_crs(crs, epsg, True)
        return exp

    def plot(self, *args, **kwargs):
        """Wrapper of the GeoDataFram.plot method"""
        self.gdf.plot(*args, **kwargs)
    plot.__doc__ = GeoDataFrame.plot.__doc__

    def copy(self, deep=True):
        """Make a copy of this Exposures object.

        Parameters
        ----------
        deep (bool): Make a deep copy, i.e. also copy data. Default True.

        Returns
        -------
            Exposures
        """
        gdf = self.gdf.copy(deep=deep)
        metadata = dict([
            (md, copy.deepcopy(self.__dict__[md])) for md in type(self)._metadata
        ])
        metadata['crs'] = self.crs
        return type(self)(
            gdf,
            **metadata
        )

    def write_raster(self, file_name, value_name='value', scheduler=None):
        """Write value data into raster file with GeoTiff format

        Parameters:
            file_name (str): name output file in tif format
        """
        if self.meta and self.meta['height'] * self.meta['width'] == len(self.gdf):
            raster = self.gdf[value_name].values.reshape((self.meta['height'],
                                                          self.meta['width']))
            # check raster starts by upper left corner
            if self.gdf.latitude.values[0] < self.gdf.latitude.values[-1]:
                raster = np.flip(raster, axis=0)
            if self.gdf.longitude.values[0] > self.gdf.longitude.values[-1]:
                raise ValueError('Points are not ordered according to meta raster.')
            u_coord.write_raster(file_name, raster, self.meta)
        else:
            raster, meta = u_coord.points_to_raster(self, [value_name], scheduler=scheduler)
            u_coord.write_raster(file_name, raster, meta)

    @staticmethod
    def concat(exposures_list):
        """Concatenates Exposures or DataFrame objectss to one Exposures object.

        Parameters
        ----------
        exposures_list : list of Exposures or DataFrames
            The list must not be empty with the first item supposed to be an Exposures object.

        Returns
        -------
        Exposures
            with the metadata of the first item in the list and the dataframes concatenated.
        """
        exp = exposures_list[0].copy(deep=False)
        df_list = [
            ex.gdf if isinstance(ex, Exposures) else ex
            for ex in exposures_list
        ]
        exp.gdf = GeoDataFrame(
            pd.concat(df_list, ignore_index=True, sort=False),
            crs=exp.crs
        )
        return exp


def add_sea(exposures, sea_res):
    """Add sea to geometry's surroundings with given resolution. region_id
    set to -1 and other variables to 0.

    Parameters:
        sea_res (tuple): (sea_coast_km, sea_res_km), where first parameter
            is distance from coast to fill with water and second parameter
            is resolution between sea points

    Returns:
        Exposures
    """
    LOGGER.info("Adding sea at %s km resolution and %s km distance from coast.",
                str(sea_res[1]), str(sea_res[0]))

    sea_res = (sea_res[0] / ONE_LAT_KM, sea_res[1] / ONE_LAT_KM)

    min_lat = max(-90, float(exposures.gdf.latitude.min()) - sea_res[0])
    max_lat = min(90, float(exposures.gdf.latitude.max()) + sea_res[0])
    min_lon = max(-180, float(exposures.gdf.longitude.min()) - sea_res[0])
    max_lon = min(180, float(exposures.gdf.longitude.max()) + sea_res[0])

    lat_arr = np.arange(min_lat, max_lat + sea_res[1], sea_res[1])
    lon_arr = np.arange(min_lon, max_lon + sea_res[1], sea_res[1])

    lon_mgrid, lat_mgrid = np.meshgrid(lon_arr, lat_arr)
    lon_mgrid, lat_mgrid = lon_mgrid.ravel(), lat_mgrid.ravel()
    on_land = ~u_coord.coord_on_land(lat_mgrid, lon_mgrid)

    sea_exp_gdf = GeoDataFrame()
    sea_exp_gdf['latitude'] = lat_mgrid[on_land]
    sea_exp_gdf['longitude'] = lon_mgrid[on_land]
    sea_exp_gdf['region_id'] = np.zeros(sea_exp_gdf.latitude.size, int) - 1

    if 'geometry' in exposures.gdf.columns:
        u_coord.set_df_geometry_points(sea_exp_gdf)

    for var_name in exposures.gdf.columns:
        if var_name not in ('latitude', 'longitude', 'region_id', 'geometry'):
            sea_exp_gdf[var_name] = np.zeros(sea_exp_gdf.latitude.size,
                                             exposures.gdf[var_name].dtype)

    return Exposures(
        pd.concat([exposures.gdf, sea_exp_gdf], ignore_index=True, sort=False),
        crs=exposures.crs,
        ref_year=exposures.ref_year,
        value_unit=exposures.value_unit,
        meta=exposures.meta,
        tag=exposures.tag
    )


def _read_mat_obligatory(exposures, data, var_names):
    """Fill obligatory variables."""
    exposures['value'] = np.squeeze(data[var_names['var_name']['val']])

    exposures['latitude'] = data[var_names['var_name']['lat']].reshape(-1)
    exposures['longitude'] = data[var_names['var_name']['lon']].reshape(-1)

    exposures[INDICATOR_IMPF] = np.squeeze(
        data[var_names['var_name']['impf']]).astype(int, copy=False)


def _read_mat_optional(exposures, data, var_names):
    """Fill optional parameters."""
    try:
        exposures['deductible'] = np.squeeze(data[var_names['var_name']['ded']])
    except KeyError:
        pass

    try:
        exposures['cover'] = np.squeeze(data[var_names['var_name']['cov']])
    except KeyError:
        pass

    try:
        exposures['category_id'] = \
            np.squeeze(data[var_names['var_name']['cat']]).astype(int, copy=False)
    except KeyError:
        pass

    try:
        exposures['region_id'] = \
        np.squeeze(data[var_names['var_name']['reg']]).astype(int, copy=False)
    except KeyError:
        pass

    try:
        assigned = np.squeeze(data[var_names['var_name']['ass']]).astype(int, copy=False)
        if assigned.size > 0:
            exposures[INDICATOR_CENTR] = assigned
    except KeyError:
        pass


def _read_mat_metadata(exposures, data, file_name, var_names):
    """Fille metadata in DataFrame object"""
    try:
        exposures.ref_year = int(np.squeeze(data[var_names['var_name']['ref']]))
    except KeyError:
        exposures.ref_year = DEF_REF_YEAR

    try:
        exposures.value_unit = u_hdf5.get_str_from_ref(
            file_name, data[var_names['var_name']['uni']][0][0])
    except KeyError:
        exposures.value_unit = DEF_VALUE_UNIT

    exposures.tag = Tag(file_name)
