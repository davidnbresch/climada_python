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

Define Centroids class.
"""

import copy
import logging
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from pyproj.crs import CRS
import rasterio
from rasterio.warp import Resampling
from scipy import sparse
from shapely.geometry.point import Point

from climada.util.constants import (DEF_CRS,
                                    ONE_LAT_KM,
                                    NATEARTH_CENTROIDS)
import climada.util.coordinates as u_coord
import climada.util.hdf5_handler as u_hdf5
import climada.util.plot as u_plot

__all__ = ['Centroids']

PROJ_CEA = CRS.from_user_input({'proj': 'cea'})

DEF_VAR_MAT = {
    'field_names': ['centroids', 'hazard'],
    'var_name': {
        'lat': 'lat',
        'lon': 'lon',
        'dist_coast': 'distance2coast_km',
        'admin0_name': 'admin0_name',
        'admin0_iso3': 'admin0_ISO3',
        'comment': 'comment',
        'region_id': 'NatId'
    }
}
"""MATLAB variable names"""

DEF_VAR_EXCEL = {
    'sheet_name': 'centroids',
    'col_name': {
        'region_id': 'region_id',
        'lat': 'latitude',
        'lon': 'longitude',
    }
}
"""Excel variable names"""

LOGGER = logging.getLogger(__name__)


class Centroids():
    """Contains raster or vector centroids.

    Raster data can be set with set_raster_file() or set_meta().
    Vector data can be set with set_lat_lon() or set_vector_file().

    Attributes
    ----------
    meta : dict, optional
        rasterio meta dictionary containing raster properties: width, height, crs and transform
        must be present at least. The affine ransformation needs to be shearless (only stretching)
        and have positive x- and negative y-orientation.
    lat : np.array, optional
        latitude of size size
    lon : np.array, optional
        longitude of size size
    geometry : gpd.GeoSeries, optional
        contains lat and lon crs. Might contain geometry points for lat and lon
    area_pixel : np.array, optional
        area of size size
    dist_coast : np.array, optional
        distance to coast of size size
    on_land : np.array, optional
        on land (True) and on sea (False) of size size
    region_id : np.array, optional
        country region code of size size
    elevation : np.array, optional
        elevation of size size
    """

    vars_check = {'lat', 'lon', 'geometry', 'area_pixel', 'dist_coast',
                  'on_land', 'region_id', 'elevation'}
    """Variables whose size will be checked"""

    def __init__(self):
        """Initialize to None raster and vector"""
        self.meta = dict()
        self.geometry = gpd.GeoSeries()
        self.lat = np.array([])
        self.lon = np.array([])
        self.area_pixel = np.array([])
        self.dist_coast = np.array([])
        self.on_land = np.array([])
        self.region_id = np.array([])
        self.elevation = np.array([])

    def check(self):
        """Check integrity of stored information.

        Checks that either `meta` attribute is set, or `lat`, `lon` and `geometry.crs`.
        Checks sizes of (optional) data attributes."""
        n_centr = self.size
        for var_name, var_val in self.__dict__.items():
            if var_name in self.vars_check:
                if var_val.size > 0 and var_val.size != n_centr:
                    raise ValueError(f'Wrong {var_name} size: {n_centr} != {var_val.size}.')
        if self.meta:
            for name in ['width', 'height', 'crs', 'transform']:
                if name not in self.meta.keys():
                    raise ValueError('Missing meta information: %s' % name)
            xres, xshear, xoff, yshear, yres, yoff = self.meta['transform'][:6]
            if xshear != 0 or yshear != 0:
                raise ValueError('Affine transformations with shearing components are not supported.')
            if yres > 0 or xres < 0:
                raise ValueError('Affine transformations with positive y-orientation '
                                 'or negative x-orientation are not supported.')

    def equal(self, centr):
        """Return True if two centroids equal, False otherwise

        Parameters
        ----------
        centr : Centroids
            centroids to compare

        Returns
        -------
        eq : bool
        """
        if self.meta and centr.meta:
            return (u_coord.equal_crs(self.meta['crs'], centr.meta['crs'])
                    and self.meta['height'] == centr.meta['height']
                    and self.meta['width'] == centr.meta['width']
                    and self.meta['transform'] == centr.meta['transform'])
        return (u_coord.equal_crs(self.geometry.crs, centr.geometry.crs)
                and self.lat.shape == centr.lat.shape
                and self.lon.shape == centr.lon.shape
                and np.allclose(self.lat, centr.lat)
                and np.allclose(self.lon, centr.lon))

    @staticmethod
    def from_base_grid(land=False, res_as=360, base_file=None):
        """Initialize from base grid data provided with CLIMADA

        Parameters
        ----------
        land : bool, optional
            If True, restrict to grid points on land. Default: False.
        res_as : int, optional
            Base grid resolution in arc-seconds (one of 150, 360). Default: 360.
        base_file : str, optional
            If set, read this file instead of one provided with climada.
        """
        centroids = Centroids()

        if base_file is None:
            base_file = NATEARTH_CENTROIDS[res_as]

        centroids.read_hdf5(base_file)
        if centroids.meta:
            xres, xshear, xoff, yshear, yres, yoff = centroids.meta['transform'][:6]
            shape = (centroids.meta['height'], centroids.meta['width'])
            if yres > 0:
                # make sure y-orientation is negative
                centroids.meta['transform'] = rasterio.Affine(xres, xshear, xoff, yshear,
                                                              -yres, yoff + (shape[0] - 1) * yres)
                # flip y-axis in data arrays
                for name in ["region_id", "dist_coast"]:
                    if not hasattr(centroids, name):
                        continue
                    data = getattr(centroids, name)
                    if data.size == 0:
                        continue
                    setattr(centroids, name, np.flipud(data.reshape(shape)).reshape(-1))
        if land:
            land_reg_ids = list(range(1, 1000))
            land_reg_ids.remove(10)  # Antarctica
            centroids = centroids.select(reg_id=land_reg_ids)

        centroids.check()
        return centroids

    @staticmethod
    def from_geodataframe(gdf, geometry_alias='geom'):
        """Create Centroids instance from GeoDataFrame.

        The geometry, lat, and lon attributes are set from the GeoDataFrame.geometry attribute,
        while the columns are copied as attributes to the Centroids object in the form of
        numpy.ndarrays using pandas.Series.to_numpy. The Series dtype will thus be respected.

        Columns named lat or lon are ignored, as they would overwrite the coordinates extracted
        from the point features. If the geometry attribute bears an alias, it can be dropped by
        setting the geometry_alias parameter.

        If the GDF includes a region_id column, but no on_land column, then on_land=True is
        inferred for those centroids that have a set region_id.

        Example
        -------
        >>> gdf = geopandas.read_file('centroids.shp')
        >>> gdf.region_id = gdf.region_id.astype(int)  # type coercion
        >>> centroids = Centroids.from_geodataframe(gdf)

        Parameters
        ----------
        gdf : GeoDataFrame
            Where the geometry column needs to consist of point features. See above for details on
            processing.
        geometry_alias : str, opt
            Alternate name for the geometry column; dropped to avoid duplicate assignment.
        """
        centroids = Centroids()

        centroids.geometry = gdf.geometry
        centroids.lat = gdf.geometry.y.to_numpy(copy=True)
        centroids.lon = gdf.geometry.x.to_numpy(copy=True)

        for col in gdf.columns:
            if col in [geometry_alias, 'geometry', 'lat', 'lon']:
                continue  # skip these, because they're already set above
            val = gdf[col].to_numpy(copy=True)
            setattr(centroids, col, val)

        if centroids.on_land.size == 0:
            try:
                centroids.on_land = ~np.isnan(centroids.region_id)
            except KeyError:
                pass

        return centroids


    def set_raster_from_pix_bounds(self, xf_lat, xo_lon, d_lat, d_lon, n_lat,
                                   n_lon, crs=DEF_CRS):
        """Set raster metadata (meta attribute) from pixel border data

        Parameters
        ----------
        xf_lat : float
            upper latitude (top)
        xo_lon : float
            left longitude
        d_lat : float
            latitude step (negative)
        d_lon : float
            longitude step (positive)
        n_lat : int
            number of latitude points
        n_lon : int
            number of longitude points
        crs : dict() or rasterio.crs.CRS, optional
            CRS. Default: DEF_CRS
        """
        self.__init__()
        self.meta = {
            'dtype': 'float32',
            'width': n_lon,
            'height': n_lat,
            'crs': crs,
            'transform': rasterio.Affine(d_lon, 0.0, xo_lon, 0.0, d_lat, xf_lat),
        }

    def set_raster_from_pnt_bounds(self, points_bounds, res, crs=DEF_CRS):
        """Set raster metadata (meta attribute) from points border data.

        Raster border = point_border + res/2

        Parameters
        ----------
        points_bounds : tuple
            points' lon_min, lat_min, lon_max, lat_max
        res : float
            desired resolution in same units as points_bounds
        crs : dict() or rasterio.crs.CRS, optional
            CRS. Default: DEF_CRS
        """
        self.__init__()
        rows, cols, ras_trans = u_coord.pts_to_raster_meta(points_bounds, (res, -res))
        self.meta = {
            'width': cols,
            'height': rows,
            'crs': crs,
            'transform': ras_trans,
        }

    def set_lat_lon(self, lat, lon, crs=DEF_CRS):
        """Set Centroids points from given latitude, longitude and CRS.

        Parameters
        ----------
        lat : np.array
            latitude
        lon : np.array
            longitude
        crs : dict() or rasterio.crs.CRS, optional
            CRS. Default: DEF_CRS
        """
        self.__init__()
        self.lat, self.lon, self.geometry = np.asarray(lat), np.asarray(lon), gpd.GeoSeries(crs=crs)

    def set_raster_file(self, file_name, band=[1], src_crs=None, window=False,
                        geometry=False, dst_crs=False, transform=None, width=None,
                        height=None, resampling=Resampling.nearest):
        """Read raster of bands and set 0 values to the masked ones.

        Each band is an event. Select region using window or geometry. Reproject input by proving
        dst_crs and/or (transform, width, height).

        Parameters
        ----------
        file_pth : str
            path of the file
        band : int, optional
            band number to read. Default: 1
        src_crs : crs, optional
            source CRS. Provide it if error without it.
        window : rasterio.windows.Window, optional
            window to read
        geometry : shapely.geometry, optional
            consider pixels only in shape
        dst_crs : crs, optional
            reproject to given crs
        transform : rasterio.Affine
            affine transformation to apply
        wdith : float
            number of lons for transform
        height : float
            number of lats for transform
        resampling : rasterio.warp,.Resampling optional
            resampling function used for reprojection to dst_crs

        Raises
        ------
            ValueError

        Returns
        -------
        inten : scipy.sparse.csr_matrix
            Each row is an event.
        """
        if not self.meta:
            self.meta, inten = u_coord.read_raster(
                file_name, band, src_crs, window, geometry, dst_crs,
                transform, width, height, resampling)
            return sparse.csr_matrix(inten)

        tmp_meta, inten = u_coord.read_raster(
            file_name, band, src_crs, window, geometry, dst_crs,
            transform, width, height, resampling)
        if (tmp_meta['crs'] != self.meta['crs']
                or tmp_meta['transform'] != self.meta['transform']
                or tmp_meta['height'] != self.meta['height']
                or tmp_meta['width'] != self.meta['width']):
            raise ValueError('Raster data is inconsistent with contained raster.')
        return sparse.csr_matrix(inten)

    def set_vector_file(self, file_name, inten_name=['intensity'], dst_crs=None):
        """Read vector file format supported by fiona.

        Each intensity name is considered an event.

        Parameters
        ----------
        file_name : str
            vector file with format supported by fiona and 'geometry' field.
        inten_name : list(str)
            list of names of the columns of the intensity of each event.
        dst_crs : crs, optional
            reproject to given crs

        Returns
        -------
        inten : scipy.sparse.csr_matrix
            Sparse intensity array of shape (len(inten_name), len(geometry)).
        """
        if not self.geometry.crs:
            self.lat, self.lon, self.geometry, inten = u_coord.read_vector(
                file_name, inten_name, dst_crs)
            return sparse.csr_matrix(inten)
        tmp_lat, tmp_lon, tmp_geometry, inten = u_coord.read_vector(
            file_name, inten_name, dst_crs)
        if not (u_coord.equal_crs(tmp_geometry.crs, self.geometry.crs)
                and np.allclose(tmp_lat, self.lat)
                and np.allclose(tmp_lon, self.lon)):
            raise ValueError('Vector data inconsistent with contained vector.')
        return sparse.csr_matrix(inten)

    def read_mat(self, file_name, var_names=DEF_VAR_MAT):
        """Read centroids from CLIMADA's MATLAB version.

        Parameters
        ----------
        file_name : str
            absolute or relative file name
        var_names : dict, default
            name of the variables

        Raises
        ------
        KeyError
        """
        LOGGER.info('Reading %s', file_name)
        if var_names is None:
            var_names = DEF_VAR_MAT

        cent = u_hdf5.read(file_name)
        # Try open encapsulating variable FIELD_NAMES
        num_try = 0
        for field in var_names['field_names']:
            try:
                cent = cent[field]
                break
            except KeyError:
                num_try += 1
        if num_try == len(var_names['field_names']):
            LOGGER.warning("Variables are not under: %s.", var_names['field_names'])

        try:
            cen_lat = np.squeeze(cent[var_names['var_name']['lat']])
            cen_lon = np.squeeze(cent[var_names['var_name']['lon']])
            self.set_lat_lon(cen_lat, cen_lon)

            try:
                self.dist_coast = np.squeeze(cent[var_names['var_name']['dist_coast']])
            except KeyError:
                pass
            try:
                self.region_id = np.squeeze(cent[var_names['var_name']['region_id']])
            except KeyError:
                pass
        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err

    def read_excel(self, file_name, var_names=DEF_VAR_EXCEL):
        """Read centroids from excel file with column names in var_names.

        Parameters
        ----------
        file_name : str
            absolute or relative file name
        var_names : dict, default
            name of the variables

        Raises
        ------
        KeyError
        """
        LOGGER.info('Reading %s', file_name)
        if var_names is None:
            var_names = DEF_VAR_EXCEL

        try:
            dfr = pd.read_excel(file_name, var_names['sheet_name'])
            self.set_lat_lon(dfr[var_names['col_name']['lat']],
                             dfr[var_names['col_name']['lon']])
            try:
                self.region_id = dfr[var_names['col_name']['region_id']]
            except KeyError:
                pass

        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err

    def clear(self):
        """Clear vector and raster data."""
        self.__init__()

    def append(self, centr):
        """Append raster or points.

        Parameters
        ----------
        centr : Centroids
            If it's a raster, it needs to have the same `meta` attribute.
            If it's of non-raster form, it's geometry needs to have the same CRS.
        """
        if self.meta and centr.meta:
            LOGGER.debug('Appending raster')
            if centr.meta['crs'] != self.meta['crs']:
                raise ValueError('Different CRS not accepted.')
            if self.meta['transform'][0] != centr.meta['transform'][0] \
               or self.meta['transform'][4] != centr.meta['transform'][4]:
                raise ValueError('Different raster resolutions.')
            left = min(self.total_bounds[0], centr.total_bounds[0])
            bottom = min(self.total_bounds[1], centr.total_bounds[1])
            right = max(self.total_bounds[2], centr.total_bounds[2])
            top = max(self.total_bounds[3], centr.total_bounds[3])
            crs = self.meta['crs']
            width = (right - left) / self.meta['transform'][0]
            height = (bottom - top) / self.meta['transform'][4]
            self.meta = {
                'dtype': 'float32',
                'width': width,
                'height': height,
                'crs': crs,
                'transform': rasterio.Affine(self.meta['transform'][0], 0.0, left,
                                             0.0, self.meta['transform'][4], top),
            }
            self.lat, self.lon = np.array([]), np.array([])
        else:
            LOGGER.debug('Appending points')
            if not u_coord.equal_crs(centr.geometry.crs, self.geometry.crs):
                raise ValueError('Different CRS not accepted.')
            self.lat = np.append(self.lat, centr.lat)
            self.lon = np.append(self.lon, centr.lon)
            self.meta = dict()

        # append all 1-dim variables
        for (var_name, var_val), centr_val in zip(self.__dict__.items(),
                                                  centr.__dict__.values()):
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 and \
            var_name not in ('lat', 'lon'):
                setattr(self, var_name, np.append(var_val, centr_val).
                        astype(var_val.dtype, copy=False))

    def get_closest_point(self, x_lon, y_lat, scheduler=None):
        """Returns closest centroid and its index to a given point.

        Parameters
        ----------
        x_lon : float
            x coord (lon)
        y_lat : float
            y coord (lat)
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”

        Returns
        -------
        x_close : float
            x-coordinate (longitude) of closest centroid.
        y_close : float
            y-coordinate (latitude) of closest centroids.
        idx_close : int
            Index of centroid in internal ordering of centroids.
        """
        if self.meta:
            if not self.lat.size or not self.lon.size:
                self.set_meta_to_lat_lon()
            i_lat, i_lon = rasterio.transform.rowcol(self.meta['transform'], x_lon, y_lat)
            i_lat = np.clip(i_lat, 0, self.meta['height'] - 1)
            i_lon = np.clip(i_lon, 0, self.meta['width'] - 1)
            close_idx = int(i_lat * self.meta['width'] + i_lon)
        else:
            self.set_geometry_points(scheduler)
            close_idx = self.geometry.distance(Point(x_lon, y_lat)).values.argmin()
        return self.lon[close_idx], self.lat[close_idx], close_idx

    def set_region_id(self, scheduler=None):
        """Set region_id as country ISO numeric code attribute for every pixel or point.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """
        ne_geom = self._ne_crs_geom(scheduler)
        LOGGER.debug('Setting region_id %s points.', str(self.lat.size))
        self.region_id = u_coord.get_country_code(
            ne_geom.geometry[:].y.values, ne_geom.geometry[:].x.values)

    def set_area_pixel(self, min_resol=1.0e-8, scheduler=None):
        """Set `area_pixel` attribute for every pixel or point (area in m*m).

        Parameters
        ----------
        min_resol : float, optional
            if centroids are points, use this minimum resolution in lat and lon. Default: 1.0e-8
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """
        if self.meta:
            if hasattr(self.meta['crs'], 'linear_units') and \
            str.lower(self.meta['crs'].linear_units) in ['m', 'metre', 'meter']:
                self.area_pixel = np.zeros((self.meta['height'], self.meta['width']))
                self.area_pixel *= abs(self.meta['transform'].a) * abs(self.meta['transform'].e)
                return
            if abs(abs(self.meta['transform'].a) -
                   abs(self.meta['transform'].e)) > 1.0e-5:
                raise ValueError('Area can not be computed for not squared pixels.')
            res = self.meta['transform'].a
        else:
            res = u_coord.get_resolution(self.lat, self.lon, min_resol=min_resol)
            res = np.abs(res).min()
        self.set_geometry_points(scheduler)
        LOGGER.debug('Setting area_pixel %s points.', str(self.lat.size))
        xy_pixels = self.geometry.buffer(res / 2).envelope
        if PROJ_CEA == self.geometry.crs:
            self.area_pixel = xy_pixels.area.values
        else:
            self.area_pixel = xy_pixels.to_crs(crs={'proj': 'cea'}).area.values

    def set_area_approx(self, min_resol=1.0e-8):
        """Set `area_pixel` attribute for every pixel or point (approximate area in m*m).

        Values are differentiated per latitude. Faster than `set_area_pixel`.

        Parameters
        ----------
        min_resol : float, optional
            if centroids are points, use this minimum resolution in lat and lon. Default: 1.0e-8
        """
        if self.meta:
            if hasattr(self.meta['crs'], 'linear_units') and \
            str.lower(self.meta['crs'].linear_units) in ['m', 'metre', 'meter']:
                self.area_pixel = np.zeros((self.meta['height'], self.meta['width']))
                self.area_pixel *= abs(self.meta['transform'].a) * abs(self.meta['transform'].e)
                return
            res_lat, res_lon = self.meta['transform'].e, self.meta['transform'].a
            lat_unique = np.arange(self.meta['transform'].f + res_lat / 2,
                                   self.meta['transform'].f + self.meta['height'] * res_lat,
                                   res_lat)
            lon_unique_len = self.meta['width']
            res_lat = abs(res_lat)
        else:
            res_lat, res_lon = np.abs(
                u_coord.get_resolution(self.lat, self.lon, min_resol=min_resol))
            lat_unique = np.array(np.unique(self.lat))
            lon_unique_len = len(np.unique(self.lon))
            if PROJ_CEA == self.geometry.crs:
                self.area_pixel = np.repeat(res_lat * res_lon, lon_unique_len)
                return

        LOGGER.debug('Setting area_pixel approx %s points.', str(self.lat.size))
        res_lat = res_lat * ONE_LAT_KM * 1000
        res_lon = res_lon * ONE_LAT_KM * 1000 * np.cos(np.radians(lat_unique))
        area_approx = np.repeat(res_lat * res_lon, lon_unique_len)
        if area_approx.size == self.size:
            self.area_pixel = area_approx
        else:
            raise ValueError('Pixel area of points can not be computed.')

    def set_elevation(self, topo_path):
        """Set elevation attribute for every pixel or point in meters.

        Parameters
        ----------
        topo_path : str
            Path to a raster file containing gridded elevation data.
        """
        if not self.coord.size:
            self.set_meta_to_lat_lon()
        self.elevation = u_coord.read_raster_sample(topo_path, self.lat, self.lon)

    def set_dist_coast(self, signed=False, precomputed=False, scheduler=None):
        """Set dist_coast attribute for every pixel or point in meters.

        Parameters
        ----------
        signed : bool
            If True, use signed distances (positive off shore and negative on land). Default: False.
        precomputed : bool
            If True, use precomputed distances (from NASA). Default: False.
        scheduler : str
            Used for dask map_partitions. "threads", "synchronous" or "processes"
        """
        if precomputed:
            if not self.lat.size or not self.lon.size:
                self.set_meta_to_lat_lon()
            self.dist_coast = u_coord.dist_to_coast_nasa(
                self.lat, self.lon, highres=True, signed=signed)
        else:
            ne_geom = self._ne_crs_geom(scheduler)
            LOGGER.debug('Computing distance to coast for %s centroids.', str(self.lat.size))
            self.dist_coast = u_coord.dist_to_coast(ne_geom, signed=signed)

    def set_on_land(self, scheduler=None):
        """Set on_land attribute for every pixel or point.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """
        ne_geom = self._ne_crs_geom(scheduler)
        LOGGER.debug('Setting on_land %s points.', str(self.lat.size))
        self.on_land = u_coord.coord_on_land(
            ne_geom.geometry[:].y.values, ne_geom.geometry[:].x.values)

    def remove_duplicate_points(self, scheduler=None):
        """Return Centroids with removed duplicated points

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”

        Returns
        -------
        cen : Centroids
            Sub-selection of this object.
        """
        self.set_geometry_points(scheduler)
        geom_wkb = self.geometry.apply(lambda geom: geom.wkb)
        sel_cen = geom_wkb.drop_duplicates().index
        return self.select(sel_cen=sel_cen)

    def select(self, reg_id=None, extent=None, sel_cen=None):
        """Return Centroids with points in the given reg_id or within mask

        Parameters
        ----------
        reg_id : int
            region to filter according to region_id values
        extent : tuple
            Format (min_lon, max_lon, min_lat, max_lat) tuple.
            If min_lon > lon_max, the extend crosses the antimeridian and is
            [lon_max, 180] + [-180, lon_min]
            Borders are inclusive.
        sel_cen : np.array
            1-dim mask, overrides reg_id and extent

        Returns
        -------
        cen : Centroids
            Sub-selection of this object
        """

        if sel_cen is None:
            sel_cen = np.ones_like(self.region_id, dtype=bool)
            if reg_id:
                sel_cen &= np.isin(self.region_id, reg_id)
            if extent:
                lon_min, lon_max, lat_min, lat_max = extent
                lon_max += 360 if lon_min > lon_max else 0
                lon_normalized = u_coord.lon_normalize(
                    self.lon.copy(), center=0.5 * (lon_min + lon_max))
                sel_cen &= (
                  (lon_normalized >= lon_min) & (lon_normalized <= lon_max) &
                  (self.lat >= lat_min) & (self.lat <= lat_max)
                )


        if not self.lat.size or not self.lon.size:
            self.set_meta_to_lat_lon()

        centr = Centroids()
        centr.set_lat_lon(self.lat[sel_cen], self.lon[sel_cen], self.geometry.crs)
        if self.area_pixel.size:
            centr.area_pixel = self.area_pixel[sel_cen]
        if self.region_id.size:
            centr.region_id = self.region_id[sel_cen]
        if self.on_land.size:
            centr.on_land = self.on_land[sel_cen]
        if self.dist_coast.size:
            centr.dist_coast = self.dist_coast[sel_cen]
        return centr

    def set_lat_lon_to_meta(self, min_resol=1.0e-8):
        """Compute meta from lat and lon values.

        Parameters
        ----------
        min_resol : float, optional
            Minimum centroids resolution to use in the raster. Default: 1.0e-8.
        """
        res = u_coord.get_resolution(self.lon, self.lat, min_resol=min_resol)
        rows, cols, ras_trans = u_coord.pts_to_raster_meta(self.total_bounds, res)
        LOGGER.debug('Resolution points: %s', str(res))
        self.meta = {
            'width': cols,
            'height': rows,
            'crs': self.crs,
            'transform': ras_trans,
        }

    def set_meta_to_lat_lon(self):
        """Compute lat and lon of every pixel center from meta raster."""
        xgrid, ygrid = u_coord.raster_to_meshgrid(
            self.meta['transform'], self.meta['width'], self.meta['height'])
        self.lon = xgrid.flatten()
        self.lat = ygrid.flatten()
        self.geometry = gpd.GeoSeries(crs=self.meta['crs'])

    def plot(self, axis=None, figsize=(9, 13), **kwargs):
        """Plot centroids scatter points over earth.

        Parameters
        ----------
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: (float, float), optional
            figure size for plt.subplots
            The default is (9, 13)
        kwargs : optional
            arguments for scatter matplotlib function

        Returns
        -------
        axis : matplotlib.axes._subplots.AxesSubplot
        """
        if self.meta and not self.coord.size:
            self.set_meta_to_lat_lon()
        pad = np.abs(u_coord.get_resolution(self.lat, self.lon)).min()

        proj_data, _ = u_plot.get_transformation(self.crs)
        proj_plot = proj_data
        if isinstance(proj_data, ccrs.PlateCarree):
            # use different projections for plot and data to shift the central lon in the plot
            xmin, ymin, xmax, ymax = u_coord.latlon_bounds(self.lat, self.lon, buffer=pad)
            proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (xmin + xmax))
        else:
            xmin, ymin, xmax, ymax = (self.lon.min() - pad, self.lat.min() - pad,
                                      self.lon.max() + pad, self.lat.max() + pad)

        if not axis:
            _, axis = u_plot.make_map(proj=proj_plot, figsize=figsize)

        axis.set_extent((xmin, xmax, ymin, ymax), crs=proj_data)
        u_plot.add_shapes(axis)
        axis.scatter(self.lon, self.lat, transform=proj_data, **kwargs)
        return axis

    def calc_pixels_polygons(self, scheduler=None):
        """Return a gpd.GeoSeries with a polygon for every pixel

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”

        Returns
        -------
        geo : gpd.GeoSeries
        """
        if not self.meta:
            self.set_lat_lon_to_meta()
        if abs(abs(self.meta['transform'].a) -
               abs(self.meta['transform'].e)) > 1.0e-5:
            raise ValueError('Area can not be computed for not squared pixels.')
        self.set_geometry_points(scheduler)
        return self.geometry.buffer(self.meta['transform'].a / 2).envelope

    def empty_geometry_points(self):
        """Removes all points in geometry.

        Useful when centroids is used in multiprocessing function."""
        self.geometry = gpd.GeoSeries(crs=self.geometry.crs)

    def write_hdf5(self, file_data):
        """Write centroids attributes into hdf5 format.

        Parameters
        ----------
        file_data : str or h5
            If string, path to write data. If h5 object, the datasets will be generated there.
        """
        if isinstance(file_data, str):
            LOGGER.info('Writting %s', file_data)
            data = h5py.File(file_data, 'w')
        else:
            data = file_data
        str_dt = h5py.special_dtype(vlen=str)
        for centr_name, centr_val in self.__dict__.items():
            if isinstance(centr_val, np.ndarray):
                data.create_dataset(centr_name, data=centr_val, compression="gzip")
            if centr_name == 'meta' and centr_val:
                centr_meta = data.create_group(centr_name)
                for key, value in centr_val.items():
                    if key not in ('crs', 'transform'):
                        if not isinstance(value, str):
                            centr_meta.create_dataset(key, (1,), data=value, dtype=type(value))
                        else:
                            hf_str = centr_meta.create_dataset(key, (1,), dtype=str_dt)
                            hf_str[0] = value
                    elif key == 'transform':
                        centr_meta.create_dataset(
                            key, (6,),
                            data=[value.a, value.b, value.c, value.d, value.e, value.f],
                            dtype=float)
        hf_str = data.create_dataset('crs', (1,), dtype=str_dt)
        hf_str[0] = CRS.from_user_input(self.crs).to_wkt()

        if isinstance(file_data, str):
            data.close()

    def read_hdf5(self, file_data):
        """Read centroids attributes from hdf5.

        Parameters
        ----------
        file_data : str or h5
            If string, path to read data. If h5 object, the datasets will be read from there.
        """
        if isinstance(file_data, (str, Path)):
            LOGGER.info('Reading %s', file_data)
            data = h5py.File(file_data, 'r')
        else:
            data = file_data
        self.clear()
        crs = DEF_CRS
        if data.get('crs'):
            crs = u_coord.to_crs_user_input(data.get('crs')[0])
        if data.get('lat') and data.get('lat').size:
            self.set_lat_lon(np.array(data.get('lat')), np.array(data.get('lon')), crs)
        elif data.get('latitude') and data.get('latitude').size:
            self.set_lat_lon(np.array(data.get('latitude')), np.array(data.get('longitude')), crs)
        else:
            centr_meta = data.get('meta')
            self.meta['crs'] = crs
            for key, value in centr_meta.items():
                if key != 'transform':
                    self.meta[key] = value[0]
                else:
                    self.meta[key] = rasterio.Affine(*value)
        for centr_name in data.keys():
            if centr_name not in ('crs', 'lat', 'lon', 'meta'):
                setattr(self, centr_name, np.array(data.get(centr_name)))
        if isinstance(file_data, str):
            data.close()

    @property
    def crs(self):
        """Get CRS of raster or vector."""
        if self.meta:
            return self.meta['crs']
        return self.geometry.crs

    @property
    def size(self):
        """Get size of pixels or points."""
        if self.meta:
            return self.meta['height'] * self.meta['width']
        return self.lat.size

    @property
    def shape(self):
        """Get shape of rastered data."""
        try:
            if self.meta:
                return (self.meta['height'], self.meta['width'])
            return (np.unique(self.lat).size, np.unique(self.lon).size)
        except AttributeError:
            return ()

    @property
    def total_bounds(self):
        """Get total bounds (left, bottom, right, top)."""
        if self.meta:
            left = self.meta['transform'].xoff
            right = left + self.meta['transform'][0] * self.meta['width']
            if left > right:
                left, right = right, left
            top = self.meta['transform'].yoff
            bottom = top + self.meta['transform'][4] * self.meta['height']
            if bottom > top:
                bottom, top = top, bottom
            return left, bottom, right, top
        return self.lon.min(), self.lat.min(), self.lon.max(), self.lat.max()

    @property
    def coord(self):
        """Get [lat, lon] array. Might take some time."""
        return np.stack([self.lat, self.lon], axis=1)

    def set_geometry_points(self, scheduler=None):
        """Set `geometry` attribute with Points from `lat`/`lon` attributes.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """
        def apply_point(df_exp):
            return df_exp.apply((lambda row: Point(row.longitude, row.latitude)), axis=1)
        if not self.geometry.size:
            LOGGER.info('Convert centroids to GeoSeries of Point shapes.')
            if not self.lat.size or not self.lon.size:
                self.set_meta_to_lat_lon()
            if not scheduler:
                self.geometry = gpd.GeoSeries(
                    gpd.points_from_xy(self.lon, self.lat), crs=self.geometry.crs)
            else:
                import dask.dataframe as dd
                from multiprocessing import cpu_count
                ddata = dd.from_pandas(self, npartitions=cpu_count())
                self.geometry = (ddata
                                 .map_partitions(apply_point, meta=Point)
                                 .compute(scheduler=scheduler))

    @staticmethod
    def _centroids_resolution(centroids):
        """ Return resolution of the centroids in their units

        Parameters:
            centroids (Centroids): centroids instance

        Returns:
            float
        """
        if centroids.meta:
            res_centr = abs(centroids.meta['transform'][4]), \
                centroids.meta['transform'][0]
        else:
            res_centr = u_coord.get_resolution(centroids.lat, centroids.lon)
        if abs(abs(res_centr[0]) - abs(res_centr[1])) > 1.0e-6:
            LOGGER.warning('Centroids do not represent regular pixels %s.', str(res_centr))
            return (res_centr[0] + res_centr[1])/2
        return res_centr[0]
    
    def _ne_crs_geom(self, scheduler=None):
        """Return `geometry` attribute in the CRS of Natural Earth.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”

        Returns
        -------
        geo : gpd.GeoSeries
        """
        if not self.lat.size or not self.lon.size:
            self.set_meta_to_lat_lon()
        if u_coord.equal_crs(self.geometry.crs, u_coord.NE_CRS) and self.geometry.size:
            return self.geometry
        self.set_geometry_points(scheduler)
        return self.geometry.to_crs(u_coord.NE_CRS)

    def __deepcopy__(self, memo):
        """Avoid error deep copy in gpd.GeoSeries by setting only the crs."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key == 'geometry':
                setattr(result, key, gpd.GeoSeries(crs=self.geometry.crs))
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result


def generate_nat_earth_centroids(res_as=360, path=None, dist_coast=False):
    """Generate hdf5 file containing Centroids of given resolution.

    For reproducibility, this is the function that generates the centroids files in
    `NATEARTH_CENTROIDS`. These files are provided with CLIMADA so that this function should never
    be called!

    Parameters
    ----------
    res_as : int
        Resolution of file in arc-seconds. Default: 360.
    path : str, optional
        If set, write resulting hdf5 file here instead of the default location.
    dist_coast : bool, optional
        If True, read distance from a NASA dataset (see util.coordinates.dist_to_coast_nasa).
        Default: False.
    """
    if path is None and res_as not in [150, 360]:
        raise ValueError("Only 150 and 360 arc-seconds are supported!")

    res_deg = res_as / 3600
    lat_dim = np.arange(-90 + res_deg, 90, res_deg)
    lon_dim = np.arange(-180 + res_deg, 180 + res_deg, res_deg)
    lon, lat = [ar.ravel() for ar in np.meshgrid(lon_dim, lat_dim)]
    natids = np.uint16(u_coord.get_country_code(lat, lon, gridded=False))

    cen = Centroids()
    cen.set_lat_lon(lat, lon)
    cen.region_id = natids
    cen.set_lat_lon_to_meta()
    cen.lat = np.array([])
    cen.lon = np.array([])

    if path is None:
        path = NATEARTH_CENTROIDS[res_as]

    if dist_coast:
        cen.set_dist_coast(precomputed=True, signed=False)
        cen.dist_coast = np.float16(cen.dist_coast)
    cen.write_hdf5(path)
