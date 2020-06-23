"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define functions to handle with coordinates
"""
import os
import copy
import logging
from multiprocessing import cpu_count
import math
import numpy as np
from cartopy.io import shapereader
import shapely.vectorized
import shapely.ops
from shapely.geometry import Polygon, MultiPolygon, Point, box
from fiona.crs import from_epsg
from iso3166 import countries as iso_cntry
import geopandas as gpd
import rasterio
import rasterio.warp
import rasterio.mask
import rasterio.crs
import rasterio.features
import shapefile
import dask.dataframe as dd
import pandas as pd
import scipy.interpolate
import zipfile

from climada.util.constants import DEF_CRS, SYSTEM_DIR, \
                                   NATEARTH_CENTROIDS_150AS, \
                                   NATEARTH_CENTROIDS_360AS, \
                                   ISIMIP_GPWV3_NATID_150AS, \
                                   ISIMIP_NATID_TO_ISO, \
                                   RIVER_FLOOD_REGIONS_CSV
from climada.util.files_handler import download_file
import climada.util.hdf5_handler as hdf5

pd.options.mode.chained_assignment = None

LOGGER = logging.getLogger(__name__)

NE_EPSG = 4326
""" Natural Earth CRS EPSG """

NE_CRS = from_epsg(NE_EPSG)
""" Natural Earth CRS """

TMP_ELEVATION_FILE = os.path.join(SYSTEM_DIR, 'tmp_elevation.tif')
""" Path of elevation file written in set_elevation """

DEM_NODATA = -9999
""" Value to use for no data values in DEM, i.e see points """

MAX_DEM_TILES_DOWN = 300
""" Maximum DEM tiles to dowload """

def grid_is_regular(coord):
    """Return True if grid is regular. If True, returns height and width.

    Parameters:
        coord (np.array):

    Returns:
        bool (is regular), int (height), int (width)
    """
    regular = False
    _, count_lat = np.unique(coord[:, 0], return_counts=True)
    _, count_lon = np.unique(coord[:, 1], return_counts=True)
    uni_lat_size = np.unique(count_lat).size
    uni_lon_size = np.unique(count_lon).size
    if uni_lat_size == uni_lon_size and uni_lat_size == 1 \
    and count_lat[0] > 1 and count_lon[0] > 1:
        regular = True
    return regular, count_lat[0], count_lon[0]

def get_coastlines(bounds=None, resolution=110):
    """ Get Polygones of coast intersecting given bounds

    Parameter:
        bounds (tuple): min_lon, min_lat, max_lon, max_lat in EPSG:4326
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            110m, i.e. 1:110.000.000

    Returns:
        GeoDataFrame
    """
    resolution = nat_earth_resolution(resolution)
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category='physical',
                                         name='coastline')
    coast_df = gpd.read_file(shp_file)
    coast_df.crs = NE_CRS
    if bounds is None:
        return coast_df[['geometry']]
    tot_coast = np.zeros(1)
    while not np.any(tot_coast):
        tot_coast = coast_df.envelope.intersects(box(*bounds))
        bounds = (bounds[0] - 20, bounds[1] - 20,
                  bounds[2] + 20, bounds[3] + 20)
    return coast_df[tot_coast][['geometry']]

def convert_wgs_to_utm(lon, lat):
    """ Get EPSG code of UTM projection for input point in EPSG 4326

    Parameter:
        lon (float): longitude point in EPSG 4326
        lat (float): latitude of point (lat, lon) in EPSG 4326

    Return:
        int
    """
    epsg_utm_base = 32601 + (0 if lat >= 0 else 100)
    return epsg_utm_base + (math.floor((lon + 180) / 6) % 60)

def utm_zones(wgs_bounds):
    """ Get EPSG code and bounds of UTM zones covering specified region

    Parameter:
        wgs_bounds (tuple): lon_min, lat_min, lon_max, lat_max

    Returns:
        list of pairs (zone_epsg, zone_wgs_bounds)
    """
    lon_min, lat_min, lon_max, lat_max = wgs_bounds
    lon_min, lon_max = max(-179.99, lon_min), min(179.99, lon_max)
    utm_min, utm_max = [math.floor((l + 180) / 6) for l in [lon_min, lon_max]]
    zones = []
    for utm in range(utm_min, utm_max + 1):
        epsg = 32601 + utm
        bounds = (-180 + 6 * utm, 0, -180 + 6 * (utm + 1), 90)
        if lat_max >= 0:
            zones.append((epsg, bounds))
        if lat_min < 0:
            bounds = (bounds[0], -90, bounds[2], 0)
            zones.append((epsg + 100, bounds))
    return zones

def dist_to_coast(coord_lat, lon=None):
    """ Compute distance to coast from input points in meters.

    Parameters:
        coord_lat (GeoDataFrame or np.array or float):
            - GeoDataFrame with geometry column in epsg:4326
            - np.array with two columns, first for latitude of each point and
                second with longitude in epsg:4326
            - np.array with one dimension containing latitudes in epsg:4326
            - float with a latitude value in epsg:4326
        lon (np.array or float, optional):
            - np.array with one dimension containing longitudes in epsg:4326
            - float with a longitude value in epsg:4326

    Returns:
        np.array
    """
    if isinstance(coord_lat, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if not equal_crs(coord_lat.crs, NE_CRS):
            LOGGER.error('Input CRS is not %s', str(NE_CRS))
            raise ValueError
        geom = coord_lat
    else:
        if lon is None:
            if isinstance(coord_lat, np.ndarray) and coord_lat.shape[1] == 2:
                lat, lon = coord_lat[:, 0], coord_lat[:, 1]
            else:
                LOGGER.error('Missing longitude values.')
                raise ValueError
        else:
            lat, lon = [np.asarray(v).reshape(-1) for v in [coord_lat, lon]]
            if lat.size != lon.size:
                LOGGER.error('Mismatching input coordinates size: %s != %s',
                             lat.size, lon.size)
                raise ValueError
        geom = gpd.GeoDataFrame(geometry=list(map(Point, lon, lat)), crs=NE_CRS)

    pad = 20
    bounds = (geom.total_bounds[0] - pad, geom.total_bounds[1] - pad,
              geom.total_bounds[2] + pad, geom.total_bounds[3] + pad)
    coast = get_coastlines(bounds, 10).geometry
    coast = gpd.GeoDataFrame(geometry=coast, crs=NE_CRS)
    dist = np.empty(geom.shape[0])
    zones = utm_zones(geom.geometry.total_bounds)
    for izone, (epsg, bounds) in enumerate(zones):
        to_crs = from_epsg(epsg)
        zone_mask = (bounds[1] <= geom.geometry.y) \
                  & (geom.geometry.y <= bounds[3]) \
                  & (bounds[0] <= geom.geometry.x) \
                  & (geom.geometry.x <= bounds[2])
        if np.count_nonzero(zone_mask) == 0:
            continue
        LOGGER.info("dist_to_coast: UTM %d (%d/%d)",
                    epsg, izone + 1, len(zones))
        bounds = geom[zone_mask].total_bounds
        bounds = (bounds[0] - pad, bounds[1] - pad,
                  bounds[2] + pad, bounds[3] + pad)
        coast_mask = coast.envelope.intersects(box(*bounds))
        utm_coast = coast[coast_mask].geometry.unary_union
        utm_coast = gpd.GeoDataFrame(geometry=[utm_coast], crs=NE_CRS)
        utm_coast = utm_coast.to_crs(to_crs).geometry[0]
        dist[zone_mask] = geom[zone_mask].to_crs(to_crs).distance(utm_coast)
    return dist

def dist_to_coast_nasa(lat, lon, highres=False):
    """ Read interpolated distance to coast (in m) from NASA data

    Note: The NASA raster file is 300 MB and will be downloaded on first run!

    Parameters:
        lat (np.array): latitudes in epsg:4326
        lon (np.array): longitudes in epsg:4326
        highres (bool, optional): Use full resolution of NASA data (much
            slower). Default: False.

    Returns:
        np.array
    """
    lat, lon = [np.asarray(ar).ravel() for ar in [lat, lon]]

    zipname = "GMT_intermediate_coast_distance_01d.zip"
    tifname = "GMT_intermediate_coast_distance_01d.tif"
    url = "https://oceancolor.gsfc.nasa.gov/docs/distfromcoast/" + zipname
    path = os.path.join(SYSTEM_DIR, tifname)
    if not os.path.isfile(path):
        cwd = os.getcwd()
        os.chdir(SYSTEM_DIR)
        path_dwn = download_file(url)
        zip_ref = zipfile.ZipFile(path_dwn, 'r')
        zip_ref.extractall(SYSTEM_DIR)
        zip_ref.close()
        os.remove(path_dwn)
        os.chdir(cwd)

    intermediate_shape = None if highres else (3600, 1800)
    dist = read_raster_sample(path, lat, lon,
        intermediate_shape=intermediate_shape, fill_value=0)
    return 1000 * np.abs(dist)

def get_land_geometry(country_names=None, extent=None, resolution=10):
    """Get union of all the countries or the provided ones or the points inside
    the extent.

    Parameters:
        country_names (list, optional): list with ISO3 names of countries, e.g
            ['ZWE', 'GBR', 'VNM', 'UZB']
        extent (tuple, optional): (min_lon, max_lon, min_lat, max_lat)
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            10m, i.e. 1:10.000.000

    Returns:
        shapely.geometry.multipolygon.MultiPolygon
    """
    resolution = nat_earth_resolution(resolution)
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category='cultural',
                                         name='admin_0_countries')
    reader = shapereader.Reader(shp_file)
    if (country_names is None) and (extent is None):
        LOGGER.info("Computing earth's land geometry ...")
        geom = list(reader.geometries())
        geom = shapely.ops.cascaded_union(geom)

    elif country_names:
        countries = list(reader.records())
        geom = [country.geometry for country in countries
                if (country.attributes['ISO_A3'] in country_names) or
                (country.attributes['WB_A3'] in country_names) or
                (country.attributes['ADM0_A3'] in country_names)]
        geom = shapely.ops.cascaded_union(geom)

    else:
        extent_poly = Polygon([(extent[0], extent[2]), (extent[0], extent[3]),
                               (extent[1], extent[3]), (extent[1], extent[2])])
        geom = []
        for cntry_geom in reader.geometries():
            inter_poly = cntry_geom.intersection(extent_poly)
            if not inter_poly.is_empty:
                geom.append(inter_poly)
        geom = shapely.ops.cascaded_union(geom)
    if not isinstance(geom, MultiPolygon):
        geom = MultiPolygon([geom])
    return geom

def coord_on_land(lat, lon, land_geom=None):
    """Check if point is on land (True) or water (False) of provided coordinates.
    All globe considered if no input countries.

    Parameters:
        lat (np.array): latitude of points in epsg:4326
        lon (np.array): longitude of points in epsg:4326
        land_geom (shapely.geometry.multipolygon.MultiPolygon, optional):
            profiles of land.

    Returns:
        np.array(bool)
    """
    if lat.size != lon.size:
        LOGGER.error('Wrong size input coordinates: %s != %s.', lat.size,
                     lon.size)
        raise ValueError
    delta_deg = 1
    if land_geom is None:
        land_geom = get_land_geometry(extent=(np.min(lon)-delta_deg, \
            np.max(lon)+delta_deg, np.min(lat)-delta_deg, \
            np.max(lat)+delta_deg), resolution=10)
    return shapely.vectorized.contains(land_geom, lon, lat)

def nat_earth_resolution(resolution):
    """Check if resolution is available in Natural Earth. Build string.

    Parameters:
        resolution (int): resolution in millions, 110 == 1:110.000.000.

    Returns:
        str

    Raises:
        ValueError
    """
    avail_res = [10, 50, 110]
    if resolution not in avail_res:
        LOGGER.error('Natural Earth does not accept resolution %s m.',
                     resolution)
        raise ValueError
    return str(resolution) + 'm'

def get_country_geometries(country_names=None, extent=None, resolution=10):
    """Returns a gpd GeoSeries of natural earth multipolygons of the
    specified countries, resp. the countries that lie within the specified
    extent. If no arguments are given, simply returns the whole natural earth
    dataset.
    Take heed: we assume WGS84 as the CRS unless the Natural Earth download
    utility from cartopy starts including the projection information. (They
    are saving a whopping 147 bytes by omitting it.) Same goes for UTF.

    Parameters:
        country_names (list, optional): list with ISO3 names of countries, e.g
            ['ZWE', 'GBR', 'VNM', 'UZB']
        extent (tuple, optional): (min_lon, max_lon, min_lat, max_lat) assumed
            to be in the same CRS as the natural earth data.
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            10m

    Returns:
        GeoDataFrame
    """
    resolution = nat_earth_resolution(resolution)
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category='cultural',
                                         name='admin_0_countries')
    nat_earth = gpd.read_file(shp_file, encoding='UTF-8')

    if not nat_earth.crs:
        nat_earth.crs = NE_CRS

    # fill gaps in nat_earth
    gap_mask = (nat_earth['ISO_A3'] == '-99')
    nat_earth.loc[gap_mask, 'ISO_A3'] = nat_earth.loc[gap_mask, 'ADM0_A3']

    gap_mask = (nat_earth['ISO_N3'] == '-99')
    for idx in nat_earth[gap_mask].index:
        for col in ['ISO_A3', 'ADM0_A3', 'NAME']:
            try:
                num = iso_cntry.get(nat_earth.loc[idx, col]).numeric
            except KeyError:
                continue
            else:
                nat_earth.loc[idx, 'ISO_N3'] = num
                break

    out = nat_earth
    if country_names:
        if isinstance(country_names, str):
            country_names = [country_names]
        out = out[out.ISO_A3.isin(country_names)]

    if extent:
        bbox = Polygon([
            (extent[0], extent[2]),
            (extent[0], extent[3]),
            (extent[1], extent[3]),
            (extent[1], extent[2])
        ])
        bbox = gpd.GeoSeries(bbox, crs=out.crs)
        bbox = gpd.GeoDataFrame({'geometry': bbox}, crs=out.crs)
        out = gpd.overlay(out, bbox, how="intersection")

    return out

def get_region_gridpoints(countries=None, regions=None, resolution=150,
                          iso=True, rect=False, basemap="natearth"):
    """ Get coordinates of gridpoints in specified countries or regions

    Parameters:
        countries (list, optional): ISO 3166-1 alpha-3 codes of countries, or
            internal numeric NatID if iso is set to False.
        regions (list, optional): Region IDs.
        resolution (float, optional): Resolution in arc-seconds. Default: 150.
        iso (bool, optional): If True, assume that countries are given by their
            ISO 3166-1 alpha-3 codes (instead of the internal NatID).
            Default: True.
        rect (bool, optional): If True, a rectangular box around the specified
            countries/regions is selected. Default: False.
        basemap (str, optional): Choose between different data sources.
            Currently available: "isimip" and "natearth". Default: "natearth".

    Returns:
        lat (np.array): latitude of points in epsg:4326
        lon (np.array): longitude of points in epsg:4326
    """
    if countries is None:
        countries = []
    if regions is None:
        regions = []

    if basemap == "natearth":
        base_file = NATEARTH_CENTROIDS_150AS
        if resolution >= 360:
            base_file = NATEARTH_CENTROIDS_360AS
        f = hdf5.read(base_file)
        meta = f['meta']
        grid_shape = (meta['height'][0], meta['width'][0])
        transform = rasterio.Affine(*meta['transform'])
        region_id = f['region_id'].reshape(grid_shape)
        lon, lat = raster_to_meshgrid(transform, grid_shape[1], grid_shape[0])
    elif basemap == "isimip":
        f = hdf5.read(ISIMIP_GPWV3_NATID_150AS)
        dim_lon, dim_lat = f['lon'], f['lat']
        bounds = dim_lon.min(), dim_lat.min(), dim_lon.max(), dim_lat.max()
        orig_res = get_resolution(dim_lon, dim_lat)
        _, _, transform = pts_to_raster_meta(bounds, orig_res)
        grid_shape = (dim_lat.size, dim_lon.size)
        region_id = f['NatIdGrid'].reshape(grid_shape).astype(int)
        region_id[region_id < 0] = 0
        natid2iso_alpha = country_natid2iso(list(range(1, 231)))
        natid2iso = country_iso_alpha2numeric(natid2iso_alpha)
        natid2iso = np.array(natid2iso, dtype=int)
        region_id = natid2iso[region_id - 1]
        lon, lat = np.meshgrid(dim_lon, dim_lat)
    else:
        raise ValueError(f"Unknown basemap: {basemap}")

    if basemap == "natearth" and resolution not in [150, 360] \
       or basemap == "isimip" and resolution != 150:
        resolution /= 3600
        region_id, transform = refine_raster_data(region_id, transform,
            resolution, method='nearest', fill_value=0)
        grid_shape = region_id.shape
        lon, lat = raster_to_meshgrid(transform, grid_shape[1], grid_shape[0])

    if not iso:
        countries = country_natid2iso(countries)
    countries += region2isos(regions)
    countries = np.unique(country_iso_alpha2numeric(countries))

    if len(countries) > 0:
        msk = np.isin(region_id, countries)
        if rect:
            msk = msk.any(axis=0)[None] * msk.any(axis=1)[:, None]
            msk |= (lat >= np.floor(lat[msk].min())) \
                 & (lon >= np.floor(lon[msk].min())) \
                 & (lat <= np.ceil(lat[msk].max())) \
                 & (lon <= np.ceil(lon[msk].max()))
        lat, lon = lat[msk], lon[msk]
    else:
        lat, lon = [ar.ravel() for ar in [lat, lon]]
    return lat, lon

def region2isos(regions):
    """ Convert region names to ISO 3166 alpha-3 codes of countries

    Parameters:
        regions (str or list of str): Region name(s).

    Returns:
        isos (list of str): Sorted list of iso codes of all countries in
            specified region(s).
    """
    regions = [regions] if isinstance(regions, str) else regions
    reg_info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
    isos = []
    for region in regions:
        region_msk = (reg_info['Reg_name'] == region)
        if not any(region_msk):
            LOGGER.error('Unknown region name: %s', region)
            raise KeyError
        isos += list(reg_info['ISO'][region_msk].values)
    return list(set(isos))

def country_iso_alpha2numeric(isos):
    """ Convert ISO 3166-1 alpha-3 to numeric-3 codes

    Parameters:
        isos (str or list of str): ISO codes of countries (or single code).

    Returns:
        int or list of int
    """
    return_int = isinstance(isos, str)
    isos = [isos] if return_int else isos
    OLD_ISO = {
        "ANT": 530,  # Netherlands Antilles: split up since 2010
        "SCG": 891,  # Serbia and Montenegro: split up since 2006
    }
    nums = []
    for iso in isos:
        if iso in OLD_ISO:
            num = OLD_ISO[iso]
        else:
            num = int(iso_cntry.get(iso).numeric)
        nums.append(num)
    return nums[0] if return_int else nums

def country_natid2iso(natids):
    """ Convert internal NatIDs to ISO 3166-1 alpha-3 codes

    Parameters:
        natids (int or list of int): Internal NatIDs of countries (or single ID).

    Returns:
        str or list of str
    """
    return_str = isinstance(natids, int)
    natids = [natids] if return_str else natids
    isos = []
    for natid in natids:
        if natid < 0 or natid >= len(ISIMIP_NATID_TO_ISO):
            LOGGER.error('Unknown country NatID: %s', natid)
            raise KeyError
        isos.append(ISIMIP_NATID_TO_ISO[natid])
    return isos[0] if return_str else isos

def country_iso2natid(isos):
    """ Convert ISO 3166-1 alpha-3 codes to internal NatIDs

    Parameters:
        isos (str or list of str): ISO codes of countries (or single code).

    Returns:
        int or list of int
    """
    return_int = isinstance(isos, str)
    isos = [isos] if return_int else isos
    natids = []
    for iso in isos:
        try:
            natids.append(ISIMIP_NATID_TO_ISO.index(iso))
        except ValueError:
            LOGGER.error('Unknown country ISO: %s', iso)
            raise KeyError
    return natids[0] if return_int else natids

NATEARTH_AREA_NONISO_NUMERIC = {
    "Akrotiri": 901,
    "Baikonur": 902,
    "Bajo Nuevo Bank": 903,
    "Clipperton I.": 904,
    "Coral Sea Is.": 905,
    "Cyprus U.N. Buffer Zone": 906,
    "Dhekelia": 907,
    "Indian Ocean Ter.": 908,
    "Kosovo": 983,  # Same as iso3166 package
    "N. Cyprus": 910,
    "Norway": 578,  # Bug in Natural Earth
    "Scarborough Reef": 912,
    "Serranilla Bank": 913,
    "Siachen Glacier": 914,
    "Somaliland": 915,
    "Spratly Is.": 916,
    "USNB Guantanamo Bay": 917,
}

def natearth_country_to_int(country):
    if country.ISO_N3 != '-99':
        return int(country.ISO_N3)
    else:
        return NATEARTH_AREA_NONISO_NUMERIC[str(country.NAME)]

def get_country_code(lat, lon, gridded=False, natid=False):
    """ Provide numeric (ISO 3166) code for every point.

    Oceans get the value zero. Areas that are not in ISO 3166 are given values
    in the range above 900 according to NATEARTH_AREA_NONISO_NUMERIC.

    Parameters:
        lat (np.array): latitude of points in epsg:4326
        lon (np.array): longitude of points in epsg:4326
        gridded (bool): If True, interpolate precomputed gridded data which
            is usually much faster. Default: False.

    Returns:
        np.array(int)
    """
    lat, lon = [np.asarray(ar).ravel() for ar in [lat, lon]]
    LOGGER.info('Setting region_id %s points.', str(lat.size))
    if gridded:
        base_file = hdf5.read(NATEARTH_CENTROIDS_150AS)
        meta, region_id = base_file['meta'], base_file['region_id']
        transform = rasterio.Affine(*meta['transform'])
        region_id = region_id.reshape(meta['height'][0], meta['width'][0])
        region_id = interp_raster_data(region_id, lat, lon, transform,
                                       method='nearest', fill_value=0)
        region_id = region_id.astype(int)
    else:
        extent = (lon.min() - 0.001, lon.max() + 0.001,
                  lat.min() - 0.001, lat.max() + 0.001)
        countries = get_country_geometries(extent=extent)
        countries['area'] = countries.geometry.area
        countries = countries.sort_values(by=['area'], ascending=False)
        region_id = np.full((lon.size,), -1, dtype=int)
        total_land = countries.geometry.unary_union
        ocean_mask = ~shapely.vectorized.contains(total_land, lon, lat)
        region_id[ocean_mask] = 0
        for i, country in enumerate(countries.itertuples()):
            unset = (region_id == -1).nonzero()[0]
            select = shapely.vectorized.contains(country.geometry,
                                                 lon[unset], lat[unset])
            region_id[unset[select]] = natearth_country_to_int(country)
        region_id[region_id == -1] = 0
    return region_id

def get_admin1_info(country_names):
    """ Provide registry info and shape files for admin1 regions

    Parameters:
        country_names (list): list with ISO3 names of countries, e.g.
                ['ZWE', 'GBR', 'VNM', 'UZB']

    Returns:
        admin1_info (dict)
        admin1_shapes (dict)
    """

    if isinstance(country_names, str):
        country_names = [country_names]
    admin1_file = shapereader.natural_earth(resolution='10m',
                                            category='cultural',
                                            name='admin_1_states_provinces')
    admin1_recs = shapefile.Reader(admin1_file)
    admin1_info = dict()
    admin1_shapes = dict()
    for iso3 in country_names:
        admin1_info[iso3] = list()
        admin1_shapes[iso3] = list()
        for rec, rec_shp in zip(admin1_recs.records(), admin1_recs.shapes()):
            if rec['adm0_a3'] == iso3:
                admin1_info[iso3].append(rec)
                admin1_shapes[iso3].append(rec_shp)
    return admin1_info, admin1_shapes

def get_resolution_1d(coords, min_resol=1.0e-8):
    """ Compute resolution of scalar grid

    Parameters:
        coords (np.array): scalar coordinates
        min_resol (float, optional): minimum resolution to consider.
            Default: 1.0e-8.

    Returns:
        float
    """
    res = np.diff(np.unique(coords))
    diff = np.diff(coords)
    mask = (res > min_resol) & np.isin(res, np.abs(diff))
    return diff[np.abs(diff) == res[mask].min()][0]


def get_resolution(*coords, min_resol=1.0e-8):
    """ Compute resolution of 2-d grid points

    Parameters:
        X, Y, ... (np.array): scalar coordinates in each axis
        min_resol (float, optional): minimum resolution to consider.
            Default: 1.0e-8.

    Returns:
        pair of floats
    """
    return tuple([get_resolution_1d(c, min_resol=min_resol) for c in coords])


def pts_to_raster_meta(points_bounds, res):
    """" Transform vector data coordinates to raster. Returns number of rows,
    columns and affine transformation

    If a raster of the given resolution doesn't exactly fit the given bounds,
    the raster might have slightly larger (but never smaller) bounds.

    Parameters:
        points_bounds (tuple): points total bounds (xmin, ymin, xmax, ymax)
        res (tuple): resolution of output raster (xres, yres)

    Returns:
        int, int, affine.Affine
    """
    Affine = rasterio.Affine
    bounds = np.asarray(points_bounds).reshape(2,2)
    res = np.asarray(res).ravel()
    if res.size == 1:
        res = np.array([res[0], res[0]])
    sizes = bounds[1,:] - bounds[0,:]
    nsteps = np.floor(sizes / np.abs(res)) + 1
    nsteps[np.abs(nsteps * res) < sizes + np.abs(res) / 2] += 1
    bounds[:,res < 0] = bounds[::-1,res < 0]
    origin = bounds[0,:] - res[:] / 2
    ras_trans = Affine.translation(*origin) * Affine.scale(*res)
    return int(nsteps[1]), int(nsteps[0]), ras_trans

def raster_to_meshgrid(transform, width, height):
    """ Get coordinates of grid points in raster

    Parameters:
        transform (affine.Affine): Affine transform defining the raster.
        width (int): Number of points in first coordinate axis.
        height (int): Number of points in second coordinate axis.

    Returns:
        x (np.array): x-coordinates of grid points
        y (np.array): y-coordinates of grid points
    """
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax = xmin + width * xres
    ymax = ymin + height * yres
    return np.meshgrid(np.arange(xmin + xres / 2, xmax, xres),
                       np.arange(ymin + yres / 2, ymax, yres))

def equal_crs(crs_one, crs_two):
    """ Compare two crs

    Parameters:
        crs_one (dict or string or wkt): user crs
        crs_two (dict or string or wkt): user crs

    Returns:
        bool
    """
    return rasterio.crs.CRS.from_user_input(crs_one) \
           == rasterio.crs.CRS.from_user_input(crs_two)

def _read_raster_reproject(src, src_crs, dst_meta,
        band=[1], geometry=None, dst_crs=None, transform=None,
        resampling=rasterio.warp.Resampling.nearest):
    """ Helper function for `read_raster` """
    if not dst_crs:
        dst_crs = src_crs
    if not transform:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds)
    else:
        transform, width, height = transform
    dst_meta.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height,
    })
    kwargs = {}
    if src.meta['nodata']:
        kwargs['src_nodata'] = src.meta['nodata']
        kwargs['dst_nodata'] = src.meta['nodata']

    intensity = np.zeros((len(band), height, width))
    for idx_band, i_band in enumerate(band):
        rasterio.warp.reproject(
            source=src.read(i_band),
            destination=intensity[idx_band, :],
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling,
            **kwargs)

        if dst_meta['nodata'] and np.isnan(dst_meta['nodata']):
            nodata_mask = np.isnan(intensity[idx_band, :])
        else:
            nodata_mask = (intensity[idx_band, :] == dst_meta['nodata'])
        intensity[idx_band, :][nodata_mask] = 0

    if geometry:
        intensity = intensity.astype('float32')
        # update driver to GTiff as netcdf does not work reliably
        dst_meta.update(driver='GTiff')
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**dst_meta) as dst:
                dst.write(intensity)

            with memfile.open() as dst:
                inten, mask_trans = rasterio.mask.mask(dst,
                    geometry, crop=True, indexes=band)
                dst_meta.update({
                    "height": inten.shape[1],
                    "width": inten.shape[2],
                    "transform": mask_trans,
                })
        intensity = inten[range(len(band)), :]
        intensity = intensity.astype('float64')

        # reset nodata values again as driver Gtiff resets them again
        if dst_meta['nodata'] and np.isnan(dst_meta['nodata']):
            intensity[np.isnan(intensity)] = 0
        else:
            intensity[intensity == dst_meta['nodata']] = 0

    return intensity

def read_raster(file_name, band=[1], src_crs=None, window=None, geometry=None,
                dst_crs=None, transform=None, width=None, height=None,
                resampling=rasterio.warp.Resampling.nearest):
    """ Read raster of bands and set 0 values to the masked ones. Each
    band is an event. Select region using window or geometry. Reproject
    input by proving dst_crs and/or (transform, width, height). Returns matrix
    in 2d: band x coordinates in 1d (can be reshaped to band x height x width)

    Parameters:
        file_name (str): name of the file
        band (list(int), optional): band number to read. Default: 1
        window (rasterio.windows.Window, optional): window to read
        geometry (shapely.geometry, optional): consider pixels only in shape
        dst_crs (crs, optional): reproject to given crs
        transform (rasterio.Affine): affine transformation to apply
        wdith (float): number of lons for transform
        height (float): number of lats for transform
        resampling (rasterio.warp.Resampling optional): resampling
            function used for reprojection to dst_crs

    Returns:
        dict (meta), np.array (band x coordinates_in_1d)
    """
    LOGGER.info('Reading %s', file_name)
    if os.path.splitext(file_name)[1] == '.gz':
        file_name = '/vsigzip/' + file_name

    with rasterio.Env():
        with rasterio.open(file_name, 'r') as src:
            src_crs = src.crs if src_crs is None else src_crs
            if not src_crs:
                src_crs = rasterio.crs.CRS.from_dict(DEF_CRS)
            dst_meta = src.meta.copy()

            if dst_crs or transform:
                LOGGER.debug('Reprojecting ...')
                transform = (transform, width, height) if transform else None
                inten = _read_raster_reproject(src, src_crs, dst_meta,
                    band=band, geometry=geometry, dst_crs=dst_crs,
                    transform=transform, resampling=resampling)
            else:
                trans = dst_meta['transform']
                if geometry:
                    inten, trans = rasterio.mask.mask(src,
                        geometry, crop=True, indexes=band)
                    if dst_meta['nodata'] and np.isnan(dst_meta['nodata']):
                        inten[np.isnan(inten)] = 0
                    else:
                        inten[inten == dst_meta['nodata']] = 0
                else:
                    masked_array = src.read(band, window=window, masked=True)
                    inten = masked_array.data
                    inten[masked_array.mask] = 0
                    if window:
                        trans = rasterio.windows.transform(window, src.transform)
                dst_meta.update({
                    "height": inten.shape[1],
                    "width": inten.shape[2],
                    "transform": trans,
                })
    if not dst_meta['crs']:
        dst_meta['crs'] = rasterio.crs.CRS.from_dict(DEF_CRS)
    intensity = inten[range(len(band)), :]
    dst_shape = (len(band), dst_meta['height']*dst_meta['width'])
    return dst_meta, intensity.reshape(dst_shape)

def read_raster_sample(path, lat, lon, intermediate_shape=None, method='linear',
                       fill_value=None):
    """ Read point samples from raster file

    Parameters:
        path (str): path of the raster file
        lat (np.array): latitudes in file's CRS
        lon (np.array): latitudes in file's CRS
        intermediate_shape (tuple, optional): If given, the raster is not read
            in its original resolution but in the given one. This can increase
            performance for files of very high resolution
        method (str, optional): The interpolation method, passed to
            scipy.interp.interpn. Default: 'linear'.
        fill_value (numeric, optional): The value used outside of the raster
            bounds. Default: The raster's nodata value or 0.

    Returns:
        np.array of same length as lat
    """
    LOGGER.info('Sampling from %s', path)
    if os.path.splitext(path)[1] == '.gz':
        path = '/vsigzip/' + path

    with rasterio.open(path, "r") as src:
        xres, _, xmin, _, yres, ymin = src.transform[:6]
        data = src.read(1, out_shape=intermediate_shape)
        xres *= src.width / data.shape[1]
        yres *= src.height / data.shape[0]
        fill_value = src.meta['nodata'] if fill_value is None else fill_value

    transform = rasterio.Affine(xres, 0, xmin, 0, yres, ymin)
    fill_value = fill_value if fill_value else 0
    return interp_raster_data(data, lat, lon, transform, method=method,
                              fill_value=fill_value)

def interp_raster_data(data, y, x, transform, method='linear', fill_value=0):
    """ Interpolate raster data, given as array and affine transform

    Parameters:
        data (np.array): 2d numpy array containing the values
        y (np.array): y-coordinates of points (corresp. to first axis of data)
        x (np.array): x-coordinates of points (corresp. to second axis of data)
        transform (affine.Affine): affine transform defining the raster
        method (str, optional): The interpolation method, passed to
            scipy.interp.interpn. Default: 'linear'.
        fill_value (numeric, optional): The value used outside of the raster
            bounds. Default: 0.

    Returns:
        np.array
    """
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax = xmin + data.shape[1] * xres
    ymax = ymin + data.shape[0] * yres
    data = np.pad(data, 1, mode='edge')

    if yres < 0:
        yres = -yres
        ymax, ymin = ymin, ymax
        data = np.flipud(data)
    if xres < 0:
        xres = -xres
        xmax, xmin = xmin, xmax
        data = np.fliplr(data)
    y_dim = ymin - yres / 2 + yres * np.arange(data.shape[0])
    x_dim = xmin - xres / 2 + xres * np.arange(data.shape[1])

    data = np.float64(data)
    data[np.isnan(data)] = fill_value
    return scipy.interpolate.interpn((y_dim, x_dim), data, np.vstack([y, x]).T,
        method=method, bounds_error=False, fill_value=fill_value)

def refine_raster_data(data, transform, res, method='linear', fill_value=0):
    """ Refine raster data, given as array and affine transform

    Parameters:
        data (np.array): 2d numpy array containing the values
        transform (affine.Affine): affine transform defining the raster
        res (float or pair of floats): new resolution
        method (str, optional): The interpolation method, passed to
            scipy.interp.interpn. Default: 'linear'.

    Return:
        np.array, affine.Affine
    """
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax = xmin + data.shape[1] * xres
    ymax = ymin + data.shape[0] * yres
    if not isinstance(res, tuple):
        res = (np.sign(xres) * res, np.sign(yres) * res)
    new_dimx = np.arange(xmin + res[0] / 2, xmax, res[0])
    new_dimy = np.arange(ymin + res[1] / 2, ymax, res[1])
    new_shape = (new_dimy.size, new_dimx.size)
    new_x, new_y = [ar.ravel() for ar in np.meshgrid(new_dimx, new_dimy)]
    new_transform = rasterio.Affine(res[0], 0, xmin, 0, res[1], ymin)
    new_data = interp_raster_data(data, new_y, new_x, transform, method=method,
                                  fill_value=fill_value)
    new_data = new_data.reshape(new_shape)
    return new_data, new_transform

def read_vector(file_name, field_name, dst_crs=None):
    """ Read vector file format supported by fiona. Each field_name name is
    considered an event.

    Parameters:
        file_name (str): vector file with format supported by fiona and
            'geometry' field.
        field_name (list(str)): list of names of the columns with values.
        dst_crs (crs, optional): reproject to given crs

    Returns:
        np.array (lat), np.array (lon), geometry (GeiSeries), np.array (value)
    """
    LOGGER.info('Reading %s', file_name)
    data_frame = gpd.read_file(file_name)
    if not data_frame.crs:
        data_frame.crs = DEF_CRS
    if dst_crs is None:
        geometry = data_frame.geometry
    else:
        geometry = data_frame.geometry.to_crs(dst_crs)
    lat, lon = geometry[:].y.values, geometry[:].x.values
    value = np.zeros([len(field_name), lat.size])
    for i_inten, inten in enumerate(field_name):
        value[i_inten, :] = data_frame[inten].values
    return lat, lon, geometry, value

def write_raster(file_name, data_matrix, meta, dtype=np.float32):
    """ Write raster in GeoTiff format

    Parameters:
        fle_name (str): file name to write
        data_matrix (np.array): 2d raster data. Either containing one band,
            or every row is a band and the column represents the grid in 1d.
        meta (dict): rasterio meta dictionary containing raster
            properties: width, height, crs and transform must be present
            at least (transform needs to contain upper left corner!)
        dtype (numpy dtype): a numpy dtype
    """
    LOGGER.info('Writting %s', file_name)
    if data_matrix.shape != (meta['height'], meta['width']):
        # every row is an event (from hazard intensity or fraction) == band
        shape = (data_matrix.shape[0], meta['height'], meta['width'])
    else:
        shape = (1, meta['height'], meta['width'])
    dst_meta = copy.deepcopy(meta)
    dst_meta.update(driver='GTiff', dtype=dtype, count=shape[0])
    data_matrix = np.asarray(data_matrix, dtype=dtype).reshape(shape)
    with rasterio.open(file_name, 'w', **dst_meta) as dst:
        dst.write(data_matrix, indexes=np.arange(1, shape[0] + 1))

def points_to_raster(points_df, val_names=['value'], res=None, raster_res=None,
                     scheduler=None):
    """ Compute raster matrix and transformation from value column

    Parameters:
        points_df (GeoDataFrame): contains columns latitude, longitude and in
            val_names
        res (float, optional): resolution of current data in units of latitude
            and longitude, approximated if not provided.
        raster_res (float, optional): desired resolution of the raster
        scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”

    Returns:
        np.array, affine.Affine

    """
    if not res:
        res = np.abs(get_resolution(points_df.latitude.values,
                                    points_df.longitude.values)).min()
    if not raster_res:
        raster_res = res

    def apply_box(df_exp):
        fun = lambda r: Point(r.longitude, r.latitude).buffer(res/2).envelope
        return df_exp.apply(fun, axis=1)

    LOGGER.info('Raster from resolution %s to %s.', res, raster_res)
    df_poly = points_df[val_names]
    if not scheduler:
        df_poly['geometry'] = apply_box(points_df)
    else:
        ddata = dd.from_pandas(points_df[['latitude', 'longitude']],
                               npartitions=cpu_count())
        df_poly['geometry'] = ddata.map_partitions(apply_box, meta=Polygon) \
                                   .compute(scheduler=scheduler)
    # construct raster
    xmin, ymin, xmax, ymax = points_df.longitude.min(), \
                             points_df.latitude.min(), \
                             points_df.longitude.max(), \
                             points_df.latitude.max()
    rows, cols, ras_trans = pts_to_raster_meta((xmin, ymin, xmax, ymax),
                                               (raster_res, -raster_res))
    raster_out = np.zeros((len(val_names), rows, cols))

    # TODO: parallel rasterize
    for i_val, val_name in enumerate(val_names):
        raster_out[i_val, :, :] = rasterio.features.rasterize(
            list(zip(df_poly.geometry, df_poly[val_name])),
            out_shape=(rows, cols),
            transform=ras_trans,
            fill=0,
            all_touched=True,
            dtype=rasterio.float32)

    meta = {
        'crs': points_df.crs,
        'height': rows,
        'width': cols,
        'transform': ras_trans,
    }
    return raster_out, meta

def set_df_geometry_points(df_val, scheduler=None):
    """ Set given geometry to given dataframe using dask if scheduler

    Parameters:
        df_val (DataFrame or GeoDataFrame): contains latitude and longitude columns
        scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
    """
    LOGGER.info('Setting geometry points.')
    def apply_point(df_exp):
        fun = lambda row: Point(row.longitude, row.latitude)
        return df_exp.apply(fun, axis=1)
    if not scheduler:
        df_val['geometry'] = apply_point(df_val)
    else:
        ddata = dd.from_pandas(df_val, npartitions=cpu_count())
        df_val['geometry'] = ddata.map_partitions(apply_point, meta=Point) \
                                  .compute(scheduler=scheduler)
